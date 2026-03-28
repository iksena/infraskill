# -----------------------------------------------------------------------------
# PLANNING SKILLS
# -----------------------------------------------------------------------------

from datetime import datetime
import json
from typing import Optional

from enums import SkillPhase
from god import AcceptanceCriterion, Constraints, ExtractedResource, GroundedObjectivesDocument
from llm_client import OpenRouterClient
from prompt import PLANNER_SYSTEM_PROMPT
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult


# ---------------------------------------------------------------------------
# Offline regex fallback constants
# ---------------------------------------------------------------------------
# These are used ONLY when no LLM client is configured (unit tests, CI).
# All production runs use the LLM path.

_RESOURCE_PATTERNS: dict[str, tuple[str, int]] = {
    r'\b(vpc|virtual private cloud)\b':           ('AWS::EC2::VPC', 10),
    r'\b(iam|role|permission)\b':                 ('AWS::IAM::Role', 11),
    r'\b(kms|encryption key)\b':                  ('AWS::KMS::Key', 12),
    r'\b(subnet|subnets)\b':                      ('AWS::EC2::Subnet', 20),
    r'\b(security group|sg|firewall)\b':          ('AWS::EC2::SecurityGroup', 21),
    r'\b(internet gateway|igw)\b':                ('AWS::EC2::InternetGateway', 22),
    r'\b(nat gateway|nat)\b':                     ('AWS::EC2::NatGateway', 23),
    r'\b(route table|routing)\b':                 ('AWS::EC2::RouteTable', 24),
    r'\b(s3|bucket|object storage)\b':            ('AWS::S3::Bucket', 30),
    r'\b(efs|elastic file)\b':                    ('AWS::EFS::FileSystem', 31),
    r'\b(rds|database|mysql|postgres|aurora)\b':  ('AWS::RDS::DBInstance', 40),
    r'\b(dynamodb|dynamo)\b':                     ('AWS::DynamoDB::Table', 41),
    r'\b(lambda|function|serverless)\b':          ('AWS::Lambda::Function', 50),
    r'\b(ec2|instance|server)\b':                 ('AWS::EC2::Instance', 51),
    r'\b(ecs|container|fargate)\b':               ('AWS::ECS::Cluster', 52),
    r'\b(api gateway|rest api)\b':                ('AWS::ApiGateway::RestApi', 60),
    r'\b(alb|elb|load balancer)\b':               ('AWS::ElasticLoadBalancingV2::LoadBalancer', 61),
    r'\b(sqs|queue)\b':                           ('AWS::SQS::Queue', 62),
    r'\b(sns|topic|notification)\b':              ('AWS::SNS::Topic', 63),
    r'\b(cloudwatch|alarm|metric)\b':             ('AWS::CloudWatch::Alarm', 70),
}

_DEPENDENCIES: dict[str, list[str]] = {
    'AWS::EC2::Subnet':                            ['AWS::EC2::VPC'],
    'AWS::EC2::SecurityGroup':                     ['AWS::EC2::VPC'],
    'AWS::EC2::InternetGateway':                   ['AWS::EC2::VPC'],
    'AWS::EC2::NatGateway':                        ['AWS::EC2::VPC', 'AWS::EC2::Subnet', 'AWS::EC2::InternetGateway'],
    'AWS::EC2::RouteTable':                        ['AWS::EC2::VPC'],
    'AWS::RDS::DBInstance':                        ['AWS::EC2::VPC', 'AWS::EC2::Subnet', 'AWS::EC2::SecurityGroup'],
    'AWS::Lambda::Function':                       ['AWS::IAM::Role'],
    'AWS::EC2::Instance':                          ['AWS::EC2::VPC', 'AWS::EC2::Subnet', 'AWS::EC2::SecurityGroup', 'AWS::IAM::Role'],
    'AWS::ECS::Cluster':                           ['AWS::EC2::VPC'],
    'AWS::ElasticLoadBalancingV2::LoadBalancer':   ['AWS::EC2::VPC', 'AWS::EC2::Subnet', 'AWS::EC2::SecurityGroup'],
}


class PlannerSkill(Skill):
    """
    The Planner transforms a natural-language prompt into a machine-checkable
    specification stored in the GOD.

    Execution strategy
    ------------------
    1. LLM path (default when an LLM client is configured) — single structured
       JSON call to extract resources, constraints AND acceptance criteria in
       one shot.  Regex fallback is attempted only on JSON decode errors.
    2. Offline regex path — used when no LLM client is present (CI / unit tests).

    Progressive disclosure
    ----------------------
    L1  metadata  — always resident (routing)
    L2  imports   — ``re`` module imported lazily in load_level2()
    L3  (none)    — no heavy assets needed
    """

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="planner",
            version="2.0.0",
            description=(
                "Transforms a natural-language infrastructure prompt into a "
                "machine-checkable GOD specification: resource list, constraints, "
                "and acceptance criteria."
            ),
            llm_description=(
                "Given a natural-language prompt describing desired cloud "
                "infrastructure, extract: (1) the list of AWS resources needed, "
                "(2) constraints such as security, HA, compliance, and (3) "
                "binary acceptance criteria that can be checked against the "
                "generated CloudFormation template."
            ),
            phase=SkillPhase.PLANNING,
            trigger_condition="intent.raw_prompt exists AND intent.resources is empty",
            writes_to=[
                "intent.resources",
                "intent.constraints",
                "intent.acceptance_criteria",
                "intent.normalized_prompt",
                "intent.parsed_at",
                "intent.parser_version",
            ],
            reads_from=["intent.raw_prompt"],
            priority=10,
            tags=["llm", "planning", "extraction"],
            examples=[
                {
                    "input": "Create a private S3 bucket with versioning for production",
                    "output": {
                        "resources": [{"resource_type": "AWS::S3::Bucket", "logical_name": "MainS3Bucket", "priority": 30}],
                        "constraints": {"environment": "production", "public_access_allowed": False},
                        "acceptance_criteria": [{"id": "AC-1", "description": "S3 bucket blocks public ACLs", "check_type": "equals"}],
                    },
                }
            ],
        )

    # ------------------------------------------------------------------
    # L2: lazy-import re so the L1 metadata load is zero-cost
    # ------------------------------------------------------------------
    def load_level2(self) -> bool:
        if not self._level2_loaded:
            import re as _re
            self._re = _re
            self._level2_loaded = True
        return True

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        return bool(god.intent.raw_prompt) and not god.intent.resources

    # ------------------------------------------------------------------
    # Main execute — LLM-first, regex fallback
    # ------------------------------------------------------------------
    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        result = SkillResult(success=True, skill_name=self.metadata.name)
        llm: Optional[OpenRouterClient] = context.get_config("llm")

        if llm is None:
            self._logger.warning("No LLM client — falling back to offline regex planner.")
            return self._execute_regex(context)

        raw = llm.complete(
            system=PLANNER_SYSTEM_PROMPT,
            user=god.intent.raw_prompt,
            temperature=0.0,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._logger.warning(f"LLM returned invalid JSON ({exc}); falling back to regex.")
            return self._execute_regex(context)

        god.intent.resources = [ExtractedResource(**r) for r in data["resources"]]
        god.intent.constraints = Constraints(**data["constraints"])
        god.intent.acceptance_criteria = [
            AcceptanceCriterion(**c) for c in data["acceptance_criteria"]
        ]
        god.intent.resources = self._resolve_dependencies(god.intent.resources)
        god.intent.normalized_prompt = god.intent.raw_prompt.lower()
        god.intent.parsed_at = datetime.now().isoformat()
        god.intent.parser_version = self.metadata.version

        result.changes_made.append(
            f"LLM extracted {len(god.intent.resources)} resources, "
            f"{len(god.intent.acceptance_criteria)} acceptance criteria"
        )
        return result

    # ------------------------------------------------------------------
    # Offline regex fallback (no LLM)
    # ------------------------------------------------------------------
    def _execute_regex(self, context: SkillContext) -> SkillResult:
        """Regex-based planner used only when no LLM client is configured."""
        god = context.god
        result = SkillResult(success=True, skill_name=self.metadata.name)
        self.load_level2()  # ensure self._re is available

        prompt = god.intent.raw_prompt.lower()
        god.intent.normalized_prompt = prompt

        resources = self._regex_extract_resources(prompt)
        resources = self._resolve_dependencies(resources)
        god.intent.resources = resources
        result.changes_made.append(f"[regex] Extracted {len(resources)} resources")

        constraints = self._regex_infer_constraints(prompt)
        god.intent.constraints = constraints
        result.changes_made.append(f"[regex] Inferred constraints (env={constraints.environment})")

        criteria = self._regex_generate_acceptance_criteria(resources, constraints)
        god.intent.acceptance_criteria = criteria
        result.changes_made.append(f"[regex] Generated {len(criteria)} acceptance criteria")

        god.intent.parsed_at = datetime.now().isoformat()
        god.intent.parser_version = f"{self.metadata.version}-regex"
        return result

    # ------------------------------------------------------------------
    # Dependency resolution (shared by both paths)
    # ------------------------------------------------------------------
    def _resolve_dependencies(
        self, resources: list[ExtractedResource]
    ) -> list[ExtractedResource]:
        result = list(resources)
        current_types = {r.resource_type for r in result}

        changed = True
        while changed:
            changed = False
            for resource in list(result):
                for dep_type in _DEPENDENCIES.get(resource.resource_type, []):
                    if dep_type not in current_types:
                        priority = next(
                            (p for _, (rt, p) in _RESOURCE_PATTERNS.items() if rt == dep_type),
                            100,
                        )
                        short = dep_type.split("::")[-1]
                        result.append(ExtractedResource(
                            resource_type=dep_type,
                            logical_name=f"Main{short}",
                            priority=priority,
                        ))
                        current_types.add(dep_type)
                        changed = True
                        self._logger.debug(f"Added dependency: {dep_type}")
                    if dep_type not in resource.dependencies:
                        resource.dependencies.append(dep_type)

        result.sort(key=lambda r: r.priority)
        return result

    # ------------------------------------------------------------------
    # Regex helpers (offline fallback only)
    # ------------------------------------------------------------------
    def _regex_extract_resources(self, prompt: str) -> list[ExtractedResource]:
        resources, seen = [], set()
        for pattern, (resource_type, priority) in _RESOURCE_PATTERNS.items():
            if self._re.search(pattern, prompt, self._re.IGNORECASE):
                if resource_type not in seen:
                    seen.add(resource_type)
                    short = resource_type.split("::")[-1]
                    resources.append(ExtractedResource(
                        resource_type=resource_type,
                        logical_name=f"Main{short}",
                        priority=priority,
                    ))
        return resources

    def _regex_infer_constraints(self, prompt: str) -> Constraints:
        re = self._re
        c = Constraints()
        if re.search(r'\b(multi-?az|high.?availability|ha|redundant|fault.?tolerant)\b', prompt, re.I):
            c.multi_az = True
        if re.search(r'\b(encrypt|secure|kms|at.?rest|in.?transit)\b', prompt, re.I):
            c.encryption_at_rest = True
            c.encryption_in_transit = True
        if re.search(r'\b(private|no.?public|internal|isolated)\b', prompt, re.I):
            c.public_access_allowed = False
        if re.search(r'\b(public.?access|internet.?facing|external)\b', prompt, re.I):
            c.public_access_allowed = True
        if re.search(r'\b(backup|disaster.?recovery|dr|snapshot|restore)\b', prompt, re.I):
            c.backup_enabled = True
            c.backup_retention_days = 30
        if re.search(r'\b(log|audit|trail|monitor)\b', prompt, re.I):
            c.logging_enabled = True
        for fw, pat in [
            ('CIS-AWS', r'\b(cis|center for internet security)\b'),
            ('SOC2',    r'\b(soc.?2|soc2)\b'),
            ('HIPAA',   r'\b(hipaa|health)\b'),
            ('PCI-DSS', r'\b(pci|payment|card)\b'),
        ]:
            if re.search(pat, prompt, re.I):
                c.compliance_frameworks.append(fw)
        if re.search(r'\b(dev|development|test|staging|sandbox)\b', prompt, re.I):
            c.environment = 'development'
            c.multi_az = False
            c.backup_retention_days = 1
        elif re.search(r'\b(prod|production)\b', prompt, re.I):
            c.environment = 'production'
            c.multi_az = True
            c.backup_enabled = True
            c.logging_enabled = True
            c.monitoring_enabled = True
        return c

    def _regex_generate_acceptance_criteria(
        self,
        resources: list[ExtractedResource],
        constraints: Constraints,
    ) -> list[AcceptanceCriterion]:
        criteria, idx = [], 1

        for resource in resources:
            rt = resource.resource_type

            if rt == 'AWS::EC2::VPC':
                for desc, prop, val in [
                    ("VPC has DNS support enabled", "EnableDnsSupport", True),
                    ("VPC has DNS hostnames enabled", "EnableDnsHostnames", True),
                ]:
                    criteria.append(AcceptanceCriterion(
                        id=f"AC-{idx}", description=desc, resource_type=rt,
                        property_path=f"Properties.{prop}", expected_value=val, check_type="equals"
                    ))
                    idx += 1

            elif rt == 'AWS::S3::Bucket':
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{idx}", description="S3 bucket blocks public ACLs",
                    resource_type=rt,
                    property_path="Properties.PublicAccessBlockConfiguration.BlockPublicAcls",
                    expected_value=True, check_type="equals"
                ))
                idx += 1
                if constraints.encryption_at_rest:
                    criteria.append(AcceptanceCriterion(
                        id=f"AC-{idx}", description="S3 bucket has server-side encryption",
                        resource_type=rt, property_path="Properties.BucketEncryption",
                        check_type="exists"
                    ))
                    idx += 1

            elif rt == 'AWS::RDS::DBInstance':
                if constraints.multi_az:
                    criteria.append(AcceptanceCriterion(
                        id=f"AC-{idx}", description="RDS instance has Multi-AZ enabled",
                        resource_type=rt, property_path="Properties.MultiAZ",
                        expected_value=True, check_type="equals"
                    ))
                    idx += 1
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{idx}", description="RDS instance has encryption enabled",
                    resource_type=rt, property_path="Properties.StorageEncrypted",
                    expected_value=True, check_type="equals"
                ))
                idx += 1
                if not constraints.public_access_allowed:
                    criteria.append(AcceptanceCriterion(
                        id=f"AC-{idx}", description="RDS instance is not publicly accessible",
                        resource_type=rt, property_path="Properties.PubliclyAccessible",
                        expected_value=False, check_type="equals"
                    ))
                    idx += 1

            elif rt == 'AWS::EC2::SecurityGroup':
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{idx}", description="Security group has a description",
                    resource_type=rt, property_path="Properties.GroupDescription",
                    check_type="exists"
                ))
                idx += 1

            elif rt == 'AWS::IAM::Role':
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{idx}", description="IAM role has assume role policy document",
                    resource_type=rt, property_path="Properties.AssumeRolePolicyDocument",
                    check_type="exists"
                ))
                idx += 1

            elif rt == 'AWS::Lambda::Function':
                for desc, prop in [
                    ("Lambda function has execution role", "Role"),
                    ("Lambda function has runtime specified", "Runtime"),
                ]:
                    criteria.append(AcceptanceCriterion(
                        id=f"AC-{idx}", description=desc,
                        resource_type=rt, property_path=f"Properties.{prop}",
                        check_type="exists"
                    ))
                    idx += 1

        return criteria
