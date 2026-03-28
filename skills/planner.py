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


class PlannerSkill(Skill):
    """
    The Planner transforms a natural-language prompt into a machine-checkable
    specification in the GOD.
    
    Three-pass decomposition:
    1. Resource Extraction - identify AWS resources needed
    2. Constraint Inference - extract security, HA, compliance requirements
    3. Acceptance Criteria Generation - create binary, checkable criteria
    """
    
    # Resource patterns: regex -> (resource_type, priority)
    # Priority determines generation order (lower = first)
    RESOURCE_PATTERNS = {
        # Foundation (priority 10-19)
        r'\b(vpc|virtual private cloud)\b': ('AWS::EC2::VPC', 10),
        r'\b(iam|role|permission)\b': ('AWS::IAM::Role', 11),
        r'\b(kms|encryption key)\b': ('AWS::KMS::Key', 12),
        
        # Network (priority 20-29)
        r'\b(subnet|subnets)\b': ('AWS::EC2::Subnet', 20),
        r'\b(security group|sg|firewall)\b': ('AWS::EC2::SecurityGroup', 21),
        r'\b(internet gateway|igw)\b': ('AWS::EC2::InternetGateway', 22),
        r'\b(nat gateway|nat)\b': ('AWS::EC2::NatGateway', 23),
        r'\b(route table|routing)\b': ('AWS::EC2::RouteTable', 24),
        
        # Storage (priority 30-39)
        r'\b(s3|bucket|object storage)\b': ('AWS::S3::Bucket', 30),
        r'\b(efs|elastic file)\b': ('AWS::EFS::FileSystem', 31),
        
        # Database (priority 40-49)
        r'\b(rds|database|mysql|postgres|aurora)\b': ('AWS::RDS::DBInstance', 40),
        r'\b(dynamodb|dynamo)\b': ('AWS::DynamoDB::Table', 41),
        
        # Compute (priority 50-59)
        r'\b(lambda|function|serverless)\b': ('AWS::Lambda::Function', 50),
        r'\b(ec2|instance|server)\b': ('AWS::EC2::Instance', 51),
        r'\b(ecs|container|fargate)\b': ('AWS::ECS::Cluster', 52),
        
        # Application (priority 60-69)
        r'\b(api gateway|rest api)\b': ('AWS::ApiGateway::RestApi', 60),
        r'\b(alb|elb|load balancer)\b': ('AWS::ElasticLoadBalancingV2::LoadBalancer', 61),
        r'\b(sqs|queue)\b': ('AWS::SQS::Queue', 62),
        r'\b(sns|topic|notification)\b': ('AWS::SNS::Topic', 63),
        
        # Monitoring (priority 70-79)
        r'\b(cloudwatch|alarm|metric)\b': ('AWS::CloudWatch::Alarm', 70),
    }
    
    # Dependencies: resource_type -> list of required resource types
    DEPENDENCIES = {
        'AWS::EC2::Subnet': ['AWS::EC2::VPC'],
        'AWS::EC2::SecurityGroup': ['AWS::EC2::VPC'],
        'AWS::EC2::InternetGateway': ['AWS::EC2::VPC'],
        'AWS::EC2::NatGateway': ['AWS::EC2::VPC', 'AWS::EC2::Subnet', 'AWS::EC2::InternetGateway'],
        'AWS::EC2::RouteTable': ['AWS::EC2::VPC'],
        'AWS::RDS::DBInstance': ['AWS::EC2::VPC', 'AWS::EC2::Subnet', 'AWS::EC2::SecurityGroup'],
        'AWS::Lambda::Function': ['AWS::IAM::Role'],
        'AWS::EC2::Instance': ['AWS::EC2::VPC', 'AWS::EC2::Subnet', 'AWS::EC2::SecurityGroup', 'AWS::IAM::Role'],
        'AWS::ECS::Cluster': ['AWS::EC2::VPC'],
        'AWS::ElasticLoadBalancingV2::LoadBalancer': ['AWS::EC2::VPC', 'AWS::EC2::Subnet', 'AWS::EC2::SecurityGroup'],
    }
    
    # Constraint patterns
    CONSTRAINT_PATTERNS = {
        'multi_az': r'\b(multi-?az|high.?availability|ha|redundant|fault.?tolerant)\b',
        'encryption': r'\b(encrypt|secure|kms|at.?rest|in.?transit)\b',
        'private': r'\b(private|no.?public|internal|isolated)\b',
        'public': r'\b(public.?access|internet.?facing|external)\b',
        'backup': r'\b(backup|disaster.?recovery|dr|snapshot|restore)\b',
        'logging': r'\b(log|audit|trail|monitor)\b',
        'compliance_cis': r'\b(cis|center for internet security)\b',
        'compliance_soc2': r'\b(soc.?2|soc2)\b',
        'compliance_hipaa': r'\b(hipaa|health)\b',
        'compliance_pci': r'\b(pci|payment|card)\b',
        'production': r'\b(prod|production)\b',
        'development': r'\b(dev|development|test|staging|sandbox)\b',
    }
    
    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="planner",
            description="Extracts resources, constraints, and acceptance criteria from user prompt",
            llm_description="Given a natural language prompt describing desired infrastructure, extract a list of AWS resources, any constraints (e.g. security, HA), and generate acceptance criteria that can be used for validation.",
            phase=SkillPhase.PLANNING,
            trigger_condition="intent.raw_prompt exists but resources is empty",
            writes_to=["intent.resources", "intent.constraints", "intent.acceptance_criteria", "intent.parsed_at"],
            reads_from=["intent.raw_prompt"],
            priority=10
        )
    
    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        return bool(god.intent.raw_prompt) and not god.intent.resources
    
    def _execute_regex(self, context: SkillContext) -> SkillResult:
        god = context.god
        result = SkillResult(success=True, skill_name=self.metadata.name)
        
        prompt = god.intent.raw_prompt.lower()
        god.intent.normalized_prompt = prompt
        
        # Pass 1: Resource Extraction
        self._logger.info("Pass 1: Extracting resources...")
        resources = self._extract_resources(prompt)
        resources = self._resolve_dependencies(resources)
        god.intent.resources = resources
        result.changes_made.append(f"Extracted {len(resources)} resources")
        
        # Pass 2: Constraint Inference
        self._logger.info("Pass 2: Inferring constraints...")
        constraints = self._infer_constraints(prompt)
        god.intent.constraints = constraints
        result.changes_made.append(f"Inferred constraints (env={constraints.environment})")
        
        # Pass 3: Acceptance Criteria Generation
        self._logger.info("Pass 3: Generating acceptance criteria...")
        criteria = self._generate_acceptance_criteria(resources, constraints)
        god.intent.acceptance_criteria = criteria
        result.changes_made.append(f"Generated {len(criteria)} acceptance criteria")
        
        god.intent.parsed_at = datetime.now().isoformat()
        god.intent.parser_version = "1.0.0"
        
        return result

    # PlannerSkill
    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        result = SkillResult(success=True, skill_name=self.metadata.name)
        llm: Optional[OpenRouterClient] = context.get_config("llm")

        if llm is None:
            # Offline fallback — existing regex logic
            return self._execute_regex(context)

        raw = llm.complete(
            system=PLANNER_SYSTEM_PROMPT,   # a module-level constant string
            user=god.intent.raw_prompt,
            temperature=0.0,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            self._logger.warning(f"LLM returned invalid JSON: {e}. Falling back to regex.")
            return self._execute_regex(context)
        god.intent.resources = [ExtractedResource(**r) for r in data["resources"]]
        god.intent.constraints = Constraints(**data["constraints"])
        god.intent.acceptance_criteria = [AcceptanceCriterion(**c) for c in data["acceptance_criteria"]]
        god.intent.resources = self._resolve_dependencies(god.intent.resources)
        god.intent.parsed_at = datetime.now().isoformat()
        result.changes_made.append(f"LLM extracted {len(god.intent.resources)} resources")
        return result
    
    def _extract_resources(self, prompt: str) -> list[ExtractedResource]:
        """Extract AWS resources mentioned in the prompt"""
        import re
        resources = []
        seen_types = set()
        
        for pattern, (resource_type, priority) in self.RESOURCE_PATTERNS.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                if resource_type not in seen_types:
                    seen_types.add(resource_type)
                    
                    short_name = resource_type.split("::")[-1]
                    resources.append(ExtractedResource(
                        resource_type=resource_type,
                        logical_name=f"Main{short_name}",
                        priority=priority
                    ))
        
        return resources
    
    def _resolve_dependencies(self, resources: list[ExtractedResource]) -> list[ExtractedResource]:
        """Add missing dependencies and set up dependency links"""
        result = list(resources)
        current_types = {r.resource_type for r in result}
        
        # Keep adding dependencies until no new ones needed
        changed = True
        while changed:
            changed = False
            for resource in list(result):
                deps = self.DEPENDENCIES.get(resource.resource_type, [])
                for dep_type in deps:
                    if dep_type not in current_types:
                        # Find priority from patterns
                        priority = 100
                        for pattern, (rt, p) in self.RESOURCE_PATTERNS.items():
                            if rt == dep_type:
                                priority = p
                                break
                        
                        short_name = dep_type.split("::")[-1]
                        result.append(ExtractedResource(
                            resource_type=dep_type,
                            logical_name=f"Main{short_name}",
                            priority=priority
                        ))
                        current_types.add(dep_type)
                        changed = True
                        self._logger.debug(f"Added dependency: {dep_type}")
                    
                    if dep_type not in resource.dependencies:
                        resource.dependencies.append(dep_type)
        
        # Sort by priority
        result.sort(key=lambda r: r.priority)
        return result
    
    def _infer_constraints(self, prompt: str) -> Constraints:
        """Infer constraints from the prompt"""
        import re
        constraints = Constraints()
        
        # Multi-AZ
        if re.search(self.CONSTRAINT_PATTERNS['multi_az'], prompt):
            constraints.multi_az = True
        
        # Encryption
        if re.search(self.CONSTRAINT_PATTERNS['encryption'], prompt):
            constraints.encryption_at_rest = True
            constraints.encryption_in_transit = True
        
        # Access
        if re.search(self.CONSTRAINT_PATTERNS['private'], prompt):
            constraints.public_access_allowed = False
        if re.search(self.CONSTRAINT_PATTERNS['public'], prompt):
            constraints.public_access_allowed = True
        
        # Backup
        if re.search(self.CONSTRAINT_PATTERNS['backup'], prompt):
            constraints.backup_enabled = True
            constraints.backup_retention_days = 30
        
        # Logging
        if re.search(self.CONSTRAINT_PATTERNS['logging'], prompt):
            constraints.logging_enabled = True
        
        # Compliance
        if re.search(self.CONSTRAINT_PATTERNS['compliance_cis'], prompt):
            constraints.compliance_frameworks.append('CIS-AWS')
        if re.search(self.CONSTRAINT_PATTERNS['compliance_soc2'], prompt):
            constraints.compliance_frameworks.append('SOC2')
        if re.search(self.CONSTRAINT_PATTERNS['compliance_hipaa'], prompt):
            constraints.compliance_frameworks.append('HIPAA')
        if re.search(self.CONSTRAINT_PATTERNS['compliance_pci'], prompt):
            constraints.compliance_frameworks.append('PCI-DSS')
        
        # Environment
        if re.search(self.CONSTRAINT_PATTERNS['development'], prompt):
            constraints.environment = 'development'
            constraints.multi_az = False
            constraints.backup_retention_days = 1
        elif re.search(self.CONSTRAINT_PATTERNS['production'], prompt):
            constraints.environment = 'production'
            constraints.multi_az = True
            constraints.backup_enabled = True
            constraints.logging_enabled = True
            constraints.monitoring_enabled = True
        
        return constraints
    
    def _generate_acceptance_criteria(
        self,
        resources: list[ExtractedResource],
        constraints: Constraints
    ) -> list[AcceptanceCriterion]:
        """Generate binary, checkable acceptance criteria"""
        criteria = []
        idx = 1
        
        for resource in resources:
            rt = resource.resource_type
            
            if rt == 'AWS::EC2::VPC':
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{idx}",
                    description="VPC has DNS support enabled",
                    resource_type=rt,
                    property_path="Properties.EnableDnsSupport",
                    expected_value=True,
                    check_type="equals"
                ))
                idx += 1
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{idx}",
                    description="VPC has DNS hostnames enabled",
                    resource_type=rt,
                    property_path="Properties.EnableDnsHostnames",
                    expected_value=True,
                    check_type="equals"
                ))
                idx += 1
                
            elif rt == 'AWS::S3::Bucket':
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{idx}",
                    description="S3 bucket blocks public ACLs",
                    resource_type=rt,
                    property_path="Properties.PublicAccessBlockConfiguration.BlockPublicAcls",
                    expected_value=True,
                    check_type="equals"
                ))
                idx += 1
                if constraints.encryption_at_rest:
                    criteria.append(AcceptanceCriterion(
                        id=f"AC-{idx}",
                        description="S3 bucket has server-side encryption",
                        resource_type=rt,
                        property_path="Properties.BucketEncryption",
                        check_type="exists"
                    ))
                    idx += 1
                if constraints.logging_enabled:
                    criteria.append(AcceptanceCriterion(
                        id=f"AC-{idx}",
                        description="S3 bucket has versioning enabled",
                        resource_type=rt,
                        property_path="Properties.VersioningConfiguration.Status",
                        expected_value="Enabled",
                        check_type="equals"
                    ))
                    idx += 1
                    
            elif rt == 'AWS::RDS::DBInstance':
                if constraints.multi_az:
                    criteria.append(AcceptanceCriterion(
                        id=f"AC-{idx}",
                        description="RDS instance has Multi-AZ enabled",
                        resource_type=rt,
                        property_path="Properties.MultiAZ",
                        expected_value=True,
                        check_type="equals"
                    ))
                    idx += 1
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{idx}",
                    description="RDS instance has encryption enabled",
                    resource_type=rt,
                    property_path="Properties.StorageEncrypted",
                    expected_value=True,
                    check_type="equals"
                ))
                idx += 1
                if not constraints.public_access_allowed:
                    criteria.append(AcceptanceCriterion(
                        id=f"AC-{idx}",
                        description="RDS instance is not publicly accessible",
                        resource_type=rt,
                        property_path="Properties.PubliclyAccessible",
                        expected_value=False,
                        check_type="equals"
                    ))
                    idx += 1
                    
            elif rt == 'AWS::EC2::SecurityGroup':
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{idx}",
                    description="Security group has a description",
                    resource_type=rt,
                    property_path="Properties.GroupDescription",
                    check_type="exists"
                ))
                idx += 1
                
            elif rt == 'AWS::EC2::Subnet':
                if not constraints.public_access_allowed:
                    criteria.append(AcceptanceCriterion(
                        id=f"AC-{idx}",
                        description="Subnet does not auto-assign public IPs",
                        resource_type=rt,
                        property_path="Properties.MapPublicIpOnLaunch",
                        expected_value=False,
                        check_type="equals"
                    ))
                    idx += 1
                    
            elif rt == 'AWS::IAM::Role':
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{idx}",
                    description="IAM role has assume role policy document",
                    resource_type=rt,
                    property_path="Properties.AssumeRolePolicyDocument",
                    check_type="exists"
                ))
                idx += 1
                
            elif rt == 'AWS::Lambda::Function':
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{idx}",
                    description="Lambda function has execution role",
                    resource_type=rt,
                    property_path="Properties.Role",
                    check_type="exists"
                ))
                idx += 1
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{idx}",
                    description="Lambda function has runtime specified",
                    resource_type=rt,
                    property_path="Properties.Runtime",
                    check_type="exists"
                ))
                idx += 1
        
        return criteria
