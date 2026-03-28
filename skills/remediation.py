# -----------------------------------------------------------------------------
# REMEDIATION SKILL
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Callable, Optional
from enums import SkillPhase
from god import GroundedObjectivesDocument, RemediationEntry, ValidationFinding
from llm_client import OpenRouterClient
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult


_TEMPLATE_LEVEL_RESOURCE_NAMES = {"(template)", "", "TEMPLATE"}

# -----------------------------------------------------------------------------
# Remediation plan produced by the LLM planning step
# -----------------------------------------------------------------------------

@dataclass
class RemediationPlan:
    """
    Structured remediation plan produced by the LLM *before* any patching.

    The LLM analyses all blocking findings against the current GOD state
    and returns a prioritised list of actions.  The plan is written back into
    the GOD (remediation_log) so downstream benchmark telemetry can inspect
    *what* was decided and *why*, independently of whether the patch succeeded.

    Fields
    ------
    strategy          : 'deterministic', 'llm_fallback', or 'escalate'
    actions           : ordered list of (rule_id, resource_name, description)
    rationale         : free-text explanation from the LLM
    confidence        : 0.0 – 1.0 self-reported confidence
    """
    strategy: str  # 'deterministic' | 'llm_fallback' | 'escalate'
    actions: list[dict] = field(default_factory=list)
    rationale: str = ""
    confidence: float = 1.0


_REMEDIATION_PLANNER_SYSTEM = """\
You are a CloudFormation remediation planner.
Given a list of validation findings and the current template, produce a JSON
remediation plan.  Respond ONLY with a single valid JSON object.

Output schema:
{
  "strategy": "deterministic" | "llm_fallback" | "escalate",
  "actions": [
    {"rule_id": "CKV_AWS_18", "resource_name": "MyBucket",
     "description": "Add server-side encryption", "patch_type": "add_property"}
  ],
  "rationale": "<why this strategy>",
  "confidence": 0.0
}

Rules:
- Use 'deterministic' when ALL findings have well-known deterministic fixes.
- Use 'llm_fallback' when at least one finding requires template-level reasoning.
- Use 'escalate' when findings cannot be auto-remediated (e.g. missing IAM policies
  that require human approval, or tool errors unrelated to template content).
- confidence is 0.0-1.0 (your self-assessed probability the plan will succeed).
"""


class RemediationSkill(Skill):
    """
    The Remediation Skill analyses validation failures and applies targeted fixes.

    Three-step execution
    --------------------
    1. **GOD-planning pass** — LLM reads all blocking findings + current template
       and writes a structured RemediationPlan into the GOD before any mutation.
       This makes the remediation strategy observable in telemetry/benchmarks.

    2. **Deterministic patches** — fast, precise, zero-LLM-cost for known rule IDs.

    3. **LLM fallback** — for unknown rule IDs or when the plan says 'llm_fallback'.

    Loop-guard invariants
    ---------------------
    - Only triggers when *fixable* failures exist (FAIL status, not ERROR).
    - Calls god.reset_validations_from(..., skip_errored=True) so validators
      that errored due to missing tools (checkov not on PATH, etc.) are NOT
      reset and do not re-trigger the remediation loop.
    - Each round appends a RemediationEntry with strategy_type so benchmark
      telemetry can distinguish deterministic vs LLM-driven rounds.
    """

    REMEDIATION_STRATEGIES: dict[str, Callable] = {}

    def __init__(self):
        super().__init__()
        self._register_strategies()

    def _register_strategies(self):
        self.REMEDIATION_STRATEGIES = {
            # S3
            "CKV_AWS_18": self._fix_s3_encryption,
            "CKV_AWS_19": self._fix_s3_public_access,
            "CKV_AWS_21": self._fix_s3_versioning,
            # RDS
            "CKV_AWS_16": self._fix_rds_encryption,
            "CKV_AWS_17": self._fix_rds_public_access,
            "CKV_AWS_157": self._fix_rds_multi_az,
            # VPC
            "CKV_AWS_10": self._fix_vpc_dns_support,
            "CKV_AWS_11": self._fix_vpc_dns_hostnames,
            # Security Groups
            "CKV_AWS_23": self._fix_sg_description,
            "CKV_AWS_24": self._fix_sg_ssh_ingress,
            "CKV_AWS_25": self._fix_sg_rdp_ingress,
            # Subnet
            "CKV_AWS_130": self._fix_subnet_public_ip,
            # Lambda
            "CKV_AWS_45": self._fix_lambda_memory,
            "CKV_AWS_56": self._fix_lambda_timeout,
            # DynamoDB
            "CKV_AWS_28": self._fix_dynamodb_pitr,
            # CloudFormation intent findings (AC-* ids from intent validator)
            # These cannot be patched deterministically — routed to LLM fallback
        }

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="remediation",
            description="Analyses validation failures and applies targeted fixes (plan → patch)",
            phase=SkillPhase.REMEDIATION,
            trigger_condition="Any FAIL validation exists and remediation rounds not exhausted",
            writes_to=["template.body", "template.resources", "remediation_log"],
            reads_from=["template.body", "validation_state"],
            priority=10,
            version="2.0.0",
            tags=["remediation", "deterministic", "llm-fallback", "god-planning"],
        )

    def load_level2(self) -> bool:
        """Lazy-import yaml once at activation."""
        if not self._level2_loaded:
            import yaml as _yaml
            self._yaml = _yaml
            self._level2_loaded = True
        return True

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        """
        Only trigger when there are genuine FAIL validations (not just ERROR).

        ERROR status means a tool is missing or crashed — re-running the
        template through the same broken tool will loop forever.  Those cases
        must be escalated, not retried.
        """
        from enums import ValidationStatus
        has_fixable_failures = any(
            v.status == ValidationStatus.FAIL
            for v in god.validation_state.values()
        )
        rounds_remaining = god.get_remediation_round() < 5
        has_template = bool(god.template.body)
        return has_fixable_failures and rounds_remaining and has_template

    # -------------------------------------------------------------------------
    # Main execution
    # -------------------------------------------------------------------------

    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        skill_result = SkillResult(success=True, skill_name=self.metadata.name)
        round_num = god.get_remediation_round() + 1
        self._logger.info(f"Remediation round {round_num}")

        blocking_findings = god.get_blocking_findings()
        if not blocking_findings:
            skill_result.warnings.append("No blocking findings to remediate")
            return skill_result

        # ------------------------------------------------------------------
        # STEP 1: GOD-planning pass (LLM analyses findings → writes plan)
        # ------------------------------------------------------------------
        llm: Optional[OpenRouterClient] = context.get_config("llm")
        plan = self._build_remediation_plan(god, blocking_findings, llm, round_num)

        # If plan says escalate, bail out immediately without touching the template
        if plan.strategy == "escalate":
            skill_result.requires_human_review = True
            skill_result.escalation_reason = (
                f"Remediation planner decided to escalate (round {round_num}): "
                f"{plan.rationale}"
            )
            god.add_remediation_entry(RemediationEntry(
                round=round_num,
                skill_name=self.metadata.name,
                action_type="escalate",
                target="(multiple)",
                description="Planner decided escalation is required",
                rationale=plan.rationale,
                findings_addressed=[],
                success=False,
                strategy_type="escalate",
            ))
            return skill_result

        # ------------------------------------------------------------------
        # STEP 2: Parse template
        # ------------------------------------------------------------------
        try:
            template = self._yaml.safe_load(god.template.body)
        except Exception as e:
            skill_result.success = False
            skill_result.errors.append(f"Cannot parse template: {e}")
            return skill_result

        resources = template.get("Resources", {})
        fixes_applied: list[str] = []
        findings_addressed: list[str] = []
        unknown_findings: list[ValidationFinding] = []

        # Group findings by resource
        findings_by_resource: dict[str, list[ValidationFinding]] = {}
        for finding in blocking_findings:
            findings_by_resource.setdefault(finding.resource_name, []).append(finding)

        # ------------------------------------------------------------------
        # STEP 3a: Deterministic patches (Tier 1)
        # ------------------------------------------------------------------
        if plan.strategy in ("deterministic", "llm_fallback"):
            for res_name, findings in findings_by_resource.items():
                is_template_level = res_name in _TEMPLATE_LEVEL_RESOURCE_NAMES
                if res_name not in resources and not is_template_level:
                    continue

                resource = resources.get(res_name, {})
                properties = resource.get("Properties", {})

                for finding in findings:
                    rule_id = finding.rule_id
                    if rule_id in self.REMEDIATION_STRATEGIES:
                        try:
                            fix_description = self.REMEDIATION_STRATEGIES[rule_id](
                                properties, finding
                            )
                            if fix_description:
                                fixes_applied.append(f"{res_name}: {fix_description}")
                                findings_addressed.append(rule_id)
                                self._logger.info(f"  [deterministic] Fixed {rule_id} on {res_name}")
                        except Exception as e:
                            self._logger.warning(f"  Failed to fix {rule_id}: {e}")
                            unknown_findings.append(finding)
                    else:
                        self._logger.warning(f"  No deterministic strategy for {rule_id}")
                        unknown_findings.append(finding)

        # ------------------------------------------------------------------
        # STEP 3b: LLM fallback for unknown rule IDs (Tier 2)
        # ------------------------------------------------------------------
        if (plan.strategy == "llm_fallback" or unknown_findings) and llm is not None:
            llm_targets = unknown_findings if unknown_findings else blocking_findings
            findings_text = "\n".join(
                f"- {f.severity.name} {f.rule_id} on {f.resource_name}: "
                f"{f.message}. Fix hint: {f.remediation_hint}"
                for f in llm_targets
            )
            try:
                fixed_yaml = llm.complete(
                    system=(
                        "You are a CloudFormation security expert. "
                        "Fix ONLY the listed findings. "
                        "Return ONLY the complete fixed YAML template. "
                        "Do not add comments or explanations."
                    ),
                    user=f"Template:\n{god.template.body}\n\nFindings to fix:\n{findings_text}",
                    temperature=0.0,
                )
                # Validate LLM output before committing
                self._yaml.safe_load(fixed_yaml)
                template = self._yaml.safe_load(fixed_yaml)
                resources = template.get("Resources", {})
                for f in llm_targets:
                    findings_addressed.append(f.rule_id)
                    fixes_applied.append(f"{f.resource_name}: LLM fixed {f.rule_id}")
                self._logger.info(
                    f"  [llm_fallback] Remediated {len(llm_targets)} findings"
                )
            except self._yaml.YAMLError as e:
                skill_result.warnings.append(
                    f"LLM remediation produced invalid YAML ({e}); skipped"
                )
            except Exception as e:
                skill_result.warnings.append(f"LLM fallback failed: {e}")

        # ------------------------------------------------------------------
        # STEP 4: Commit fixes to GOD
        # ------------------------------------------------------------------
        if fixes_applied:
            try:
                god.template.body = self._yaml.dump(
                    template, default_flow_style=False, sort_keys=False
                )
                god.template.update_checksum()
                god.template.increment_version(self.metadata.name)

                # KEY FIX: skip_errored=True ensures validators that errored
                # due to missing tools (checkov not on PATH, etc.) are NOT
                # reset and do not re-enter the loop.
                god.reset_validations_from("yaml_syntax", self.metadata.name, skip_errored=True)

                strategy_type = (
                    "deterministic" if not unknown_findings
                    else "llm_fallback" if llm is not None
                    else "deterministic_partial"
                )

                god.add_remediation_entry(RemediationEntry(
                    round=round_num,
                    skill_name=self.metadata.name,
                    action_type="patch",
                    target=", ".join(findings_by_resource.keys()),
                    description="; ".join(fixes_applied),
                    rationale=plan.rationale or "Two-tier automated remediation",
                    findings_addressed=findings_addressed,
                    success=True,
                    strategy_type=strategy_type,
                ))
                skill_result.changes_made = fixes_applied

            except Exception as e:
                skill_result.success = False
                skill_result.errors.append(f"Failed to serialize fixed template: {e}")
                return skill_result

        # ------------------------------------------------------------------
        # STEP 5: Escalate anything still unresolved
        # ------------------------------------------------------------------
        remaining = [
            f for f in blocking_findings if f.rule_id not in findings_addressed
        ]
        if remaining:
            skill_result.requires_human_review = True
            skill_result.escalation_reason = (
                f"Cannot automatically remediate {len(remaining)} findings: "
                f"{[f.rule_id for f in remaining[:5]]}"
            )
            god.add_remediation_entry(RemediationEntry(
                round=round_num,
                skill_name=self.metadata.name,
                action_type="escalate",
                target="(multiple)",
                description="Could not automatically remediate all findings",
                rationale=skill_result.escalation_reason,
                findings_addressed=[],
                success=False,
                strategy_type="escalate",
            ))

        return skill_result

    # -------------------------------------------------------------------------
    # GOD-planning: LLM decides strategy before patching
    # -------------------------------------------------------------------------

    def _build_remediation_plan(
        self,
        god: GroundedObjectivesDocument,
        findings: list[ValidationFinding],
        llm: Optional[OpenRouterClient],
        round_num: int,
    ) -> RemediationPlan:
        """
        Ask the LLM to produce a structured remediation plan and write it back
        into the GOD before any template mutation occurs.

        Falls back to a deterministic plan when:
        - No LLM client is configured, OR
        - All findings have registered deterministic strategies, OR
        - The LLM call fails.
        """
        import json

        # Fast path: all findings have deterministic strategies
        all_deterministic = all(
            f.rule_id in self.REMEDIATION_STRATEGIES for f in findings
        )
        if all_deterministic:
            plan = RemediationPlan(
                strategy="deterministic",
                actions=[
                    {"rule_id": f.rule_id, "resource_name": f.resource_name,
                     "description": f"Apply registered patch for {f.rule_id}",
                     "patch_type": "deterministic"}
                    for f in findings
                ],
                rationale="All findings have registered deterministic patches.",
                confidence=1.0,
            )
            self._logger.info(
                f"  [plan] strategy=deterministic "
                f"({len(findings)} findings, round {round_num})"
            )
            # Write plan as a PLANNING-type remediation entry so telemetry picks it up
            god.add_remediation_entry(RemediationEntry(
                round=round_num,
                skill_name=self.metadata.name,
                action_type="plan",
                target="(planning)",
                description=f"GOD-planning: strategy=deterministic, "
                            f"actions={len(plan.actions)}, confidence={plan.confidence}",
                rationale=plan.rationale,
                findings_addressed=[f.rule_id for f in findings],
                success=True,
                strategy_type="deterministic",
            ))
            return plan

        # LLM planning pass
        if llm is not None:
            findings_text = "\n".join(
                f"- [{f.severity.name}] {f.rule_id} on {f.resource_name}: "
                f"{f.message}"
                for f in findings
            )
            known_ids = list(self.REMEDIATION_STRATEGIES.keys())
            try:
                raw = llm.complete(
                    system=_REMEDIATION_PLANNER_SYSTEM,
                    user=(
                        f"Findings (round {round_num}):\n{findings_text}\n\n"
                        f"Known deterministic rule IDs: {known_ids}\n\n"
                        f"Template (first 1500 chars):\n"
                        f"{god.template.body[:1500]}"
                    ),
                    temperature=0.0,
                )
                data = json.loads(raw)
                plan = RemediationPlan(
                    strategy=data.get("strategy", "llm_fallback"),
                    actions=data.get("actions", []),
                    rationale=data.get("rationale", ""),
                    confidence=float(data.get("confidence", 0.5)),
                )
                self._logger.info(
                    f"  [plan] LLM strategy={plan.strategy} "
                    f"confidence={plan.confidence:.2f} round={round_num}"
                )
                # Write plan into GOD so telemetry can inspect it
                god.add_remediation_entry(RemediationEntry(
                    round=round_num,
                    skill_name=self.metadata.name,
                    action_type="plan",
                    target="(planning)",
                    description=(
                        f"GOD-planning: strategy={plan.strategy}, "
                        f"actions={len(plan.actions)}, "
                        f"confidence={plan.confidence:.2f}"
                    ),
                    rationale=plan.rationale,
                    findings_addressed=[f.rule_id for f in findings],
                    success=True,
                    strategy_type=plan.strategy,
                ))
                return plan
            except (json.JSONDecodeError, KeyError, Exception) as e:
                self._logger.warning(
                    f"  [plan] LLM planning failed ({e}), defaulting to llm_fallback"
                )

        # Default fallback plan
        plan = RemediationPlan(
            strategy="llm_fallback",
            rationale="LLM planning unavailable; applying LLM fallback for unknown findings.",
            confidence=0.5,
        )
        god.add_remediation_entry(RemediationEntry(
            round=round_num,
            skill_name=self.metadata.name,
            action_type="plan",
            target="(planning)",
            description="GOD-planning: strategy=llm_fallback (default)",
            rationale=plan.rationale,
            findings_addressed=[f.rule_id for f in findings],
            success=True,
            strategy_type="llm_fallback",
        ))
        return plan

    # -------------------------------------------------------------------------
    # Deterministic fix strategies
    # -------------------------------------------------------------------------

    def _fix_s3_encryption(self, props: dict, finding: ValidationFinding) -> str:
        props["BucketEncryption"] = {
            "ServerSideEncryptionConfiguration": [
                {"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
            ]
        }
        return "Added server-side encryption AES256"

    def _fix_s3_public_access(self, props: dict, finding: ValidationFinding) -> str:
        props["PublicAccessBlockConfiguration"] = {
            "BlockPublicAcls": True,
            "BlockPublicPolicy": True,
            "IgnorePublicAcls": True,
            "RestrictPublicBuckets": True,
        }
        return "Added public access block configuration"

    def _fix_s3_versioning(self, props: dict, finding: ValidationFinding) -> str:
        props["VersioningConfiguration"] = {"Status": "Enabled"}
        return "Enabled versioning"

    def _fix_rds_encryption(self, props: dict, finding: ValidationFinding) -> str:
        props["StorageEncrypted"] = True
        return "Enabled storage encryption"

    def _fix_rds_public_access(self, props: dict, finding: ValidationFinding) -> str:
        props["PubliclyAccessible"] = False
        return "Disabled public accessibility"

    def _fix_rds_multi_az(self, props: dict, finding: ValidationFinding) -> str:
        props["MultiAZ"] = True
        return "Enabled Multi-AZ"

    def _fix_vpc_dns_support(self, props: dict, finding: ValidationFinding) -> str:
        props["EnableDnsSupport"] = True
        return "Enabled DNS support"

    def _fix_vpc_dns_hostnames(self, props: dict, finding: ValidationFinding) -> str:
        props["EnableDnsHostnames"] = True
        return "Enabled DNS hostnames"

    def _fix_sg_description(self, props: dict, finding: ValidationFinding) -> str:
        if not props.get("GroupDescription"):
            props["GroupDescription"] = "Security group managed by INFRA-SKILL"
        return "Added group description"

    def _fix_sg_ssh_ingress(self, props: dict, finding: ValidationFinding) -> str:
        ingress = props.get("SecurityGroupIngress", [])
        props["SecurityGroupIngress"] = [
            rule for rule in ingress
            if not (rule.get("CidrIp") == "0.0.0.0/0"
                    and rule.get("FromPort", 0) == 22)
        ]
        return "Removed unrestricted SSH (0.0.0.0/0:22) ingress rule"

    def _fix_sg_rdp_ingress(self, props: dict, finding: ValidationFinding) -> str:
        ingress = props.get("SecurityGroupIngress", [])
        props["SecurityGroupIngress"] = [
            rule for rule in ingress
            if not (rule.get("CidrIp") == "0.0.0.0/0"
                    and rule.get("FromPort", 0) == 3389)
        ]
        return "Removed unrestricted RDP (0.0.0.0/0:3389) ingress rule"

    def _fix_subnet_public_ip(self, props: dict, finding: ValidationFinding) -> str:
        props["MapPublicIpOnLaunch"] = False
        return "Disabled auto-assign public IP"

    def _fix_lambda_memory(self, props: dict, finding: ValidationFinding) -> str:
        if "MemorySize" not in props:
            props["MemorySize"] = 256
        return "Set memory size to 256MB"

    def _fix_lambda_timeout(self, props: dict, finding: ValidationFinding) -> str:
        if "Timeout" not in props:
            props["Timeout"] = 30
        return "Set timeout to 30s"

    def _fix_dynamodb_pitr(self, props: dict, finding: ValidationFinding) -> str:
        props["PointInTimeRecoverySpecification"] = {"PointInTimeRecoveryEnabled": True}
        return "Enabled Point-in-Time Recovery"
