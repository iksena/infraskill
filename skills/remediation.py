# -----------------------------------------------------------------------------
# REMEDIATION SKILL
# -----------------------------------------------------------------------------

from typing import Callable, Optional
from enums import SkillPhase
from god import GroundedObjectivesDocument, RemediationEntry, ValidationFinding
from llm_client import OpenRouterClient
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult


_TEMPLATE_LEVEL_RESOURCE_NAMES = {"(template)", "", "TEMPLATE"}


class RemediationSkill(Skill):
    """
    Analyzes validation failures and applies targeted fixes.

    Two-tier strategy
    -----------------
    1. Deterministic patches — fast, precise, zero LLM cost for known rule IDs.
    2. LLM fallback — for unknown rule IDs that have no registered strategy.

    Progressive disclosure
    ----------------------
    L1  metadata — always resident
    L2  yaml     — imported lazily in load_level2()
    L3  (none)
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
        }

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="remediation",
            version="1.1.0",
            description=(
                "Analyzes validation failures and applies targeted fixes. "
                "Uses deterministic patches for known rule IDs and LLM fallback "
                "for unknown rules."
            ),
            llm_description=(
                "When deterministic strategies are exhausted, sends the full "
                "template and remaining findings to the LLM and asks for a "
                "corrected YAML output."
            ),
            phase=SkillPhase.REMEDIATION,
            trigger_condition="Any validation has FAILED (god.has_failed_validations() is True)",
            writes_to=["template.body", "template.resources", "remediation_log"],
            reads_from=["template.body", "validation_state"],
            priority=10,
            tags=["remediation", "deterministic", "llm-fallback"],
        )

    # L2: lazy-import yaml
    def load_level2(self) -> bool:
        if not self._level2_loaded:
            import yaml as _yaml
            self._yaml = _yaml
            self._level2_loaded = True
        return True

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        return god.has_failed_validations()

    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        skill_result = SkillResult(success=True, skill_name=self.metadata.name)
        round_num = god.get_remediation_round() + 1
        self._logger.info(f"Remediation round {round_num}")

        blocking_findings = god.get_blocking_findings()
        if not blocking_findings:
            skill_result.warnings.append("No blocking findings to remediate")
            return skill_result

        try:
            template = self._yaml.safe_load(god.template.body)
        except Exception as exc:
            skill_result.success = False
            skill_result.errors.append(f"Cannot parse template: {exc}")
            return skill_result

        resources = template.get("Resources", {})
        fixes_applied: list[str] = []
        findings_addressed: list[str] = []
        unknown_findings: list[ValidationFinding] = []

        findings_by_resource: dict[str, list[ValidationFinding]] = {}
        for finding in blocking_findings:
            findings_by_resource.setdefault(finding.resource_name, []).append(finding)

        # --- Tier 1: Deterministic patches --------------------------------
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
                        fix_desc = self.REMEDIATION_STRATEGIES[rule_id](properties, finding)
                        if fix_desc:
                            fixes_applied.append(f"{res_name}: {fix_desc}")
                            findings_addressed.append(rule_id)
                            self._logger.info(f"  Fixed {rule_id} on {res_name}")
                    except Exception as exc:
                        self._logger.warning(f"  Failed to apply fix {rule_id}: {exc}")
                        unknown_findings.append(finding)
                else:
                    self._logger.warning(f"  No deterministic strategy for {rule_id}")
                    unknown_findings.append(finding)

        # --- Tier 2: LLM fallback -----------------------------------------
        llm: Optional[OpenRouterClient] = context.get_config("llm")
        if unknown_findings and llm is not None:
            findings_text = "\n".join(
                f"- {f.severity.name} {f.rule_id} on {f.resource_name}: "
                f"{f.message}. Fix hint: {f.remediation_hint}"
                for f in unknown_findings
            )
            try:
                fixed_yaml = llm.complete(
                    system=(
                        "You are a CloudFormation security expert. "
                        "Fix ONLY the listed findings. "
                        "Return ONLY the complete fixed YAML template. "
                        "Do not add comments or explanations."
                    ),
                    user=(
                        f"Template:\n{god.template.body}\n\n"
                        f"Findings:\n{findings_text}"
                    ),
                    temperature=0.0,
                )
                self._yaml.safe_load(fixed_yaml)  # validate before committing
                template = self._yaml.safe_load(fixed_yaml)
                resources = template.get("Resources", {})
                for f in unknown_findings:
                    findings_addressed.append(f.rule_id)
                    fixes_applied.append(f"{f.resource_name}: LLM fixed {f.rule_id}")
                self._logger.info(
                    f"  LLM remediated {len(unknown_findings)} unknown findings"
                )
            except self._yaml.YAMLError as exc:
                skill_result.warnings.append(
                    f"LLM remediation produced invalid YAML ({exc}); skipped"
                )
            except Exception as exc:
                skill_result.warnings.append(f"LLM fallback failed: {exc}")

        # --- Commit -------------------------------------------------------
        if fixes_applied:
            try:
                god.template.body = self._yaml.dump(
                    template, default_flow_style=False, sort_keys=False
                )
                god.template.update_checksum()
                god.template.increment_version(self.metadata.name)
                god.reset_validations_from("yaml_syntax", self.metadata.name)

                god.add_remediation_entry(RemediationEntry(
                    round=round_num,
                    skill_name=self.metadata.name,
                    action_type="patch",
                    target=", ".join(findings_by_resource.keys()),
                    description="; ".join(fixes_applied),
                    rationale="Two-tier automated remediation (deterministic + LLM fallback)",
                    findings_addressed=findings_addressed,
                    success=True,
                ))
                skill_result.changes_made = fixes_applied

            except Exception as exc:
                skill_result.success = False
                skill_result.errors.append(f"Failed to serialize fixed template: {exc}")
                return skill_result

        # --- Escalate remaining -------------------------------------------
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
            ))

        return skill_result

    # ---- Deterministic fix strategies ------------------------------------

    def _fix_s3_encryption(self, props: dict, _: ValidationFinding) -> str:
        props["BucketEncryption"] = {
            "ServerSideEncryptionConfiguration": [
                {"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
            ]
        }
        return "Added server-side encryption AES256"

    def _fix_s3_public_access(self, props: dict, _: ValidationFinding) -> str:
        props["PublicAccessBlockConfiguration"] = {
            "BlockPublicAcls": True,
            "BlockPublicPolicy": True,
            "IgnorePublicAcls": True,
            "RestrictPublicBuckets": True,
        }
        return "Added public access block configuration"

    def _fix_s3_versioning(self, props: dict, _: ValidationFinding) -> str:
        props["VersioningConfiguration"] = {"Status": "Enabled"}
        return "Enabled versioning"

    def _fix_rds_encryption(self, props: dict, _: ValidationFinding) -> str:
        props["StorageEncrypted"] = True
        return "Enabled storage encryption"

    def _fix_rds_public_access(self, props: dict, _: ValidationFinding) -> str:
        props["PubliclyAccessible"] = False
        return "Disabled public accessibility"

    def _fix_rds_multi_az(self, props: dict, _: ValidationFinding) -> str:
        props["MultiAZ"] = True
        return "Enabled Multi-AZ"

    def _fix_vpc_dns_support(self, props: dict, _: ValidationFinding) -> str:
        props["EnableDnsSupport"] = True
        return "Enabled DNS support"

    def _fix_vpc_dns_hostnames(self, props: dict, _: ValidationFinding) -> str:
        props["EnableDnsHostnames"] = True
        return "Enabled DNS hostnames"

    def _fix_sg_description(self, props: dict, _: ValidationFinding) -> str:
        if not props.get("GroupDescription"):
            props["GroupDescription"] = "Security group managed by INFRA-SKILL"
        return "Added group description"

    def _fix_sg_ssh_ingress(self, props: dict, _: ValidationFinding) -> str:
        ingress = props.get("SecurityGroupIngress", [])
        props["SecurityGroupIngress"] = [
            rule for rule in ingress
            if not (
                rule.get("CidrIp") == "0.0.0.0/0"
                and rule.get("FromPort", 0) == 22
            )
        ]
        return "Removed unrestricted SSH (0.0.0.0/0:22) ingress rule"

    def _fix_sg_rdp_ingress(self, props: dict, _: ValidationFinding) -> str:
        ingress = props.get("SecurityGroupIngress", [])
        props["SecurityGroupIngress"] = [
            rule for rule in ingress
            if not (
                rule.get("CidrIp") == "0.0.0.0/0"
                and rule.get("FromPort", 0) == 3389
            )
        ]
        return "Removed unrestricted RDP (0.0.0.0/0:3389) ingress rule"

    def _fix_subnet_public_ip(self, props: dict, _: ValidationFinding) -> str:
        props["MapPublicIpOnLaunch"] = False
        return "Disabled auto-assign public IP"

    def _fix_lambda_memory(self, props: dict, _: ValidationFinding) -> str:
        if "MemorySize" not in props:
            props["MemorySize"] = 256
        return "Set memory size to 256MB"

    def _fix_lambda_timeout(self, props: dict, _: ValidationFinding) -> str:
        if "Timeout" not in props:
            props["Timeout"] = 30
        return "Set timeout to 30s"
