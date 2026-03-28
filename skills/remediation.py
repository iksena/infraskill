# -----------------------------------------------------------------------------
# REMEDIATION SKILL  v2.4.0
# -----------------------------------------------------------------------------
#
# Execution order inside execute():
#
#   0. YAML syntax triage    — YAML001 findings -> LLM YAML-repair, return early
#   1. Intent triage         — INTENT/COVERAGE/AC-* findings -> re-engineer
#                              (clear template, reset resources, add context hints)
#   2. GOD-planning pass     — LLM writes RemediationPlan for security findings
#   3. Deterministic Tier    — known rule IDs patched without LLM
#   4. LLM fallback Tier     — unknown rule IDs patched by LLM
#   5. Commit to GOD
#   6. Escalate remainder    — only if nothing worked AND max rounds exceeded
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Callable, Optional
from enums import SkillPhase, ValidationStatus
from god import GroundedObjectivesDocument, RemediationEntry, ValidationFinding, ValidationResult
from llm_client import OpenRouterClient
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult


_TEMPLATE_LEVEL_RESOURCE_NAMES = {"(template)", "", "TEMPLATE", "template"}

# Rule ID prefixes produced by IntentAlignmentValidatorSkill
_INTENT_RULE_PREFIXES = ("INTENT", "COVERAGE", "AC-")


# =============================================================================
# Remediation plan
# =============================================================================

@dataclass
class RemediationPlan:
    strategy: str
    actions: list[dict] = field(default_factory=list)
    rationale: str = ""
    confidence: float = 1.0


_REMEDIATION_PLANNER_SYSTEM = """\
You are a CloudFormation *security* remediation planner.
Given a list of validation findings and the current template, produce a JSON
remediation plan.  Respond ONLY with a single valid JSON object.

IMPORTANT: This planner handles SECURITY and SCHEMA findings ONLY.
- If ANY finding has rule_id == "YAML001" the template is syntactically broken
  and cannot be security-patched.  In that case you MUST return
  strategy=escalate with rationale explaining that YAML repair is needed first.

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
- Use 'escalate' when findings cannot be auto-remediated.
- confidence is 0.0-1.0.
"""


_YAML_REPAIR_SYSTEM = """\
You are an AWS CloudFormation expert.
The template below has a YAML syntax error.
Fix the YAML structure so it becomes valid CloudFormation.

Rules:
- Return ONLY the complete fixed YAML.  No markdown fences, no prose.
- Preserve ALL resources and their intended properties.
- Do NOT add or remove resources.
- Do NOT add security changes — only fix YAML structure.
- The output must start with 'AWSTemplateFormatVersion:'.
- CRITICAL: Do NOT use YAML tag shorthand (!Ref, !Sub, !GetAtt, !Select,
  !Join, !If, !Equals, !Split, !FindInMap, !Base64, !Condition).
  These tags are rejected by the YAML parser used by this system.
  Use the equivalent Fn:: dict form instead:
    !Ref LogicalName          -> {Ref: LogicalName}
    !Sub 'string ${Var}'      -> {Fn::Sub: 'string ${Var}'}
    !GetAtt Res.Attr          -> {Fn::GetAtt: [Res, Attr]}
    !Select [0, list]         -> {Fn::Select: [0, list]}
    !Join [",", list]         -> {Fn::Join: [",", list]}
    !If [cond, a, b]          -> {Fn::If: [cond, a, b]}
    !Split [",", str]         -> {Fn::Split: [",", str]}
    !FindInMap [M, K1, K2]    -> {Fn::FindInMap: [M, K1, K2]}
    !Base64 value             -> {Fn::Base64: value}
"""


class RemediationSkill(Skill):
    """
    The Remediation Skill analyses validation failures and applies targeted fixes.

    Three distinct remediation paths:
    1. YAML repair     — YAML001 findings (syntax broken, LLM repairs in-place)
    2. Re-engineering  — INTENT/COVERAGE/AC-* findings (template missing resources
                         or structurally wrong; clear and re-engineer with hints)
    3. Security patch  — CKV_*/E-* findings (deterministic then LLM fallback)

    can_trigger() fires only when a FAIL validator has CRITICAL/HIGH findings
    AND remediation rounds < 5.
    """

    REMEDIATION_STRATEGIES: dict[str, Callable] = {}

    def __init__(self):
        super().__init__()
        self._register_strategies()

    def _register_strategies(self):
        self.REMEDIATION_STRATEGIES = {
            "CKV_AWS_18": self._fix_s3_encryption,
            "CKV_AWS_19": self._fix_s3_public_access,
            "CKV_AWS_21": self._fix_s3_versioning,
            "CKV_AWS_16": self._fix_rds_encryption,
            "CKV_AWS_17": self._fix_rds_public_access,
            "CKV_AWS_157": self._fix_rds_multi_az,
            "CKV_AWS_10": self._fix_vpc_dns_support,
            "CKV_AWS_11": self._fix_vpc_dns_hostnames,
            "CKV_AWS_23": self._fix_sg_description,
            "CKV_AWS_24": self._fix_sg_ssh_ingress,
            "CKV_AWS_25": self._fix_sg_rdp_ingress,
            "CKV_AWS_130": self._fix_subnet_public_ip,
            "CKV_AWS_45": self._fix_lambda_memory,
            "CKV_AWS_56": self._fix_lambda_timeout,
            "CKV_AWS_28": self._fix_dynamodb_pitr,
        }

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="remediation",
            description="Analyses validation failures and applies targeted fixes",
            phase=SkillPhase.REMEDIATION,
            trigger_condition=(
                "Any FAIL validator has blocking findings AND "
                "remediation rounds not exhausted"
            ),
            writes_to=["template.body", "template.resources", "remediation_log"],
            reads_from=["template.body", "validation_state"],
            priority=10,
            version="2.4.0",
            tags=["remediation", "deterministic", "llm-fallback", "god-planning"],
        )

    def load_level2(self) -> bool:
        if not self._level2_loaded:
            import yaml as _yaml
            self._yaml = _yaml
            self._level2_loaded = True
        return True

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        return (
            god.has_remediable_failures()
            and god.get_remediation_round() < 5
            and bool(god.template.body)
        )

    # =========================================================================
    # Main execution
    # =========================================================================

    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        skill_result = SkillResult(success=True, skill_name=self.metadata.name)
        round_num = god.get_remediation_round() + 1
        self._logger.info(f"Remediation round {round_num}")

        blocking_findings = god.get_blocking_findings()
        if not blocking_findings:
            skill_result.warnings.append("No blocking findings to remediate")
            return skill_result

        llm: Optional[OpenRouterClient] = context.get_config("llm")

        # ------------------------------------------------------------------
        # STEP 0: YAML syntax triage
        # ------------------------------------------------------------------
        yaml_findings = [f for f in blocking_findings if f.rule_id == "YAML001"]
        if yaml_findings:
            return self._handle_yaml_syntax_error(
                god, skill_result, llm, round_num, yaml_findings
            )

        # ------------------------------------------------------------------
        # STEP 1: Intent/coverage triage
        # Intent findings mean the template is structurally correct but missing
        # resources or intent fulfilment.  The right fix is re-engineering, not
        # patching individual properties.
        # ------------------------------------------------------------------
        intent_findings = [
            f for f in blocking_findings
            if any(f.rule_id.startswith(p) for p in _INTENT_RULE_PREFIXES)
        ]
        if intent_findings:
            return self._handle_intent_findings(
                god, skill_result, llm, round_num, intent_findings
            )

        # ------------------------------------------------------------------
        # STEP 2: GOD-planning pass (security/schema findings)
        # ------------------------------------------------------------------
        security_findings = [
            f for f in blocking_findings
            if not any(f.rule_id.startswith(p) for p in _INTENT_RULE_PREFIXES)
        ]
        plan = self._build_remediation_plan(god, security_findings, llm, round_num)

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
        # STEP 3: Parse template
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

        findings_by_resource: dict[str, list[ValidationFinding]] = {}
        for finding in security_findings:
            findings_by_resource.setdefault(finding.resource_name, []).append(finding)

        # ------------------------------------------------------------------
        # STEP 3a: Deterministic patches
        # ------------------------------------------------------------------
        if plan.strategy in ("deterministic", "llm_fallback"):
            for res_name, findings in findings_by_resource.items():
                is_template_level = res_name in _TEMPLATE_LEVEL_RESOURCE_NAMES
                if res_name not in resources and not is_template_level:
                    continue
                resource = resources.get(res_name, {})
                properties = resource.get("Properties", {})
                for finding in findings:
                    if finding.rule_id in self.REMEDIATION_STRATEGIES:
                        try:
                            fix_desc = self.REMEDIATION_STRATEGIES[finding.rule_id](
                                properties, finding
                            )
                            if fix_desc:
                                fixes_applied.append(f"{res_name}: {fix_desc}")
                                findings_addressed.append(finding.rule_id)
                                self._logger.info(
                                    f"  [deterministic] Fixed {finding.rule_id} on {res_name}"
                                )
                        except Exception as e:
                            self._logger.warning(f"  Failed to fix {finding.rule_id}: {e}")
                            unknown_findings.append(finding)
                    else:
                        self._logger.warning(
                            f"  No deterministic strategy for {finding.rule_id}"
                        )
                        unknown_findings.append(finding)

        # ------------------------------------------------------------------
        # STEP 3b: LLM fallback
        # ------------------------------------------------------------------
        if (plan.strategy == "llm_fallback" or unknown_findings) and llm is not None:
            llm_targets = unknown_findings if unknown_findings else security_findings
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
                        "Do not add comments or explanations.\n"
                        "CRITICAL: Do NOT use YAML tag shorthand (!Ref, !Sub, !GetAtt, etc.). "
                        "Use Fn:: dict form instead: "
                        "!Ref X -> {Ref: X}, "
                        "!Sub S -> {Fn::Sub: S}, "
                        "!GetAtt R.A -> {Fn::GetAtt: [R, A]}"
                    ),
                    user=(
                        f"Template:\n{god.template.body}\n\n"
                        f"Findings to fix:\n{findings_text}"
                    ),
                    temperature=0.0,
                )
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
        # STEP 4: Commit
        # ------------------------------------------------------------------
        if fixes_applied:
            try:
                import yaml as _yaml
                god.template.body = _yaml.dump(
                    template, default_flow_style=False, sort_keys=False
                )
                god.template.update_checksum()
                god.template.increment_version(self.metadata.name)
                god.reset_validations_from(
                    "yaml_syntax", self.metadata.name, skip_errored=True
                )

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
            f for f in security_findings if f.rule_id not in findings_addressed
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

    # =========================================================================
    # STEP 0: YAML syntax repair
    # =========================================================================

    def _handle_yaml_syntax_error(
        self,
        god: GroundedObjectivesDocument,
        skill_result: SkillResult,
        llm: Optional[OpenRouterClient],
        round_num: int,
        yaml_findings: list[ValidationFinding],
    ) -> SkillResult:
        self._logger.info("  [yaml-repair] YAML syntax error detected — using YAML repair path")
        error_msg = yaml_findings[0].message if yaml_findings else "unknown YAML error"

        god.add_remediation_entry(RemediationEntry(
            round=round_num,
            skill_name=self.metadata.name,
            action_type="plan",
            target="(planning)",
            description="GOD-planning: strategy=yaml_repair (YAML syntax error detected)",
            rationale=f"YAML001: {error_msg[:200]}",
            findings_addressed=["YAML001"],
            success=True,
            strategy_type="yaml_repair",
        ))

        if llm is None:
            self._logger.warning(
                "  [yaml-repair] No LLM available — resetting template for re-generation"
            )
            return self._reset_for_reengineering(god, skill_result, round_num, error_msg)

        try:
            repaired = llm.complete(
                system=_YAML_REPAIR_SYSTEM,
                user=(
                    f"YAML error: {error_msg}\n\n"
                    f"Broken template:\n{god.template.body}"
                ),
                temperature=0.0,
            )

            import re as _re
            repaired = _re.sub(
                r'^```[\w]*\n?|\n?```$', '', repaired.strip(),
                flags=_re.MULTILINE
            ).strip()

            self._yaml.safe_load(repaired)

            if "AWSTemplateFormatVersion" not in repaired and "Resources" not in repaired:
                raise ValueError(
                    "Repaired template missing AWSTemplateFormatVersion/Resources"
                )

            god.template.body = repaired
            god.template.update_checksum()
            god.template.increment_version(self.metadata.name)
            god.reset_validations_from(
                "yaml_syntax", self.metadata.name, skip_errored=True
            )

            god.add_remediation_entry(RemediationEntry(
                round=round_num,
                skill_name=self.metadata.name,
                action_type="patch",
                target="template",
                description="LLM repaired YAML syntax error",
                rationale=f"Fixed: {error_msg[:200]}",
                findings_addressed=["YAML001"],
                success=True,
                strategy_type="yaml_repair",
            ))
            skill_result.changes_made = ["YAML syntax repaired by LLM"]
            self._logger.info("  [yaml-repair] YAML repaired and committed to GOD")
            return skill_result

        except self._yaml.YAMLError as e:
            self._logger.warning(
                f"  [yaml-repair] LLM produced invalid YAML ({e}); "
                "falling back to re-engineering"
            )
        except Exception as e:
            self._logger.warning(
                f"  [yaml-repair] LLM repair failed ({e}); "
                "falling back to re-engineering"
            )

        return self._reset_for_reengineering(god, skill_result, round_num, error_msg)

    # =========================================================================
    # STEP 1: Intent/coverage re-engineering
    # =========================================================================

    def _handle_intent_findings(
        self,
        god: GroundedObjectivesDocument,
        skill_result: SkillResult,
        llm: Optional[OpenRouterClient],
        round_num: int,
        intent_findings: list[ValidationFinding],
    ) -> SkillResult:
        """
        Intent/coverage failures mean the template is missing resources or
        violates the user's intent.  Patching individual properties cannot fix
        this.  Instead we:
          1. Build a context hint string from the findings.
          2. Store the hint on the GOD so the engineer can use it.
          3. Clear the template and reset resource.generated flags so the
             engineer re-runs from scratch with the extra context.
        """
        self._logger.info(
            f"  [intent-reengineering] {len(intent_findings)} intent/coverage "
            "findings — triggering re-engineering"
        )

        hints = "\n".join(
            f"- [{f.rule_id}] {f.message}. Hint: {f.remediation_hint}"
            for f in intent_findings
        )

        # Attach hints to the GOD so the engineer system prompt picks them up.
        # Store under god.template.remediation_hints (str, may be empty initially).
        existing_hints = getattr(god.template, "remediation_hints", "") or ""
        god.template.remediation_hints = (
            (existing_hints + "\n" if existing_hints else "") +
            f"[Round {round_num} intent findings]\n{hints}"
        )

        god.add_remediation_entry(RemediationEntry(
            round=round_num,
            skill_name=self.metadata.name,
            action_type="plan",
            target="template",
            description=(
                f"Intent/coverage failure ({len(intent_findings)} findings) — "
                "clearing template for re-engineering with context hints"
            ),
            rationale=hints[:500],
            findings_addressed=[f.rule_id for f in intent_findings],
            success=True,
            strategy_type="intent_reengineering",
        ))

        return self._reset_for_reengineering(
            god, skill_result, round_num,
            f"{len(intent_findings)} intent/coverage findings"
        )

    def _reset_for_reengineering(
        self,
        god: GroundedObjectivesDocument,
        skill_result: SkillResult,
        round_num: int,
        error_msg: str,
    ) -> SkillResult:
        self._logger.info(
            "  Clearing template — general-engineer will re-run"
        )
        god.template.body = ""
        god.template.resources = {}
        god.template.increment_version(self.metadata.name)
        for resource in god.intent.resources:
            resource.generated = False
        god.reset_validations_from("yaml_syntax", self.metadata.name, skip_errored=False)
        god.add_remediation_entry(RemediationEntry(
            round=round_num,
            skill_name=self.metadata.name,
            action_type="patch",
            target="template",
            description="Cleared template; re-engineering will regenerate",
            rationale=f"Re-engineering triggered for: {error_msg[:200]}",
            findings_addressed=[],
            success=True,
            strategy_type="reengineering_reset",
        ))
        skill_result.changes_made = ["Template cleared — re-engineering triggered"]
        return skill_result

    # =========================================================================
    # GOD-planning: LLM decides strategy
    # =========================================================================

    def _build_remediation_plan(
        self,
        god: GroundedObjectivesDocument,
        findings: list[ValidationFinding],
        llm: Optional[OpenRouterClient],
        round_num: int,
    ) -> RemediationPlan:
        import json

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
            god.add_remediation_entry(RemediationEntry(
                round=round_num,
                skill_name=self.metadata.name,
                action_type="plan",
                target="(planning)",
                description=(
                    f"GOD-planning: strategy=deterministic, "
                    f"actions={len(plan.actions)}, confidence={plan.confidence}"
                ),
                rationale=plan.rationale,
                findings_addressed=[f.rule_id for f in findings],
                success=True,
                strategy_type="deterministic",
            ))
            return plan

        if llm is not None:
            findings_text = "\n".join(
                f"- [{f.severity.name}] {f.rule_id} on {f.resource_name}: {f.message}"
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

        plan = RemediationPlan(
            strategy="llm_fallback",
            rationale="LLM planning unavailable; using LLM fallback for unknown findings.",
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

    # =========================================================================
    # Deterministic fix strategies
    # =========================================================================

    def _fix_s3_encryption(self, props: dict, finding: ValidationFinding) -> str:
        props["BucketEncryption"] = {
            "ServerSideEncryptionConfiguration": [
                {"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
            ]
        }
        return "Added server-side encryption AES256"

    def _fix_s3_public_access(self, props: dict, finding: ValidationFinding) -> str:
        props["PublicAccessBlockConfiguration"] = {
            "BlockPublicAcls": True, "BlockPublicPolicy": True,
            "IgnorePublicAcls": True, "RestrictPublicBuckets": True,
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
            r for r in ingress
            if not (r.get("CidrIp") == "0.0.0.0/0" and r.get("FromPort", 0) == 22)
        ]
        return "Removed unrestricted SSH ingress"

    def _fix_sg_rdp_ingress(self, props: dict, finding: ValidationFinding) -> str:
        ingress = props.get("SecurityGroupIngress", [])
        props["SecurityGroupIngress"] = [
            r for r in ingress
            if not (r.get("CidrIp") == "0.0.0.0/0" and r.get("FromPort", 0) == 3389)
        ]
        return "Removed unrestricted RDP ingress"

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
        props["PointInTimeRecoverySpecification"] = {
            "PointInTimeRecoveryEnabled": True
        }
        return "Enabled Point-in-Time Recovery"
