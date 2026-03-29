# -----------------------------------------------------------------------------
# REMEDIATION SKILL  v3.0.0  — pure LLM, no deterministic patches
# -----------------------------------------------------------------------------
#
# All remediation is performed by the LLM. The skill sends the full template
# and all blocking validation findings to the LLM and expects back a COMPLETE
# corrected CloudFormation template.
#
# For intent/coverage failures the template AND intent.resources are cleared
# so PlannerSkill.can_trigger() fires on the next iteration — the planner
# re-runs with all accumulated remediation_hints as extra context.
#
# There is no deterministic patch dict, no regex, no yaml.dump, no triage on
# rule_id prefixes. The LLM handles all finding types uniformly.
# -----------------------------------------------------------------------------

from typing import Optional

from enums import SkillPhase
from god import GroundedObjectivesDocument, RemediationEntry, ValidationFinding
from llm_client import OpenRouterClient
from prompt import REMEDIATION_SYSTEM_PROMPT
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult

# Rule ID prefixes that indicate the template must be regenerated from scratch
# (missing resources / intent violations) rather than patched in place.
_REPLAN_PREFIXES = ("INTENT", "COVERAGE", "AC-")


class RemediationSkill(Skill):
    """
    Fixes a CloudFormation template that has failed validation.

    Two paths, both LLM-driven:

    1. In-place fix  — security, schema, and YAML syntax findings.
       The LLM receives the full template + findings and returns the
       complete corrected template.

    2. Re-plan       — intent/coverage/AC-* findings.
       The template AND intent.resources are cleared so PlannerSkill
       re-triggers on the next pipeline iteration, incorporating all
       accumulated remediation_hints.
    """

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="remediation",
            description=(
                "Fixes a CloudFormation template that has failed validation. "
                "Uses the LLM to produce a corrected complete template."
            ),
            phase=SkillPhase.REMEDIATION,
            trigger_condition=(
                "Any FAIL validator has blocking findings AND "
                "remediation rounds < 5 AND template.body is non-empty"
            ),
            writes_to=["template.body", "template.version", "remediation_log"],
            reads_from=["template.body", "validation_state"],
            priority=10,
            version="3.0.0",
            tags=["llm", "remediation", "cloudformation"],
        )

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
        if llm is None:
            return SkillResult.failure(
                self.metadata.name,
                "No LLM client configured. RemediationSkill requires an LLM.",
            )

        # ------------------------------------------------------------------
        # Decide path: intent/coverage findings require a full re-plan;
        # all other findings (YAML syntax, security, schema) are fixed
        # in-place by the LLM.
        # ------------------------------------------------------------------
        intent_findings = [
            f for f in blocking_findings
            if any(f.rule_id.startswith(p) for p in _REPLAN_PREFIXES)
        ]
        fixable_findings = [
            f for f in blocking_findings
            if not any(f.rule_id.startswith(p) for p in _REPLAN_PREFIXES)
        ]

        # Accumulate remediation hints for both paths.
        self._accumulate_hints(god, round_num, blocking_findings)

        if intent_findings and not fixable_findings:
            # All findings are intent-level — re-plan from scratch.
            return self._reset_for_replanning(god, skill_result, round_num, intent_findings)

        # For mixed findings, include intent findings in the LLM fix prompt
        # so the LLM can add missing resources as well as fix security issues.
        findings_to_fix = fixable_findings or blocking_findings

        return self._llm_fix(
            god, skill_result, llm, round_num, findings_to_fix
        )

    # =========================================================================
    # LLM in-place fix
    # =========================================================================

    def _llm_fix(
        self,
        god: GroundedObjectivesDocument,
        skill_result: SkillResult,
        llm: OpenRouterClient,
        round_num: int,
        findings: list[ValidationFinding],
    ) -> SkillResult:
        findings_text = "\n".join(
            f"- [{f.severity.name}] {f.rule_id} on '{f.resource_name}': "
            f"{f.message}. Remediation hint: {f.remediation_hint or 'n/a'}"
            for f in findings
        )

        remediation_hints = getattr(god.template, "remediation_hints", "") or "(none)"

        self._logger.info(
            f"  [llm-fix] Sending {len(findings)} findings to LLM (round {round_num})"
        )

        try:
            fixed_template = llm.complete(
                system=REMEDIATION_SYSTEM_PROMPT,
                user=(
                    f"## Current template\n"
                    f"```yaml\n{god.template.body}\n```\n\n"
                    f"## Validation findings to fix (round {round_num})\n"
                    f"{findings_text}\n\n"
                    f"## Accumulated context from previous rounds\n"
                    f"{remediation_hints}"
                ),
                temperature=0.0,
            )
        except Exception as e:
            skill_result.success = False
            skill_result.errors.append(f"LLM call failed: {e}")
            return skill_result

        fixed_template = self._strip_fences(fixed_template)

        if "AWSTemplateFormatVersion" not in fixed_template:
            skill_result.success = False
            skill_result.errors.append(
                "LLM remediation did not return a valid CloudFormation template "
                "(missing AWSTemplateFormatVersion)."
            )
            return skill_result

        god.template.body = fixed_template
        god.template.update_checksum()
        god.template.increment_version(self.metadata.name)
        god.reset_validations_from("yaml_syntax", self.metadata.name, skip_errored=True)

        god.add_remediation_entry(RemediationEntry(
            round=round_num,
            skill_name=self.metadata.name,
            action_type="patch",
            target="template",
            description=f"LLM fixed {len(findings)} findings in-place",
            rationale=f"Findings: {[f.rule_id for f in findings]}",
            findings_addressed=[f.rule_id for f in findings],
            success=True,
            strategy_type="llm",
        ))

        skill_result.changes_made = [
            f"LLM remediated {len(findings)} findings (round {round_num})"
        ]
        self._logger.info(f"  [llm-fix] Template updated successfully")
        return skill_result

    # =========================================================================
    # Re-plan path (intent / coverage failures)
    # =========================================================================

    def _reset_for_replanning(
        self,
        god: GroundedObjectivesDocument,
        skill_result: SkillResult,
        round_num: int,
        intent_findings: list[ValidationFinding],
    ) -> SkillResult:
        """
        Clear both the assembled template AND intent.resources so that
        PlannerSkill.can_trigger() returns True on the next iteration.
        The planner re-runs with god.template.remediation_hints as context.
        """
        self._logger.info(
            f"  [replan] {len(intent_findings)} intent/coverage findings — "
            "clearing template and resources for re-planning"
        )

        god.template.body = ""
        god.template.resources = {}
        god.template.increment_version(self.metadata.name)

        # Clear intent so PlannerSkill.can_trigger() fires.
        god.intent.resources = []
        god.intent.acceptance_criteria = []
        god.intent.parsed_at = None

        god.reset_validations_from("yaml_syntax", self.metadata.name, skip_errored=False)

        god.add_remediation_entry(RemediationEntry(
            round=round_num,
            skill_name=self.metadata.name,
            action_type="plan",
            target="template",
            description=(
                f"{len(intent_findings)} intent/coverage findings — "
                "cleared template and resources; planner will re-run"
            ),
            rationale="\n".join(
                f"- [{f.rule_id}] {f.message}" for f in intent_findings
            )[:500],
            findings_addressed=[f.rule_id for f in intent_findings],
            success=True,
            strategy_type="replan",
        ))

        skill_result.changes_made = [
            "Template and resources cleared — re-planning triggered"
        ]
        return skill_result

    # =========================================================================
    # Helpers
    # =========================================================================

    def _accumulate_hints(
        self,
        god: GroundedObjectivesDocument,
        round_num: int,
        findings: list[ValidationFinding],
    ) -> None:
        """Append current findings as remediation hints on the GOD template."""
        new_hints = "\n".join(
            f"- [{f.rule_id}] on '{f.resource_name}': {f.message}. "
            f"Hint: {f.remediation_hint or 'n/a'}"
            for f in findings
        )
        existing = getattr(god.template, "remediation_hints", "") or ""
        god.template.remediation_hints = (
            (existing + "\n" if existing else "") +
            f"[Round {round_num}]\n{new_hints}"
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove optional markdown code fences from an LLM response."""
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
