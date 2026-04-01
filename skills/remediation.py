# -----------------------------------------------------------------------------
# REMEDIATION SKILL  v3.5.0  — checkov policy context injection
# -----------------------------------------------------------------------------
# Changes from v3.4.0:
#   - _llm_fix now calls get_checkov_policy_context() for any CKV_* findings
#     and appends the policy source code block to the user message.
#     This mirrors the generate_security_remediation_feedback pattern from
#     IaCGen (Code/main.py) so the LLM sees the exact scan_resource_conf()
#     logic and knows precisely which CFN property path to fix.
# -----------------------------------------------------------------------------

import json
import time
from typing import Optional

from enums import SkillPhase
from god import GroundedObjectivesDocument, RemediationEntry, ValidationFinding
from llm_client import OpenRouterClient
from prompt import REMEDIATION_SYSTEM_PROMPT
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult
from telemetry import TelemetryRecorder
from checkov_context import get_checkov_policy_context

_REPLAN_PREFIXES = ("INTENT", "COVERAGE", "AC-")


class RemediationSkill(Skill):
    """
    Fixes a CloudFormation template that has failed validation.

    Two paths, both LLM-driven:
    1. In-place fix  -- security, schema, and YAML syntax findings.
    2. Re-plan       -- intent/coverage/AC-* findings.

    There is no round cap here.  The orchestrator's max_total_iterations is
    the only mechanism that stops the remediation loop.  This ensures the
    pipeline always attempts to reach SUCCEEDED (all validations PASS) while
    iterations remain.
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
                "Any FAIL validator has blocking findings AND template.body is non-empty"
            ),
            writes_to=["template.body", "template.version", "remediation_log"],
            reads_from=["template.body", "validation_state", "intent"],
            priority=10,
            version="3.5.0",
            tags=["llm", "remediation", "cloudformation"],
        )

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        # No round cap -- the orchestrator iteration budget is the only stop.
        return god.has_remediable_failures() and bool(god.template.body)

    # =========================================================================
    # Main execution
    # =========================================================================

    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        skill_result = SkillResult(success=True, skill_name=self.metadata.name)
        round_num = god.get_remediation_round() + 1
        self._logger.info(f"Remediation round {round_num}")
        tel: Optional[TelemetryRecorder] = context.get_config("telemetry")
        iteration: int = context.iteration

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

        intent_findings = [
            f for f in blocking_findings
            if any(f.rule_id.startswith(p) for p in _REPLAN_PREFIXES)
        ]
        fixable_findings = [
            f for f in blocking_findings
            if not any(f.rule_id.startswith(p) for p in _REPLAN_PREFIXES)
        ]

        self._accumulate_hints(god, round_num, blocking_findings)

        if intent_findings and not fixable_findings:
            return self._reset_for_replanning(
                god, skill_result, round_num, intent_findings, tel, iteration
            )

        findings_to_fix = fixable_findings or blocking_findings
        return self._llm_fix(
            god, skill_result, llm, round_num, findings_to_fix, tel, iteration
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
        tel: Optional[TelemetryRecorder],
        iteration: int,
    ) -> SkillResult:
        # ------------------------------------------------------------------
        # Build findings text (blocking) + all_findings text (full picture)
        # ------------------------------------------------------------------
        findings_text = "\n".join(
            f"- [{f.severity.name}] {f.rule_id} on '{f.resource_name}': "
            f"{f.message}. Remediation hint: {f.remediation_hint or 'n/a'}"
            for f in findings
        )

        # All findings across every validator give the LLM the full picture
        # so it avoids introducing new issues while fixing existing ones.
        all_findings = [
            finding
            for result in god.validation_state.values()
            for finding in result.findings
        ]
        all_findings_text = "\n".join(
            f"- [{f.severity.name}] {f.rule_id} on '{f.resource_name}': {f.message}"
            for f in all_findings
        ) or "(none)"

        remediation_hints = getattr(god.template, "remediation_hints", "") or "(none)"

        # Full GOD snapshot so the LLM sees resources, constraints, and AC
        god_snapshot_json = json.dumps(god.snapshot(), indent=2, default=str)

        system = REMEDIATION_SYSTEM_PROMPT.format(
            god_snapshot=god_snapshot_json,
        )
        
        def _build_checkov_fallback_context(checkov_findings: list[ValidationFinding]) -> str:
            parts: list[str] = []
            for f in checkov_findings:
                parts.append(
                    "\n".join([
                        f"### {f.rule_id} on {f.resource_name}",
                        f"- Resource type: {f.resource_type}",
                        f"- Human-readable issue: {f.message or f.rule_id}",
                        f"- Fix target: {f.remediation_hint or 'See finding details in validator output'}",
                        (
                            f"- Check ID: {f.check_id}"
                            if getattr(f, "check_id", None) else ""
                        ),
                    ]).strip()
                )
            return "\n\n".join(parts)

        # ------------------------------------------------------------------
        # Checkov policy context — inject source code for CKV_* findings so
        # the LLM understands exactly which CFN property path must be fixed.
        # Mirrors generate_security_remediation_feedback from IaCGen.
        # ------------------------------------------------------------------
        checkov_findings = [f for f in findings if (f.rule_id or "").startswith("CKV_")]
        policy_context_block = ""
        policy_context_injected = False
        if checkov_findings:
            policy_context = get_checkov_policy_context(checkov_findings)
            fallback_context = _build_checkov_fallback_context(checkov_findings)
            effective_context = (policy_context or "").strip() or fallback_context
            policy_context_block = (
                "\n\n## Security Policy Reference\n"
                "Use the following security context to determine the exact "
                "CloudFormation property path and compliant value(s) that must be fixed.\n\n"
                f"{effective_context}\n"
            )
            policy_context_injected = bool(effective_context.strip())

        user_message = (
            f"## Current template (round {round_num})\n"
            f"```yaml\n{god.template.body}\n```\n\n"
            f"## Blocking findings to fix\n"
            f"{findings_text}\n\n"
            f"## All validation findings (context — do not introduce new ones)\n"
            f"{all_findings_text}\n\n"
            f"## Accumulated hints from previous rounds\n"
            f"{remediation_hints}"
            f"{policy_context_block}"
        )

        if checkov_findings and "## Security Policy Reference" not in user_message:
            self._logger.warning(
                "  [llm-fix] Expected security policy context in prompt but section "
                "header is missing after prompt assembly"
            )

        self._logger.info(
            f"  [llm-fix] Sending {len(findings)} findings to LLM (round {round_num})"
            + (f", {len(checkov_findings)} with policy context" if checkov_findings else "")
        )

        t0 = time.monotonic()
        try:
            fixed_template = llm.complete(
                system=system,
                user=user_message,
                temperature=0.0,
            )
            llm_ok = True
            llm_err = None
        except Exception as e:
            fixed_template = ""
            llm_ok = False
            llm_err = str(e)
        llm_ms = (time.monotonic() - t0) * 1000

        if tel:
            tel.record_llm_call(
                skill_name=self.metadata.name,
                iteration=iteration,
                call_purpose=f"remediate_round_{round_num}",
                system_prompt=system,
                user_message=user_message,
                raw_response=fixed_template,
                duration_ms=llm_ms,
                success=llm_ok,
                error=llm_err,
                extra={
                    "round_num": round_num,
                    "finding_count": len(findings),
                    "finding_ids": [f.rule_id for f in findings],
                    "all_finding_count": len(all_findings),
                    "checkov_policy_context_count": len(checkov_findings),
                    "checkov_finding_count": len(checkov_findings),
                    "checkov_policy_context_injected": policy_context_injected,
                    "checkov_policy_reference_present": "## Security Policy Reference" in user_message,
                },
            )

        if not llm_ok:
            skill_result.success = False
            skill_result.errors.append(f"LLM call failed: {llm_err}")
            return skill_result

        fixed_template = self._strip_fences(fixed_template)

        if "AWSTemplateFormatVersion" not in fixed_template:
            skill_result.success = False
            skill_result.errors.append(
                "LLM remediation did not return a valid CloudFormation template "
                "(missing AWSTemplateFormatVersion)."
            )
            return skill_result

        before_body = god.template.body

        # ------------------------------------------------------------------
        # No-op guard: detect when the LLM returned the template unchanged
        # ------------------------------------------------------------------
        if fixed_template == before_body:
            self._logger.warning(
                f"  [llm-fix] No-op detected on round {round_num}: "
                "LLM returned an identical template. Marking as failure."
            )
            if tel:
                tel.record_god_change(
                    field="template.body",
                    before=before_body,
                    after=fixed_template,
                    changed_by=self.metadata.name,
                    iteration=iteration,
                    noop=True,
                )
            god.add_remediation_entry(RemediationEntry(
                round=round_num,
                skill_name=self.metadata.name,
                action_type="patch",
                target="template",
                description=f"LLM returned unchanged template (no-op) on round {round_num}",
                rationale=f"Findings: {[f.rule_id for f in findings]}",
                findings_addressed=[f.rule_id for f in findings],
                success=False,
                strategy_type="llm_noop",
            ))
            skill_result.success = False
            skill_result.errors.append(
                f"Remediation round {round_num} produced no changes to the template. "
                "The LLM returned an identical body. Check findings and prompt."
            )
            return skill_result

        # ------------------------------------------------------------------
        # Apply the fix
        # ------------------------------------------------------------------
        god.template.body = fixed_template
        god.template.update_checksum()
        god.template.increment_version(self.metadata.name)
        god.reset_validations_from("yaml_syntax", self.metadata.name, skip_errored=True)

        if tel:
            tel.record_god_change(
                field="template.body",
                before=before_body,
                after=fixed_template,
                changed_by=self.metadata.name,
                iteration=iteration,
            )

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
    # Re-plan path
    # =========================================================================

    def _reset_for_replanning(
        self,
        god: GroundedObjectivesDocument,
        skill_result: SkillResult,
        round_num: int,
        intent_findings: list[ValidationFinding],
        tel: Optional[TelemetryRecorder],
        iteration: int,
    ) -> SkillResult:
        self._logger.info(
            f"  [replan] {len(intent_findings)} intent/coverage findings -- "
            "clearing template and resources for re-planning"
        )

        before_body = god.template.body
        before_resources = list(god.intent.resources) if god.intent.resources else []

        god.template.body = ""
        god.template.resources = {}
        god.template.increment_version(self.metadata.name)
        god.intent.resources = []
        god.intent.acceptance_criteria = []
        god.intent.parsed_at = None
        god.reset_validations_from("yaml_syntax", self.metadata.name, skip_errored=False)

        if tel:
            tel.record_god_change(
                field="template.body",
                before=before_body,
                after="(cleared for replan)",
                changed_by=self.metadata.name,
                iteration=iteration,
            )
            tel.record_god_change(
                field="intent.resources",
                before=[r.resource_type for r in before_resources],
                after="(cleared for replan)",
                changed_by=self.metadata.name,
                iteration=iteration,
            )

        god.add_remediation_entry(RemediationEntry(
            round=round_num,
            skill_name=self.metadata.name,
            action_type="plan",
            target="template",
            description=(
                f"{len(intent_findings)} intent/coverage findings -- "
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
            "Template and resources cleared -- re-planning triggered"
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
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
