# -----------------------------------------------------------------------------
# REMEDIATION SKILL  v4.1.0  — LLM-assisted remediation routing
# -----------------------------------------------------------------------------

import json
import time
from typing import Optional

from checkov_context import get_checkov_policy_context
from enums import SkillPhase, ValidationStatus
from god import GroundedObjectivesDocument, RemediationEntry, ValidationFinding
from llm_client import OpenRouterClient
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult
from telemetry import TelemetryRecorder

_REPLAN_PREFIXES = ("INTENT", "COVERAGE", "AC-", "OBJ-")


_REMEDIATION_HINT_SYSTEM_PROMPT = """\
You are an AWS CloudFormation remediation assistant.

Task:
- Analyze validation failures (yaml syntax, cfn-lint, checkov).
- Produce concrete remediation guidance for the next template generation pass.

Output rules:
- Return plain text only (no markdown code fences).
- Be concise and actionable.
- Group by validator: YAML, CFN-LINT, CHECKOV.
- For each finding include:
    1) probable root cause,
    2) exact CloudFormation property path(s) to change,
    3) expected safe target value or structure.
- Preserve architecture intent; suggest minimal edits.
"""


class RemediationSkill(Skill):
    """
    Determines the next correction path after validation failures.

    The skill is deterministic and does not directly rewrite template bodies.
    It builds remediation hints from validator findings and then chooses one path:
    1) re-plan: clear intent + template so planner regenerates GOD
    2) re-engineer: keep intent, clear template so engineer regenerates template
    """

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="remediation",
            description=(
                "Builds remediation hints from validation failures with LLM "
                "assistance and chooses whether the next iteration should "
                "re-plan GOD or re-engineer the template."
            ),
            phase=SkillPhase.REMEDIATION,
            trigger_condition=(
                "Any FAIL validator has findings/errors AND template.body is non-empty"
            ),
            writes_to=[
                "template.remediation_hints",
                "intent.* (on re-plan)",
                "template.body (cleared)",
                "validation_state.* (reset)",
                "remediation_log",
            ],
            reads_from=["template.body", "validation_state", "intent"],
            priority=10,
            version="4.1.0",
            tags=["deterministic", "llm", "remediation", "routing"],
        )

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        if not god.template.body:
            return False
        return any(
            vr.status == ValidationStatus.FAIL and bool(vr.findings or vr.errors)
            for vr in god.validation_state.values()
        )

    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        result = SkillResult(success=True, skill_name=self.metadata.name)
        round_num = god.get_remediation_round() + 1
        tel: Optional[TelemetryRecorder] = context.get_config("telemetry")
        llm: Optional[OpenRouterClient] = context.get_config("llm")
        iteration: int = context.iteration
        max_output_tokens: int = context.get_config("remediation_max_output_tokens", 8192)

        relevant_findings = self._collect_relevant_findings(god)
        if not relevant_findings:
            result.warnings.append("No FAIL findings to remediate")
            return result

        llm_hints = self._build_llm_hints(
            god=god,
            findings=relevant_findings,
            llm=llm,
            telemetry=tel,
            iteration=iteration,
            max_output_tokens=max_output_tokens,
        )
        self._accumulate_hints(god, round_num, relevant_findings, llm_hints)
        if llm_hints:
            result.changes_made.append(
                "Generated LLM-assisted remediation guidance from YAML/cfn-lint/checkov failures"
            )
        else:
            result.warnings.append(
                "LLM remediation guidance unavailable; using deterministic findings-only hints"
            )

        intent_findings = [
            f for f in relevant_findings
            if any((f.rule_id or "").startswith(prefix) for prefix in _REPLAN_PREFIXES)
        ]

        if intent_findings:
            return self._route_to_planner(
                god=god,
                result=result,
                round_num=round_num,
                findings=intent_findings,
                telemetry=tel,
                iteration=iteration,
            )

        return self._route_to_engineer(
            god=god,
            result=result,
            round_num=round_num,
            findings=relevant_findings,
            telemetry=tel,
            iteration=iteration,
        )

    @staticmethod
    def _collect_relevant_findings(
        god: GroundedObjectivesDocument,
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        for validation in god.validation_state.values():
            if validation.status == ValidationStatus.FAIL:
                findings.extend(validation.findings)
        return findings

    def _build_llm_hints(
        self,
        god: GroundedObjectivesDocument,
        findings: list[ValidationFinding],
        llm: Optional[OpenRouterClient],
        telemetry: Optional[TelemetryRecorder],
        iteration: int,
        max_output_tokens: int,
    ) -> str:
        if llm is None:
            return ""

        validator_failures = {
            name: {
                "status": vr.status.value,
                "errors": vr.errors,
                "findings": [
                    {
                        "rule_id": f.rule_id,
                        "resource_name": f.resource_name,
                        "resource_type": f.resource_type,
                        "severity": f.severity.name,
                        "message": f.message,
                        "check_id": f.check_id,
                        "line_number": f.line_number,
                    }
                    for f in vr.findings
                ],
            }
            for name, vr in god.validation_state.items()
            if vr.status == ValidationStatus.FAIL
        }
        findings_json = json.dumps(validator_failures, indent=2)
        god_context = json.dumps(god.llm_context(), indent=2, default=str)
        template_text = god.template.body #[:_MAX_TEMPLATE_CHARS_FOR_LLM]
        checkov_policy_ctx = get_checkov_policy_context(findings)

        user_message = (
            "Provide remediation guidance for the failing IaC template.\n\n"
            "Use both grounded objectives and the template together when proposing fixes.\n\n"
            "## Validation failures\n"
            f"```json\n{findings_json}\n```\n\n"
            "## Current template\n"
            f"```yaml\n{template_text}\n```\n\n"
            "## Grounded objectives context\n"
            f"```json\n{god_context}\n```\n\n"
            "## Checkov policy source context\n"
            f"{checkov_policy_ctx or '(none)'}"
        )

        t0 = time.monotonic()
        try:
            hints = llm.complete(
                system=_REMEDIATION_HINT_SYSTEM_PROMPT,
                user=user_message,
                temperature=0.0,
                max_output_tokens=max_output_tokens,
            ).strip()
            llm_ok = True
            llm_err = None
        except Exception as exc:
            hints = ""
            llm_ok = False
            llm_err = str(exc)
        llm_ms = (time.monotonic() - t0) * 1000

        if telemetry:
            telemetry.record_llm_call(
                skill_name=self.metadata.name,
                iteration=iteration,
                call_purpose="remediation_hints",
                system_prompt=_REMEDIATION_HINT_SYSTEM_PROMPT,
                user_message=user_message,
                raw_response=hints,
                duration_ms=llm_ms,
                success=llm_ok,
                error=llm_err,
                extra={
                    "findings_count": len(findings),
                    "max_output_tokens": max_output_tokens,
                },
            )

        return hints if llm_ok else ""

    def _route_to_planner(
        self,
        god: GroundedObjectivesDocument,
        result: SkillResult,
        round_num: int,
        findings: list[ValidationFinding],
        telemetry: Optional[TelemetryRecorder],
        iteration: int,
    ) -> SkillResult:
        before_body = god.template.body
        before_objectives = [o.description for o in god.intent.objectives]
        god.template.previous_body = before_body

        god.template.body = ""
        god.template.resources = {}
        god.template.increment_version(self.metadata.name)

        god.intent.objectives = []
        god.intent.parsed_at = None

        god.reset_validations_from("yaml_syntax", self.metadata.name, skip_errored=False)

        if telemetry:
            telemetry.record_god_change(
                field="template.body",
                before=before_body,
                after="(cleared for planner re-run)",
                changed_by=self.metadata.name,
                iteration=iteration,
            )
            telemetry.record_god_change(
                field="intent.objectives",
                before=before_objectives,
                after="(cleared for planner re-run)",
                changed_by=self.metadata.name,
                iteration=iteration,
            )

        god.add_remediation_entry(RemediationEntry(
            round=round_num,
            skill_name=self.metadata.name,
            action_type="plan",
            target="intent+template",
            description="Remediation routed to planner (GOD changes required)",
            rationale=self._findings_summary(findings),
            findings_addressed=[f.rule_id for f in findings],
            success=True,
            strategy_type="deterministic_replan",
        ))

        result.next_skill_hint = "planner"
        result.changes_made.append(
            "Constructed remediation prompt and routed to planner"
        )
        return result

    def _route_to_engineer(
        self,
        god: GroundedObjectivesDocument,
        result: SkillResult,
        round_num: int,
        findings: list[ValidationFinding],
        telemetry: Optional[TelemetryRecorder],
        iteration: int,
    ) -> SkillResult:
        before_body = god.template.body
        god.template.previous_body = before_body

        # Keep intent/GOD as-is and force regeneration from engineer.
        god.template.body = ""
        god.template.resources = {}
        god.template.increment_version(self.metadata.name)
        god.reset_validations_from("yaml_syntax", self.metadata.name, skip_errored=True)

        if telemetry:
            telemetry.record_god_change(
                field="template.body",
                before=before_body,
                after="(cleared for engineer re-run)",
                changed_by=self.metadata.name,
                iteration=iteration,
            )

        god.add_remediation_entry(RemediationEntry(
            round=round_num,
            skill_name=self.metadata.name,
            action_type="patch",
            target="template",
            description="Remediation routed to engineer (template-only changes)",
            rationale=self._findings_summary(findings),
            findings_addressed=[f.rule_id for f in findings],
            success=True,
            strategy_type="deterministic_reengineer",
        ))

        result.next_skill_hint = "engineer"
        result.changes_made.append(
            "Constructed remediation prompt and routed to engineer"
        )
        return result

    @staticmethod
    def _findings_summary(findings: list[ValidationFinding]) -> str:
        return "\n".join(
            f"- [{f.rule_id}] {f.resource_name}: {f.message}" for f in findings
        )[:800]

    def _accumulate_hints(
        self,
        god: GroundedObjectivesDocument,
        round_num: int,
        findings: list[ValidationFinding],
        llm_hints: str,
    ) -> None:
        deterministic_hints = "\n".join(
            f"- [{f.rule_id}] on '{f.resource_name}': {f.message}"
            for f in findings
        )
        sections = [
            f"[Round {round_num}]",
            "[Deterministic findings]",
            deterministic_hints,
        ]
        if llm_hints:
            sections.extend([
                "[LLM-assisted guidance]",
                llm_hints,
            ])
        new_hints = "\n".join(sections)

        existing = getattr(god.template, "remediation_hints", "") or ""
        god.template.remediation_hints = (
            (existing + "\n" if existing else "") +
            new_hints
        )
