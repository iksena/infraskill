# -----------------------------------------------------------------------------
# REMEDIATION SKILL  v3.4.0
# -----------------------------------------------------------------------------
# Changes vs v3.3.0:
#   - Added _deterministic_sanitise() pre-pass that regex-replaces known-bad
#     YAML patterns BEFORE the LLM output is written to the GOD.
#     Handles:
#       1. List-item inline dict: `- Key: Val: ...` -> `- Key:\n    Val: ...`
#       2. YAML tag shorthand:   `!Ref X`  -> `Ref: X`
#                                `!Sub X`  -> `Fn::Sub: X`
#                                `!GetAtt X.Y` -> `Fn::GetAtt: [X, Y]`
#   - _llm_fix() calls _deterministic_sanitise() after _strip_fences() and
#     logs how many substitutions were made.
#   - No round cap (inherited from v3.3.0).
# -----------------------------------------------------------------------------

import re
import time
from typing import Optional

from enums import SkillPhase
from god import GroundedObjectivesDocument, RemediationEntry, ValidationFinding
from llm_client import OpenRouterClient
from prompt import REMEDIATION_SYSTEM_PROMPT
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult
from telemetry import TelemetryRecorder

_REPLAN_PREFIXES = ("INTENT", "COVERAGE", "AC-")

# ---------------------------------------------------------------------------
# Regex patterns for the deterministic YAML sanitiser
# ---------------------------------------------------------------------------

# Pattern 1 -- list-item with an inline key:value that itself contains a colon
# e.g.  `          - AWS::StackName: Ref: AWS::StackName`
#  ->   `          - Ref: AWS::StackName`
# More generally:  `- <ns>::<Name>: <fn>: <value>`  where the outer key is
# a pseudo-namespace that should not be there at all.
_RE_LIST_INLINE_NS = re.compile(
    r'^([ \t]*)-[ \t]+[A-Za-z0-9_:]+:[ \t]+((?:Ref|Fn::[A-Za-z]+):[ \t]*.+)$',
    re.MULTILINE,
)

# Pattern 2 -- YAML tag shorthand  !Ref X
_RE_TAG_REF = re.compile(r'!Ref[ \t]+(\S+)')

# Pattern 3 -- YAML tag shorthand  !Sub '...'  or  !Sub "..."
_RE_TAG_SUB = re.compile(r"!Sub[ \t]+(['\"]?.+?['\"]?)(?=$|\n)")

# Pattern 4 -- YAML tag shorthand  !GetAtt Logical.Attr
_RE_TAG_GETATT = re.compile(r'!GetAtt[ \t]+([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)')

# Pattern 5 -- YAML tag shorthand  !Join, !Select, !If, !Split, !FindInMap,
#              !Base64, !ImportValue, !Equals, !Condition, !Transform
#              (convert to flow-mapping dict form on the same line)
_RE_TAG_OTHER = re.compile(
    r'!(Join|Select|If|Split|FindInMap|Base64|ImportValue|Equals|Condition|Transform)'
    r'([ \t]+)',
)


def _deterministic_sanitise(template: str) -> tuple[str, int]:
    """
    Apply regex-based fixes for known-bad YAML patterns that LLMs commonly
    produce.  Returns (sanitised_template, substitution_count).

    This runs AFTER _strip_fences() and BEFORE the template is written to the
    GOD.  It is a best-effort pass: it catches the high-frequency regressions
    without attempting to fully parse the YAML.
    """
    count = 0
    out = template

    # --- 1. List-item with spurious namespace key -------------------------
    # `          - AWS::StackName: Ref: AWS::StackName`
    # becomes
    # `          - Ref: AWS::StackName`
    def _fix_list_inline(m: re.Match) -> str:
        indent = m.group(1)
        intrinsic = m.group(2)  # e.g. "Ref: AWS::StackName"
        return f"{indent}- {intrinsic}"

    new_out, n = _RE_LIST_INLINE_NS.subn(_fix_list_inline, out)
    count += n
    out = new_out

    # --- 2. !Ref X  ->  Ref: X ------------------------------------------
    new_out, n = _RE_TAG_REF.subn(lambda m: f"Ref: {m.group(1)}", out)
    count += n
    out = new_out

    # --- 3. !Sub '...'  ->  Fn::Sub: '...' -------------------------------
    new_out, n = _RE_TAG_SUB.subn(lambda m: f"Fn::Sub: {m.group(1)}", out)
    count += n
    out = new_out

    # --- 4. !GetAtt Logical.Attr  ->  Fn::GetAtt: [Logical, Attr] --------
    new_out, n = _RE_TAG_GETATT.subn(
        lambda m: f"Fn::GetAtt: [{m.group(1)}, {m.group(2)}]", out
    )
    count += n
    out = new_out

    # --- 5. Other YAML tags  ->  Fn:: prefix (keep remainder as-is) ------
    new_out, n = _RE_TAG_OTHER.subn(
        lambda m: f"Fn::{m.group(1)}{m.group(2)}", out
    )
    count += n
    out = new_out

    return out, count


class RemediationSkill(Skill):
    """
    Fixes a CloudFormation template that has failed validation.

    Two paths, both LLM-driven:
    1. In-place fix  -- security, schema, and YAML syntax findings.
    2. Re-plan       -- intent/coverage/AC-* findings.

    After receiving the LLM output, _deterministic_sanitise() is applied to
    catch the most common YAML anti-patterns before writing to the GOD.  This
    breaks infinite loops caused by the LLM repeatedly producing the same
    broken pattern.

    There is no round cap.  The orchestrator's max_total_iterations is the
    only mechanism that stops the remediation loop.
    """

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="remediation",
            description=(
                "Fixes a CloudFormation template that has failed validation. "
                "Uses the LLM + deterministic sanitiser to produce a corrected template."
            ),
            phase=SkillPhase.REMEDIATION,
            trigger_condition=(
                "Any FAIL validator has blocking findings AND template.body is non-empty"
            ),
            writes_to=["template.body", "template.version", "remediation_log"],
            reads_from=["template.body", "validation_state"],
            priority=10,
            version="3.4.0",
            tags=["llm", "remediation", "cloudformation"],
        )

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
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
        findings_text = "\n".join(
            f"- [{f.severity.name}] {f.rule_id} on '{f.resource_name}': "
            f"{f.message}. Remediation hint: {f.remediation_hint or 'n/a'}"
            for f in findings
        )

        remediation_hints = getattr(god.template, "remediation_hints", "") or "(none)"
        user_message = (
            f"## Current template\n"
            f"```yaml\n{god.template.body}\n```\n\n"
            f"## Validation findings to fix (round {round_num})\n"
            f"{findings_text}\n\n"
            f"## Accumulated context from previous rounds\n"
            f"{remediation_hints}"
        )

        self._logger.info(
            f"  [llm-fix] Sending {len(findings)} findings to LLM (round {round_num})"
        )

        t0 = time.monotonic()
        try:
            fixed_template = llm.complete(
                system=REMEDIATION_SYSTEM_PROMPT,
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
                system_prompt=REMEDIATION_SYSTEM_PROMPT,
                user_message=user_message,
                raw_response=fixed_template,
                duration_ms=llm_ms,
                success=llm_ok,
                error=llm_err,
                extra={
                    "round_num": round_num,
                    "finding_count": len(findings),
                    "finding_ids": [f.rule_id for f in findings],
                },
            )

        if not llm_ok:
            skill_result.success = False
            skill_result.errors.append(f"LLM call failed: {llm_err}")
            return skill_result

        # -- post-process: strip fences then apply deterministic sanitiser --
        fixed_template = self._strip_fences(fixed_template)
        fixed_template, sanitise_count = _deterministic_sanitise(fixed_template)
        if sanitise_count:
            self._logger.info(
                f"  [sanitise] Applied {sanitise_count} deterministic YAML fix(es)"
            )

        if "AWSTemplateFormatVersion" not in fixed_template:
            skill_result.success = False
            skill_result.errors.append(
                "LLM remediation did not return a valid CloudFormation template "
                "(missing AWSTemplateFormatVersion)."
            )
            return skill_result

        before_body = god.template.body
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
            description=(
                f"LLM fixed {len(findings)} findings in-place"
                + (f"; sanitiser applied {sanitise_count} fix(es)" if sanitise_count else "")
            ),
            rationale=f"Findings: {[f.rule_id for f in findings]}",
            findings_addressed=[f.rule_id for f in findings],
            success=True,
            strategy_type="llm",
        ))

        skill_result.changes_made = [
            f"LLM remediated {len(findings)} findings (round {round_num})"
            + (f"; {sanitise_count} deterministic sanitise fix(es)" if sanitise_count else "")
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
