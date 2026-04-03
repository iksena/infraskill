# -----------------------------------------------------------------------------
# PLANNING SKILL  v3.1.0  — pure LLM + telemetry instrumentation
# -----------------------------------------------------------------------------

from datetime import datetime
import json
import time
from typing import Optional

from enums import SkillPhase
from god import GroundedObjective, GroundedObjectivesDocument
from llm_client import OpenRouterClient
from prompt import PLANNER_SYSTEM_PROMPT
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult
from telemetry import TelemetryRecorder


class PlannerSkill(Skill):
    """
    Transforms a natural-language infrastructure prompt into a compact,
    objectives-only GOD specification.
    """

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="planner",
            version="3.4.0",
            description=(
                "Transforms a natural-language infrastructure prompt into a "
                "compact GOD specification with prompt and grounded objectives "
                "as the single source of truth."
            ),
            llm_description=(
                "Given a natural-language prompt describing desired cloud "
                "infrastructure, extract a concise list of grounded objectives "
                "that are actionable for template generation and validation."
            ),
            phase=SkillPhase.PLANNING,
            trigger_condition=(
                "intent.raw_prompt exists AND intent.objectives is empty"
            ),
            writes_to=[
                "intent.objectives",
                "intent.parsed_at",
                "intent.parser_version",
                "template.body (cleared only during re-plan)",
            ],
            reads_from=["intent.raw_prompt"],
            priority=10,
            tags=["llm", "planning", "objectives"],
        )

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        return bool(god.intent.raw_prompt) and not bool(god.intent.objectives)

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------
    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        result = SkillResult(success=True, skill_name=self.metadata.name)
        llm: Optional[OpenRouterClient] = context.get_config("llm")
        tel: Optional[TelemetryRecorder] = context.get_config("telemetry")
        iteration: int = context.iteration

        if llm is None:
            return SkillResult.failure(
                self.metadata.name,
                "No LLM client configured. PlannerSkill requires an LLM to extract "
                "objectives from the prompt.",
            )

        remediation_hints = getattr(god.template, "remediation_hints", "") or ""
        user_message = god.intent.raw_prompt
        if remediation_hints:
            user_message = (
                f"{god.intent.raw_prompt}\n\n"
                f"## Corrections from previous generation rounds\n"
                f"{remediation_hints}"
            )

        # Record GOD reads
        if tel:
            tel.record_god_change(
                field="intent.raw_prompt",
                before=None,
                after=god.intent.raw_prompt,
                changed_by=self.metadata.name,
                iteration=iteration,
            )

        t0 = time.monotonic()
        try:
            raw = llm.complete(
                system=PLANNER_SYSTEM_PROMPT,
                user=user_message,
                temperature=0.0,
            )
            llm_ok = True
            llm_err = None
        except Exception as e:
            raw = ""
            llm_ok = False
            llm_err = str(e)
        llm_ms = (time.monotonic() - t0) * 1000

        if tel:
            tel.record_llm_call(
                skill_name=self.metadata.name,
                iteration=iteration,
                call_purpose="plan",
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_message=user_message,
                raw_response=raw,
                duration_ms=llm_ms,
                success=llm_ok,
                error=llm_err,
            )

        if not llm_ok:
            return SkillResult.failure(self.metadata.name, f"LLM call failed: {llm_err}")

        data = self._parse_json_with_retry(raw, user_message, llm, tel, iteration)
        if data is None:
            return SkillResult.failure(
                self.metadata.name,
                "Planner LLM did not return valid JSON after retry. "
                "Check model availability or prompt.",
            )

        is_replan = bool(god.template.body)
        before_objectives = list(god.intent.objectives) if god.intent.objectives else []
        before_template_body = god.template.body
        god.intent.objectives = self._parse_objectives(data.get("objectives", []))
        god.intent.parsed_at = datetime.now().isoformat()
        god.intent.parser_version = self.metadata.version

        if tel:
            tel.record_god_change(
                field="intent.objectives",
                before=[o.description for o in before_objectives],
                after=[o.description for o in god.intent.objectives],
                changed_by=self.metadata.name,
                iteration=iteration,
            )

        if is_replan:
            god.template.body = ""
            god.template.resources = {}
            god.template.increment_version(self.metadata.name)
            god.reset_validations_from("yaml_syntax", self.metadata.name, skip_errored=True)
            if tel:
                tel.record_god_change(
                    field="template.body",
                    before=before_template_body,
                    after="(cleared for regeneration after re-plan)",
                    changed_by=self.metadata.name,
                    iteration=iteration,
                )
            result.changes_made.append(
                "Detected intent drift findings; cleared template to force regeneration"
            )

        result.changes_made.append(f"LLM extracted {len(god.intent.objectives)} grounded objectives")
        return result

    @staticmethod
    def _parse_objectives(raw_objectives: list) -> list[GroundedObjective]:
        """
        Accepts either the new compact schema (list[str]) or legacy object
        schema and normalizes both to description-only objectives.
        """
        parsed: list[GroundedObjective] = []
        for item in raw_objectives:
            if isinstance(item, str):
                desc = item.strip()
            elif isinstance(item, dict):
                desc = str(item.get("description", "")).strip()
            else:
                desc = ""
            if desc:
                parsed.append(GroundedObjective(description=desc))
        return parsed

    # ------------------------------------------------------------------
    # JSON parse with single correction retry
    # ------------------------------------------------------------------
    def _parse_json_with_retry(
        self,
        raw: str,
        original_message: str,
        llm: OpenRouterClient,
        tel: Optional[TelemetryRecorder],
        iteration: int,
    ) -> Optional[dict]:
        cleaned = self._strip_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            self._logger.warning(
                f"LLM returned invalid JSON ({exc}); retrying with correction prompt."
            )

        correction_system = (
            "Your previous response was not valid JSON. "
            "Return ONLY a valid JSON object — no markdown, no prose, no code fences. "
            "The JSON must match the schema in the original system prompt."
        )
        correction_user = (
            f"Original request:\n{original_message}\n\n"
            f"Your previous (invalid) response:\n{raw[:2000]}\n\n"
            "Please return ONLY the corrected JSON object."
        )
        t0 = time.monotonic()
        try:
            corrected = llm.complete(
                system=correction_system,
                user=correction_user,
                temperature=0.0,
            )
            retry_ok = True
            retry_err = None
        except Exception as exc2:
            corrected = ""
            retry_ok = False
            retry_err = str(exc2)
        retry_ms = (time.monotonic() - t0) * 1000

        if tel:
            tel.record_llm_call(
                skill_name=self.metadata.name,
                iteration=iteration,
                call_purpose="plan_json_retry",
                system_prompt=correction_system,
                user_message=correction_user,
                raw_response=corrected,
                duration_ms=retry_ms,
                success=retry_ok,
                error=retry_err,
            )

        if not retry_ok:
            self._logger.error(f"LLM correction retry failed: {retry_err}")
            return None

        try:
            return json.loads(self._strip_fences(corrected))
        except json.JSONDecodeError as exc2:
            self._logger.error(
                f"LLM correction retry also returned invalid JSON ({exc2}). Giving up."
            )
            return None

    @staticmethod
    def _strip_fences(text: str) -> str:
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
