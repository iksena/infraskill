# -----------------------------------------------------------------------------
# ENGINEERING SKILL  v2.3.0  — compact grounded-objectives context
# -----------------------------------------------------------------------------
# Changes from v2.2.0:
#   - uses god.llm_context() instead of full god.snapshot() to keep LLM payload
#     compact and reduce truncation pressure on objectives.
# -----------------------------------------------------------------------------

import json
import time
from typing import Optional

from enums import SkillPhase
from god import GroundedObjectivesDocument
from llm_client import OpenRouterClient
from prompt import ENGINEER_SYSTEM_PROMPT
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult
from telemetry import TelemetryRecorder


class GeneralEngineerSkill(Skill):
    """
    Generates a complete AWS CloudFormation template from the GOD specification.

    One LLM call produces the entire template — all resources, parameters,
    outputs, and cross-resource references are handled by the LLM in a single
    context window.
    """

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="general-engineer",
            version="2.4.0",
            description=(
                "Generates a complete AWS CloudFormation template from the GOD "
                "specification in a single LLM call."
            ),
            llm_description=(
                "Given prompt + grounded objectives + remediation hints, produce a complete, "
                "deployment-ready CloudFormation YAML template. All resources, "
                "parameters, outputs, and cross-resource references must be included."
            ),
            phase=SkillPhase.ENGINEERING,
            trigger_condition=(
                "intent.objectives is non-empty AND template.body is empty"
            ),
            writes_to=["template.body", "template.version"],
            reads_from=[
                "intent.raw_prompt",
                "intent.objectives",
                "template.remediation_hints",
            ],
            priority=10,
            tags=["llm", "engineering", "cloudformation"],
        )

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        has_objectives = bool(god.intent.objectives)
        no_template = not god.template.body
        return has_objectives and no_template

    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        result = SkillResult(success=True, skill_name=self.metadata.name)
        llm: Optional[OpenRouterClient] = context.get_config("llm")
        tel: Optional[TelemetryRecorder] = context.get_config("telemetry")
        iteration: int = context.iteration
        max_output_tokens: int = context.get_config("engineer_max_output_tokens", 8192)

        if llm is None:
            return SkillResult.failure(
                self.metadata.name,
                "No LLM client configured — cannot generate CloudFormation template.",
            )

        # ------------------------------------------------------------------
        # Build prompt context from prompt + objective source of truth.
        # ------------------------------------------------------------------
        objectives_spec = json.dumps([o.description for o in god.intent.objectives], indent=2)
        prompt_text = god.intent.raw_prompt or "(empty)"

        remediation_hints = getattr(god.template, "remediation_hints", "") or "(none)"

        system = ENGINEER_SYSTEM_PROMPT.format(
            prompt_text=prompt_text,
            objectives_spec=objectives_spec,
            remediation_hints=remediation_hints,
        )

        # Attach compact LLM context that includes objectives, previous template,
        # and remediation history so regeneration avoids repeated mistakes.
        god_context_json = json.dumps(god.llm_context(), indent=2, default=str)
        previous_template = getattr(god.template, "previous_body", "") or "(none)"
        user_message = (
            "Generate the complete CloudFormation template.\n\n"
            "## Previous failed template\n"
            f"```yaml\n{previous_template}\n```\n\n"
            "## Grounded objectives context\n"
            f"```json\n{god_context_json}\n```"
        )

        t0 = time.monotonic()
        try:
            template_body = llm.complete(
                system=system,
                user=user_message,
                temperature=0.1,
                max_output_tokens=max_output_tokens,
            )
            llm_ok = True
            llm_err = None
        except Exception as e:
            template_body = ""
            llm_ok = False
            llm_err = str(e)
        llm_ms = (time.monotonic() - t0) * 1000

        if tel:
            tel.record_llm_call(
                skill_name=self.metadata.name,
                iteration=iteration,
                call_purpose="engineer",
                system_prompt=system,
                user_message=user_message,
                raw_response=template_body,
                duration_ms=llm_ms,
                success=llm_ok,
                error=llm_err,
                extra={
                    "objective_count": len(god.intent.objectives),
                    "max_output_tokens": max_output_tokens,
                    "has_remediation_hints": bool(
                        getattr(god.template, "remediation_hints", "")
                    ),
                },
            )

        if not llm_ok:
            return SkillResult.failure(
                self.metadata.name,
                f"LLM call failed during template generation: {llm_err}",
            )

        template_body = self._strip_fences(template_body)

        if "AWSTemplateFormatVersion" not in template_body:
            return SkillResult.failure(
                self.metadata.name,
                "LLM output does not appear to be a CloudFormation template "
                "(missing AWSTemplateFormatVersion). Raw output logged at DEBUG.",
            )

        before_body = god.template.body  # empty string on first run
        god.template.body = template_body
        god.template.resources = {}
        god.template.update_checksum()
        god.template.increment_version(self.metadata.name)

        if tel:
            tel.record_god_change(
                field="template.body",
                before=before_body or "(empty)",
                after=template_body,
                changed_by=self.metadata.name,
                iteration=iteration,
            )

        result.changes_made.append(
            f"Generated complete CFN template ({len(template_body)} chars, "
            f"{len(god.intent.objectives)} objectives)"
        )
        return result

    @staticmethod
    def _strip_fences(text: str) -> str:
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
