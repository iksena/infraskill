# -----------------------------------------------------------------------------
# ENGINEERING SKILL  v2.3.0  — compact grounded-objectives context
# -----------------------------------------------------------------------------
# Changes from v2.2.0:
#   - user message now carries the main GOD snapshot plus current/previous
#     template bodies and remediation details.
# -----------------------------------------------------------------------------

import json
import time
from typing import Optional

from enums import SkillPhase
from god import GroundedObjectivesDocument
from llm_client import OpenRouterClient
from prompt import ENGINEER_SYSTEM_PROMPT, GOD_BLACKBOARD_PRIMER
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
        skill_inputs = {
            "prompt": god.intent.raw_prompt,
            "objectives": [o.description for o in god.intent.objectives],
            "objective_count": len(god.intent.objectives),
            "remediation_hints": getattr(god.template, "remediation_hints", "") or "",
            "previous_template_length": len(getattr(god.template, "previous_body", "") or ""),
            "max_output_tokens": max_output_tokens,
        }

        def emit_skill_telemetry(outputs: dict | None = None):
            if not tel:
                return
            tel.record_skill_execution(
                skill_name=self.metadata.name,
                phase=self.metadata.phase.name,
                iteration=iteration,
                success=result.success,
                duration_ms=result.duration_ms,
                changes_made=result.changes_made,
                errors=result.errors,
                warnings=result.warnings,
                inputs=skill_inputs,
                outputs=outputs or {},
            )

        if llm is None:
            failure = SkillResult.failure(
                self.metadata.name,
                "No LLM client configured — cannot generate CloudFormation template.",
            )
            emit_skill_telemetry({"error": failure.errors[0]})
            return failure

        god_snapshot_json = json.dumps(god.llm_context(), indent=2, default=str)
        system = ENGINEER_SYSTEM_PROMPT.format(
            god_blackboard=GOD_BLACKBOARD_PRIMER,
        )

        user_message = (
            "Generate the deployment-ready CloudFormation template from the main GOD snapshot.\n\n"
            "## Main GOD Snapshot\n"
            f"{god_snapshot_json}"
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
                },
            )

        if not llm_ok:
            failure = SkillResult.failure(
                self.metadata.name,
                f"LLM call failed during template generation: {llm_err}",
            )
            emit_skill_telemetry({"error": llm_err, "raw_response": template_body})
            return failure

        template_body = self._strip_fences(template_body)

        if "AWSTemplateFormatVersion" not in template_body:
            failure = SkillResult.failure(
                self.metadata.name,
                "LLM output does not appear to be a CloudFormation template "
                "(missing AWSTemplateFormatVersion). Raw output logged at DEBUG.",
            )
            emit_skill_telemetry({"error": failure.errors[0], "raw_response": template_body})
            return failure

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
        emit_skill_telemetry({
            "template_length": len(template_body),
            "template_version": god.template.version,
            "template_checksum": god.template.checksum,
        })
        return result

    @staticmethod
    def _strip_fences(text: str) -> str:
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
