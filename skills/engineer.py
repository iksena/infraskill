# -----------------------------------------------------------------------------
# ENGINEERING SKILL  v2.2.0  — full GOD context in every LLM call
# -----------------------------------------------------------------------------
# Changes from v2.1.0:
#   - acceptance_criteria and remediation_hints are now injected into the
#     system prompt so the LLM has the full GOD picture on first generation
#     and on every re-plan round.
#   - god_snapshot is serialised as JSON and appended to the user message
#     so the LLM can cross-reference resource types, constraints, and AC.
#   - record_god_change now fired even when before_body is empty so the
#     llm_conversations and god_changes logs are always in sync.
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
            version="2.2.0",
            description=(
                "Generates a complete AWS CloudFormation template from the GOD "
                "specification in a single LLM call."
            ),
            llm_description=(
                "Given a full infrastructure specification (resources, constraints, "
                "acceptance criteria, remediation hints), produce a complete, "
                "deployment-ready CloudFormation YAML template. All resources, "
                "parameters, outputs, and cross-resource references must be included."
            ),
            phase=SkillPhase.ENGINEERING,
            trigger_condition=(
                "intent.resources is non-empty AND template.body is empty"
            ),
            writes_to=["template.body", "template.version"],
            reads_from=[
                "intent.resources",
                "intent.constraints",
                "intent.acceptance_criteria",
                "template.remediation_hints",
            ],
            priority=10,
            tags=["llm", "engineering", "cloudformation"],
            examples=[
                {
                    "input": {
                        "resources": ["AWS::S3::Bucket", "AWS::IAM::Role"],
                        "constraints": {"encryption_at_rest": True},
                    },
                    "output": "Complete CFN YAML with all resources, parameters, outputs",
                }
            ],
        )

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        has_resources = bool(god.intent.resources)
        no_template = not god.template.body
        return has_resources and no_template

    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        result = SkillResult(success=True, skill_name=self.metadata.name)
        llm: Optional[OpenRouterClient] = context.get_config("llm")
        tel: Optional[TelemetryRecorder] = context.get_config("telemetry")
        iteration: int = context.iteration

        if llm is None:
            return SkillResult.failure(
                self.metadata.name,
                "No LLM client configured — cannot generate CloudFormation template.",
            )

        # ------------------------------------------------------------------
        # Build prompt context from the full GOD
        # ------------------------------------------------------------------
        resources_spec = json.dumps(
            [
                {
                    "resource_type": r.resource_type,
                    "logical_name": r.logical_name,
                    "priority": r.priority,
                    "dependencies": r.dependencies,
                    "properties_hints": r.properties_hints or {},
                }
                for r in god.intent.resources
            ],
            indent=2,
        )

        constraints = god.intent.constraints
        constraints_spec = json.dumps(
            {
                "environment": constraints.environment,
                "multi_az": constraints.multi_az,
                "encryption_at_rest": constraints.encryption_at_rest,
                "encryption_in_transit": constraints.encryption_in_transit,
                "public_access_allowed": constraints.public_access_allowed,
                "backup_enabled": constraints.backup_enabled,
                "backup_retention_days": constraints.backup_retention_days,
                "logging_enabled": constraints.logging_enabled,
                "monitoring_enabled": constraints.monitoring_enabled,
                "compliance_frameworks": constraints.compliance_frameworks,
                "cost_optimization": constraints.cost_optimization,
            },
            indent=2,
        )

        acceptance_criteria = json.dumps(
            [ac.to_dict() for ac in god.intent.acceptance_criteria],
            indent=2,
        ) if god.intent.acceptance_criteria else "(none defined)"

        remediation_hints = getattr(god.template, "remediation_hints", "") or "(none)"

        system = ENGINEER_SYSTEM_PROMPT.format(
            resources_spec=resources_spec,
            constraints_spec=constraints_spec,
            acceptance_criteria=acceptance_criteria,
            remediation_hints=remediation_hints,
        )

        # Attach a compact GOD snapshot as additional user-turn context so the
        # LLM can see resource counts, validation state, and prior round info.
        god_snapshot_json = json.dumps(god.snapshot(), indent=2, default=str)
        user_message = (
            "Generate the complete CloudFormation template.\n\n"
            "## Full GOD snapshot (for reference)\n"
            f"```json\n{god_snapshot_json}\n```"
        )

        t0 = time.monotonic()
        try:
            template_body = llm.complete(
                system=system,
                user=user_message,
                temperature=0.1,
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
                    "resource_count": len(god.intent.resources),
                    "ac_count": len(god.intent.acceptance_criteria),
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
            f"{len(god.intent.resources)} resources)"
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
