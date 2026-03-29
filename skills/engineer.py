# -----------------------------------------------------------------------------
# ENGINEERING SKILL  v2.0.0  — pure LLM, whole-template generation
# -----------------------------------------------------------------------------
#
# The engineer is a pure LLM skill. A single LLM call receives the full
# infrastructure specification (resources + constraints + remediation hints)
# and produces a COMPLETE, deployment-ready CloudFormation template.
#
# There is no block-by-block generation loop and no string manipulation of
# YAML. Cross-resource references (Fn::Sub, Fn::GetAtt, Ref) are resolved by
# the LLM natively.
#
# TemplateAssemblerSkill is removed. The engineer output IS the assembled
# template. The orchestrator should transition directly from ENGINEERING to
# VALIDATION after this skill succeeds.
# -----------------------------------------------------------------------------

import json
from typing import Optional

from enums import SkillPhase
from god import GroundedObjectivesDocument
from llm_client import OpenRouterClient
from prompt import ENGINEER_SYSTEM_PROMPT
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult


class GeneralEngineerSkill(Skill):
    """
    Generates a complete AWS CloudFormation template from the GOD specification.

    One LLM call produces the entire template — all resources, parameters,
    outputs, and cross-resource references are handled by the LLM in a single
    context window. This eliminates the stitching and indentation problems
    caused by per-resource block generation.
    """

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="general-engineer",
            version="2.0.0",
            description=(
                "Generates a complete AWS CloudFormation template from the GOD "
                "specification in a single LLM call."
            ),
            llm_description=(
                "Given a full infrastructure specification (resources, constraints, "
                "remediation hints), produce a complete, deployment-ready "
                "CloudFormation YAML template. All resources, parameters, outputs, "
                "and cross-resource references must be included."
            ),
            phase=SkillPhase.ENGINEERING,
            trigger_condition=(
                "intent.resources is non-empty AND template.body is empty"
            ),
            writes_to=["template.body", "template.version"],
            reads_from=[
                "intent.resources",
                "intent.constraints",
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
        if llm is None:
            return SkillResult.failure(
                self.metadata.name,
                "No LLM client configured — cannot generate CloudFormation template.",
            )

        # Build the resources spec as a readable JSON block for the prompt.
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

        remediation_hints = getattr(god.template, "remediation_hints", "") or "(none)"

        system = ENGINEER_SYSTEM_PROMPT.format(
            resources_spec=resources_spec,
            constraints_spec=constraints_spec,
            remediation_hints=remediation_hints,
        )

        try:
            template_body = llm.complete(
                system=system,
                user="Generate the complete CloudFormation template.",
                temperature=0.1,
            )
        except Exception as e:
            return SkillResult.failure(
                self.metadata.name,
                f"LLM call failed during template generation: {e}",
            )

        # Strip any markdown fences the LLM may have wrapped the output in.
        template_body = self._strip_fences(template_body)

        if "AWSTemplateFormatVersion" not in template_body:
            return SkillResult.failure(
                self.metadata.name,
                "LLM output does not appear to be a CloudFormation template "
                "(missing AWSTemplateFormatVersion). Raw output logged at DEBUG.",
            )

        god.template.body = template_body
        god.template.resources = {}  # not used in whole-template mode
        god.template.update_checksum()
        god.template.increment_version(self.metadata.name)

        result.changes_made.append(
            f"Generated complete CFN template ({len(template_body)} chars, "
            f"{len(god.intent.resources)} resources)"
        )
        return result

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove optional markdown code fences from an LLM response."""
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
