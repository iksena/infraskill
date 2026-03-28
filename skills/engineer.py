# -----------------------------------------------------------------------------
# ENGINEERING SKILLS
# -----------------------------------------------------------------------------

from typing import Optional

from enums import SkillPhase
from god import Constraints, GroundedObjectivesDocument
from llm_client import OpenRouterClient
from prompt import ENGINEER_SYSTEM_PROMPT
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult


class BaseEngineerSkill(Skill):
    """Base class for resource engineering skills."""

    def __init__(self):
        super().__init__()
        self.resource_types: list[str] = []

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        if not god.intent.resources:
            return False
        for resource in god.intent.resources:
            if resource.resource_type in self.resource_types and not resource.generated:
                return True
        return False

    def _mark_generated(self, god: GroundedObjectivesDocument, resource_type: str):
        for r in god.intent.resources:
            if r.resource_type == resource_type:
                r.generated = True

    def _generate_block(
        self,
        context: SkillContext,
        resource_type: str,
        logical_name: str,
        constraints: Constraints,
        existing_refs: list[str],
        properties_hints: dict | None = None,
    ) -> str:
        llm: OpenRouterClient = context.get_config("llm")
        c = constraints
        hints_text = ""
        if properties_hints:
            hints_text = "\n## Property hints from planner\n" + "\n".join(
                f"  {k}: {v}" for k, v in properties_hints.items()
            )

        system = ENGINEER_SYSTEM_PROMPT.format(
            resource_type=resource_type,
            logical_name=logical_name,
            environment=c.environment,
            multi_az=c.multi_az,
            encryption_at_rest=c.encryption_at_rest,
            encryption_in_transit=c.encryption_in_transit,
            public_access_allowed=c.public_access_allowed,
            backup_enabled=c.backup_enabled,
            backup_retention_days=c.backup_retention_days,
            logging_enabled=c.logging_enabled,
            compliance_frameworks=", ".join(c.compliance_frameworks) or "none",
            existing_refs="\n".join(f"  - {r}" for r in existing_refs) or "  (none yet)",
            properties_hints=hints_text,
        )

        return llm.complete(
            system=system,
            user=f"Generate the {resource_type} block.",
            temperature=0.1,
        )


class GeneralEngineerSkill(BaseEngineerSkill):
    """
    Universal engineer skill — generates ANY AWS CloudFormation resource block.

    Progressive disclosure
    ----------------------
    L1  metadata   — always resident
    L2  re module  — imported lazily on first activation
    L3  (none)     — no heavy assets
    """

    def __init__(self):
        super().__init__()
        self.resource_types = []  # empty = accepts all

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="general-engineer",
            version="1.1.0",
            description=(
                "Generates any AWS CloudFormation resource block using the LLM. "
                "Processes resources in priority order as defined in the GOD intent."
            ),
            llm_description=(
                "Given a request to generate a specific AWS CloudFormation resource "
                "block, produce the YAML snippet that defines that resource. Use the "
                "provided constraints and any hints from the planner to inform the "
                "generation."
            ),
            phase=SkillPhase.ENGINEERING,
            trigger_condition="Any ungenerated resource exists in intent.resources",
            writes_to=["template.resources", "template.version"],
            reads_from=["intent.resources", "intent.constraints"],
            priority=10,
            tags=["llm", "engineering", "cloudformation"],
            examples=[
                {
                    "input": {"resource_type": "AWS::S3::Bucket", "logical_name": "MainS3Bucket"},
                    "output": "YAML block with BucketEncryption, PublicAccessBlockConfiguration, Tags",
                }
            ],
        )

    # L2: lazily import re
    def load_level2(self) -> bool:
        if not self._level2_loaded:
            import re as _re
            self._re = _re
            self._level2_loaded = True
        return True

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        return bool(god.intent.get_ungenerated_resources())

    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        result = SkillResult(success=True, skill_name=self.metadata.name)

        llm: Optional[OpenRouterClient] = context.get_config("llm")
        if llm is None:
            return SkillResult.failure(
                self.metadata.name,
                "No LLM client configured — cannot generate resource blocks",
            )

        constraints = god.intent.constraints
        pending = god.intent.get_ungenerated_resources()

        for resource in pending:
            existing_refs = list(god.template.resources.keys())
            properties_hints = resource.properties_hints

            try:
                block = self._generate_block(
                    context=context,
                    resource_type=resource.resource_type,
                    logical_name=resource.logical_name,
                    constraints=constraints,
                    existing_refs=existing_refs,
                    properties_hints=properties_hints,
                )
                # Strip trailing code-fence markers the LLM sometimes emits
                clean_block = self._re.sub(r'\n?```\s*$', '', block)
                if not clean_block.endswith('\n'):
                    clean_block += '\n'

            except Exception as e:
                result.success = False
                result.errors.append(
                    f"Failed to generate {resource.logical_name} "
                    f"({resource.resource_type}): {e}"
                )
                continue

            god.template.resources[resource.logical_name] = clean_block
            resource.generated = True
            god.template.increment_version(self.metadata.name)
            result.changes_made.append(
                f"Generated {resource.resource_type} \u2192 {resource.logical_name}"
            )

        return result


# -----------------------------------------------------------------------------
# ASSEMBLY SKILL
# -----------------------------------------------------------------------------

class TemplateAssemblerSkill(Skill):
    """
    Assembles individual resource blocks into a complete CloudFormation template.

    Resource ordering follows the GOD priority (set by PlannerSkill) rather than
    a hardcoded list, so any resource type the planner adds is handled correctly.

    IMPORTANT — YAML tag safety
    ---------------------------
    We use yaml.safe_load everywhere for validation, which rejects YAML custom
    tags like !Ref, !Sub, !GetAtt.  The header and outputs sections below
    therefore use only the Fn:: dict form so safe_load never sees unknown tags.
    The LLM-generated resource blocks may still contain tag shorthand, but those
    are stitched in between our safe header/footer so the assembler itself stays
    tag-free.
    """

    # -------------------------------------------------------------------------
    # Top-level template skeleton
    # -------------------------------------------------------------------------
    # Rules:
    #   1. NO leading spaces on top-level keys — yaml.safe_load is strict about
    #      indentation; any indent here produces the 'expected <block end>,
    #      found <block mapping start>' error.
    #   2. NO !Ref / !Sub / !GetAtt shorthand — use Fn:: dict form so that
    #      yaml.safe_load (used by YAMLSyntaxValidatorSkill and RemediationSkill)
    #      can parse the header without a custom Loader.
    # -------------------------------------------------------------------------

    CFN_HEADER = """\
AWSTemplateFormatVersion: '2010-09-09'
Description: Infrastructure generated by INFRA-SKILL

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues:
      - development
      - staging
      - production
    Description: Environment name

Resources:
"""

    CFN_OUTPUTS = """\
Outputs:
  StackName:
    Description: Stack Name
    Value:
      Ref: AWS::StackName
"""

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="template-assembler",
            version="1.2.0",
            description="Assembles resource blocks into a complete CFN template in dependency order.",
            llm_description=(
                "Combines individually generated CloudFormation resource YAML blocks "
                "into a single valid template with header, Resources section, and Outputs."
            ),
            phase=SkillPhase.ASSEMBLY,
            trigger_condition=(
                "template.resources is non-empty AND "
                "template.body does not contain AWSTemplateFormatVersion"
            ),
            writes_to=["template.body", "template.version"],
            reads_from=["template.resources"],
            priority=10,
            tags=["assembly", "cloudformation"],
        )

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        has_resources = bool(god.template.resources)
        no_body = not god.template.body or "AWSTemplateFormatVersion" not in god.template.body
        return has_resources and no_body

    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        result = SkillResult(success=True, skill_name=self.metadata.name)

        # Order by the priority already set on each ExtractedResource by PlannerSkill.
        priority_map: dict[str, int] = {
            r.logical_name: r.priority for r in god.intent.resources
        }

        def _sort_key(name: str) -> int:
            return priority_map.get(name, 999)

        resources_yaml = ""
        for name in sorted(god.template.resources.keys(), key=_sort_key):
            block = god.template.resources[name]
            # Indent each resource block by 2 spaces so it sits inside Resources:
            indented = "\n".join(
                "  " + line if line.strip() else line
                for line in block.splitlines()
            )
            resources_yaml += indented.rstrip() + "\n"

        god.template.body = self.CFN_HEADER + resources_yaml + "\n" + self.CFN_OUTPUTS
        god.template.update_checksum()
        god.template.increment_version(self.metadata.name)

        result.changes_made.append(
            f"Assembled {len(god.template.resources)} resources into template"
        )
        return result
