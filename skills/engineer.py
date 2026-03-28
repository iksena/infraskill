# -----------------------------------------------------------------------------
# ENGINEERING SKILLS
# -----------------------------------------------------------------------------

import re
from typing import Optional

from enums import SkillPhase
from god import Constraints, GroundedObjectivesDocument
from llm_client import OpenRouterClient
from prompt import ENGINEER_SYSTEM_PROMPT
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult


class BaseEngineerSkill(Skill):
    """Base class for resource engineering skills"""
    
    def __init__(self):
        super().__init__()
        self.resource_types: list[str] = []
    
    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        if not god.intent.resources:
            return False
        
        # Check if any of our resource types are needed and not yet generated
        for resource in god.intent.resources:
            if resource.resource_type in self.resource_types and not resource.generated:
                return True
        return False
    
    def _mark_generated(self, god: GroundedObjectivesDocument, resource_type: str):
        """Mark a resource type as generated"""
        for r in god.intent.resources:
            if r.resource_type == resource_type:
                r.generated = True
    
    # BaseEngineerSkill — shared helper used by all 7 engineer subclasses
    def _generate_block(
        self,
        context: SkillContext,
        resource_type: str,
        logical_name: str,
        constraints: Constraints,
        existing_refs: list[str],
        properties_hints: dict | None = None,   # ← new
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

    Replaces the 7 per-domain subclasses. The ENGINEER_SYSTEM_PROMPT is already
    resource-agnostic; routing is done by priority order from the GOD intent.
    """

    def __init__(self):
        super().__init__()
        # Accepts all resource types — no allowlist needed
        self.resource_types = []  # empty = accept all

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="general-engineer",
            description=(
                "Generates any AWS CloudFormation resource block using the LLM. "
                "Processes resources in priority order as defined in the GOD intent."
            ),
            llm_description=(
                "Given a request to generate a specific AWS CloudFormation resource block, produce the YAML snippet that defines that resource. Use the provided constraints and any hints from the planner to inform the generation."
            ),
            phase=SkillPhase.ENGINEERING,
            trigger_condition="Any ungenerated resource exists in intent",
            writes_to="template.resources",
            reads_from="intent.resources, intent.constraints",
            priority=10,
        )

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
        # Resources are already priority-sorted from PlannerSkill
        pending = god.intent.get_ungenerated_resources()

        for resource in pending:
            existing_refs = list(god.template.resources.keys())
            properties_hints = resource.properties_hints  # ← feed planner hints back in

            try:
                block = self._generate_block(
                    context=context,
                    resource_type=resource.resource_type,
                    logical_name=resource.logical_name,
                    constraints=constraints,
                    existing_refs=existing_refs,
                    properties_hints=properties_hints,
                )
                clean_block = re.sub(r'\n?```\s*$', '', block)
                
                # Ensure the block ends with a newline so the assembler concatenates cleanly
                if not clean_block.endswith('\n'):
                    clean_block += '\n'

            except Exception as e:
                result.success = False
                result.errors.append(
                    f"Failed to generate {resource.logical_name} "
                    f"({resource.resource_type}): {e}"
                )
                continue

            # 3. Save the cleaned block to the GOD template
            god.template.resources[resource.logical_name] = clean_block
            resource.generated = True
            god.template.increment_version(self.metadata.name)
            
            result.changes_made.append(
                f"Generated {resource.resource_type} → {resource.logical_name}"
            )

        return result

# -----------------------------------------------------------------------------
# ASSEMBLY SKILL
# -----------------------------------------------------------------------------

class TemplateAssemblerSkill(Skill):
    """Assembles resource blocks into a complete CloudFormation template"""
    
    CFN_HEADER = """AWSTemplateFormatVersion: '2010-09-09'
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

Resources:"""

    CFN_OUTPUTS = """
Outputs:
  VPCId:
    Description: VPC ID
    Value: !Ref VPC
    Export:
      Name: !Sub '${AWS::StackName}-VPCId'
  
  StackName:
    Description: Stack Name
    Value: !Ref AWS::StackName
"""
    
    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="template-assembler",
            description="Assembles resource blocks into a complete CFN template",
            phase=SkillPhase.ASSEMBLY,
            trigger_condition="Resource blocks exist but template body is empty",
            writes_to=["template.body"],
            reads_from=["template.resources"],
            priority=10
        )
    
    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        has_resources = bool(god.template.resources)
        no_body = not god.template.body or "AWSTemplateFormatVersion" not in god.template.body
        return has_resources and no_body
    
    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        result = SkillResult(success=True, skill_name=self.metadata.name)
        
        # Sort resources by dependency order (simple alphabetical for now)
        resource_order = [
            'VPC', 'InternetGateway', 'VPCGatewayAttachment',
            'PrivateSubnetA', 'PrivateSubnetB', 'PublicSubnetA', 'PublicSubnetB',
            'PrivateRouteTable', 'PublicRouteTable', 'PublicRoute',
            'AppSecurityGroup', 'DatabaseSecurityGroup',
            'LambdaExecutionRole', 'EC2InstanceRole', 'EC2InstanceProfile', 'ApplicationRole',
            'DBSubnetGroup', 'RDSInstance',
            'S3Bucket', 'LambdaFunction'
        ]
        
        # Build resources section
        resources_yaml = ""
        added = set()
        
        # First, add resources in preferred order
        for name in resource_order:
            if name in god.template.resources:
                resources_yaml += god.template.resources[name]
                added.add(name)
        
        # Then add any remaining resources
        for name, block in god.template.resources.items():
            if name not in added:
                resources_yaml += block
        
        # Assemble complete template
        god.template.body = self.CFN_HEADER + resources_yaml + self.CFN_OUTPUTS
        god.template.update_checksum()
        god.template.increment_version(self.metadata.name)
        
        result.changes_made.append(f"Assembled {len(god.template.resources)} resources into template")
        
        return result

