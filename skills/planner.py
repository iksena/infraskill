# -----------------------------------------------------------------------------
# PLANNING SKILL  v3.0.0  — pure LLM, no regex fallback
# -----------------------------------------------------------------------------
#
# The planner is a pure LLM skill. It sends the user's natural-language prompt
# to the LLM with a structured JSON schema and populates the GOD intent.
#
# If no LLM client is configured the skill fails immediately with a clear error
# rather than silently producing a regex-inferred plan.
#
# JSON parse failure: one correction retry is attempted before giving up.
# -----------------------------------------------------------------------------

from datetime import datetime
import json
from typing import Optional

from enums import SkillPhase
from god import AcceptanceCriterion, Constraints, ExtractedResource, GroundedObjectivesDocument
from llm_client import OpenRouterClient
from prompt import PLANNER_SYSTEM_PROMPT
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult


class PlannerSkill(Skill):
    """
    Transforms a natural-language infrastructure prompt into a machine-checkable
    GOD specification: resource list, constraints, and acceptance criteria.

    This skill is a pure LLM agent — it has no deterministic fallback.
    A configured LLM client is required for execution.
    """

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="planner",
            version="3.0.0",
            description=(
                "Transforms a natural-language infrastructure prompt into a "
                "machine-checkable GOD specification: resource list, constraints, "
                "and acceptance criteria."
            ),
            llm_description=(
                "Given a natural-language prompt describing desired cloud "
                "infrastructure, extract: (1) the list of AWS resources needed, "
                "(2) constraints such as security, HA, compliance, and (3) "
                "binary acceptance criteria that can be checked against the "
                "generated CloudFormation template."
            ),
            phase=SkillPhase.PLANNING,
            trigger_condition="intent.raw_prompt exists AND intent.resources is empty",
            writes_to=[
                "intent.resources",
                "intent.constraints",
                "intent.acceptance_criteria",
                "intent.normalized_prompt",
                "intent.parsed_at",
                "intent.parser_version",
            ],
            reads_from=["intent.raw_prompt"],
            priority=10,
            tags=["llm", "planning", "extraction"],
        )

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        return bool(god.intent.raw_prompt) and not god.intent.resources

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------
    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        result = SkillResult(success=True, skill_name=self.metadata.name)
        llm: Optional[OpenRouterClient] = context.get_config("llm")

        if llm is None:
            return SkillResult.failure(
                self.metadata.name,
                "No LLM client configured. PlannerSkill requires an LLM to extract "
                "resources and acceptance criteria from the prompt.",
            )

        # Append any accumulated remediation hints as additional context
        # so the planner incorporates corrections from previous rounds.
        remediation_hints = getattr(god.template, "remediation_hints", "") or ""
        user_message = god.intent.raw_prompt
        if remediation_hints:
            user_message = (
                f"{god.intent.raw_prompt}\n\n"
                f"## Corrections from previous generation rounds\n"
                f"{remediation_hints}"
            )

        raw = llm.complete(
            system=PLANNER_SYSTEM_PROMPT,
            user=user_message,
            temperature=0.0,
        )

        data = self._parse_json_with_retry(raw, user_message, llm)
        if data is None:
            return SkillResult.failure(
                self.metadata.name,
                "Planner LLM did not return valid JSON after retry. "
                "Check model availability or prompt.",
            )

        god.intent.resources = [ExtractedResource(**r) for r in data["resources"]]
        god.intent.constraints = Constraints(**data["constraints"])
        god.intent.acceptance_criteria = [
            AcceptanceCriterion(**c) for c in data["acceptance_criteria"]
        ]
        god.intent.normalized_prompt = god.intent.raw_prompt.lower()
        god.intent.parsed_at = datetime.now().isoformat()
        god.intent.parser_version = self.metadata.version

        result.changes_made.append(
            f"LLM extracted {len(god.intent.resources)} resources, "
            f"{len(god.intent.acceptance_criteria)} acceptance criteria"
        )
        return result

    # ------------------------------------------------------------------
    # JSON parse with single correction retry
    # ------------------------------------------------------------------
    def _parse_json_with_retry(
        self,
        raw: str,
        original_message: str,
        llm: OpenRouterClient,
    ) -> Optional[dict]:
        """Parse JSON from LLM output; on failure ask the LLM to correct it once."""
        cleaned = self._strip_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            self._logger.warning(
                f"LLM returned invalid JSON ({exc}); retrying with correction prompt."
            )

        try:
            corrected = llm.complete(
                system=(
                    "Your previous response was not valid JSON. "
                    "Return ONLY a valid JSON object — no markdown, no prose, no code fences. "
                    "The JSON must match the schema in the original system prompt."
                ),
                user=(
                    f"Original request:\n{original_message}\n\n"
                    f"Your previous (invalid) response:\n{raw[:2000]}\n\n"
                    "Please return ONLY the corrected JSON object."
                ),
                temperature=0.0,
            )
            return json.loads(self._strip_fences(corrected))
        except json.JSONDecodeError as exc2:
            self._logger.error(
                f"LLM correction retry also returned invalid JSON ({exc2}). Giving up."
            )
            return None
        except Exception as exc2:
            self._logger.error(f"LLM correction retry failed: {exc2}")
            return None

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove optional markdown code fences from an LLM response."""
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
