# =============================================================================
# PART 3: SKILL FRAMEWORK
# =============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any, Optional, Self

from enums import OrchestratorState, SkillPhase
from god import GroundedObjectivesDocument


@dataclass
class SkillMetadata:
    """
    Level 1 Metadata — Always in memory for routing decisions.

    Progressive disclosure levels:
      L1 (this dataclass) — always resident: name, phase, priority, trigger_condition
      L2 (load_level2)   — loaded on activation: procedural logic / imported modules
      L3 (load_level3)   — loaded on demand:  large schemas, embedding indexes, etc.
    """
    # ---- Required -------------------------------------------------------
    name: str
    description: str
    phase: SkillPhase
    trigger_condition: str
    writes_to: list[str]
    reads_from: list[str]

    # ---- LLM routing ----------------------------------------------------
    llm_description: str = ""   # Human-readable hint for LLM-based orchestrators

    # ---- Execution parameters -------------------------------------------
    priority: int = 100         # Lower = higher priority within the same phase
    timeout_seconds: int = 60
    retryable: bool = True
    max_retries: int = 2

    # ---- Provenance & discoverability -----------------------------------
    version: str = "1.0.0"      # Semver; bump on behaviour-breaking changes
    tags: list[str] = field(default_factory=list)  # e.g. ["llm", "static", "security"]
    examples: list[dict] = field(default_factory=list)  # [{"input": ..., "output": ...}]

    # ---- Graph wiring ---------------------------------------------------
    requires_skills: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "llm_description": self.llm_description,
            "phase": self.phase.name,
            "priority": self.priority,
            "tags": self.tags,
            "trigger_condition": self.trigger_condition,
            "writes_to": self.writes_to,
            "reads_from": self.reads_from,
            "requires_skills": self.requires_skills,
        }


@dataclass
class SkillContext:
    """Context passed to a skill during execution"""
    god: GroundedObjectivesDocument
    orchestrator_state: OrchestratorState
    iteration: int
    triggered_by: str = "orchestrator"
    retry_count: int = 0
    parent_skill: Optional[str] = None
    config: dict = field(default_factory=dict)

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


@dataclass
class SkillResult:
    """Result returned from skill execution"""
    success: bool
    skill_name: str

    changes_made: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    duration_ms: float = 0
    metrics: dict = field(default_factory=dict)

    next_skill_hint: Optional[str] = None
    should_retry: bool = False
    requires_human_review: bool = False
    escalation_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "skill_name": self.skill_name,
            "changes_made": self.changes_made,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
            "requires_human_review": self.requires_human_review,
        }

    @classmethod
    def failure(cls, skill_name: str, error: str) -> Self:
        return cls(success=False, skill_name=skill_name, errors=[error])

    @classmethod
    def success_with_changes(cls, skill_name: str, changes: list[str]) -> Self:
        return cls(success=True, skill_name=skill_name, changes_made=changes)


class Skill(ABC):
    """
    Abstract base class for all skills.

    Three-level progressive disclosure pattern
    ==========================================
    Level 1 — Metadata (always resident)
        _define_metadata() is called once in __init__ and cached.
        Orchestrator routing decisions never pay more than a dict lookup.

    Level 2 — Procedural logic (loaded on first activation)
        Override load_level2() to do one-time imports / lightweight setup.
        Called automatically by pre_execute().

    Level 3 — Heavy assets (loaded on demand, unloaded after use)
        Override load_level3() for large schemas, embedding indexes, etc.
        Override unload_level3() to release memory after execution.

    Subclasses must implement:
        _define_metadata()  → SkillMetadata
        can_trigger(god)    → bool
        execute(context)    → SkillResult
    """

    def __init__(self):
        self._logger = logging.getLogger(f"INFRA-SKILL.Skill.{self.__class__.__name__}")
        self.metadata = self._define_metadata()
        self._level2_loaded = False
        self._level3_loaded = False
        self._execution_count = 0
        self._total_duration_ms = 0

    @abstractmethod
    def _define_metadata(self) -> SkillMetadata:
        pass

    @abstractmethod
    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        pass

    @abstractmethod
    def execute(self, context: SkillContext) -> SkillResult:
        pass

    # ------------------------------------------------------------------
    # Progressive disclosure hooks
    # ------------------------------------------------------------------

    def load_level2(self) -> bool:
        """Load Level-2 assets (imports, caches). Called before first execute."""
        if not self._level2_loaded:
            self._logger.debug(f"[L2] Loading {self.metadata.name}")
            self._level2_loaded = True
        return True

    def load_level3(self) -> bool:
        """Load Level-3 assets (heavy resources). Override for lazy loading."""
        if not self._level3_loaded:
            self._logger.debug(f"[L3] Loading {self.metadata.name}")
            self._level3_loaded = True
        return True

    def unload_level3(self):
        """Release Level-3 assets to free memory after execution."""
        if self._level3_loaded:
            self._logger.debug(f"[L3] Unloading {self.metadata.name}")
            self._level3_loaded = False

    def pre_execute(self, context: SkillContext) -> tuple[bool, str]:
        """Pre-execution hook. Loads L2 and returns (proceed, reason)."""
        self.load_level2()
        return True, ""

    def post_execute(self, context: SkillContext, result: SkillResult):
        """Post-execution hook."""
        self._execution_count += 1
        self._total_duration_ms += result.duration_ms

    def get_stats(self) -> dict:
        return {
            "name": self.metadata.name,
            "execution_count": self._execution_count,
            "total_duration_ms": self._total_duration_ms,
            "avg_duration_ms": (
                self._total_duration_ms / self._execution_count
                if self._execution_count > 0 else 0
            ),
        }


class SkillRegistry:
    """
    Registry of all available skills.

    Maintains Level-1 metadata for all skills always in memory.
    Provides methods for skill lookup and routing.
    """

    def __init__(self):
        self._logger = logging.getLogger("INFRA-SKILL.SkillRegistry")
        self._skills: dict[str, Skill] = {}
        self._by_phase: dict[SkillPhase, list[Skill]] = {phase: [] for phase in SkillPhase}

    def register(self, skill: Skill) -> Self:
        name = skill.metadata.name
        if name in self._skills:
            self._logger.warning(f"Overwriting existing skill: {name}")
        self._skills[name] = skill
        self._by_phase[skill.metadata.phase].append(skill)
        self._by_phase[skill.metadata.phase].sort(key=lambda s: s.metadata.priority)
        self._logger.info(
            f"Registered: {name} "
            f"(phase={skill.metadata.phase.name}, priority={skill.metadata.priority}, "
            f"version={skill.metadata.version}, tags={skill.metadata.tags})"
        )
        return self

    def register_all(self, skills: list[Skill]) -> Self:
        for skill in skills:
            self.register(skill)
        return self

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def get_by_phase(self, phase: SkillPhase) -> list[Skill]:
        return self._by_phase.get(phase, [])

    def get_all(self) -> list[Skill]:
        return list(self._skills.values())

    def get_triggerable(self, god: GroundedObjectivesDocument) -> list[Skill]:
        triggerable = [s for s in self._skills.values() if s.can_trigger(god)]
        triggerable.sort(key=lambda s: (s.metadata.phase.value, s.metadata.priority))
        return triggerable

    def get_metadata_table(self) -> list[dict]:
        return [s.metadata.to_dict() for s in self._skills.values()]

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
