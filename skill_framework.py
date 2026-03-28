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
    Level 1 Metadata - Always in memory for routing decisions.
    
    This is the lightweight descriptor that the orchestrator uses to
    decide which skill to invoke. It contains no actual logic.
    """
    name: str
    description: str
    phase: SkillPhase
    
    # Trigger condition (human-readable, for documentation)
    trigger_condition: str
    
    # Field access control
    writes_to: list[str]    # GOD fields this skill can write
    reads_from: list[str]   # GOD fields this skill needs to read
    
    # Execution parameters
    llm_description: str = ""
    priority: int = 100     # Lower = higher priority within phase
    timeout_seconds: int = 60
    retryable: bool = True
    max_retries: int = 2
    
    # Dependencies
    requires_skills: list[str] = field(default_factory=list)  # Skills that must run first
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "phase": self.phase.name,
            "priority": self.priority,
            "writes_to": self.writes_to,
            "reads_from": self.reads_from
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
    
    # What happened
    changes_made: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    # Metrics
    duration_ms: float = 0
    metrics: dict = field(default_factory=dict)
    
    # Control flow hints
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
            "requires_human_review": self.requires_human_review
        }
    
    @classmethod
    def failure(cls, skill_name: str, error: str) -> Self:
        """Create a failure result"""
        return cls(success=False, skill_name=skill_name, errors=[error])
    
    @classmethod
    def success_with_changes(cls, skill_name: str, changes: list[str]) -> Self:
        """Create a success result with changes"""
        return cls(success=True, skill_name=skill_name, changes_made=changes)


class Skill(ABC):
    """
    Abstract base class for all skills.
    
    Skills implement the three-level progressive disclosure pattern:
    - Level 1: Metadata (always loaded) - lightweight, for routing
    - Level 2: Procedural logic (loaded on activation) - the execute() method
    - Level 3: Heavy assets (loaded on demand) - schemas, templates, etc.
    
    Subclasses must implement:
    - _define_metadata(): Return the skill's metadata
    - can_trigger(god): Check if the skill should run
    - execute(context): Perform the skill's work
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
        """Define Level 1 metadata. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        """
        Check if this skill should be activated based on GOD state.
        This is called frequently during routing, so it should be fast.
        """
        pass
    
    @abstractmethod
    def execute(self, context: SkillContext) -> SkillResult:
        """
        Execute the skill's main logic.
        
        Must only write to GOD fields specified in metadata.writes_to.
        Should return a SkillResult indicating success/failure.
        """
        pass
    
    def load_level2(self) -> bool:
        """Load Level 2 assets (procedural logic). Override if needed."""
        if not self._level2_loaded:
            self._logger.debug(f"Loading Level 2 assets for {self.metadata.name}")
            self._level2_loaded = True
        return True
    
    def load_level3(self) -> bool:
        """Load Level 3 assets (heavy resources). Override for lazy loading."""
        if not self._level3_loaded:
            self._logger.debug(f"Loading Level 3 assets for {self.metadata.name}")
            self._level3_loaded = True
        return True
    
    def unload_level3(self):
        """Unload Level 3 assets to free memory."""
        if self._level3_loaded:
            self._logger.debug(f"Unloading Level 3 assets for {self.metadata.name}")
            self._level3_loaded = False
    
    def pre_execute(self, context: SkillContext) -> tuple[bool, str]:
        """
        Pre-execution hook. Return (True, "") to proceed, (False, reason) to abort.
        """
        self.load_level2()
        return True, ""
    
    def post_execute(self, context: SkillContext, result: SkillResult):
        """Post-execution hook. Called after execute() completes."""
        self._execution_count += 1
        self._total_duration_ms += result.duration_ms
    
    def get_stats(self) -> dict:
        """Get execution statistics for this skill"""
        return {
            "name": self.metadata.name,
            "execution_count": self._execution_count,
            "total_duration_ms": self._total_duration_ms,
            "avg_duration_ms": (
                self._total_duration_ms / self._execution_count 
                if self._execution_count > 0 else 0
            )
        }


class SkillRegistry:
    """
    Registry of all available skills.
    
    Maintains Level 1 metadata for all skills always in memory.
    Provides methods for skill lookup and routing.
    """
    
    def __init__(self):
        self._logger = logging.getLogger("INFRA-SKILL.SkillRegistry")
        self._skills: dict[str, Skill] = {}
        self._by_phase: dict[SkillPhase, list[Skill]] = {phase: [] for phase in SkillPhase}
    
    def register(self, skill: Skill) -> Self:
        """Register a skill. Returns self for chaining."""
        name = skill.metadata.name
        
        if name in self._skills:
            self._logger.warning(f"Overwriting existing skill: {name}")
        
        self._skills[name] = skill
        self._by_phase[skill.metadata.phase].append(skill)
        
        # Sort by priority within phase
        self._by_phase[skill.metadata.phase].sort(key=lambda s: s.metadata.priority)
        
        self._logger.info(f"Registered: {name} (phase={skill.metadata.phase.name}, priority={skill.metadata.priority})")
        
        return self
    
    def register_all(self, skills: list[Skill]) -> Self:
        """Register multiple skills. Returns self for chaining."""
        for skill in skills:
            self.register(skill)
        return self
    
    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name"""
        return self._skills.get(name)
    
    def get_by_phase(self, phase: SkillPhase) -> list[Skill]:
        """Get all skills for a phase, sorted by priority"""
        return self._by_phase.get(phase, [])
    
    def get_all(self) -> list[Skill]:
        """Get all registered skills"""
        return list(self._skills.values())
    
    def get_triggerable(self, god: GroundedObjectivesDocument) -> list[Skill]:
        """Get all skills that can currently trigger, sorted by phase then priority"""
        triggerable = [s for s in self._skills.values() if s.can_trigger(god)]
        triggerable.sort(key=lambda s: (s.metadata.phase.value, s.metadata.priority))
        return triggerable
    
    def get_metadata_table(self) -> list[dict]:
        """Get Level 1 metadata for all skills as a table"""
        return [s.metadata.to_dict() for s in self._skills.values()]
    
    def __len__(self) -> int:
        return len(self._skills)
    
    def __contains__(self, name: str) -> bool:
        return name in self._skills

