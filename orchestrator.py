"""
INFRA-SKILL Orchestrator
========================

A robust, deterministic state machine that coordinates skills for AWS CloudFormation generation.

The Orchestrator is NOT an LLM agent - it is a pure state machine that:
1. Manages the GOD (Grounded Objectives Document) lifecycle
2. Routes to skills based on GOD state
3. Enforces validation gates (no forward progress on failures)
4. Implements feedback loops with loop guards
5. Produces complete audit trails

Author: INFRA-SKILL
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
from datetime import datetime
from pathlib import Path
import sys
from enums import OrchestratorState, SkillPhase
from god import GroundedObjectivesDocument
from llm_client import OpenRouterClient
from logger import setup_logging
from prompt import SKILL_SELECTOR_PROMPT
from skill_framework import Skill, SkillContext, SkillRegistry, SkillResult
from skills.engineer import GeneralEngineerSkill, TemplateAssemblerSkill
from skills.planner import PlannerSkill
from skills.remediation import RemediationSkill
from skills.validator import CFNLintValidatorSkill, CheckovValidatorSkill, IntentAlignmentValidatorSkill, YAMLSyntaxValidatorSkill

@dataclass
class OrchestratorConfig:
    """Configuration for the Orchestrator"""
    
    # Loop guards
    max_remediation_rounds: int = 5
    max_total_iterations: int = 50
    max_skill_retries: int = 2
    
    # Validation behavior
    fail_on_critical_findings: bool = True
    fail_on_high_findings: bool = True
    allow_medium_findings: bool = True
    
    # Checkpointing
    enable_checkpoints: bool = True
    checkpoint_on_phase_change: bool = True
    checkpoint_on_validation_failure: bool = True
    
    # Logging
    verbose_logging: bool = True
    log_god_snapshots: bool = False
    
    # Execution
    skill_timeout_seconds: int = 120

    llm_client: Optional[object] = None  
    
    def to_dict(self) -> dict:
        return {
            "max_remediation_rounds": self.max_remediation_rounds,
            "max_total_iterations": self.max_total_iterations,
            "fail_on_critical_findings": self.fail_on_critical_findings,
            "fail_on_high_findings": self.fail_on_high_findings,
            "enable_checkpoints": self.enable_checkpoints
        }


# =============================================================================
# PART 5: ORCHESTRATOR EVENT SYSTEM
# =============================================================================

class OrchestratorEventType(Enum):
    """Events emitted by the orchestrator"""
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    STATE_CHANGED = "state_changed"
    PHASE_CHANGED = "phase_changed"
    SKILL_STARTED = "skill_started"
    SKILL_COMPLETED = "skill_completed"
    SKILL_FAILED = "skill_failed"
    VALIDATION_GATE_PASSED = "validation_gate_passed"
    VALIDATION_GATE_FAILED = "validation_gate_failed"
    REMEDIATION_STARTED = "remediation_started"
    LOOP_GUARD_TRIGGERED = "loop_guard_triggered"
    ESCALATION_REQUIRED = "escalation_required"


@dataclass
class OrchestratorEvent:
    """An event emitted by the orchestrator"""
    event_type: OrchestratorEventType
    timestamp: str
    data: dict = field(default_factory=dict)


# Type for event handlers
EventHandler = Callable[[OrchestratorEvent], None]


class EventEmitter:
    """Simple event emitter for orchestrator events"""
    
    def __init__(self):
        self._handlers: dict[OrchestratorEventType, list[EventHandler]] = {
            et: [] for et in OrchestratorEventType
        }
        self._global_handlers: list[EventHandler] = []
    
    def on(self, event_type: OrchestratorEventType, handler: EventHandler):
        """Register a handler for a specific event type"""
        self._handlers[event_type].append(handler)
    
    def on_any(self, handler: EventHandler):
        """Register a handler for all events"""
        self._global_handlers.append(handler)
    
    def emit(self, event_type: OrchestratorEventType, data: dict = None):
        """Emit an event"""
        event = OrchestratorEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            data=data or {}
        )
        
        # Call specific handlers
        for handler in self._handlers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                pass  # Don't let handler errors break the pipeline
        
        # Call global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception:
                pass


# =============================================================================
# PART 6: THE ORCHESTRATOR
# =============================================================================

class Orchestrator:
    """
    The Orchestrator is the central coordinator of the INFRA-SKILL system.
    
    It is a DETERMINISTIC STATE MACHINE, not an LLM agent. It makes routing
    decisions based purely on the GOD state and configuration rules.
    
    Responsibilities:
    1. Manage the GOD lifecycle
    2. Route to skills based on GOD state and skill triggers
    3. Enforce validation gates (no forward progress on failures)
    4. Implement feedback loops with loop guards
    5. Emit events for observability
    6. Produce complete audit trails
    
    Key principle: The orchestrator NEVER generates content. It only decides
    WHICH skill should run next and WHETHER to continue the pipeline.
    """
    
    def __init__(self, config: OrchestratorConfig = None):
        self._logger = logging.getLogger("INFRA-SKILL.Orchestrator")
        self.config = config or OrchestratorConfig()
        
        # Core components
        self.registry = SkillRegistry()
        self.events = EventEmitter()
        
        # Runtime state
        self.god: Optional[GroundedObjectivesDocument] = None
        self.state = OrchestratorState.UNINITIALIZED
        self._iteration = 0
        self._current_phase: Optional[SkillPhase] = None
        
        # Execution tracking
        self._execution_log: list[dict] = []
        self._state_history: list[tuple[OrchestratorState, str, str]] = []  # (state, reason, timestamp)
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
    
    # -------------------------------------------------------------------------
    # Builder Pattern for Configuration
    # -------------------------------------------------------------------------
    
    def with_config(self, config: OrchestratorConfig) -> Orchestrator:
        """Set configuration. Returns self for chaining."""
        self.config = config
        return self
    
    def with_skill(self, skill: Skill) -> Orchestrator:
        """Register a skill. Returns self for chaining."""
        self.registry.register(skill)
        return self
    
    def with_skills(self, skills: list[Skill]) -> Orchestrator:
        """Register multiple skills. Returns self for chaining."""
        self.registry.register_all(skills)
        return self
    
    def on_event(self, event_type: OrchestratorEventType, handler: EventHandler) -> Orchestrator:
        """Register an event handler. Returns self for chaining."""
        self.events.on(event_type, handler)
        return self
    
    # -------------------------------------------------------------------------
    # State Machine
    # -------------------------------------------------------------------------
    
    def _transition_to(self, new_state: OrchestratorState, reason: str = ""):
        """Transition to a new state with audit trail"""
        old_state = self.state
        self.state = new_state
        timestamp = datetime.now().isoformat()
        
        self._state_history.append((new_state, reason, timestamp))
        
        self._logger.info(f"State: {old_state.name} → {new_state.name}" + (f" ({reason})" if reason else ""))
        
        self.events.emit(OrchestratorEventType.STATE_CHANGED, {
            "old_state": old_state.name,
            "new_state": new_state.name,
            "reason": reason
        })
        
        # Checkpoint on major transitions
        if self.config.checkpoint_on_phase_change and self.god:
            self.god.save_checkpoint(f"state:{new_state.name}", "orchestrator")
    
    def _determine_target_state(self) -> OrchestratorState:
        """
        Determine what state we should be in based on GOD.
        This is the core routing logic.
        """
        if self.god is None:
            return OrchestratorState.UNINITIALIZED
        
        # Terminal states are sticky
        if self.state.is_terminal():
            return self.state
        
        # Check loop guard
        if self.god.get_remediation_round() >= self.config.max_remediation_rounds:
            return OrchestratorState.ESCALATED
        
        # Check iteration limit
        if self._iteration >= self.config.max_total_iterations:
            return OrchestratorState.ESCALATED
        
        # If all validations passed, we're done
        if self.god.all_validations_passed():
            return OrchestratorState.SUCCEEDED
        
        # If we have blocking failures, go to remediation
        if self.god.has_failed_validations():
            # But only if we haven't exhausted retries
            if self.god.get_remediation_round() < self.config.max_remediation_rounds:
                return OrchestratorState.REMEDIATING
            else:
                return OrchestratorState.ESCALATED
        
        # Determine state based on what's missing/pending
        if not self.god.intent.resources:
            return OrchestratorState.PLANNING
        
        if not self.god.template.resources and not self.god.template.body:
            return OrchestratorState.ENGINEERING
        
        # Check if template needs assembly
        if self.god.template.resources and not self.god.template.body:
            return OrchestratorState.ASSEMBLING
        
        if "AWSTemplateFormatVersion" not in self.god.template.body:
            return OrchestratorState.ASSEMBLING
        
        if self.god.has_pending_validations():
            return OrchestratorState.VALIDATING
        
        # Default to validating if we have a template
        if self.god.template.body:
            return OrchestratorState.VALIDATING
        
        return OrchestratorState.ENGINEERING
    
    def _state_to_phase(self, state: OrchestratorState) -> Optional[SkillPhase]:
        """Map orchestrator state to skill phase"""
        mapping = {
            OrchestratorState.PLANNING: SkillPhase.PLANNING,
            OrchestratorState.ENGINEERING: SkillPhase.ENGINEERING,
            OrchestratorState.ASSEMBLING: SkillPhase.ASSEMBLY,
            OrchestratorState.VALIDATING: SkillPhase.VALIDATION,
            OrchestratorState.REMEDIATING: SkillPhase.REMEDIATION,
        }
        return mapping.get(state)
    
    # -------------------------------------------------------------------------
    # Skill Selection and Execution
    # -------------------------------------------------------------------------

    def select_next_skill(self) -> Optional[Skill]:
        triggerable = self.registry.get_triggerable(self.god)
        if not triggerable:
            return None

        # Fast path: only one option — no need to invoke LLM
        if len(triggerable) == 1:
            return triggerable[0]

        llm = self.config.llm_client
        if llm is None:
            # Graceful fallback to deterministic routing if no LLM configured
            self._logger.warning("No LLM client — falling back to priority-based routing")
            return triggerable[0]

        god_snapshot = json.dumps(self.god.intent.to_dict(), indent=2)[:2000]  # truncate
        skill_table = json.dumps(self.registry.get_metadata_table(), indent=2)

        try:
            raw = llm.complete(
                system=SKILL_SELECTOR_PROMPT.format(
                    god_snapshot=god_snapshot,
                    skill_metadata_table=skill_table,
                ),
                user="Which skill should run next?",
                temperature=0.0,
            )
            decision = json.loads(raw)
            chosen_name = decision["skill_name"]
            rationale = decision.get("rationale", "")

            skill = self.registry.get(chosen_name)
            if skill and skill.can_trigger(self.god):
                self._logger.info(
                    f"LLM selected skill '{chosen_name}': {rationale}"
                )
                return skill

            # LLM hallucinated a skill name or chose a non-triggerable skill
            self._logger.warning(
                f"LLM chose invalid skill '{chosen_name}', "
                f"falling back to priority routing"
            )
            return triggerable[0]

        except (json.JSONDecodeError, KeyError) as e:
            self._logger.warning(f"Skill selector LLM failed ({e}), using priority fallback")
            return triggerable[0]
    
    def _check_phase_advancement(self, current_phase: SkillPhase) -> Optional[Skill]:
        """Check if we should advance to the next phase and return a skill from it"""
        
        # Phase advancement rules
        if current_phase == SkillPhase.ENGINEERING:
            # All engineering done, move to assembly
            assembly_skills = self.registry.get_by_phase(SkillPhase.ASSEMBLY)
            for skill in assembly_skills:
                if skill.can_trigger(self.god):
                    self._transition_to(OrchestratorState.ASSEMBLING, "engineering complete")
                    return skill
        
        elif current_phase == SkillPhase.ASSEMBLY:
            # Assembly done, move to validation
            self._transition_to(OrchestratorState.VALIDATING, "assembly complete")
            validation_skills = self.registry.get_by_phase(SkillPhase.VALIDATION)
            for skill in validation_skills:
                if skill.can_trigger(self.god):
                    return skill
        
        elif current_phase == SkillPhase.VALIDATION:
            # Check if all validations passed or if we need remediation
            if self.god.all_validations_passed():
                self._transition_to(OrchestratorState.SUCCEEDED, "all validations passed")
            elif self.god.has_failed_validations():
                self._transition_to(OrchestratorState.REMEDIATING, "validation failures detected")
                remediation_skills = self.registry.get_by_phase(SkillPhase.REMEDIATION)
                for skill in remediation_skills:
                    if skill.can_trigger(self.god):
                        return skill
        
        elif current_phase == SkillPhase.REMEDIATION:
            # After remediation, go back to validation
            self._transition_to(OrchestratorState.VALIDATING, "remediation complete")
            validation_skills = self.registry.get_by_phase(SkillPhase.VALIDATION)
            for skill in validation_skills:
                if skill.can_trigger(self.god):
                    return skill
        
        return None
    
    def _execute_skill(self, skill: Skill) -> SkillResult:
        """Execute a skill with full lifecycle management"""
        
        context = SkillContext(
            god=self.god,
            orchestrator_state=self.state,
            iteration=self._iteration,
            config={"llm": self.config.llm_client}
        )
        
        self.events.emit(OrchestratorEventType.SKILL_STARTED, {
            "skill_name": skill.metadata.name,
            "phase": skill.metadata.phase.name,
            "iteration": self._iteration
        })
        
        start_time = datetime.now()
        
        # Pre-execution check
        can_proceed, abort_reason = skill.pre_execute(context)
        if not can_proceed:
            result = SkillResult.failure(skill.metadata.name, f"Pre-execution failed: {abort_reason}")
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._record_execution(skill, result)
            return result
        
        # Execute
        try:
            result = skill.execute(context)
        except Exception as e:
            self._logger.error(f"Skill {skill.metadata.name} raised exception: {e}")
            self._logger.debug(traceback.format_exc())
            result = SkillResult.failure(skill.metadata.name, f"Exception: {str(e)}")
        
        result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Post-execution
        skill.post_execute(context, result)
        
        # Record execution
        self._record_execution(skill, result)
        
        # Emit event
        event_type = (
            OrchestratorEventType.SKILL_COMPLETED if result.success 
            else OrchestratorEventType.SKILL_FAILED
        )
        self.events.emit(event_type, {
            "skill_name": skill.metadata.name,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "changes_made": result.changes_made,
            "errors": result.errors
        })
        
        return result
    
    def _record_execution(self, skill: Skill, result: SkillResult):
        """Record skill execution for audit trail"""
        self._execution_log.append({
            "iteration": self._iteration,
            "timestamp": datetime.now().isoformat(),
            "skill_name": skill.metadata.name,
            "phase": skill.metadata.phase.name,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "changes_made": result.changes_made,
            "errors": result.errors,
            "warnings": result.warnings
        })
    
    # -------------------------------------------------------------------------
    # Main Execution Loop
    # -------------------------------------------------------------------------
    
    def run(self, user_prompt: str) -> dict:
        """
        Execute the full pipeline for a user prompt.
        
        This is the main entry point. It:
        1. Initializes the GOD with the user prompt
        2. Runs the skill selection/execution loop
        3. Returns a result dictionary
        
        Args:
            user_prompt: Natural language description of desired infrastructure
            
        Returns:
            Result dictionary with success status, template, and audit trail
        """
        self._logger.info("=" * 70)
        self._logger.info("INFRA-SKILL Pipeline Starting")
        self._logger.info("=" * 70)
        self._logger.info(f"Prompt: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")
        
        # Initialize
        self._start_time = datetime.now()
        self._iteration = 0
        self._execution_log = []
        self._state_history = []
        
        # Create and initialize GOD
        self.god = GroundedObjectivesDocument()
        self.god.intent.raw_prompt = user_prompt
        self.god.lock_field("intent.raw_prompt", "orchestrator")  # Immutable
        self.god.save_checkpoint("initialized", "orchestrator")
        
        self._transition_to(OrchestratorState.INITIALIZING, "pipeline start")
        
        self.events.emit(OrchestratorEventType.PIPELINE_STARTED, {
            "prompt_length": len(user_prompt),
            "registered_skills": len(self.registry)
        })
        
        # Main execution loop
        self._transition_to(OrchestratorState.PLANNING, "initialization complete")
        
        while self._iteration < self.config.max_total_iterations:
            self._iteration += 1
            
            if self.config.verbose_logging:
                self._log_iteration_status()
            
            # Check terminal states
            if self.state.is_terminal():
                self._logger.info(f"Terminal state reached: {self.state.name}")
                break
            
            # Select next skill
            skill = self.select_next_skill()
            
            if skill is None:
                self._logger.info("No skill can trigger")
                
                # Determine final state
                if self.god.all_validations_passed():
                    self._transition_to(OrchestratorState.SUCCEEDED, "all validations passed")
                elif self.god.has_failed_validations():
                    if self.god.get_remediation_round() >= self.config.max_remediation_rounds:
                        self._transition_to(OrchestratorState.ESCALATED, "max remediation rounds exceeded")
                        self.events.emit(OrchestratorEventType.LOOP_GUARD_TRIGGERED, {
                            "remediation_rounds": self.god.get_remediation_round()
                        })
                    else:
                        self._transition_to(OrchestratorState.FAILED, "no skills can remediate failures")
                else:
                    self._transition_to(OrchestratorState.FAILED, "pipeline stuck - no triggerable skills")
                break
            
            # Execute skill
            self._logger.info(f"Executing: {skill.metadata.name}")
            result = self._execute_skill(skill)
            
            # Handle special results
            if result.requires_human_review:
                self._transition_to(OrchestratorState.ESCALATED, result.escalation_reason or "human review required")
                self.events.emit(OrchestratorEventType.ESCALATION_REQUIRED, {
                    "skill_name": skill.metadata.name,
                    "reason": result.escalation_reason
                })
                break
            
            # Log result
            if result.success:
                for change in result.changes_made:
                    self._logger.info(f"  ✓ {change}")
            else:
                for error in result.errors:
                    self._logger.warning(f"  ✗ {error}")
        
        # Finalize
        self._end_time = datetime.now()
        
        self.events.emit(OrchestratorEventType.PIPELINE_COMPLETED, {
            "final_state": self.state.name,
            "iterations": self._iteration,
            "duration_ms": (self._end_time - self._start_time).total_seconds() * 1000
        })
        
        return self._build_result()
    
    def _log_iteration_status(self):
        """Log current iteration status"""
        self._logger.info(f"\n{'─' * 50}")
        self._logger.info(f"Iteration {self._iteration} | State: {self.state.name}")
        self._logger.info(f"Validations: {self.god.get_validation_summary()}")
        self._logger.info(f"Remediation Rounds: {self.god.get_remediation_round()}")
    
    def _build_result(self) -> dict:
        """Build the final result dictionary"""
        success = self.state == OrchestratorState.SUCCEEDED
        
        duration_ms = 0
        if self._start_time and self._end_time:
            duration_ms = (self._end_time - self._start_time).total_seconds() * 1000
        
        return {
            # Status
            "success": success,
            "state": self.state.name,
            
            # Output
            "template": self.god.template.body if success else None,
            "template_checksum": self.god.template.checksum if success else None,
            
            # Metrics
            "iterations": self._iteration,
            "duration_ms": duration_ms,
            "remediation_rounds": self.god.get_remediation_round(),
            
            # Validation summary
            "validation_summary": self.god.get_validation_summary(),
            "findings_summary": self.god.get_findings_summary(),
            "blocking_findings": [f.to_dict() for f in self.god.get_blocking_findings()],
            
            # Audit trail
            "execution_log": self._execution_log,
            "state_history": [
                {"state": s.name, "reason": r, "timestamp": t}
                for s, r, t in self._state_history
            ],
            
            # Full GOD state (for debugging)
            "god_snapshot": self.god.snapshot() if self.config.log_god_snapshots else None
        }
    
    # -------------------------------------------------------------------------
    # Inspection and Debugging
    # -------------------------------------------------------------------------
    
    def get_status(self) -> dict:
        """Get current orchestrator status"""
        return {
            "state": self.state.name,
            "iteration": self._iteration,
            "current_phase": self._current_phase.name if self._current_phase else None,
            "god_summary": self.god.snapshot()["summary"] if self.god else None,
            "registered_skills": len(self.registry),
            "recent_executions": self._execution_log[-5:] if self._execution_log else []
        }
    
    def print_skill_registry(self):
        """Print registered skills in a formatted table"""
        print("\n" + "=" * 70)
        print("REGISTERED SKILLS")
        print("=" * 70)
        
        for phase in SkillPhase:
            skills = self.registry.get_by_phase(phase)
            if skills:
                print(f"\n{phase.name}:")
                print("-" * 60)
                for skill in skills:
                    m = skill.metadata
                    print(f"  [{m.priority:3}] {m.name:25} - {m.description[:40]}")
        
        print("\n" + "=" * 70)
    
    def print_result(self, result: dict):
        """Print a formatted result summary"""
        print("\n" + "=" * 70)
        print("PIPELINE RESULT")
        print("=" * 70)
        
        status_icon = "✓" if result["success"] else "✗"
        status_color = "\033[32m" if result["success"] else "\033[31m"
        reset = "\033[0m"
        
        print(f"\nStatus: {status_color}{status_icon} {result['state']}{reset}")
        print(f"Iterations: {result['iterations']}")
        print(f"Duration: {result['duration_ms']:.1f}ms")
        print(f"Remediation Rounds: {result['remediation_rounds']}")
        
        print("\nValidation Summary:")
        for validator, status in result["validation_summary"].items():
            icon = "✓" if status == "PASS" else ("✗" if status == "FAIL" else "○")
            print(f"  {icon} {validator}: {status}")
        
        print("\nFindings Summary:")
        for severity, count in result["findings_summary"].items():
            if count > 0:
                print(f"  {severity}: {count}")
        
        if result["blocking_findings"]:
            print("\nBlocking Findings:")
            for finding in result["blocking_findings"][:5]:
                print(f"  [{finding['severity']}] {finding['rule_id']}: {finding['message']}")
        
        if result["success"] and result["template"]:
            print("\nGenerated Template:")
            print("-" * 50)
            # Print first 50 lines
            lines = result["template"].split("\n")
            for line in lines[:50]:
                print(line)
            if len(lines) > 50:
                print(f"... ({len(lines) - 50} more lines)")
        
        print("\n" + "=" * 70)

# =============================================================================
# PART 8: SKILL FACTORY
# =============================================================================

def create_default_skills() -> list[Skill]:
    """Create the default set of skills for AWS CloudFormation generation"""
    return [
        # Planning
        PlannerSkill(),
        
        # Engineering
        # VPCEngineerSkill(),
        # SecurityGroupEngineerSkill(),
        # IAMEngineerSkill(),
        # S3EngineerSkill(),
        # RDSEngineerSkill(),
        # LambdaEngineerSkill(),
        GeneralEngineerSkill(),
        
        # Assembly
        TemplateAssemblerSkill(),
        
        # Validation
        YAMLSyntaxValidatorSkill(),
        CFNLintValidatorSkill(),
        CheckovValidatorSkill(),
        IntentAlignmentValidatorSkill(),
        
        # Remediation
        RemediationSkill(),
    ]


def create_orchestrator(
    config: OrchestratorConfig = None,
    skills: list[Skill] = None
) -> Orchestrator:
    """
    Factory function to create a fully configured orchestrator.
    
    Args:
        config: Orchestrator configuration (uses defaults if not provided)
        skills: List of skills to register (uses default set if not provided)
    
    Returns:
        Configured Orchestrator instance
    """
    config = config or OrchestratorConfig()
    skills = skills if skills is not None else create_default_skills()
    
    orchestrator = Orchestrator(config)
    orchestrator.with_skills(skills)
    
    return orchestrator


# =============================================================================
# PART 9: CLI AND MAIN EXECUTION
# =============================================================================

def demo_event_handler(event: OrchestratorEvent):
    """Demo event handler that prints events"""
    icon = {
        OrchestratorEventType.PIPELINE_STARTED: "🚀",
        OrchestratorEventType.PIPELINE_COMPLETED: "🏁",
        OrchestratorEventType.STATE_CHANGED: "⚡",
        OrchestratorEventType.SKILL_STARTED: "▶️",
        OrchestratorEventType.SKILL_COMPLETED: "✅",
        OrchestratorEventType.SKILL_FAILED: "❌",
        OrchestratorEventType.VALIDATION_GATE_PASSED: "✓",
        OrchestratorEventType.VALIDATION_GATE_FAILED: "✗",
        OrchestratorEventType.LOOP_GUARD_TRIGGERED: "⚠️",
        OrchestratorEventType.ESCALATION_REQUIRED: "🚨",
    }.get(event.event_type, "•")
    
    # Only print important events
    if event.event_type in [
        OrchestratorEventType.PIPELINE_STARTED,
        OrchestratorEventType.PIPELINE_COMPLETED,
        OrchestratorEventType.SKILL_COMPLETED,
        OrchestratorEventType.SKILL_FAILED,
        OrchestratorEventType.ESCALATION_REQUIRED,
    ]:
        print(f"  {icon} {event.event_type.value}: {event.data}")


def main():
    """Main entry point demonstrating the INFRA-SKILL orchestrator"""
    
    # Setup logging
    setup_logging(logging.INFO)
    logger = logging.getLogger("INFRA-SKILL.Main")
    
    # Create orchestrator with default configuration
    config = OrchestratorConfig(
        max_remediation_rounds=5,
        max_total_iterations=30,
        verbose_logging=True,
        enable_checkpoints=True,
        llm_client=OpenRouterClient(
            model="arcee-ai/trinity-large-preview:free",
        ),
    )
    
    orchestrator = create_orchestrator(config)
    
    # Register demo event handler
    orchestrator.on_event(OrchestratorEventType.PIPELINE_COMPLETED, demo_event_handler)
    orchestrator.on_event(OrchestratorEventType.ESCALATION_REQUIRED, demo_event_handler)
    
    # Print registered skills
    orchestrator.print_skill_registry()
    
    # Example prompts
    prompts = [
        """
        Create a production VPC with multi-AZ subnets, an S3 bucket for application logs,
        and a PostgreSQL RDS database. The environment should be highly available,
        encrypted, and follow CIS security best practices. No public access should be allowed.
        """,
        
        # Additional examples (uncomment to try):
        "Create a simple S3 bucket for storing static assets with encryption enabled",
        "Set up a Lambda function with an IAM role for processing data",
        "Create a development VPC with a single subnet and basic security group",
    ]
    
    for prompt in prompts:
        print("\n" + "=" * 80)
        print("USER PROMPT:")
        print("-" * 80)
        print(prompt.strip())
        print("=" * 80)
        
        # Run the pipeline
        result = orchestrator.run(prompt.strip())
        
        # Print results
        orchestrator.print_result(result)
        
        # Export template to file if successful
        if result["success"]:
            output_file = Path("generated_template.yaml")
            output_file.write_text(result["template"])
            logger.info(f"Template written to: {output_file}")
            
            # Also export the audit trail
            audit_file = Path("audit_trail.json")
            audit_file.write_text(json.dumps(orchestrator.god.export_audit_trail(), indent=2))
            logger.info(f"Audit trail written to: {audit_file}")


def run_interactive():
    """Interactive mode for testing prompts"""
    setup_logging(logging.INFO)
    
    print("\n" + "=" * 70)
    print("INFRA-SKILL Interactive Mode")
    print("=" * 70)
    print("Enter your infrastructure requirements, or 'quit' to exit.\n")
    
    orchestrator = create_orchestrator()
    orchestrator.print_skill_registry()
    
    while True:
        print("\n" + "-" * 50)
        prompt = input("Prompt> ").strip()
        
        if prompt.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not prompt:
            continue
        
        result = orchestrator.run(prompt)
        orchestrator.print_result(result)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive()
    else:
        main()
