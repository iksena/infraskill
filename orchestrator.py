"""
INFRA-SKILL Orchestrator
========================

A robust, deterministic state machine that coordinates skills for AWS CloudFormation generation.

The Orchestrator is NOT an LLM agent - it is a pure state machine that:
1. Manages the GOD (Grounded Objectives Document) lifecycle
2. Routes to skills based on GOD state
3. Enforces validation gates (no forward progress on failures)
4. Implements feedback loops — the ONLY stop condition is max_total_iterations
5. Produces complete audit trails

Loop-guard policy (v1.5.0)
--------------------------
There is exactly ONE stop condition: ``max_total_iterations``.

The old ``max_remediation_rounds`` check has been removed from the routing
logic.  As long as iterations remain, the pipeline WILL keep trying to fix
the template.  ESCALATED is only reachable when the iteration counter is
exhausted.  All validations MUST be PASS (or SKIPPED/ERROR for unavailable
tools) before the pipeline can reach SUCCEEDED.

Author: INFRA-SKILL
Version: 1.5.0
"""

from __future__ import annotations

import logging
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
from datetime import datetime
from enums import OrchestratorState, SkillPhase
from god import GroundedObjectivesDocument
from skill_framework import Skill, SkillContext, SkillRegistry, SkillResult
from skills.engineer import GeneralEngineerSkill
from skills.planner import PlannerSkill
from skills.remediation import RemediationSkill
from skills.validator import CFNLintValidatorSkill, CheckovValidatorSkill, IntentAlignmentValidatorSkill, YAMLSyntaxValidatorSkill
from telemetry import TelemetryRecorder


# =============================================================================
# MODULE-LEVEL FACTORY
# =============================================================================

def create_default_skills() -> list[Skill]:
    """
    Return the canonical ordered list of skills for the INFRA-SKILL pipeline.

    Call order within each phase is controlled by SkillMetadata.priority (lower
    number = higher priority).  The registry resolves ordering; this list just
    needs to contain every skill the pipeline requires.

    Phases & skills
    ---------------
    PLANNING    : PlannerSkill
    ENGINEERING : GeneralEngineerSkill
    VALIDATION  : YAMLSyntaxValidatorSkill (priority 10)
                  CFNLintValidatorSkill    (priority 20)
                  CheckovValidatorSkill    (priority 30)
                  IntentAlignmentValidatorSkill (priority 40)
    REMEDIATION : RemediationSkill
    """
    return [
        PlannerSkill(),
        GeneralEngineerSkill(),
        YAMLSyntaxValidatorSkill(),
        CFNLintValidatorSkill(),
        CheckovValidatorSkill(),
        IntentAlignmentValidatorSkill(),
        RemediationSkill(),
    ]


@dataclass
class OrchestratorConfig:
    """Configuration for the Orchestrator"""

    # Loop guard — the ONLY stop condition.
    max_total_iterations: int = 50
    max_skill_retries: int = 50

    # Checkpointing
    enable_checkpoints: bool = True
    checkpoint_on_phase_change: bool = True
    checkpoint_on_validation_failure: bool = True

    # Logging
    verbose_logging: bool = True
    log_god_snapshots: bool = False

    # Execution
    skill_timeout_seconds: int = 3600

    # Passed through to skills that require LLM calls for generation/validation.
    llm_client: Optional[object] = None

    # Telemetry
    telemetry_dir: str = "telemetry"
    enable_telemetry: bool = True

    def to_dict(self) -> dict:
        return {
            "max_total_iterations": self.max_total_iterations,
            "enable_checkpoints": self.enable_checkpoints,
            "enable_telemetry": self.enable_telemetry,
            "telemetry_dir": self.telemetry_dir,
        }


# =============================================================================
# ORCHESTRATOR EVENT SYSTEM
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
    VALIDATOR_SKIPPED = "validator_skipped"


@dataclass
class OrchestratorEvent:
    """An event emitted by the orchestrator"""
    event_type: OrchestratorEventType
    timestamp: str
    data: dict = field(default_factory=dict)


EventHandler = Callable[[OrchestratorEvent], None]


class EventEmitter:
    """Simple event emitter for orchestrator events"""

    def __init__(self):
        self._handlers: dict[OrchestratorEventType, list[EventHandler]] = {
            et: [] for et in OrchestratorEventType
        }
        self._global_handlers: list[EventHandler] = []

    def on(self, event_type: OrchestratorEventType, handler: EventHandler):
        self._handlers[event_type].append(handler)

    def on_any(self, handler: EventHandler):
        self._global_handlers.append(handler)

    def emit(self, event_type: OrchestratorEventType, data: dict = None):
        event = OrchestratorEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            data=data or {}
        )
        for handler in self._handlers.get(event_type, []):
            try:
                handler(event)
            except Exception:
                pass
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception:
                pass


# =============================================================================
# THE ORCHESTRATOR
# =============================================================================

class Orchestrator:
    """
    Deterministic state machine coordinating the INFRA-SKILL pipeline.

    v1.5.0 — single loop guard
    --------------------------
    The only way to reach ESCALATED is by exhausting ``max_total_iterations``.
    RemediationSkill.can_trigger() no longer has a round cap, and
    _determine_target_state() no longer checks get_remediation_round().
    The pipeline will keep cycling VALIDATING → REMEDIATING → VALIDATING
    until either every validator returns PASS/SKIPPED/ERROR (→ SUCCEEDED)
    or the iteration budget is spent (→ ESCALATED).

    v1.4.1 — restored create_default_skills() at module level.
    v1.4.0 — TelemetryRecorder injected per-run via SkillContext.
    """

    _STAGE_ORDER: list[OrchestratorState] = [
        OrchestratorState.PLANNING,
        OrchestratorState.ENGINEERING,
        OrchestratorState.ASSEMBLING,
        OrchestratorState.VALIDATING,
        OrchestratorState.REMEDIATING,
    ]

    def __init__(self, config: OrchestratorConfig = None):
        self._logger = logging.getLogger("INFRA-SKILL.Orchestrator")
        self.config = config or OrchestratorConfig()
        self.registry = SkillRegistry()
        self.events = EventEmitter()
        self.god: Optional[GroundedObjectivesDocument] = None
        self.state = OrchestratorState.UNINITIALIZED
        self._iteration = 0
        self._current_phase: Optional[SkillPhase] = None
        self._execution_log: list[dict] = []
        self._state_history: list[tuple[OrchestratorState, str, str]] = []
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._consecutive_failures: dict[str, int] = defaultdict(int)
        self._telemetry: Optional[TelemetryRecorder] = None

    # -------------------------------------------------------------------------
    # Builder Pattern
    # -------------------------------------------------------------------------

    def with_config(self, config: OrchestratorConfig) -> Orchestrator:
        self.config = config
        return self

    def with_skill(self, skill: Skill) -> Orchestrator:
        self.registry.register(skill)
        return self

    def with_skills(self, skills: list[Skill]) -> Orchestrator:
        self.registry.register_all(skills)
        return self

    def on_event(self, event_type: OrchestratorEventType, handler: EventHandler) -> Orchestrator:
        self.events.on(event_type, handler)
        return self

    # -------------------------------------------------------------------------
    # State Machine
    # -------------------------------------------------------------------------

    def _transition_to(self, new_state: OrchestratorState, reason: str = ""):
        old_state = self.state
        self.state = new_state
        timestamp = datetime.now().isoformat()
        self._state_history.append((new_state, reason, timestamp))
        self._logger.info(f"State: {old_state.name} -> {new_state.name}" + (f" ({reason})" if reason else ""))
        self.events.emit(OrchestratorEventType.STATE_CHANGED, {
            "old_state": old_state.name,
            "new_state": new_state.name,
            "reason": reason
        })
        if self._telemetry:
            self._telemetry.record_orchestrator_event(
                event_type="STATE_CHANGED",
                iteration=self._iteration,
                data={"old_state": old_state.name, "new_state": new_state.name, "reason": reason},
            )
        if self.config.checkpoint_on_phase_change and self.god:
            self.god.save_checkpoint(f"state:{new_state.name}", "orchestrator")

    # -------------------------------------------------------------------------
    # Deterministic target-state resolution
    # -------------------------------------------------------------------------

    def _determine_target_state(self) -> OrchestratorState:
        """
        Resolve the next state from GOD facts alone.

        Stop conditions (in priority order)
        ------------------------------------
        1. Already terminal                    → stay terminal
        2. max_total_iterations exhausted      → ESCALATED  (ONLY stop condition)
        3. All validators PASS/SKIPPED/ERROR   → SUCCEEDED
        4. Any FAIL with blocking findings     → REMEDIATING
        5. No intent resources yet             → PLANNING
        6. No template body yet                → ENGINEERING / ASSEMBLING
        7. Default                             → VALIDATING
        """
        if self.god is None:
            return OrchestratorState.UNINITIALIZED

        if self.state.is_terminal():
            return self.state

        # ── Single stop condition ────────────────────────────────────────────
        if self._iteration >= self.config.max_total_iterations:
            return OrchestratorState.ESCALATED

        # ── Success ─────────────────────────────────────────────────────────
        if self.god.all_validations_passed():
            if self.god.has_error_validations():
                error_names = [
                    name for name, v in self.god.validation_state.items()
                    if v.status.value == "ERROR"
                ]
                self._logger.warning(
                    f"Pipeline succeeded with skipped validators "
                    f"(tool unavailable): {error_names}"
                )
            return OrchestratorState.SUCCEEDED

        # ── Remediation loop ─────────────────────────────────────────────────
        if self.god.has_remediable_failures():
            return OrchestratorState.REMEDIATING

        # ── Forward planning / engineering ───────────────────────────────────
        if not self.god.intent.resources:
            return OrchestratorState.PLANNING

        if not self.god.template.resources and not self.god.template.body:
            return OrchestratorState.ENGINEERING

        if self.god.template.resources and not self.god.template.body:
            return OrchestratorState.ASSEMBLING

        if "AWSTemplateFormatVersion" not in self.god.template.body:
            return OrchestratorState.ASSEMBLING

        return OrchestratorState.VALIDATING

    def _state_to_phase(self, state: OrchestratorState) -> Optional[SkillPhase]:
        mapping = {
            OrchestratorState.PLANNING: SkillPhase.PLANNING,
            OrchestratorState.ENGINEERING: SkillPhase.ENGINEERING,
            OrchestratorState.ASSEMBLING: SkillPhase.ASSEMBLY,
            OrchestratorState.VALIDATING: SkillPhase.VALIDATION,
            OrchestratorState.REMEDIATING: SkillPhase.REMEDIATION,
        }
        return mapping.get(state)

    def _phase_to_state(self, phase: SkillPhase) -> Optional[OrchestratorState]:
        mapping = {
            SkillPhase.PLANNING: OrchestratorState.PLANNING,
            SkillPhase.ENGINEERING: OrchestratorState.ENGINEERING,
            SkillPhase.ASSEMBLY: OrchestratorState.ASSEMBLING,
            SkillPhase.VALIDATION: OrchestratorState.VALIDATING,
            SkillPhase.REMEDIATION: OrchestratorState.REMEDIATING,
        }
        return mapping.get(phase)

    def _set_current_phase(self, target_phase: SkillPhase):
        if target_phase == self._current_phase:
            return
        old_phase = self._current_phase
        self._current_phase = target_phase
        self.events.emit(OrchestratorEventType.PHASE_CHANGED, {
            "old_phase": old_phase.name if old_phase else None,
            "new_phase": target_phase.name,
        })
        if self._telemetry:
            self._telemetry.record_orchestrator_event(
                event_type="PHASE_CHANGED",
                iteration=self._iteration,
                data={
                    "old_phase": old_phase.name if old_phase else None,
                    "new_phase": target_phase.name,
                },
            )

    # -------------------------------------------------------------------------
    # Skill selection
    # -------------------------------------------------------------------------

    def select_next_skill(self) -> Optional[Skill]:
        target_state = self._determine_target_state()

        if target_state != self.state:
            reason_map = {
                OrchestratorState.PLANNING:    "intent missing -- re-plan",
                OrchestratorState.ENGINEERING: "template missing -- engineer",
                OrchestratorState.ASSEMBLING:  "resources ready -- assemble",
                OrchestratorState.VALIDATING:  "template ready -- validate",
                OrchestratorState.REMEDIATING: "validation failures -- remediate",
                OrchestratorState.SUCCEEDED:   "all validations passed",
                OrchestratorState.ESCALATED:   "max iterations exhausted",
            }
            self._transition_to(target_state, reason_map.get(target_state, "auto-transition"))

        if self.state.is_terminal():
            return None

        target_phase = self._state_to_phase(self.state)
        if target_phase is None:
            return None

        self._set_current_phase(target_phase)

        # Fully deterministic skill selection: phase-gated and priority-ordered.
        phase_skills = self.registry.get_by_phase(target_phase)
        candidates = [s for s in phase_skills if s.can_trigger(self.god)]
        if not candidates:
            return None

        selected_skill = candidates[0]

        if selected_skill is None:
            return None

        # Keep state/phase coherent with selected skill.
        selected_phase = selected_skill.metadata.phase
        self._set_current_phase(selected_phase)
        desired_state = self._phase_to_state(selected_phase)
        if desired_state and not self.state.is_terminal() and desired_state != self.state:
            self._transition_to(
                desired_state,
                f"selected skill '{selected_skill.metadata.name}' ({selected_phase.name})",
            )

        return selected_skill

    def _execute_skill(self, skill: Skill) -> SkillResult:
        timeout_s = self.config.skill_timeout_seconds
        context = SkillContext(
            god=self.god,
            orchestrator_state=self.state,
            iteration=self._iteration,
            config={
                "llm": self.config.llm_client,
                "llm_timeout": max(30, timeout_s - 10),
                "telemetry": self._telemetry,
            }
        )
        self.events.emit(OrchestratorEventType.SKILL_STARTED, {
            "skill_name": skill.metadata.name,
            "phase": skill.metadata.phase.name,
            "iteration": self._iteration
        })
        if self._telemetry:
            self._telemetry.record_orchestrator_event(
                event_type="SKILL_STARTED",
                iteration=self._iteration,
                data={"skill_name": skill.metadata.name, "phase": skill.metadata.phase.name},
            )
        start_time = datetime.now()
        can_proceed, abort_reason = skill.pre_execute(context)
        if not can_proceed:
            result = SkillResult.failure(
                skill.metadata.name, f"Pre-execution failed: {abort_reason}"
            )
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._record_execution(skill, result)
            return result
        result: SkillResult
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(skill.execute, context)
                try:
                    result = future.result(timeout=timeout_s)
                except FuturesTimeoutError:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    self._logger.error(
                        f"Skill '{skill.metadata.name}' timed out after "
                        f"{elapsed:.1f}s (limit={timeout_s}s)"
                    )
                    result = SkillResult.failure(
                        skill.metadata.name,
                        f"Skill timed out after {timeout_s}s"
                    )
        except Exception as e:
            self._logger.error(f"Skill {skill.metadata.name} raised exception: {e}")
            self._logger.debug(traceback.format_exc())
            result = SkillResult.failure(skill.metadata.name, f"Exception: {str(e)}")
        result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        skill.post_execute(context, result)
        self._record_execution(skill, result)
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
        entry = {
            "iteration": self._iteration,
            "timestamp": datetime.now().isoformat(),
            "skill_name": skill.metadata.name,
            "phase": skill.metadata.phase.name,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "changes_made": result.changes_made,
            "errors": result.errors,
            "warnings": result.warnings
        }
        self._execution_log.append(entry)
        if self._telemetry:
            self._telemetry.record_skill_execution(
                skill_name=skill.metadata.name,
                phase=skill.metadata.phase.name,
                iteration=self._iteration,
                success=result.success,
                duration_ms=result.duration_ms or 0.0,
                changes_made=result.changes_made,
                errors=result.errors,
                warnings=result.warnings,
            )

    # -------------------------------------------------------------------------
    # Main Execution Loop
    # -------------------------------------------------------------------------

    def run(self, user_prompt: str) -> dict:
        self._logger.info("=" * 70)
        self._logger.info("INFRA-SKILL Pipeline Starting")
        self._logger.info("=" * 70)
        self._logger.info(f"Prompt: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")
        self._start_time = datetime.now()
        self._iteration = 0
        self._execution_log = []
        self._state_history = []
        self._consecutive_failures = defaultdict(int)

        # ------------------------------------------------------------------
        # Telemetry: create recorder and start the run
        # ------------------------------------------------------------------
        run_id = self._start_time.strftime("%Y%m%dT%H%M%S") + f"_{id(self):x}"
        if self.config.enable_telemetry:
            self._telemetry = TelemetryRecorder(base_dir=self.config.telemetry_dir)
            self._telemetry.start_run(
                run_id=run_id,
                prompt=user_prompt,
                config=self.config.to_dict(),
            )
        else:
            self._telemetry = None

        self.god = GroundedObjectivesDocument()
        self.god.intent.raw_prompt = user_prompt
        self.god.lock_field("intent.raw_prompt", "orchestrator")
        self.god.save_checkpoint("initialized", "orchestrator")
        self._transition_to(OrchestratorState.INITIALIZING, "pipeline start")
        self.events.emit(OrchestratorEventType.PIPELINE_STARTED, {
            "prompt_length": len(user_prompt),
            "registered_skills": len(self.registry)
        })
        if self._telemetry:
            self._telemetry.record_orchestrator_event(
                event_type="PIPELINE_STARTED",
                iteration=0,
                data={"run_id": run_id, "prompt_length": len(user_prompt),
                      "registered_skills": len(self.registry)},
            )
        self._transition_to(OrchestratorState.PLANNING, "initialization complete")

        while self._iteration < self.config.max_total_iterations:
            self._iteration += 1
            if self.config.verbose_logging:
                self._log_iteration_status()
            if self.state.is_terminal():
                self._logger.info(f"Terminal state reached: {self.state.name}")
                break
            skill = self.select_next_skill()
            if skill is None:
                self._logger.info("No skill can trigger")
                if self.state.is_terminal():
                    pass
                elif self.god.all_validations_passed():
                    self._transition_to(OrchestratorState.SUCCEEDED, "all validations passed")
                else:
                    # No skill could trigger but we are not yet succeeded and
                    # not yet at max_total_iterations.  This means the pipeline
                    # is genuinely stuck (e.g. all validators ERROR/SKIPPED and
                    # no findings to remediate, yet template body is missing).
                    self._transition_to(
                        OrchestratorState.FAILED,
                        f"pipeline stuck -- no triggerable skills in phase {self._current_phase}"
                    )
                break
            self._logger.info(f"Executing: {skill.metadata.name}")
            result = self._execute_skill(skill)

            if result.requires_human_review:
                self._transition_to(
                    OrchestratorState.ESCALATED,
                    result.escalation_reason or "human review required"
                )
                self.events.emit(OrchestratorEventType.ESCALATION_REQUIRED, {
                    "skill_name": skill.metadata.name,
                    "reason": result.escalation_reason
                })
                break

            if result.success:
                self._consecutive_failures[skill.metadata.name] = 0
                for change in result.changes_made:
                    self._logger.info(f"  v {change}")
            else:
                self._consecutive_failures[skill.metadata.name] += 1
                consec = self._consecutive_failures[skill.metadata.name]
                for error in result.errors:
                    self._logger.warning(f"  x {error}")
                if consec >= self.config.max_skill_retries:
                    self._logger.error(
                        f"Skill '{skill.metadata.name}' failed {consec} consecutive "
                        f"time(s) (max={self.config.max_skill_retries}). "
                        "Aborting pipeline."
                    )
                    self._transition_to(
                        OrchestratorState.FAILED,
                        f"{skill.metadata.name} exceeded max consecutive failures"
                    )
                    break
                self._logger.info(
                    f"  Skill failed (attempt {consec}/{self.config.max_skill_retries}), "
                    "retrying same phase next iteration."
                )

        # Out of iteration budget
        if not self.state.is_terminal():
            self._logger.warning(
                f"max_total_iterations ({self.config.max_total_iterations}) exhausted "
                f"with unresolved findings. Escalating."
            )
            self._transition_to(
                OrchestratorState.ESCALATED,
                f"max_total_iterations ({self.config.max_total_iterations}) exhausted"
            )
            self.events.emit(OrchestratorEventType.LOOP_GUARD_TRIGGERED, {
                "iterations": self._iteration,
                "max_total_iterations": self.config.max_total_iterations,
            })

        self._end_time = datetime.now()
        duration_ms = (self._end_time - self._start_time).total_seconds() * 1000
        self.events.emit(OrchestratorEventType.PIPELINE_COMPLETED, {
            "final_state": self.state.name,
            "iterations": self._iteration,
            "duration_ms": duration_ms,
        })
        if self._telemetry:
            self._telemetry.record_orchestrator_event(
                event_type="PIPELINE_COMPLETED",
                iteration=self._iteration,
                data={"final_state": self.state.name, "iterations": self._iteration,
                      "duration_ms": duration_ms},
            )
            self._telemetry.finish_run(
                final_state=self.state.name,
                iterations=self._iteration,
                duration_ms=duration_ms,
                remediation_rounds=self.god.get_remediation_round(),
                validation_summary=self.god.get_validation_summary(),
            )

        return self._build_result()

    def _log_iteration_status(self):
        self._logger.info(f"\n{'--' * 25}")
        self._logger.info(f"Iteration {self._iteration} | State: {self.state.name}")
        self._logger.info(f"Validations: {self.god.get_validation_summary()}")
        self._logger.info(f"Remediation Rounds: {self.god.get_remediation_round()}")

    def _build_result(self) -> dict:
        success = self.state == OrchestratorState.SUCCEEDED
        duration_ms = 0
        if self._start_time and self._end_time:
            duration_ms = (self._end_time - self._start_time).total_seconds() * 1000
        return {
            "success": success,
            "state": self.state.name,
            "template": self.god.template.body if success else None,
            "template_checksum": self.god.template.checksum if success else None,
            "iterations": self._iteration,
            "duration_ms": duration_ms,
            "remediation_rounds": self.god.get_remediation_round(),
            "validation_summary": self.god.get_validation_summary(),
            "findings_summary": self.god.get_findings_summary(),
            "blocking_findings": [f.to_dict() for f in self.god.get_blocking_findings()],
            "execution_log": self._execution_log,
            "state_history": [
                {"state": s.name, "reason": r, "timestamp": t}
                for s, r, t in self._state_history
            ],
            "god_snapshot": self.god.snapshot() if self.config.log_god_snapshots else None
        }

    # -------------------------------------------------------------------------
    # Inspection helpers
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "state": self.state.name,
            "iteration": self._iteration,
            "current_phase": self._current_phase.name if self._current_phase else None,
            "god_summary": self.god.snapshot()["summary"] if self.god else None,
            "registered_skills": len(self.registry),
            "recent_executions": self._execution_log[-5:] if self._execution_log else []
        }

    def print_skill_registry(self):
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
        print("\n" + "=" * 70)
        print("PIPELINE RESULT")
        print("=" * 70)
        status_icon = "v" if result["success"] else "x"
        status_color = "\033[32m" if result["success"] else "\033[31m"
        reset = "\033[0m"
        print(f"\nStatus: {status_color}{status_icon} {result['state']}{reset}")
        print(f"Iterations: {result['iterations']}")
        print(f"Duration: {result['duration_ms']:.1f}ms")
        print(f"Remediation Rounds: {result['remediation_rounds']}")
        print("\nValidation Summary:")
        for validator, status in result["validation_summary"].items():
            icon = "v" if status == "PASS" else ("x" if status == "FAIL" else "o")
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
            print("\nGenerated Template (first 20 lines):")
            print("-" * 50)
            lines = result["template"].split("\n")
            for line in lines[:20]:
                print(line)
            if len(lines) > 20:
                print(f"... ({len(lines) - 20} more lines)")
        print("\n" + "=" * 70)
