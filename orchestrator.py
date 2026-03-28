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
Version: 1.2.0  (FAIL vs ERROR routing fix)
"""

from __future__ import annotations

import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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

    Pipeline stages (in order):
        PLANNING → ENGINEERING → ASSEMBLING → VALIDATING
                                                   ↓ (FAIL findings)
                                              REMEDIATING
                                                   ↓
                                              VALIDATING  (repeat up to max_remediation_rounds)
                                                   ↓ (all pass or ERROR only)
                                              SUCCEEDED

    Key routing rule
    ----------------
    _determine_target_state() routes to REMEDIATING only when
    god.has_remediable_failures() is True — meaning at least one validator
    is FAIL with blocking (CRITICAL/HIGH) findings.

    Validators in ERROR state (tool not on PATH) are treated as SKIPPED:
    they do NOT trigger remediation and do NOT block SUCCEEDED.  This
    prevents the infinite loop where a missing tool (e.g. checkov) causes
    perpetual VALIDATING → REMEDIATING → no-op → FAILED.
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
        self._logger.info(f"State: {old_state.name} → {new_state.name}" + (f" ({reason})" if reason else ""))
        self.events.emit(OrchestratorEventType.STATE_CHANGED, {
            "old_state": old_state.name,
            "new_state": new_state.name,
            "reason": reason
        })
        if self.config.checkpoint_on_phase_change and self.god:
            self.god.save_checkpoint(f"state:{new_state.name}", "orchestrator")

    # -------------------------------------------------------------------------
    # Deterministic target-state resolution
    # -------------------------------------------------------------------------

    def _determine_target_state(self) -> OrchestratorState:
        """
        Derive the required orchestrator state purely from GOD content.

        REMEDIATING gate
        ----------------
        Uses has_remediable_failures() (FAIL + blocking findings) NOT
        has_failed_validations() (FAIL or ERROR).  This is the critical
        distinction:

          FAIL   → template has a real defect → REMEDIATING
          ERROR  → tool missing from PATH    → treat as SKIPPED, continue

        Without this distinction, a missing checkov binary causes permanent
        oscillation: VALIDATING → REMEDIATING → (no blocking findings) →
        no skill triggers → FAILED.
        """
        if self.god is None:
            return OrchestratorState.UNINITIALIZED
        
        if self.state.is_terminal():
            return self.state
        
        # Loop guards
        if self.god.get_remediation_round() >= self.config.max_remediation_rounds:
            return OrchestratorState.ESCALATED
        if self._iteration >= self.config.max_total_iterations:
            return OrchestratorState.ESCALATED
        
        # Success: all validators PASS, SKIPPED, or ERROR (tool absent)
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
        
        # Real failures with blocking findings → remediate
        if self.god.has_remediable_failures():
            return OrchestratorState.REMEDIATING
        
        # Forward progress: what is still missing in GOD?
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

    # -------------------------------------------------------------------------
    # Deterministic skill selection (LLM as tiebreaker only)
    # -------------------------------------------------------------------------

    def select_next_skill(self) -> Optional[Skill]:
        """
        Two-step skill selection:
          1. Resolve required state via _determine_target_state() and transition.
          2. Select from phase-gated candidates; LLM only as within-phase tiebreaker.
        """
        target_state = self._determine_target_state()

        if target_state != self.state:
            reason_map = {
                OrchestratorState.PLANNING:    "intent missing — re-plan",
                OrchestratorState.ENGINEERING: "template missing — engineer",
                OrchestratorState.ASSEMBLING:  "resources ready — assemble",
                OrchestratorState.VALIDATING:  "template ready — validate",
                OrchestratorState.REMEDIATING: "validation failures — remediate",
                OrchestratorState.SUCCEEDED:   "all validations passed",
                OrchestratorState.ESCALATED:   "loop guard triggered",
            }
            self._transition_to(target_state, reason_map.get(target_state, "auto-transition"))

        if self.state.is_terminal():
            return None

        target_phase = self._state_to_phase(self.state)
        if target_phase is None:
            return None

        if target_phase != self._current_phase:
            old_phase = self._current_phase
            self._current_phase = target_phase
            self.events.emit(OrchestratorEventType.PHASE_CHANGED, {
                "old_phase": old_phase.name if old_phase else None,
                "new_phase": target_phase.name,
            })

        phase_skills = self.registry.get_by_phase(target_phase)
        candidates = [s for s in phase_skills if s.can_trigger(self.god)]

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # LLM tiebreaker within phase
        llm = self.config.llm_client
        if llm is None:
            return candidates[0]

        god_snapshot = json.dumps(self.god.intent.to_dict(), indent=2)[:2000]
        candidate_table = json.dumps(
            [s.metadata.to_dict() for s in candidates], indent=2
        )

        try:
            raw = llm.complete(
                system=SKILL_SELECTOR_PROMPT.format(
                    god_snapshot=god_snapshot,
                    skill_metadata_table=candidate_table,
                ),
                user="Which skill should run next?",
                temperature=0.0,
                timeout=15,
            )
            decision = json.loads(raw)
            chosen_name = decision["skill_name"]
            rationale = decision.get("rationale", "")
            skill = self.registry.get(chosen_name)
            if skill and skill in candidates:
                self._logger.info(f"LLM selected skill '{chosen_name}': {rationale}")
                return skill
            self._logger.warning(
                f"LLM chose '{chosen_name}' not in candidates for phase "
                f"{target_phase.name}; falling back to priority routing"
            )
            return candidates[0]
        except (json.JSONDecodeError, KeyError):
            return candidates[0]
        except Exception:
            return candidates[0]

    def _execute_skill(self, skill: Skill) -> SkillResult:
        timeout_s = self.config.skill_timeout_seconds
        context = SkillContext(
            god=self.god,
            orchestrator_state=self.state,
            iteration=self._iteration,
            config={
                "llm": self.config.llm_client,
                "llm_timeout": max(30, timeout_s - 10),
            }
        )
        self.events.emit(OrchestratorEventType.SKILL_STARTED, {
            "skill_name": skill.metadata.name,
            "phase": skill.metadata.phase.name,
            "iteration": self._iteration
        })
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
        self._logger.info("=" * 70)
        self._logger.info("INFRA-SKILL Pipeline Starting")
        self._logger.info("=" * 70)
        self._logger.info(f"Prompt: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")
        self._start_time = datetime.now()
        self._iteration = 0
        self._execution_log = []
        self._state_history = []
        self.god = GroundedObjectivesDocument()
        self.god.intent.raw_prompt = user_prompt
        self.god.lock_field("intent.raw_prompt", "orchestrator")
        self.god.save_checkpoint("initialized", "orchestrator")
        self._transition_to(OrchestratorState.INITIALIZING, "pipeline start")
        self.events.emit(OrchestratorEventType.PIPELINE_STARTED, {
            "prompt_length": len(user_prompt),
            "registered_skills": len(self.registry)
        })
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
                elif self.god.has_remediable_failures():
                    if self.god.get_remediation_round() >= self.config.max_remediation_rounds:
                        self._transition_to(OrchestratorState.ESCALATED, "max remediation rounds exceeded")
                        self.events.emit(OrchestratorEventType.LOOP_GUARD_TRIGGERED, {
                            "remediation_rounds": self.god.get_remediation_round()
                        })
                    else:
                        self._transition_to(
                            OrchestratorState.FAILED,
                            f"no skills can remediate failures in phase {self._current_phase}"
                        )
                else:
                    self._transition_to(
                        OrchestratorState.FAILED,
                        f"pipeline stuck — no triggerable skills in phase {self._current_phase}"
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
                for change in result.changes_made:
                    self._logger.info(f"  ✓ {change}")
            else:
                for error in result.errors:
                    self._logger.warning(f"  ✗ {error}")
        
        self._end_time = datetime.now()
        self.events.emit(OrchestratorEventType.PIPELINE_COMPLETED, {
            "final_state": self.state.name,
            "iterations": self._iteration,
            "duration_ms": (self._end_time - self._start_time).total_seconds() * 1000
        })
        return self._build_result()
    
    def _log_iteration_status(self):
        self._logger.info(f"\n{'─' * 50}")
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
            lines = result["template"].split("\n")
            for line in lines[:50]:
                print(line)
            if len(lines) > 50:
                print(f"... ({len(lines) - 50} more lines)")
        print("\n" + "=" * 70)


# =============================================================================
# SKILL FACTORY
# =============================================================================

def create_default_skills() -> list[Skill]:
    return [
        PlannerSkill(),
        GeneralEngineerSkill(),
        TemplateAssemblerSkill(),
        YAMLSyntaxValidatorSkill(),
        CFNLintValidatorSkill(),
        CheckovValidatorSkill(),
        IntentAlignmentValidatorSkill(),
        RemediationSkill(),
    ]


def create_orchestrator(
    config: OrchestratorConfig = None,
    skills: list[Skill] = None
) -> Orchestrator:
    config = config or OrchestratorConfig()
    skills = skills if skills is not None else create_default_skills()
    orchestrator = Orchestrator(config)
    orchestrator.with_skills(skills)
    return orchestrator


# =============================================================================
# CLI AND MAIN EXECUTION
# =============================================================================

def demo_event_handler(event: OrchestratorEvent):
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
    if event.event_type in [
        OrchestratorEventType.PIPELINE_STARTED,
        OrchestratorEventType.PIPELINE_COMPLETED,
        OrchestratorEventType.SKILL_COMPLETED,
        OrchestratorEventType.SKILL_FAILED,
        OrchestratorEventType.ESCALATION_REQUIRED,
    ]:
        print(f"  {icon} {event.event_type.value}: {event.data}")


def main():
    setup_logging(logging.INFO)
    logger = logging.getLogger("INFRA-SKILL.Main")
    config = OrchestratorConfig(
        max_remediation_rounds=5,
        max_total_iterations=30,
        verbose_logging=True,
        enable_checkpoints=True,
        skill_timeout_seconds=120,
        llm_client=OpenRouterClient(
            model="arcee-ai/trinity-large-preview:free",
            default_timeout=90,
            max_retries=3,
        ),
    )
    orchestrator = create_orchestrator(config)
    orchestrator.on_event(OrchestratorEventType.PIPELINE_COMPLETED, demo_event_handler)
    orchestrator.on_event(OrchestratorEventType.ESCALATION_REQUIRED, demo_event_handler)
    orchestrator.print_skill_registry()
    prompts = [
        """
        Create a production VPC with multi-AZ subnets, an S3 bucket for application logs,
        and a PostgreSQL RDS database. The environment should be highly available,
        encrypted, and follow CIS security best practices. No public access should be allowed.
        """,
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
        result = orchestrator.run(prompt.strip())
        orchestrator.print_result(result)
        if result["success"]:
            output_file = Path("generated_template.yaml")
            output_file.write_text(result["template"])
            logger.info(f"Template written to: {output_file}")
            audit_file = Path("audit_trail.json")
            audit_file.write_text(
                json.dumps(orchestrator.god.export_audit_trail(), indent=2)
            )
            logger.info(f"Audit trail written to: {audit_file}")


def run_interactive():
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
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive()
    else:
        main()
