# god.py — Grounded Objectives Document
# Patched: reset_validations_from now accepts skip_errored (default True)
# and RemediationEntry gains strategy_type field for telemetry.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from enums import Severity, ValidationStatus


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationFinding:
    """A single finding from a validator"""
    rule_id: str
    resource_name: str
    resource_type: str
    severity: Severity
    message: str
    remediation_hint: str = ""
    check_id: str = ""
    file_path: str = ""
    line_number: int = 0

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "resource_name": self.resource_name,
            "resource_type": self.resource_type,
            "severity": self.severity.name,
            "message": self.message,
            "remediation_hint": self.remediation_hint,
            "check_id": self.check_id,
        }

    def __str__(self) -> str:
        return (
            f"[{self.severity.name}] {self.rule_id}: {self.message} "
            f"(resource: {self.resource_name})"
        )


@dataclass
class ValidationResult:
    """Result of a validation phase"""
    status: ValidationStatus = ValidationStatus.PENDING
    validator_name: str = ""
    errors: list[str] = field(default_factory=list)
    findings: list[ValidationFinding] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0
    tool_version: str = ""
    raw_output: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "validator_name": self.validator_name,
            "errors": self.errors,
            "findings_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings[:10]],
            "duration_ms": self.duration_ms,
            "tool_version": self.tool_version,
        }

    def count_by_severity(self) -> dict[str, int]:
        counts = {s.name: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.name] += 1
        return counts

    def has_blocking_findings(self) -> bool:
        return any(
            f.severity in [Severity.CRITICAL, Severity.HIGH]
            for f in self.findings
        )

    def get_findings_by_severity(self, severity: Severity) -> list[ValidationFinding]:
        return [f for f in self.findings if f.severity == severity]


@dataclass
class AcceptanceCriterion:
    """A single acceptance criterion"""
    id: str
    description: str
    resource_type: Optional[str] = None
    property_path: Optional[str] = None
    expected_value: Any = None
    check_type: str = "exists"
    is_met: Optional[bool] = None
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "resource_type": self.resource_type,
            "property_path": self.property_path,
            "check_type": self.check_type,
            "is_met": self.is_met,
            "failure_reason": self.failure_reason,
        }


@dataclass
class ExtractedResource:
    """A resource extracted from user intent"""
    resource_type: str
    logical_name: str
    priority: int = 100
    dependencies: list[str] = field(default_factory=list)
    properties_hints: dict = field(default_factory=dict)
    generated: bool = False

    def to_dict(self) -> dict:
        return {
            "resource_type": self.resource_type,
            "logical_name": self.logical_name,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "generated": self.generated,
        }


@dataclass
class Constraints:
    """Infrastructure constraints extracted from user intent"""
    multi_az: bool = False
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    public_access_allowed: bool = False
    compliance_frameworks: list[str] = field(default_factory=list)
    environment: str = "production"
    backup_enabled: bool = True
    backup_retention_days: int = 7
    logging_enabled: bool = True
    monitoring_enabled: bool = True
    cost_optimization: bool = False

    def to_dict(self) -> dict:
        return {
            "multi_az": self.multi_az,
            "encryption_at_rest": self.encryption_at_rest,
            "encryption_in_transit": self.encryption_in_transit,
            "public_access_allowed": self.public_access_allowed,
            "compliance_frameworks": self.compliance_frameworks,
            "environment": self.environment,
            "backup_enabled": self.backup_enabled,
            "logging_enabled": self.logging_enabled,
        }

    def is_production(self) -> bool:
        return self.environment.lower() in ["production", "prod"]


@dataclass
class Intent:
    """The intent section of the GOD"""
    raw_prompt: str = ""
    normalized_prompt: str = ""
    resources: list[ExtractedResource] = field(default_factory=list)
    constraints: Constraints = field(default_factory=Constraints)
    acceptance_criteria: list[AcceptanceCriterion] = field(default_factory=list)
    parsed_at: Optional[str] = None
    parser_version: str = ""

    def to_dict(self) -> dict:
        return {
            "raw_prompt": (
                self.raw_prompt[:200] + "..."
                if len(self.raw_prompt) > 200
                else self.raw_prompt
            ),
            "resources": [r.to_dict() for r in self.resources],
            "resource_count": len(self.resources),
            "constraints": self.constraints.to_dict(),
            "acceptance_criteria_count": len(self.acceptance_criteria),
            "parsed_at": self.parsed_at,
        }

    def get_resource_types(self) -> set[str]:
        return {r.resource_type for r in self.resources}

    def get_ungenerated_resources(self) -> list[ExtractedResource]:
        return [r for r in self.resources if not r.generated]

    def mark_resource_generated(self, resource_type: str):
        for r in self.resources:
            if r.resource_type == resource_type:
                r.generated = True


@dataclass
class Template:
    """The template section of the GOD"""
    format: str = "CFN"
    body: str = ""
    version: int = 0
    last_modified_by: str = ""
    last_modified_at: Optional[str] = None
    parameters: dict[str, dict] = field(default_factory=dict)
    resources: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, dict] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    checksum: str = ""

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "version": self.version,
            "last_modified_by": self.last_modified_by,
            "last_modified_at": self.last_modified_at,
            "body_length": len(self.body),
            "resource_blocks": list(self.resources.keys()),
            "checksum": self.checksum,
        }

    def update_checksum(self):
        self.checksum = hashlib.sha256(self.body.encode()).hexdigest()[:16]

    def increment_version(self, modified_by: str):
        self.version += 1
        self.last_modified_by = modified_by
        self.last_modified_at = datetime.now().isoformat()


@dataclass
class RemediationEntry:
    """
    A single remediation action in the audit log.

    strategy_type is new in v2: records whether the action was produced by
    a deterministic rule, llm_fallback, llm_plan, or an escalation.  Benchmark
    telemetry reads this field to compute per-strategy success rates.
    """
    round: int
    skill_name: str
    action_type: str          # 'plan', 'patch', 'escalate'
    target: str
    description: str
    rationale: str
    findings_addressed: list[str]
    success: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    strategy_type: str = "unknown"  # NEW: 'deterministic'|'llm_fallback'|'llm_plan'|'escalate'

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "skill_name": self.skill_name,
            "action_type": self.action_type,
            "target": self.target,
            "description": self.description,
            "findings_addressed": self.findings_addressed,
            "success": self.success,
            "timestamp": self.timestamp,
            "strategy_type": self.strategy_type,
        }


class GODEventType(Enum):
    CREATED = "created"
    FIELD_UPDATED = "field_updated"
    CHECKPOINT_SAVED = "checkpoint_saved"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    REMEDIATION_APPLIED = "remediation_applied"
    LOCKED = "locked"


@dataclass
class GODEvent:
    event_type: GODEventType
    field_path: str
    old_value_hash: Optional[str]
    new_value_hash: Optional[str]
    actor: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


# =============================================================================
# GROUNDED OBJECTIVES DOCUMENT
# =============================================================================

class GroundedObjectivesDocument:
    """
    The Grounded Objectives Document (GOD) is the central shared artifact.

    All skills read and write through the GOD.  It maintains a complete audit
    trail of every mutation, supports checkpoint/restore, and enforces field
    locking after the planning phase.
    """

    VALIDATION_PIPELINE = [
        "yaml_syntax",
        "cfn_lint",
        "checkov",
        "intent_alignment",
    ]

    def __init__(self):
        import logging
        self._logger = logging.getLogger("INFRA-SKILL.GOD")

        self.intent = Intent()
        self.template = Template()
        self.validation_state: dict[str, ValidationResult] = {
            name: ValidationResult(validator_name=name)
            for name in self.VALIDATION_PIPELINE
        }
        self.remediation_log: list[RemediationEntry] = []
        self._audit_trail: list[GODEvent] = []
        self._locked_fields: set[str] = set()
        self._checkpoints: dict[str, dict] = {}

    # -------------------------------------------------------------------------
    # Validation state helpers
    # -------------------------------------------------------------------------

    def set_validation_result(
        self, validator_name: str, result: ValidationResult, actor: str = "unknown"
    ):
        if validator_name not in self.validation_state:
            self._logger.warning(f"Unknown validator: {validator_name}")
            return
        self.validation_state[validator_name] = result
        self._record_event(
            GODEventType.VALIDATION_COMPLETED,
            f"validation_state.{validator_name}",
            None,
            result.status.value,
            actor,
        )

    def has_failed_validations(self) -> bool:
        """True when any validator is in FAIL or ERROR status."""
        return any(
            v.status.blocks_progress()
            for v in self.validation_state.values()
        )

    def has_pending_validations(self) -> bool:
        return any(
            v.status == ValidationStatus.PENDING
            for v in self.validation_state.values()
        )

    def all_validations_passed(self) -> bool:
        return all(
            v.status in (ValidationStatus.PASS, ValidationStatus.SKIPPED)
            for v in self.validation_state.values()
        )

    def get_blocking_findings(self) -> list[ValidationFinding]:
        findings = []
        for result in self.validation_state.values():
            findings.extend(
                f for f in result.findings
                if f.severity in (Severity.CRITICAL, Severity.HIGH)
            )
        return findings

    def get_validation_summary(self) -> dict[str, str]:
        return {
            name: result.status.value
            for name, result in self.validation_state.items()
        }

    def get_findings_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {s.name: 0 for s in Severity}
        for result in self.validation_state.values():
            for finding in result.findings:
                counts[finding.severity.name] += 1
        return counts

    def reset_validations_from(
        self,
        from_validator: str,
        actor: str = "unknown",
        skip_errored: bool = True,
    ):
        """
        Reset validators at and after *from_validator* back to PENDING so they
        re-run after a remediation patch.

        skip_errored (default True)
        ---------------------------
        When True, validators currently in ERROR status are NOT reset.  This
        prevents the infinite-loop pattern where a tool that is missing from
        PATH (e.g. checkov) errors on every run and immediately re-triggers
        remediation, which resets validators, which errors again, ad infinitum.

        Only set skip_errored=False in tests or when you explicitly want to
        force a re-run of an errored validator.
        """
        pipeline = self.VALIDATION_PIPELINE
        try:
            start_idx = pipeline.index(from_validator)
        except ValueError:
            self._logger.warning(
                f"reset_validations_from: unknown validator '{from_validator}'"
            )
            return

        for name in pipeline[start_idx:]:
            current = self.validation_state[name]
            if skip_errored and current.status == ValidationStatus.ERROR:
                self._logger.debug(
                    f"  Skipping reset of '{name}' (status=ERROR, tool unavailable)"
                )
                continue
            self.validation_state[name] = ValidationResult(validator_name=name)
            self._record_event(
                GODEventType.FIELD_UPDATED,
                f"validation_state.{name}",
                current.status.value,
                ValidationStatus.PENDING.value,
                actor,
            )

    # -------------------------------------------------------------------------
    # Remediation helpers
    # -------------------------------------------------------------------------

    def add_remediation_entry(self, entry: RemediationEntry):
        self.remediation_log.append(entry)
        self._record_event(
            GODEventType.REMEDIATION_APPLIED,
            "remediation_log",
            None,
            f"round={entry.round},action={entry.action_type},strategy={entry.strategy_type}",
            entry.skill_name,
        )

    def get_remediation_round(self) -> int:
        """Number of completed patch rounds (action_type='patch' or 'escalate', not 'plan')."""
        return sum(
            1 for e in self.remediation_log
            if e.action_type in ("patch", "escalate")
        )

    # -------------------------------------------------------------------------
    # Field locking
    # -------------------------------------------------------------------------

    def lock_field(self, field_path: str, actor: str):
        self._locked_fields.add(field_path)
        self._record_event(
            GODEventType.LOCKED, field_path, None, "locked", actor
        )

    def is_locked(self, field_path: str) -> bool:
        return field_path in self._locked_fields

    # -------------------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------------------

    def save_checkpoint(self, name: str, actor: str):
        self._checkpoints[name] = {
            "template_body": self.template.body,
            "template_version": self.template.version,
            "validation_state": {
                k: v.status.value for k, v in self.validation_state.items()
            },
            "remediation_rounds": self.get_remediation_round(),
            "timestamp": datetime.now().isoformat(),
        }
        self._record_event(
            GODEventType.CHECKPOINT_SAVED, "checkpoint", None, name, actor
        )

    # -------------------------------------------------------------------------
    # Audit trail
    # -------------------------------------------------------------------------

    def _record_event(
        self,
        event_type: GODEventType,
        field_path: str,
        old_value: Optional[str],
        new_value: Optional[str],
        actor: str,
    ):
        self._audit_trail.append(
            GODEvent(
                event_type=event_type,
                field_path=field_path,
                old_value_hash=(
                    hashlib.md5(old_value.encode()).hexdigest()[:8]
                    if old_value
                    else None
                ),
                new_value_hash=(
                    hashlib.md5(new_value.encode()).hexdigest()[:8]
                    if new_value
                    else None
                ),
                actor=actor,
            )
        )

    def export_audit_trail(self) -> list[dict]:
        return [
            {
                "event_type": e.event_type.value,
                "field_path": e.field_path,
                "actor": e.actor,
                "timestamp": e.timestamp,
                "old_value_hash": e.old_value_hash,
                "new_value_hash": e.new_value_hash,
            }
            for e in self._audit_trail
        ]

    # -------------------------------------------------------------------------
    # Snapshot / summary
    # -------------------------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "summary": {
                "template_version": self.template.version,
                "template_length": len(self.template.body),
                "remediation_rounds": self.get_remediation_round(),
                "validation_summary": self.get_validation_summary(),
                "findings_summary": self.get_findings_summary(),
                "locked_fields": list(self._locked_fields),
                "checkpoints": list(self._checkpoints.keys()),
            },
            "intent": self.intent.to_dict(),
            "template": self.template.to_dict(),
            "validation_state": {
                k: v.to_dict() for k, v in self.validation_state.items()
            },
            "remediation_log": [e.to_dict() for e in self.remediation_log],
        }
