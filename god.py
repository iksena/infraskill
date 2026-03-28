
# =============================================================================
# PART 2: GROUNDED OBJECTIVES DOCUMENT (GOD)
# =============================================================================

import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import logging
from typing import Any, Optional

from enums import Severity, ValidationStatus
import logger


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
        return f"[{self.severity.name}] {self.rule_id}: {self.message} (resource: {self.resource_name})"


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
            "findings": [f.to_dict() for f in self.findings[:10]],  # Limit for readability
            "duration_ms": self.duration_ms,
            "tool_version": self.tool_version
        }
    
    def count_by_severity(self) -> dict[str, int]:
        counts = {s.name: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.name] += 1
        return counts
    
    def has_blocking_findings(self) -> bool:
        """Check if any findings should block progress"""
        return any(f.severity in [Severity.CRITICAL, Severity.HIGH] for f in self.findings)
    
    def get_findings_by_severity(self, severity: Severity) -> list[ValidationFinding]:
        return [f for f in self.findings if f.severity == severity]


@dataclass
class AcceptanceCriterion:
    """A single acceptance criterion - must be binary and checkable"""
    id: str
    description: str
    resource_type: Optional[str] = None
    property_path: Optional[str] = None
    expected_value: Any = None
    check_type: str = "exists"  # exists, equals, contains, not_exists, not_equals, regex
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
            "failure_reason": self.failure_reason
        }


@dataclass
class ExtractedResource:
    """A resource extracted from user intent"""
    resource_type: str
    logical_name: str
    priority: int = 100  # Lower = generated first
    dependencies: list[str] = field(default_factory=list)
    properties_hints: dict = field(default_factory=dict)
    generated: bool = False
    
    def to_dict(self) -> dict:
        return {
            "resource_type": self.resource_type,
            "logical_name": self.logical_name,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "generated": self.generated
        }


@dataclass
class Constraints:
    """Infrastructure constraints extracted from user intent"""
    # Availability
    multi_az: bool = False
    
    # Security
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    public_access_allowed: bool = False
    
    # Compliance
    compliance_frameworks: list[str] = field(default_factory=list)
    
    # Operational
    environment: str = "production"
    backup_enabled: bool = True
    backup_retention_days: int = 7
    logging_enabled: bool = True
    monitoring_enabled: bool = True
    
    # Cost
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
            "logging_enabled": self.logging_enabled
        }
    
    def is_production(self) -> bool:
        return self.environment.lower() in ["production", "prod"]


@dataclass
class Intent:
    """The intent section of the GOD - what the user wants"""
    raw_prompt: str = ""
    normalized_prompt: str = ""
    resources: list[ExtractedResource] = field(default_factory=list)
    constraints: Constraints = field(default_factory=Constraints)
    acceptance_criteria: list[AcceptanceCriterion] = field(default_factory=list)
    parsed_at: Optional[str] = None
    parser_version: str = ""
    
    def to_dict(self) -> dict:
        return {
            "raw_prompt": self.raw_prompt[:200] + "..." if len(self.raw_prompt) > 200 else self.raw_prompt,
            "resources": [r.to_dict() for r in self.resources],
            "resource_count": len(self.resources),
            "constraints": self.constraints.to_dict(),
            "acceptance_criteria_count": len(self.acceptance_criteria),
            "parsed_at": self.parsed_at
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
    """The template section of the GOD - the generated artifact"""
    format: str = "CFN"  # CFN (CloudFormation) or TF (Terraform)
    body: str = ""
    version: int = 0
    last_modified_by: str = ""
    last_modified_at: Optional[str] = None
    
    # Structured components (merged into body during assembly)
    parameters: dict[str, dict] = field(default_factory=dict)
    resources: dict[str, str] = field(default_factory=dict)  # name -> YAML block
    outputs: dict[str, dict] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    
    # Tracking
    checksum: str = ""
    
    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "version": self.version,
            "last_modified_by": self.last_modified_by,
            "last_modified_at": self.last_modified_at,
            "body_length": len(self.body),
            "resource_blocks": list(self.resources.keys()),
            "checksum": self.checksum
        }
    
    def update_checksum(self):
        """Update the checksum based on current body"""
        self.checksum = hashlib.sha256(self.body.encode()).hexdigest()[:16]
        
    def increment_version(self, modified_by: str):
        """Increment version and update metadata"""
        self.version += 1
        self.last_modified_by = modified_by
        self.last_modified_at = datetime.now().isoformat()


@dataclass
class RemediationEntry:
    """A single remediation action in the audit log"""
    round: int
    skill_name: str
    action_type: str  # "patch", "regenerate", "config_change", "escalate"
    target: str       # What was changed (resource name, property path, etc.)
    description: str
    rationale: str
    findings_addressed: list[str]  # List of finding rule_ids addressed
    success: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "skill_name": self.skill_name,
            "action_type": self.action_type,
            "target": self.target,
            "description": self.description,
            "findings_addressed": self.findings_addressed,
            "success": self.success,
            "timestamp": self.timestamp
        }


class GODEventType(Enum):
    """Types of events that can occur on the GOD"""
    CREATED = "created"
    FIELD_UPDATED = "field_updated"
    CHECKPOINT_SAVED = "checkpoint_saved"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    REMEDIATION_APPLIED = "remediation_applied"
    LOCKED = "locked"


@dataclass
class GODEvent:
    """An event that occurred on the GOD - for audit trail"""
    event_type: GODEventType
    field_path: str
    old_value_hash: Optional[str]
    new_value_hash: Optional[str]
    actor: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


class GroundedObjectivesDocument:
    """
    The Grounded Objectives Document (GOD) is the central artifact.
    
    Design principles:
    1. Single source of truth - all agents read/write through GOD
    2. Structured sections with clear ownership
    3. Complete audit trail of all changes
    4. Field locking to prevent unauthorized modifications
    5. Checkpoint/restore capability
    
    Sections:
    - intent: What the user wants (immutable after planning)
    - template: The generated artifact (mutable during engineering/remediation)
    - validation_state: Results of all validators (mutable by validators only)
    - remediation_log: History of fixes applied (append-only)
    """
    
    # Validation pipeline order - validators run in this sequence
    VALIDATION_PIPELINE = [
        "yaml_syntax",
        "cfn_lint", 
        "checkov",
        "intent_alignment"
    ]
    
    def __init__(self):
        self._logger = logging.getLogger("INFRA-SKILL.GOD")
        
        # Core sections
        self.intent = Intent()
        self.template = Template()
        self.validation_state: dict[str, ValidationResult] = {
            name: ValidationResult(validator_name=name) 
            for name in self.VALIDATION_PIPELINE
        }
        self.remediation_log: list[RemediationEntry] = []
        
        # Metadata
        self._created_at = datetime.now().isoformat()
        self._id = hashlib.sha256(self._created_at.encode()).hexdigest()[:12]
        
        # Audit and state management
        self._events: list[GODEvent] = []
        self._checkpoints: list[dict] = []
        self._locked_fields: set[str] = set()
        
        self._record_event(GODEventType.CREATED, "root", None, None, "system")
    
    # -------------------------------------------------------------------------
    # Event Recording
    # -------------------------------------------------------------------------
    
    def _record_event(
        self, 
        event_type: GODEventType, 
        field_path: str,
        old_value: Any,
        new_value: Any,
        actor: str,
        metadata: dict = None
    ):
        """Record an event for audit trail"""
        def hash_value(v):
            if v is None:
                return None
            return hashlib.sha256(str(v).encode()).hexdigest()[:8]
        
        event = GODEvent(
            event_type=event_type,
            field_path=field_path,
            old_value_hash=hash_value(old_value),
            new_value_hash=hash_value(new_value),
            actor=actor,
            metadata=metadata or {}
        )
        self._events.append(event)
    
    # -------------------------------------------------------------------------
    # Field Access Control
    # -------------------------------------------------------------------------
    
    def lock_field(self, field_path: str, actor: str = "system"):
        """Lock a field from further modification"""
        self._locked_fields.add(field_path)
        self._record_event(GODEventType.LOCKED, field_path, None, None, actor)
        self._logger.debug(f"Locked field: {field_path}")
    
    def is_field_locked(self, field_path: str) -> bool:
        """Check if a field is locked"""
        # Check exact match and parent paths
        parts = field_path.split(".")
        for i in range(len(parts)):
            check_path = ".".join(parts[:i+1])
            if check_path in self._locked_fields:
                return True
        return False
    
    def can_write(self, field_path: str, skill_writes_to: list[str]) -> bool:
        """Check if a skill is allowed to write to a field"""
        if self.is_field_locked(field_path):
            return False
        
        # Check if field is in skill's allowed write list
        for allowed in skill_writes_to:
            if field_path.startswith(allowed):
                return True
        return False
    
    # -------------------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------------------
    
    def save_checkpoint(self, label: str = "", actor: str = "system") -> int:
        """Save a checkpoint of current state. Returns checkpoint index."""
        checkpoint = {
            "index": len(self._checkpoints),
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "state": {
                "intent": copy.deepcopy(self.intent),
                "template": copy.deepcopy(self.template),
                "validation_state": copy.deepcopy(self.validation_state),
                "remediation_log": copy.deepcopy(self.remediation_log)
            }
        }
        self._checkpoints.append(checkpoint)
        self._record_event(GODEventType.CHECKPOINT_SAVED, "root", None, None, actor, {"label": label})
        self._logger.debug(f"Checkpoint saved: {label} (index: {checkpoint['index']})")
        return checkpoint["index"]
    
    def restore_checkpoint(self, index: int) -> bool:
        """Restore state from a checkpoint"""
        if index < 0 or index >= len(self._checkpoints):
            return False
        
        checkpoint = self._checkpoints[index]
        state = checkpoint["state"]
        
        self.intent = copy.deepcopy(state["intent"])
        self.template = copy.deepcopy(state["template"])
        self.validation_state = copy.deepcopy(state["validation_state"])
        self.remediation_log = copy.deepcopy(state["remediation_log"])
        
        self._logger.info(f"Restored checkpoint: {checkpoint['label']} (index: {index})")
        return True
    
    def get_checkpoint_labels(self) -> list[tuple[int, str, str]]:
        """Get list of (index, label, timestamp) for all checkpoints"""
        return [(c["index"], c["label"], c["timestamp"]) for c in self._checkpoints]
    
    # -------------------------------------------------------------------------
    # Validation State Management
    # -------------------------------------------------------------------------
    
    def get_validation_summary(self) -> dict[str, str]:
        """Get status of all validators"""
        return {name: result.status.value for name, result in self.validation_state.items()}
    
    def get_first_pending_validator(self) -> Optional[str]:
        """Get the first validator that hasn't run yet"""
        for name in self.VALIDATION_PIPELINE:
            if self.validation_state[name].status == ValidationStatus.PENDING:
                return name
        return None
    
    def get_first_failed_validator(self) -> Optional[tuple[str, ValidationResult]]:
        """Get the first failed validator and its result"""
        for name in self.VALIDATION_PIPELINE:
            result = self.validation_state[name]
            if result.status.blocks_progress():
                return (name, result)
        return None
    
    def has_pending_validations(self) -> bool:
        """Check if any validation is pending"""
        return any(r.status == ValidationStatus.PENDING for r in self.validation_state.values())
    
    def has_failed_validations(self) -> bool:
        """Check if any validation has failed"""
        return any(r.status.blocks_progress() for r in self.validation_state.values())
    
    def all_validations_passed(self) -> bool:
        """Check if all validations have passed"""
        return all(
            r.status in [ValidationStatus.PASS, ValidationStatus.SKIPPED] 
            for r in self.validation_state.values()
        )
    
    def reset_validations_from(self, validator_name: str, actor: str = "system"):
        """Reset a validator and all downstream validators to PENDING"""
        try:
            start_idx = self.VALIDATION_PIPELINE.index(validator_name)
        except ValueError:
            self._logger.warning(f"Unknown validator: {validator_name}")
            return
        
        for name in self.VALIDATION_PIPELINE[start_idx:]:
            old_status = self.validation_state[name].status
            self.validation_state[name] = ValidationResult(validator_name=name)
            self._record_event(
                GODEventType.FIELD_UPDATED,
                f"validation_state.{name}",
                old_status.value,
                ValidationStatus.PENDING.value,
                actor
            )
        
        self._logger.debug(f"Reset validations from {validator_name} onwards")
    
    def set_validation_result(self, validator_name: str, result: ValidationResult, actor: str):
        """Set the result for a validator"""
        if validator_name not in self.validation_state:
            raise ValueError(f"Unknown validator: {validator_name}")
        
        old_result = self.validation_state[validator_name]
        result.validator_name = validator_name
        self.validation_state[validator_name] = result
        
        self._record_event(
            GODEventType.VALIDATION_COMPLETED,
            f"validation_state.{validator_name}",
            old_result.status.value,
            result.status.value,
            actor,
            {"findings_count": len(result.findings)}
        )
    
    # -------------------------------------------------------------------------
    # Findings Aggregation
    # -------------------------------------------------------------------------
    
    def get_all_findings(self) -> list[ValidationFinding]:
        """Get all findings from all validators"""
        findings = []
        for result in self.validation_state.values():
            findings.extend(result.findings)
        return findings
    
    def get_findings_by_severity(self, severity: Severity) -> list[ValidationFinding]:
        """Get all findings of a specific severity"""
        return [f for f in self.get_all_findings() if f.severity == severity]
    
    def get_blocking_findings(self) -> list[ValidationFinding]:
        """Get findings that block progress (CRITICAL and HIGH)"""
        return [f for f in self.get_all_findings() if f.severity in [Severity.CRITICAL, Severity.HIGH]]
    
    def get_findings_summary(self) -> dict[str, int]:
        """Get count of findings by severity"""
        summary = {s.name: 0 for s in Severity}
        for f in self.get_all_findings():
            summary[f.severity.name] += 1
        return summary
    
    # -------------------------------------------------------------------------
    # Remediation
    # -------------------------------------------------------------------------
    
    def add_remediation_entry(self, entry: RemediationEntry):
        """Add a remediation entry to the log"""
        self.remediation_log.append(entry)
        self._record_event(
            GODEventType.REMEDIATION_APPLIED,
            "remediation_log",
            None,
            entry.description,
            entry.skill_name,
            {"round": entry.round, "success": entry.success}
        )
    
    def get_remediation_round(self) -> int:
        """Get the current remediation round number"""
        return len(self.remediation_log)
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def snapshot(self) -> dict:
        """Create a complete snapshot of current state"""
        return {
            "id": self._id,
            "created_at": self._created_at,
            "snapshot_at": datetime.now().isoformat(),
            "intent": self.intent.to_dict(),
            "template": self.template.to_dict(),
            "validation_state": {k: v.to_dict() for k, v in self.validation_state.items()},
            "remediation_log": [e.to_dict() for e in self.remediation_log],
            "summary": {
                "resources_planned": len(self.intent.resources),
                "resources_generated": len(self.template.resources),
                "template_version": self.template.version,
                "validations": self.get_validation_summary(),
                "findings": self.get_findings_summary(),
                "remediation_rounds": len(self.remediation_log),
                "checkpoints": len(self._checkpoints),
                "events": len(self._events)
            }
        }
    
    def export_template(self) -> str:
        """Export the final template body"""
        return self.template.body
    
    def export_audit_trail(self) -> list[dict]:
        """Export the complete audit trail"""
        return [
            {
                "event_type": e.event_type.value,
                "field_path": e.field_path,
                "actor": e.actor,
                "timestamp": e.timestamp,
                "metadata": e.metadata
            }
            for e in self._events
        ]
    
    def __str__(self) -> str:
        return json.dumps(self.snapshot(), indent=2, default=str)

