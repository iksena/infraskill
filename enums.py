from enum import Enum, auto

class ValidationStatus(Enum):
    """Status of a validation gate"""
    PENDING = "PENDING"      # Not yet run
    RUNNING = "RUNNING"      # Currently executing
    PASS = "PASS"            # Validation passed
    FAIL = "FAIL"            # Validation failed (blocks progress)
    SKIPPED = "SKIPPED"      # Intentionally skipped
    ERROR = "ERROR"          # Execution error (different from FAIL)
    
    def is_terminal(self) -> bool:
        return self in [ValidationStatus.PASS, ValidationStatus.FAIL, 
                        ValidationStatus.SKIPPED, ValidationStatus.ERROR]
    
    def blocks_progress(self) -> bool:
        return self in [ValidationStatus.FAIL, ValidationStatus.ERROR]


class Severity(Enum):
    """Severity levels for findings"""
    CRITICAL = 1   # Must fix, blocks deployment
    HIGH = 2       # Should fix, security risk
    MEDIUM = 3     # Recommended fix
    LOW = 4        # Minor issue
    INFO = 5       # Informational only
    
    def __lt__(self, other):
        if isinstance(other, Severity):
            return self.value < other.value
        return NotImplemented


class SkillPhase(Enum):
    """
    Pipeline phases - skills are grouped and ordered by phase.
    The orchestrator processes phases in order.
    """
    PLANNING = 10       # Intent extraction, resource identification
    ENGINEERING = 20    # Resource block generation
    ASSEMBLY = 30       # Template assembly and formatting
    VALIDATION = 40     # All validation gates
    REMEDIATION = 50    # Fixing validation failures


class OrchestratorState(Enum):
    """
    States of the orchestrator state machine.
    These represent the high-level state of the pipeline.
    """
    UNINITIALIZED = auto()   # Before run() is called
    INITIALIZING = auto()    # Setting up GOD
    PLANNING = auto()        # Extracting intent
    ENGINEERING = auto()     # Generating resources
    ASSEMBLING = auto()      # Building template
    VALIDATING = auto()      # Running validators
    REMEDIATING = auto()     # Fixing failures
    SUCCEEDED = auto()       # All validations passed
    FAILED = auto()          # Unrecoverable failure
    ESCALATED = auto()       # Requires human intervention
    ABORTED = auto()         # Externally cancelled
    
    def is_terminal(self) -> bool:
        return self in [
            OrchestratorState.SUCCEEDED,
            OrchestratorState.FAILED,
            OrchestratorState.ESCALATED,
            OrchestratorState.ABORTED
        ]
    
    def is_active(self) -> bool:
        return not self.is_terminal() and self != OrchestratorState.UNINITIALIZED

