# -----------------------------------------------------------------------------
# VALIDATION SKILLS
# -----------------------------------------------------------------------------
#
# Validator chain (in priority order):
#   10  YAMLSyntaxValidatorSkill   — pure Python, always fast
#   20  CFNLintValidatorSkill      — real cfn-lint CLI (no hand-rolled schemas)
#   30  CheckovValidatorSkill      — real Checkov CLI (unchanged)
#   40  IntentAlignmentValidatorSkill — LLM scoring (accuracy + coverage)
#
# Gate semantics for can_trigger()
# ---------------------------------
# Each validator requires its upstream predecessor to be PASS or ERROR.
# ERROR means the tool is absent from PATH — it is treated as “skipped”
# so the pipeline keeps moving rather than stalling forever.
#
# Checkov → Remediation policy context
# ---------------------------------------
# CheckovValidatorSkill populates ValidationFinding.check_id for every failed
# CKV_* check.  RemediationSkill uses checkov_context.get_checkov_policy_context()
# to look up the Checkov source code for each failing check_id from
# Data/checkov_cfn_policy_map.csv and appends it to the LLM fix prompt.
# This mirrors the generate_security_remediation_feedback pattern in IaCGen
# (Code/main.py) so the LLM sees the exact scan_resource_conf() logic and
# knows precisely which CFN property path to fix.
#
# Runtime loading:
#   Heavy imports (yaml, subprocess, shutil, json, re) are initialized lazily
#   on the first execute() call for each validator.
# -----------------------------------------------------------------------------

from datetime import datetime
import os
import shutil
import tempfile

from enums import Severity, SkillPhase, ValidationStatus
from god import GroundedObjectivesDocument, ValidationFinding, ValidationResult
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult


_PASS_OR_ERROR = (ValidationStatus.PASS, ValidationStatus.ERROR)

# -----------------------------------------------------------------------------
# CloudFormation-aware YAML loader
# -----------------------------------------------------------------------------
# PyYAML’s safe_load rejects all custom tags (e.g. !Ref, !Sub, !GetAtt).
# CloudFormation templates use these tags extensively, so we register
# passthrough constructors for every known CFN intrinsic function tag.
# This approach is taken directly from the IaCGen evaluation pipeline
# (Code/evaluation/cloud_evaluation.py :: get_required_resource_types).
# -----------------------------------------------------------------------------

_CFN_TAGS = [
    "!Ref", "!Sub", "!GetAtt", "!Join", "!Select", "!Split",
    "!Equals", "!If", "!FindInMap", "!GetAZs", "!Base64", "!Cidr",
    "!Transform", "!ImportValue", "!Not", "!And", "!Or", "!Condition",
    "!ForEach", "!ValueOf", "!Rain::Embed",
]


def _build_cfn_loader():
    """Return a yaml.SafeLoader subclass that accepts all CFN intrinsic tags."""
    import yaml

    class CloudFormationLoader(yaml.SafeLoader):
        pass

    def _construct_cfn_tag(loader, node):
        if isinstance(node, yaml.ScalarNode):
            return node.value
        elif isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        elif isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)

    for tag in _CFN_TAGS:
        CloudFormationLoader.add_constructor(tag, _construct_cfn_tag)

    return CloudFormationLoader


def cfn_safe_load(template_body: str):
    """
    Parse a CloudFormation YAML template string.

    Uses a custom SafeLoader subclass that registers passthrough constructors
    for all common CFN intrinsic function tags (!Ref, !Sub, !GetAtt, etc.)
    so they are preserved as plain strings/lists/dicts rather than causing a
    ConstructorError. Falls back to standard yaml.safe_load for JSON templates.
    """
    import yaml
    loader_cls = _build_cfn_loader()
    return yaml.load(template_body, Loader=loader_cls)  # noqa: S506 — custom safe loader


# =============================================================================
# 1. YAML SYNTAX VALIDATOR
# =============================================================================

class YAMLSyntaxValidatorSkill(Skill):
    """
    Validates YAML syntax of the generated CloudFormation template.

    Uses yamllint for structural YAML correctness (indentation, trailing
    spaces, etc.) and then parses with a CFN-aware loader that accepts all
    intrinsic function tags (!Ref, !Sub, !GetAtt, …).

    Static / pure-Python — no LLM, no CLI tool.
    YAML tooling is imported lazily on first execute().
    """

    # yamllint config mirrors the IaCGen evaluation pipeline settings so that
    # YAML correctness is judged by the same rules as the research baseline.
    _YAMLLINT_CONFIG = """
        extends: default
        rules:
            document-start: disable
            line-length: disable
            trailing-spaces: disable
            new-line-at-end-of-file: disable
            indentation:
                spaces: consistent
                indent-sequences: consistent
            truthy:
                allowed-values: ['true', 'false', 'yes', 'no']
    """

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="yaml-validator",
            version="2.0.0",
            description=(
                "Validates that the template body is syntactically valid YAML "
                "and parseable as a CloudFormation template (supports !Ref, "
                "!Sub, !GetAtt and all other CFN intrinsic tags)."
            ),
            llm_description=(
                "Runs yamllint for structural YAML correctness, then parses "
                "with a CFN-aware SafeLoader that accepts intrinsic function "
                "tags. Fails if the YAML is malformed, empty, or the root "
                "node is not a mapping."
            ),
            phase=SkillPhase.VALIDATION,
            trigger_condition="template.body exists AND validation_state.yaml_syntax is PENDING",
            writes_to=["validation_state.yaml_syntax"],
            reads_from=["template.body"],
            priority=10,
            tags=["static", "syntax", "yaml"],
        )

    def _ensure_runtime(self):
        if getattr(self, "_runtime_initialized", False):
            return

        import yaml as _yaml
        self._yaml = _yaml
        try:
            from yamllint import linter as _linter
            from yamllint.config import YamlLintConfig as _YamlLintConfig
            self._linter = _linter
            self._YamlLintConfig = _YamlLintConfig
            self._yamllint_available = True
        except ImportError:
            self._logger.warning(
                "yamllint not installed; falling back to PyYAML CFN parse only. "
                "Install with: pip install yamllint"
            )
            self._yamllint_available = False
        self._runtime_initialized = True

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        return (
            bool(god.template.body)
            and god.validation_state["yaml_syntax"].status == ValidationStatus.PENDING
        )

    def execute(self, context: SkillContext) -> SkillResult:
        self._ensure_runtime()
        god = context.god
        template_body = god.template.body

        result = ValidationResult(validator_name="yaml_syntax")
        result.started_at = datetime.now().isoformat()

        # ------------------------------------------------------------------
        # Step 1: yamllint structural check (mirrors IaCGen evaluation)
        # ------------------------------------------------------------------
        if self._yamllint_available:
            lint_errors = self._run_yamllint(template_body)
            if lint_errors:
                error_msg = "\n".join(lint_errors)
                result.status = ValidationStatus.FAIL
                result.errors.append(error_msg)
                result.findings.append(ValidationFinding(
                    rule_id="YAML001",
                    resource_name="template",
                    resource_type="template",
                    severity=Severity.CRITICAL,
                    message=error_msg,
                ))
                god.set_validation_result("yaml_syntax", result, actor="yaml-validator")
                return SkillResult.failure(self.metadata.name, f"YAML lint errors:\n{error_msg}")

        # ------------------------------------------------------------------
        # Step 2: CFN-aware parse — accepts !Ref, !Sub, !GetAtt, etc.
        # ------------------------------------------------------------------
        try:
            parsed = cfn_safe_load(template_body)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Template root must be a YAML mapping, got {type(parsed).__name__}"
                )
            result.status = ValidationStatus.PASS
            god.set_validation_result("yaml_syntax", result, actor="yaml-validator")
            return SkillResult.success_with_changes(
                self.metadata.name, ["yaml_syntax: PASS"]
            )
        except Exception as e:
            result.status = ValidationStatus.FAIL
            result.errors.append(str(e))
            result.findings.append(ValidationFinding(
                rule_id="YAML001",
                resource_name="template",
                resource_type="template",
                severity=Severity.CRITICAL,
                message=str(e),
            ))
            god.set_validation_result("yaml_syntax", result, actor="yaml-validator")
            return SkillResult.failure(self.metadata.name, f"YAML parse error: {e}")

    def _run_yamllint(self, template_body: str) -> list[str]:
        """Return a list of error-level yamllint messages (warnings ignored)."""
        try:
            config = self._YamlLintConfig(self._YAMLLINT_CONFIG)
            problems = list(self._linter.run(template_body, config))
            return [
                f"Line {p.line}: {p.desc}"
                for p in problems
                if p.level == "error"
            ]
        except Exception as exc:
            self._logger.warning(f"yamllint check failed unexpectedly: {exc}")
            return []


# =============================================================================
# 2. CFN-LINT VALIDATOR
# =============================================================================

class CFNLintValidatorSkill(Skill):
    """
    Validates CloudFormation template structure and schema using the real
    cfn-lint CLI tool.

    Gate: yaml_syntax must be PASS (YAML errors must be fixed first).
    Install: pip install cfn-lint
    """

    _SEVERITY_MAP = {
        "E": Severity.CRITICAL,
        "W": Severity.HIGH,
        "I": Severity.INFO,
    }

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="cfn-lint-validator",
            version="2.0.0",
            description=(
                "Validates CloudFormation template schema, resource types, and "
                "required properties using the real cfn-lint CLI."
            ),
            llm_description=(
                "Runs cfn-lint against the generated template and maps each "
                "finding to a ValidationFinding. Blocks on E (error) severity; "
                "W (warning) findings are recorded but non-blocking."
            ),
            phase=SkillPhase.VALIDATION,
            trigger_condition=(
                "validation_state.yaml_syntax is PASS "
                "AND validation_state.cfn_lint is PENDING"
            ),
            writes_to=["validation_state.cfn_lint"],
            reads_from=["template.body"],
            priority=20,
            tags=["static", "cfn-lint", "schema", "cloudformation"],
        )

    def _ensure_runtime(self):
        if getattr(self, "_runtime_initialized", False):
            return

        import shutil as _shutil
        import subprocess as _subprocess
        import json as _json
        self._shutil = _shutil
        self._subprocess = _subprocess
        self._json = _json
        self._runtime_initialized = True

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        # yaml_syntax must be PASS (not just non-ERROR) — broken YAML cannot
        # be linted.  cfn_lint itself must be PENDING.
        return (
            god.validation_state["yaml_syntax"].status == ValidationStatus.PASS
            and god.validation_state["cfn_lint"].status == ValidationStatus.PENDING
        )

    def execute(self, context: SkillContext) -> SkillResult:
        self._ensure_runtime()
        god = context.god
        skill_result = SkillResult(success=True, skill_name=self.metadata.name)

        vr = ValidationResult(validator_name="cfn_lint")
        vr.started_at = datetime.now().isoformat()

        cfn_lint_bin = self._shutil.which("cfn-lint")
        if cfn_lint_bin is None:
            self._logger.error("cfn-lint not found on PATH. Install: pip install cfn-lint")
            vr.status = ValidationStatus.ERROR
            vr.errors = ["cfn-lint not installed. Run: pip install cfn-lint"]
            vr.completed_at = datetime.now().isoformat()
            god.set_validation_result("cfn_lint", vr, self.metadata.name)
            skill_result.success = False
            return skill_result

        tmp_dir = tempfile.mkdtemp(prefix="cfnlint_skill_")
        template_path = os.path.join(tmp_dir, "template.yaml")

        try:
            with open(template_path, "w", encoding="utf-8") as fh:
                fh.write(god.template.body)

            cmd = [
                cfn_lint_bin,
                "--template", template_path,
                "--format", "json",
                "--include-checks", "W",
            ]
            self._logger.info(f"Running: {' '.join(cmd)}")

            proc = self._subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=context.config.get("cfn_lint_timeout_seconds", 60),
            )
            raw = proc.stdout.strip()
            vr.raw_output = raw[:4000]
            vr.tool_version = self._extract_version(proc.stderr)

            if proc.returncode not in (0, 2, 4, 6):
                raise RuntimeError(
                    f"cfn-lint exited with unexpected code {proc.returncode}: "
                    f"{proc.stderr[:200]}"
                )

            findings = self._parse_cfn_lint_output(raw)
            vr.findings = findings

            error_findings = [f for f in findings if f.severity == Severity.CRITICAL]
            if error_findings:
                vr.status = ValidationStatus.FAIL
                skill_result.success = False
                skill_result.errors.append(
                    f"{len(error_findings)} cfn-lint E-level errors"
                )
            else:
                vr.status = ValidationStatus.PASS
                skill_result.changes_made.append(
                    f"CFN schema validated by cfn-lint "
                    f"({len(findings)} warning(s))"
                )

        except self._subprocess.TimeoutExpired:
            vr.status = ValidationStatus.ERROR
            vr.errors = ["cfn-lint timed out"]
            skill_result.success = False
        except Exception as exc:
            vr.status = ValidationStatus.ERROR
            vr.errors = [str(exc)]
            skill_result.success = False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        vr.completed_at = datetime.now().isoformat()
        god.set_validation_result("cfn_lint", vr, self.metadata.name)
        return skill_result

    def _parse_cfn_lint_output(self, raw: str) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if not raw:
            return findings
        try:
            data = self._json.loads(raw)
        except self._json.JSONDecodeError:
            self._logger.warning(f"Could not parse cfn-lint JSON output: {raw[:200]}")
            return findings

        if isinstance(data, dict):
            data = data.get("matches", data.get("issues", []))

        normalized_matches: list[dict] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    normalized_matches.append(item)
                elif isinstance(item, list):
                    normalized_matches.extend(x for x in item if isinstance(x, dict))

        for match in normalized_matches:
            rule = match.get("Rule", {})
            rule_id = rule.get("Id", "UNKNOWN")
            message = match.get("Message", "")
            first_char = rule_id[0].upper() if rule_id else "W"
            severity = self._SEVERITY_MAP.get(first_char, Severity.MEDIUM)
            location = match.get("Location", {}) or {}
            path_obj = location.get("Path", {}) if isinstance(location, dict) else {}
            cfn_path = path_obj.get("CfnPath", []) if isinstance(path_obj, dict) else []
            resource_name = cfn_path[1] if len(cfn_path) > 1 else "(template)"
            resource_type = cfn_path[0] if cfn_path else "Template"
            start_obj = location.get("Start", {}) if isinstance(location, dict) else {}
            line_number = start_obj.get("LineNumber", 0) if isinstance(start_obj, dict) else 0
            findings.append(ValidationFinding(
                rule_id=rule_id,
                resource_name=resource_name,
                resource_type=resource_type,
                severity=severity,
                message=message,
                remediation_hint=rule.get("ShortDescription", ""),
                line_number=line_number,
            ))

        return findings

    @staticmethod
    def _extract_version(stderr: str) -> str:
        import re
        match = re.search(r"cfn-lint/(\d+\.\d+\.\d+)", stderr, re.I)
        if match:
            return f"cfn-lint-{match.group(1)}"
        match = re.search(r"version\s+(\d[\d.]+)", stderr, re.I)
        return f"cfn-lint-{match.group(1)}" if match else "cfn-lint-unknown"


# =============================================================================
# 3. CHECKOV VALIDATOR
# =============================================================================

class CheckovValidatorSkill(Skill):
    """
    Validates security best practices using the real Checkov CLI.

    Gate: cfn_lint must be PASS **or ERROR** (missing cfn-lint tool is treated
    as skipped — the pipeline must not stall here).
    Install: pip install checkov

    Policy context integration
    --------------------------
    Every failed check is stored as a ValidationFinding with check_id populated.
    RemediationSkill reads check_id and calls
    checkov_context.get_checkov_policy_context() to look up the Checkov source
    code from Data/checkov_cfn_policy_map.csv, then appends it to the LLM
    remediation prompt.  No changes to this skill are required — check_id is
    already populated by _check_to_finding() and the remediation_hint is
    augmented below to surface the inspected CFN property path.
    """

    _SEVERITY_MAP: dict[str, Severity] = {
        "CRITICAL": Severity.CRITICAL,
        "HIGH":     Severity.HIGH,
        "MEDIUM":   Severity.MEDIUM,
        "LOW":      Severity.LOW,
        "INFO":     Severity.INFO,
    }
    _CHECK_SEVERITY_FALLBACK: dict[str, Severity] = {
        "CKV_AWS_17": Severity.CRITICAL,
        "CKV_AWS_19": Severity.CRITICAL,
        "CKV_AWS_24": Severity.CRITICAL,
        "CKV_AWS_25": Severity.CRITICAL,
        "CKV_AWS_18": Severity.HIGH,
        "CKV_AWS_16": Severity.HIGH,
    }

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="checkov-validator",
            version="1.2.0",
            description="Validates security best practices using the Checkov CLI.",
            llm_description=(
                "Runs Checkov against the CloudFormation template. Blocks on "
                "CRITICAL/HIGH findings; records MEDIUM/LOW as non-blocking. "
                "Each finding carries check_id so RemediationSkill can look up "
                "the policy source code from checkov_cfn_policy_map.csv via "
                "checkov_context.get_checkov_policy_context()."
            ),
            phase=SkillPhase.VALIDATION,
            trigger_condition=(
                "validation_state.cfn_lint is PASS or ERROR "
                "AND validation_state.checkov is PENDING"
            ),
            writes_to=["validation_state.checkov"],
            reads_from=["template.body", "intent.constraints"],
            priority=30,
            tags=["static", "security", "checkov"],
        )

    def _ensure_runtime(self):
        if getattr(self, "_runtime_initialized", False):
            return

        import shutil as _shutil
        import subprocess as _subprocess
        import json as _json
        import re as _re
        self._shutil = _shutil
        self._subprocess = _subprocess
        self._json = _json
        self._re = _re
        self._runtime_initialized = True

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        # Accept cfn_lint=ERROR (tool absent) as equivalent to PASS so the
        # chain does not stall when cfn-lint is not installed.
        return (
            god.validation_state["cfn_lint"].status in _PASS_OR_ERROR
            and god.validation_state["checkov"].status == ValidationStatus.PENDING
        )

    def execute(self, context: SkillContext) -> SkillResult:
        self._ensure_runtime()
        god = context.god
        skill_result = SkillResult(success=True, skill_name=self.metadata.name)

        vr = ValidationResult(validator_name="checkov")
        vr.started_at = datetime.now().isoformat()

        checkov_bin = self._shutil.which("checkov")
        if checkov_bin is None:
            self._logger.error("checkov not found on PATH. Install: pip install checkov")
            vr.status = ValidationStatus.ERROR
            vr.errors = ["checkov not installed. Run: pip install checkov"]
            vr.completed_at = datetime.now().isoformat()
            god.set_validation_result("checkov", vr, self.metadata.name)
            skill_result.success = False
            return skill_result

        tmp_dir = tempfile.mkdtemp(prefix="checkov_skill_")
        template_path = os.path.join(tmp_dir, "template.yaml")

        try:
            with open(template_path, "w", encoding="utf-8") as fh:
                fh.write(god.template.body)

            cmd = [
                checkov_bin,
                "--file", template_path,
                "--framework", "cloudformation",
                "--output", "json",
                "--quiet",
                "--compact",
            ]
            self._logger.info(f"Running: {' '.join(cmd)}")

            proc = self._subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=context.config.get("checkov_timeout_seconds", 120),
            )

            raw_stdout = proc.stdout.strip()
            raw_stderr = proc.stderr.strip()
            vr.raw_output = raw_stdout[:4000]
            vr.tool_version = self._extract_version(raw_stderr)

            if proc.returncode == 2:
                raise RuntimeError(f"Checkov process error (exit 2): {raw_stderr[:300]}")

            findings = self._parse_checkov_output(raw_stdout)
            vr.findings = findings

            blocking = [f for f in findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
            if blocking:
                vr.status = ValidationStatus.FAIL
                skill_result.success = False
                skill_result.errors.append(f"{len(blocking)} critical/high security findings")
            else:
                vr.status = ValidationStatus.PASS
                skill_result.changes_made.append(
                    f"Security validated by Checkov ({len(findings)} low/medium findings)"
                )

        except self._subprocess.TimeoutExpired:
            vr.status = ValidationStatus.ERROR
            vr.errors = ["Checkov timed out"]
            skill_result.success = False
        except Exception as exc:
            vr.status = ValidationStatus.ERROR
            vr.errors = [str(exc)]
            skill_result.success = False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        vr.completed_at = datetime.now().isoformat()
        god.set_validation_result("checkov", vr, self.metadata.name)
        return skill_result

    def _parse_checkov_output(self, raw: str) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if not raw:
            return findings
        try:
            data = self._json.loads(raw)
        except self._json.JSONDecodeError:
            match = self._re.search(r"\{.*\}", raw, self._re.DOTALL)
            if not match:
                return findings
            try:
                data = self._json.loads(match.group())
            except self._json.JSONDecodeError:
                return findings

        all_failed: list[dict] = []
        if isinstance(data, list):
            for entry in data:
                all_failed.extend(entry.get("results", {}).get("failed_checks", []))
        else:
            all_failed = data.get("results", {}).get("failed_checks", [])

        for check in all_failed:
            findings.append(self._check_to_finding(check))
        return findings

    def _check_to_finding(self, check: dict) -> ValidationFinding:
        check_id = check.get("check_id", "UNKNOWN")
        check_name = (
            check.get("check_name")
            or check.get("check")
            or check.get("check_title")
            or check.get("description")
            or check_id
        )
        if isinstance(check_name, dict):
            check_name = (
                check_name.get("name")
                or check_name.get("title")
                or check_name.get("description")
                or check_id
            )
        if not isinstance(check_name, str):
            check_name = str(check_name)

        raw_resource = check.get("resource", "")
        resource_type, resource_name = (
            raw_resource.rsplit(".", 1) if "." in raw_resource
            else (raw_resource, raw_resource)
        )
        sev_str = (check.get("severity") or "").upper()
        severity = (
            self._SEVERITY_MAP.get(sev_str)
            or self._CHECK_SEVERITY_FALLBACK.get(check_id)
            or Severity.MEDIUM
        )
        line_range = check.get("file_line_range", [0, 0])
        guideline = check.get("guideline", "")
        evaluated_keys = check.get("check_result", {}).get("evaluated_keys", [])
        # Surface the evaluated CFN property path in the hint so that even
        # without the full policy context the LLM has a concrete target.
        property_hint = (
            "Fix CFN property path(s): " + ", ".join(str(k) for k in evaluated_keys[:5])
            if evaluated_keys else ""
        )
        remediation_parts = [p for p in [property_hint, guideline] if p]
        remediation = " | ".join(remediation_parts)

        return ValidationFinding(
            rule_id=check_id,
            resource_name=resource_name,
            resource_type=resource_type,
            severity=severity,
            message=check_name,
            remediation_hint=remediation,
            check_id=check_id,
            file_path=check.get("file_path", ""),
            line_number=line_range[0] if line_range else 0,
        )

    @staticmethod
    def _extract_version(stderr: str) -> str:
        import re
        match = re.search(r"checkov[/ ](\d+\.\d+\.\d+)", stderr, re.I)
        if match:
            return f"checkov-{match.group(1)}"
        match = re.search(r"version\s+(\d[\d.]+)", stderr, re.I)
        return f"checkov-{match.group(1)}" if match else "checkov-unknown"


# =============================================================================
# 4. INTENT ALIGNMENT VALIDATOR
# =============================================================================

_INTENT_ALIGNMENT_PROMPT = """
You are an expert AWS CloudFormation reviewer.
You will be given:
  1. The original user intent (what infrastructure was requested)
  2. The list of acceptance criteria that the template must satisfy
  3. The full generated CloudFormation YAML template

Your job is to evaluate the template on TWO dimensions:

## Dimension 1 — Accuracy
Does each resource that IS present match the user's intent?
- Wrong resource type for the stated goal
- Incorrect property values (e.g. wrong engine, wrong runtime)
- Missing critical properties implied by the intent

## Dimension 2 — Coverage
Are ALL resources and requirements from the intent present in the template?
- Missing resources (user asked for X but X is absent)
- Missing acceptance criteria fulfilment

Return a JSON object ONLY (no prose, no markdown fences) with this exact schema:
{
  "accuracy_score": <float 0.0-1.0>,
  "coverage_score": <float 0.0-1.0>,
  "accuracy_findings": [
    {
      "criterion_id": "AC-N or ACCURACY-N",
      "resource_type": "AWS::...",
      "resource_name": "LogicalName",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "message": "<what is wrong>",
      "remediation_hint": "<how to fix>"
    }
  ],
  "coverage_findings": [
    {
      "criterion_id": "COVERAGE-N",
      "resource_type": "AWS::... or empty string",
      "resource_name": "(missing) or LogicalName",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "message": "<what is missing>",
      "remediation_hint": "<how to fix>"
    }
  ],
  "overall_pass": <true if accuracy_score >= 0.8 AND coverage_score >= 0.8 else false>
}
"""


class IntentAlignmentValidatorSkill(Skill):
    """
    Validates that the generated template satisfies the user's original intent
    using an LLM-based two-dimensional scoring approach.

    Gate: checkov must be PASS **or ERROR** (missing checkov tool is treated
    as skipped — the pipeline must not stall here).

    Dimensions
    ----------
    Accuracy  — each resource present in the template is correct and complete
    Coverage  — all resources / requirements from the intent are present

    Pass threshold: accuracy_score >= 0.8 AND coverage_score >= 0.8.
    """

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="intent-validator",
            version="2.0.0",
            description=(
                "LLM-based validator that scores the generated template on "
                "accuracy (resources are correct) and coverage (all intent "
                "requirements are present)."
            ),
            llm_description=(
                "Sends the user intent, acceptance criteria, and the full "
                "CloudFormation template to the LLM for structured evaluation. "
                "Returns per-finding details and pass/fail scores on accuracy "
                "and coverage dimensions."
            ),
            phase=SkillPhase.VALIDATION,
            trigger_condition=(
                "validation_state.checkov is PASS or ERROR "
                "AND validation_state.intent_alignment is PENDING"
            ),
            writes_to=["validation_state.intent_alignment"],
            reads_from=[
                "template.body",
                "intent.raw_prompt",
                "intent.acceptance_criteria",
                "intent.resources",
            ],
            priority=40,
            tags=["llm", "intent", "accuracy", "coverage"],
        )

    def _ensure_runtime(self):
        if getattr(self, "_runtime_initialized", False):
            return
        import json as _json
        self._json = _json
        self._runtime_initialized = True

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        # Accept checkov=ERROR (tool absent) as equivalent to PASS.
        return (
            god.validation_state["checkov"].status in _PASS_OR_ERROR
            and god.validation_state["intent_alignment"].status == ValidationStatus.PENDING
        )

    def execute(self, context: SkillContext) -> SkillResult:
        self._ensure_runtime()
        god = context.god
        skill_result = SkillResult(success=True, skill_name=self.metadata.name)

        vr = ValidationResult(validator_name="intent_alignment")
        vr.started_at = datetime.now().isoformat()

        from llm_client import OpenRouterClient
        llm: OpenRouterClient | None = context.get_config("llm")

        if llm is None:
            self._logger.warning(
                "No LLM client; falling back to structural intent check."
            )
            return self._fallback_structural_check(god, vr, skill_result)

        threshold: float = context.config.get("intent_pass_threshold", 0.8)

        criteria_text = "\n".join(
            f"  [{c.id}] {c.description}"
            for c in god.intent.acceptance_criteria
        )
        resources_text = "\n".join(
            f"  - {r.resource_type} ({r.logical_name})"
            for r in god.intent.resources
        )

        user_message = (
            f"## User Intent\n{god.intent.raw_prompt}\n\n"
            f"## Expected Resources\n{resources_text}\n\n"
            f"## Acceptance Criteria\n{criteria_text}\n\n"
            f"## Generated CloudFormation Template\n```yaml\n{god.template.body}\n```"
        )

        try:
            raw = llm.complete(
                system=_INTENT_ALIGNMENT_PROMPT,
                user=user_message,
                temperature=0.0,
            )

            import re as _re
            raw_clean = _re.sub(r'^```[\w]*\n?|\n?```$', '', raw.strip(), flags=_re.MULTILINE)
            data: dict = self._json.loads(raw_clean)

        except self._json.JSONDecodeError as exc:
            self._logger.warning(
                f"LLM returned non-JSON intent evaluation ({exc}); "
                "falling back to structural check."
            )
            return self._fallback_structural_check(god, vr, skill_result)
        except Exception as exc:
            vr.status = ValidationStatus.ERROR
            vr.errors = [f"LLM intent evaluation error: {exc}"]
            vr.completed_at = datetime.now().isoformat()
            god.set_validation_result("intent_alignment", vr, self.metadata.name)
            skill_result.success = False
            return skill_result

        accuracy_score: float = float(data.get("accuracy_score", 0.0))
        coverage_score: float = float(data.get("coverage_score", 0.0))
        overall_pass: bool = data.get("overall_pass", False)

        vr.metrics = {
            "accuracy_score": accuracy_score,
            "coverage_score": coverage_score,
        }

        findings: list[ValidationFinding] = []
        for raw_f in data.get("accuracy_findings", []) + data.get("coverage_findings", []):
            sev_str = (raw_f.get("severity") or "HIGH").upper()
            sev_map = {
                "CRITICAL": Severity.CRITICAL,
                "HIGH": Severity.HIGH,
                "MEDIUM": Severity.MEDIUM,
                "LOW": Severity.LOW,
            }
            findings.append(ValidationFinding(
                rule_id=raw_f.get("criterion_id", "INTENT"),
                resource_name=raw_f.get("resource_name", "(template)"),
                resource_type=raw_f.get("resource_type", "Template"),
                severity=sev_map.get(sev_str, Severity.HIGH),
                message=raw_f.get("message", ""),
                remediation_hint=raw_f.get("remediation_hint", ""),
            ))
        vr.findings = findings

        finding_criterion_ids = {
            f.rule_id for f in findings if f.severity in (Severity.CRITICAL, Severity.HIGH)
        }
        for criterion in god.intent.acceptance_criteria:
            criterion.is_met = criterion.id not in finding_criterion_ids

        if overall_pass and accuracy_score >= threshold and coverage_score >= threshold:
            vr.status = ValidationStatus.PASS
            skill_result.changes_made.append(
                f"Intent validated — accuracy={accuracy_score:.2f}, "
                f"coverage={coverage_score:.2f}"
            )
        else:
            vr.status = ValidationStatus.FAIL
            skill_result.success = False
            skill_result.errors.append(
                f"Intent misalignment: accuracy={accuracy_score:.2f} "
                f"coverage={coverage_score:.2f} "
                f"(threshold={threshold}) — "
                f"{len(findings)} finding(s)"
            )

        vr.completed_at = datetime.now().isoformat()
        god.set_validation_result("intent_alignment", vr, self.metadata.name)
        return skill_result

    def _fallback_structural_check(
        self,
        god: GroundedObjectivesDocument,
        vr: ValidationResult,
        skill_result: SkillResult,
    ) -> SkillResult:
        try:
            template = cfn_safe_load(god.template.body)
        except Exception as exc:
            vr.status = ValidationStatus.ERROR
            vr.errors = [f"Cannot parse template for structural check: {exc}"]
            vr.completed_at = datetime.now().isoformat()
            god.set_validation_result("intent_alignment", vr, self.metadata.name)
            skill_result.success = False
            return skill_result

        resources = template.get("Resources", {})
        present_types = {r.get("Type") for r in resources.values()}
        findings: list[ValidationFinding] = []

        for expected in god.intent.resources:
            if expected.resource_type not in present_types:
                findings.append(ValidationFinding(
                    rule_id="COVERAGE-MISSING",
                    resource_name=expected.logical_name,
                    resource_type=expected.resource_type,
                    severity=Severity.HIGH,
                    message=f"Expected resource type {expected.resource_type} not found in template",
                    remediation_hint=f"Add an {expected.resource_type} resource to the template",
                ))

        vr.findings = findings
        if findings:
            vr.status = ValidationStatus.FAIL
            skill_result.success = False
            skill_result.errors.append(
                f"{len(findings)} missing resource type(s) in structural check"
            )
        else:
            vr.status = ValidationStatus.PASS
            skill_result.changes_made.append(
                "Structural intent check passed (LLM not available)"
            )

        vr.completed_at = datetime.now().isoformat()
        god.set_validation_result("intent_alignment", vr, self.metadata.name)
        return skill_result
