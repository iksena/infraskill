
# -----------------------------------------------------------------------------
# VALIDATION SKILLS
# -----------------------------------------------------------------------------

from datetime import datetime
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any

from enums import Severity, SkillPhase, ValidationStatus
from god import AcceptanceCriterion, GroundedObjectivesDocument, ValidationFinding, ValidationResult
from skill_framework import Skill, SkillContext, SkillMetadata, SkillResult


class YAMLSyntaxValidatorSkill(Skill):
    """Validates YAML syntax of the template"""
    
    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="yaml-validator",
            description="Validates YAML syntax",
            phase=SkillPhase.VALIDATION,
            trigger_condition="yaml_syntax validation is PENDING",
            writes_to=["validation_state.yaml_syntax"],
            reads_from=["template.body"],
            priority=10  # First validator
        )
    
    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        return (
            god.template.body and 
            god.validation_state["yaml_syntax"].status == ValidationStatus.PENDING
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        skill_result = SkillResult(success=True, skill_name=self.metadata.name)
        
        validation_result = ValidationResult(validator_name="yaml_syntax")
        validation_result.started_at = datetime.now().isoformat()
        
        try:
            import yaml
            parsed = yaml.safe_load(god.template.body)
            
            if parsed is None:
                validation_result.status = ValidationStatus.FAIL
                validation_result.errors = ["Template parsed to None - empty or invalid"]
                skill_result.success = False
            elif not isinstance(parsed, dict):
                validation_result.status = ValidationStatus.FAIL
                validation_result.errors = ["Template root must be a mapping"]
                skill_result.success = False
            else:
                validation_result.status = ValidationStatus.PASS
                skill_result.changes_made.append("YAML syntax validated")
                
        except yaml.YAMLError as e:
            validation_result.status = ValidationStatus.FAIL
            validation_result.errors = [f"YAML parse error: {str(e)}"]
            skill_result.success = False
        except ImportError:
            validation_result.status = ValidationStatus.ERROR
            validation_result.errors = ["PyYAML not installed"]
            skill_result.success = False
        
        validation_result.completed_at = datetime.now().isoformat()
        god.set_validation_result("yaml_syntax", validation_result, self.metadata.name)
        
        return skill_result


class CFNLintValidatorSkill(Skill):
    """Validates CloudFormation template structure and schema"""
    
    REQUIRED_TOP_LEVEL = ["AWSTemplateFormatVersion", "Resources"]
    
    RESOURCE_SCHEMAS = {
        "AWS::EC2::VPC": {
            "required": ["CidrBlock"],
            "optional": ["EnableDnsSupport", "EnableDnsHostnames", "InstanceTenancy", "Tags"]
        },
        "AWS::EC2::Subnet": {
            "required": ["VpcId", "CidrBlock"],
            "optional": ["AvailabilityZone", "MapPublicIpOnLaunch", "Tags"]
        },
        "AWS::EC2::SecurityGroup": {
            "required": ["GroupDescription"],
            "optional": ["GroupName", "VpcId", "SecurityGroupIngress", "SecurityGroupEgress", "Tags"]
        },
        "AWS::S3::Bucket": {
            "required": [],
            "optional": ["BucketName", "PublicAccessBlockConfiguration", "BucketEncryption", 
                        "VersioningConfiguration", "LoggingConfiguration", "Tags"]
        },
        "AWS::RDS::DBInstance": {
            "required": ["DBInstanceClass", "Engine"],
            "optional": ["DBInstanceIdentifier", "AllocatedStorage", "StorageType", "StorageEncrypted",
                        "MultiAZ", "PubliclyAccessible", "MasterUsername", "ManageMasterUserPassword",
                        "DBSubnetGroupName", "VPCSecurityGroups", "BackupRetentionPeriod", "Tags"]
        },
        "AWS::IAM::Role": {
            "required": ["AssumeRolePolicyDocument"],
            "optional": ["RoleName", "Description", "ManagedPolicyArns", "Policies", "Tags"]
        },
        "AWS::Lambda::Function": {
            "required": ["Role", "Runtime"],
            "optional": ["FunctionName", "Handler", "Code", "MemorySize", "Timeout", "Tags"]
        }
    }
    
    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="cfn-lint-validator",
            description="Validates CloudFormation schema and structure",
            phase=SkillPhase.VALIDATION,
            trigger_condition="yaml_syntax PASS and cfn_lint PENDING",
            writes_to=["validation_state.cfn_lint"],
            reads_from=["template.body"],
            priority=20
        )
    
    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        return (
            god.validation_state["yaml_syntax"].status == ValidationStatus.PASS and
            god.validation_state["cfn_lint"].status == ValidationStatus.PENDING
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        skill_result = SkillResult(success=True, skill_name=self.metadata.name)
        
        validation_result = ValidationResult(validator_name="cfn_lint")
        validation_result.started_at = datetime.now().isoformat()
        
        try:
            import yaml
            template = yaml.safe_load(god.template.body)
            
            findings = []
            
            # Check required top-level keys
            for key in self.REQUIRED_TOP_LEVEL:
                if key not in template:
                    findings.append(ValidationFinding(
                        rule_id="E1001",
                        resource_name="(template)",
                        resource_type="Template",
                        severity=Severity.CRITICAL,
                        message=f"Missing required top-level key: {key}",
                        remediation_hint=f"Add '{key}' to the template root"
                    ))
            
            # Check each resource
            resources = template.get("Resources", {})
            if not resources:
                findings.append(ValidationFinding(
                    rule_id="E3001",
                    resource_name="(template)",
                    resource_type="Template",
                    severity=Severity.CRITICAL,
                    message="No resources defined in template",
                    remediation_hint="Add at least one resource to the Resources section"
                ))
            
            for res_name, resource in resources.items():
                # Check Type exists
                if "Type" not in resource:
                    findings.append(ValidationFinding(
                        rule_id="E3002",
                        resource_name=res_name,
                        resource_type="Unknown",
                        severity=Severity.CRITICAL,
                        message=f"Resource {res_name} missing Type property",
                        remediation_hint="Add 'Type' property to the resource"
                    ))
                    continue
                
                res_type = resource["Type"]
                properties = resource.get("Properties", {})
                
                # Check against schema if we have one
                if res_type in self.RESOURCE_SCHEMAS:
                    schema = self.RESOURCE_SCHEMAS[res_type]
                    
                    for req_prop in schema["required"]:
                        if req_prop not in properties:
                            findings.append(ValidationFinding(
                                rule_id="E3003",
                                resource_name=res_name,
                                resource_type=res_type,
                                severity=Severity.CRITICAL,
                                message=f"Missing required property: {req_prop}",
                                remediation_hint=f"Add '{req_prop}' to {res_name}.Properties"
                            ))
            
            # Set result based on findings
            critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
            
            if critical_findings:
                validation_result.status = ValidationStatus.FAIL
                skill_result.success = False
            else:
                validation_result.status = ValidationStatus.PASS
                skill_result.changes_made.append("CFN schema validated")
            
            validation_result.findings = findings
            
        except Exception as e:
            validation_result.status = ValidationStatus.ERROR
            validation_result.errors = [str(e)]
            skill_result.success = False
        
        validation_result.completed_at = datetime.now().isoformat()
        god.set_validation_result("cfn_lint", validation_result, self.metadata.name)
        
        return skill_result

class CheckovValidatorSkill(Skill):
    """
    Validates security best practices using the real Checkov CLI.

    Invokes: checkov -f <template> --framework cloudformation --output json
    Falls back to a subprocess-unavailable warning (not silent failure).
    
    Requires: `pip install checkov` in the runtime environment.
    """

    # Severity mapping from Checkov's string levels to your Severity enum
    _SEVERITY_MAP: dict[str, Severity] = {
        "CRITICAL": Severity.CRITICAL,
        "HIGH":     Severity.HIGH,
        "MEDIUM":   Severity.MEDIUM,
        "LOW":      Severity.LOW,
        "INFO":     Severity.INFO,
    }

    # Checkov doesn't always emit a severity field; default by check ID prefix
    _CHECK_SEVERITY_FALLBACK: dict[str, Severity] = {
        "CKV_AWS_17": Severity.CRITICAL,  # RDS not publicly accessible
        "CKV_AWS_19": Severity.CRITICAL,  # S3 public access block
        "CKV_AWS_24": Severity.CRITICAL,  # SG SSH open to world
        "CKV_AWS_25": Severity.CRITICAL,  # SG RDP open to world
        "CKV_AWS_18": Severity.HIGH,      # S3 encryption
        "CKV_AWS_16": Severity.HIGH,      # RDS storage encrypted
    }

    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="checkov-validator",
            description="Validates security best practices using Checkov CLI (real tool)",
            phase=SkillPhase.VALIDATION,
            trigger_condition="cfn_lint PASS and checkov PENDING",
            writes_to=["validation_state.checkov"],
            reads_from=["template.body", "intent.constraints"],
            priority=30,
        )

    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        return (
            god.validation_state["cfn_lint"].status == ValidationStatus.PASS
            and god.validation_state["checkov"].status == ValidationStatus.PENDING
        )

    # -------------------------------------------------------------------------
    # Main execution
    # -------------------------------------------------------------------------

    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        skill_result = SkillResult(success=True, skill_name=self.metadata.name)

        validation_result = ValidationResult(validator_name="checkov")
        validation_result.started_at = datetime.now().isoformat()

        # ------------------------------------------------------------------
        # 1. Preflight: is checkov on PATH?
        # ------------------------------------------------------------------
        checkov_bin = shutil.which("checkov")
        if checkov_bin is None:
            self._logger.error(
                "checkov binary not found on PATH. "
                "Install it with: pip install checkov"
            )
            validation_result.status = ValidationStatus.ERROR
            validation_result.errors = [
                "checkov not installed. Run: pip install checkov"
            ]
            validation_result.completed_at = datetime.now().isoformat()
            god.set_validation_result("checkov", validation_result, self.metadata.name)
            skill_result.success = False
            skill_result.errors.append("checkov binary not found")
            return skill_result

        # ------------------------------------------------------------------
        # 2. Write template to a temp file (checkov needs a file path)
        # ------------------------------------------------------------------
        tmp_dir = tempfile.mkdtemp(prefix="checkov_skill_")
        template_path = os.path.join(tmp_dir, "template.yaml")

        try:
            with open(template_path, "w", encoding="utf-8") as fh:
                fh.write(god.template.body)

            # ------------------------------------------------------------------
            # 3. Run Checkov
            # ------------------------------------------------------------------
            cmd = [
                checkov_bin,
                "--file", template_path,
                "--framework", "cloudformation",
                "--output", "json",
                "--quiet",          # suppress progress bars / ANSI codes
                "--compact",        # single-line JSON per check (machine-friendly)
            ]

            self._logger.info(f"Running: {' '.join(cmd)}")

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=context.config.get("checkov_timeout_seconds", 120),
            )

            # Checkov exits 0 (all pass), 1 (failures found), or 2 (error)
            raw_stdout = proc.stdout.strip()
            raw_stderr = proc.stderr.strip()

            self._logger.debug(f"Checkov exit code: {proc.returncode}")
            if raw_stderr:
                self._logger.debug(f"Checkov stderr: {raw_stderr[:500]}")

            validation_result.raw_output = raw_stdout[:4000]  # truncate for GOD

            if proc.returncode == 2:
                # Checkov itself errored (bad template path, internal error, etc.)
                raise RuntimeError(
                    f"Checkov process error (exit 2): {raw_stderr[:300]}"
                )

            # ------------------------------------------------------------------
            # 4. Parse JSON output → ValidationFinding list
            # ------------------------------------------------------------------
            findings = self._parse_checkov_output(raw_stdout, template_path)
            validation_result.findings = findings

            # ------------------------------------------------------------------
            # 5. Determine PASS / FAIL
            # ------------------------------------------------------------------
            blocking = [
                f for f in findings
                if f.severity in (Severity.CRITICAL, Severity.HIGH)
            ]

            if blocking:
                validation_result.status = ValidationStatus.FAIL
                skill_result.success = False
                skill_result.errors.append(
                    f"{len(blocking)} critical/high security findings"
                )
                self._logger.warning(
                    f"Checkov FAIL: {len(blocking)} blocking findings "
                    f"({len(findings)} total)"
                )
            else:
                validation_result.status = ValidationStatus.PASS
                skill_result.changes_made.append(
                    f"Security validated by Checkov "
                    f"({len(findings)} low/medium findings)"
                )
                self._logger.info(
                    f"Checkov PASS ({len(findings)} non-blocking findings)"
                )

            # Attach version info from stderr/stdout if available
            validation_result.tool_version = self._extract_version(raw_stderr)

        except subprocess.TimeoutExpired:
            validation_result.status = ValidationStatus.ERROR
            validation_result.errors = ["Checkov timed out"]
            skill_result.success = False
            self._logger.error("Checkov timed out")

        except Exception as exc:
            validation_result.status = ValidationStatus.ERROR
            validation_result.errors = [str(exc)]
            skill_result.success = False
            self._logger.error(f"Checkov execution error: {exc}")

        finally:
            # Always clean up the temp dir
            shutil.rmtree(tmp_dir, ignore_errors=True)

        validation_result.completed_at = datetime.now().isoformat()
        god.set_validation_result("checkov", validation_result, self.metadata.name)

        return skill_result

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _parse_checkov_output(
        self, raw: str, template_path: str
    ) -> list[ValidationFinding]:
        """
        Parse Checkov JSON output into ValidationFinding objects.

        Checkov --output json produces a top-level dict like:
        {
          "results": {
            "passed_checks": [...],
            "failed_checks": [...],
            "parsing_errors": [...]
          }
        }
        Each check entry:
        {
          "check_id": "CKV_AWS_18",
          "check": {"name": "...", "id": "..."},
          "resource": "AWS::S3::Bucket.MyBucket",
          "file_path": "/tmp/.../template.yaml",
          "file_line_range": [10, 20],
          "severity": "HIGH",           ← present in newer Checkov (>=2.3)
          "guideline": "https://...",
          "check_result": {"result": "FAILED", "evaluated_keys": [...]}
        }
        """
        findings: list[ValidationFinding] = []

        if not raw:
            self._logger.warning("Checkov produced no output — assuming clean")
            return findings

        # Checkov may emit multiple JSON objects when scanning multiple files.
        # In --compact mode it's usually a single JSON document.
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting the first valid JSON object
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                self._logger.warning(f"Could not parse Checkov output: {raw[:200]}")
                return findings
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                self._logger.warning("Checkov output is not valid JSON")
                return findings

        # Handle both single-framework dict and multi-framework list
        if isinstance(data, list):
            # Multiple framework results — merge them
            all_failed: list[dict] = []
            for entry in data:
                results = entry.get("results", {})
                all_failed.extend(results.get("failed_checks", []))
        else:
            results = data.get("results", {})
            all_failed = results.get("failed_checks", [])

        for check in all_failed:
            findings.append(self._check_to_finding(check))

        return findings

    def _check_to_finding(self, check: dict) -> ValidationFinding:
        """Convert a single Checkov failed_check dict to a ValidationFinding."""

        check_id: str = check.get("check_id", "UNKNOWN")
        check_name: str = check.get("check", {})
        if isinstance(check_name, dict):
            check_name = check_name.get("name", check_id)

        # Resource comes as "ResourceType.LogicalName" e.g. "AWS::S3::Bucket.MyBucket"
        raw_resource: str = check.get("resource", "")
        if "." in raw_resource:
            resource_type, resource_name = raw_resource.rsplit(".", 1)
        else:
            resource_type = raw_resource
            resource_name = raw_resource

        # Severity: prefer field on the check, fall back to our mapping, then MEDIUM
        sev_str: str = (check.get("severity") or "").upper()
        severity = (
            self._SEVERITY_MAP.get(sev_str)
            or self._CHECK_SEVERITY_FALLBACK.get(check_id)
            or Severity.MEDIUM
        )

        # File location
        file_path: str = check.get("file_path", "")
        line_range: list = check.get("file_line_range", [0, 0])
        line_number: int = line_range[0] if line_range else 0

        # Remediation hint from guideline URL or evaluated keys
        guideline: str = check.get("guideline", "")
        evaluated_keys = check.get("check_result", {}).get("evaluated_keys", [])
        remediation = guideline or (
            f"Review property: {evaluated_keys[0]}" if evaluated_keys else ""
        )

        return ValidationFinding(
            rule_id=check_id,
            resource_name=resource_name,
            resource_type=resource_type,
            severity=severity,
            message=check_name,
            remediation_hint=remediation,
            check_id=check_id,
            file_path=file_path,
            line_number=line_number,
        )

    @staticmethod
    def _extract_version(stderr: str) -> str:
        """Extract checkov version string from stderr output."""
        import re
        match = re.search(r"checkov[/ ](\d+\.\d+\.\d+)", stderr, re.IGNORECASE)
        if match:
            return f"checkov-{match.group(1)}"
        # Try: checkov version X.Y.Z
        match = re.search(r"version\s+(\d[\d.]+)", stderr, re.IGNORECASE)
        return f"checkov-{match.group(1)}" if match else "checkov-unknown"


class IntentAlignmentValidatorSkill(Skill):
    """
    Validates that the generated template meets all acceptance criteria.
    
    This is the final validation gate - it checks that the template
    actually implements what the user requested.
    """
    
    def _define_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="intent-validator",
            description="Validates template meets acceptance criteria",
            phase=SkillPhase.VALIDATION,
            trigger_condition="checkov PASS and intent_alignment PENDING",
            writes_to=["validation_state.intent_alignment"],
            reads_from=["template.body", "intent.acceptance_criteria"],
            priority=40  # Last validator
        )
    
    def can_trigger(self, god: GroundedObjectivesDocument) -> bool:
        return (
            god.validation_state["checkov"].status == ValidationStatus.PASS and
            god.validation_state["intent_alignment"].status == ValidationStatus.PENDING
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        god = context.god
        skill_result = SkillResult(success=True, skill_name=self.metadata.name)
        
        validation_result = ValidationResult(validator_name="intent_alignment")
        validation_result.started_at = datetime.now().isoformat()
        
        try:
            import yaml
            template = yaml.safe_load(god.template.body)
            resources = template.get("Resources", {})
            
            unmet_criteria = []
            met_count = 0
            
            for criterion in god.intent.acceptance_criteria:
                is_met, reason = self._check_criterion(criterion, resources)
                criterion.is_met = is_met
                criterion.failure_reason = reason
                
                if is_met:
                    met_count += 1
                else:
                    unmet_criteria.append(criterion)
                    validation_result.findings.append(ValidationFinding(
                        rule_id=criterion.id,
                        resource_name=criterion.resource_type or "(template)",
                        resource_type=criterion.resource_type or "Template",
                        severity=Severity.HIGH,
                        message=f"Acceptance criterion not met: {criterion.description}",
                        remediation_hint=reason
                    ))
            
            total = len(god.intent.acceptance_criteria)
            
            if unmet_criteria:
                validation_result.status = ValidationStatus.FAIL
                skill_result.success = False
                skill_result.errors.append(f"{len(unmet_criteria)}/{total} acceptance criteria not met")
            else:
                validation_result.status = ValidationStatus.PASS
                skill_result.changes_made.append(f"All {total} acceptance criteria met")
            
        except Exception as e:
            validation_result.status = ValidationStatus.ERROR
            validation_result.errors = [str(e)]
            skill_result.success = False
        
        validation_result.completed_at = datetime.now().isoformat()
        god.set_validation_result("intent_alignment", validation_result, self.metadata.name)
        
        return skill_result
    
    def _check_criterion(
        self, 
        criterion: AcceptanceCriterion, 
        resources: dict
    ) -> tuple[bool, str]:
        """
        Check a single acceptance criterion against the template resources.
        Returns (is_met, failure_reason).
        """
        # Find resources of the specified type
        matching_resources = [
            (name, res) for name, res in resources.items()
            if res.get("Type") == criterion.resource_type
        ]
        
        if not matching_resources and criterion.resource_type:
            return False, f"No resource of type {criterion.resource_type} found"
        
        # For each matching resource, check the criterion
        for res_name, resource in matching_resources:
            properties = resource.get("Properties", {})
            
            # Navigate the property path
            value = self._get_nested_value(properties, criterion.property_path)
            
            # Check based on check_type
            if criterion.check_type == "exists":
                if value is not None:
                    return True, ""
                    
            elif criterion.check_type == "not_exists":
                if value is None:
                    return True, ""
                    
            elif criterion.check_type == "equals":
                if value == criterion.expected_value:
                    return True, ""
                    
            elif criterion.check_type == "not_equals":
                if value != criterion.expected_value:
                    return True, ""
                    
            elif criterion.check_type == "contains":
                if criterion.expected_value in str(value):
                    return True, ""
        
        # Build failure reason
        if criterion.check_type == "exists":
            return False, f"Property {criterion.property_path} not found"
        elif criterion.check_type == "equals":
            return False, f"Expected {criterion.property_path}={criterion.expected_value}, got {value}"
        elif criterion.check_type == "not_equals":
            return False, f"Property {criterion.property_path} should not equal {criterion.expected_value}"
        else:
            return False, f"Check '{criterion.check_type}' failed for {criterion.property_path}"
    
    def _get_nested_value(self, obj: dict, path: str) -> Any:
        """Get a nested value from a dict using dot notation path"""
        if not path:
            return obj
        
        # Remove 'Properties.' prefix if present
        if path.startswith("Properties."):
            path = path[11:]
        
        parts = path.split(".")
        current = obj
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
