"""
INFRA-SKILL Telemetry
=====================

IaCGen-style data collection. Every LLM call, skill invocation, tool use,
error, validation result, remediation round, and the final template + GOD
are written to a newline-delimited JSON (JSONL) file under data/.

One file per pipeline run, named by the run_id (timestamp + prompt hash).
Each line is a self-contained JSON record with a ``record_type`` discriminator.

Record types
------------
``run_start``        – pipeline initialised
``llm_call``         – every LLM request/response (all skills + skill-selector)
``skill_start``      – skill about to execute
``skill_end``        – skill finished (success or failure)
``tool_use``         – deterministic tool/patch applied inside a skill
``error``            – any caught exception or LLM-parse failure
``validation_result``– result of each validator
``remediation``      – one remediation round summary
``god_snapshot``     – full GOD state (saved at key checkpoints)
``template_snapshot``– the assembled CloudFormation template body
``run_end``          – final result summary
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id(prompt: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ph = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    return f"{ts}_{ph}"


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------

class TelemetryCollector:
    """
    Thread-safe, append-only telemetry sink.

    Usage
    -----
    >>> tc = TelemetryCollector(run_id="...", data_dir="data")
    >>> tc.record_run_start(prompt="...")
    >>> tc.record_llm_call(skill_name="planner", ...)
    >>> tc.record_skill_end(skill_name="planner", ...)
    >>> tc.record_run_end(result_dict)
    """

    def __init__(self, run_id: str, data_dir: str = "data"):
        self.run_id = run_id
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{run_id}.jsonl"
        self._lock = threading.Lock()
        self._seq = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, record: dict) -> None:
        with self._lock:
            self._seq += 1
            record["_run_id"] = self.run_id
            record["_seq"] = self._seq
            record["_ts"] = _now_iso()
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_run_start(self, prompt: str, config: dict | None = None) -> None:
        self._write({
            "record_type": "run_start",
            "prompt": prompt,
            "prompt_length": len(prompt),
            "config": config or {},
        })

    def record_llm_call(
        self,
        *,
        skill_name: str,
        purpose: str,          # e.g. "skill_selection", "planning", "engineering", "remediation"
        model: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        temperature: float,
        latency_ms: float,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        self._write({
            "record_type": "llm_call",
            "skill_name": skill_name,
            "purpose": purpose,
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response,
            "temperature": temperature,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "error": error,
        })

    def record_skill_start(
        self,
        *,
        skill_name: str,
        phase: str,
        iteration: int,
        orchestrator_state: str,
    ) -> None:
        self._write({
            "record_type": "skill_start",
            "skill_name": skill_name,
            "phase": phase,
            "iteration": iteration,
            "orchestrator_state": orchestrator_state,
        })

    def record_skill_end(
        self,
        *,
        skill_name: str,
        phase: str,
        success: bool,
        duration_ms: float,
        changes_made: list[str],
        errors: list[str],
        warnings: list[str],
        requires_human_review: bool = False,
        escalation_reason: Optional[str] = None,
    ) -> None:
        self._write({
            "record_type": "skill_end",
            "skill_name": skill_name,
            "phase": phase,
            "success": success,
            "duration_ms": duration_ms,
            "changes_made": changes_made,
            "errors": errors,
            "warnings": warnings,
            "requires_human_review": requires_human_review,
            "escalation_reason": escalation_reason,
        })

    def record_tool_use(
        self,
        *,
        skill_name: str,
        tool_name: str,           # e.g. "cfn_lint", "checkov", "yaml_safe_load"
        input_summary: str,
        output_summary: str,
        success: bool,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        self._write({
            "record_type": "tool_use",
            "skill_name": skill_name,
            "tool_name": tool_name,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "success": success,
            "duration_ms": duration_ms,
            "error": error,
        })

    def record_error(
        self,
        *,
        skill_name: str,
        error_type: str,
        message: str,
        traceback: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> None:
        self._write({
            "record_type": "error",
            "skill_name": skill_name,
            "error_type": error_type,
            "message": message,
            "traceback": traceback,
            "context": context or {},
        })

    def record_validation_result(
        self,
        *,
        validator_name: str,
        status: str,
        findings: list[dict],
        duration_ms: float,
        raw_output: str = "",
        tool_version: str = "",
    ) -> None:
        self._write({
            "record_type": "validation_result",
            "validator_name": validator_name,
            "status": status,
            "findings": findings,
            "findings_count": len(findings),
            "duration_ms": duration_ms,
            "raw_output": raw_output[:4000],  # cap raw output
            "tool_version": tool_version,
        })

    def record_remediation(
        self,
        *,
        round_num: int,
        skill_name: str,
        findings_count: int,
        findings_addressed: list[str],
        fixes_applied: list[str],
        llm_used: bool,
        success: bool,
        escalated: bool = False,
        escalation_reason: Optional[str] = None,
    ) -> None:
        self._write({
            "record_type": "remediation",
            "round_num": round_num,
            "skill_name": skill_name,
            "findings_count": findings_count,
            "findings_addressed": findings_addressed,
            "fixes_applied": fixes_applied,
            "llm_used": llm_used,
            "success": success,
            "escalated": escalated,
            "escalation_reason": escalation_reason,
        })

    def record_god_snapshot(
        self,
        *,
        label: str,
        snapshot: dict,
    ) -> None:
        self._write({
            "record_type": "god_snapshot",
            "label": label,
            "snapshot": snapshot,
        })

    def record_template_snapshot(
        self,
        *,
        label: str,
        template_body: str,
        template_version: int,
        checksum: str,
    ) -> None:
        self._write({
            "record_type": "template_snapshot",
            "label": label,
            "template_body": template_body,
            "template_version": template_version,
            "checksum": checksum,
        })

    def record_run_end(
        self,
        *,
        success: bool,
        final_state: str,
        iterations: int,
        duration_ms: float,
        remediation_rounds: int,
        validation_summary: dict,
        findings_summary: dict,
        template_body: Optional[str],
        god_snapshot: Optional[dict],
    ) -> None:
        self._write({
            "record_type": "run_end",
            "success": success,
            "final_state": final_state,
            "iterations": iterations,
            "duration_ms": duration_ms,
            "remediation_rounds": remediation_rounds,
            "validation_summary": validation_summary,
            "findings_summary": findings_summary,
            "template_body": template_body,
            "god_snapshot": god_snapshot,
        })


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_collector(prompt: str, data_dir: str = "data") -> TelemetryCollector:
    """Create a TelemetryCollector for a new pipeline run."""
    rid = _run_id(prompt)
    return TelemetryCollector(run_id=rid, data_dir=data_dir)
