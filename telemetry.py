"""
telemetry.py  —  InfraSkill run-level telemetry recorder
=========================================================

Writes five files per pipeline run into  ./telemetry/<run_id>/:

  llm_conversations.jsonl   — every LLM call (prompt, response, timing)
  god_changes.jsonl         — every mutation to the GOD (field, before, after)
  skill_executions.jsonl    — per-skill timing, inputs, outputs, errors
  orchestrator_events.jsonl — every orchestrator state-machine event
  run_summary.csv           — one row per run (append mode)

Usage (inside orchestrator.run):

    from telemetry import TelemetryRecorder
    tel = TelemetryRecorder(base_dir="telemetry")
    tel.start_run(run_id, prompt, config_dict)
    ...  # pass `tel` into SkillContext via context.config["telemetry"]
    tel.finish_run(final_state, iterations, duration_ms, validation_summary)
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def _safe_truncate(value: Any, max_chars: int = 8000) -> Any:
    """Truncate long strings so JSONL rows stay manageable."""
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"
    return value


# ---------------------------------------------------------------------------
# TelemetryRecorder
# ---------------------------------------------------------------------------

class TelemetryRecorder:
    """
    Central recorder.  One instance per pipeline run.

    All public methods are safe to call from multiple threads; each write
    is a single os.write() call on a line-buffered file, which is atomic
    on POSIX for lines < PIPE_BUF (~4 KB).  For longer payloads we acquire
    a simple file lock via a threading.Lock.
    """

    # CSV columns written to run_summary.csv
    _SUMMARY_COLS = [
        "run_id", "started_at", "finished_at", "prompt_snippet",
        "final_state", "iterations", "duration_ms", "remediation_rounds",
        "llm_calls", "total_prompt_tokens", "total_response_tokens",
        "yaml_syntax", "cfn_lint", "checkov",
    ]

    def __init__(self, base_dir: str = "telemetry"):
        self._base_dir = Path(base_dir)
        self._run_id: Optional[str] = None
        self._run_dir: Optional[Path] = None
        self._started_at: Optional[str] = None
        self._prompt: str = ""
        self._config: dict = {}
        # Counters accumulated during the run
        self._llm_call_count: int = 0
        self._total_prompt_tokens: int = 0
        self._total_response_tokens: int = 0
        # File handles (opened in start_run)
        self._fh: dict[str, Any] = {}
        import threading
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_id: str, prompt: str, config: dict) -> None:
        """Open output files and write the run header."""
        self._run_id = run_id
        self._started_at = _now_iso()
        self._prompt = prompt
        self._config = config
        self._llm_call_count = 0
        self._total_prompt_tokens = 0
        self._total_response_tokens = 0

        self._run_dir = self._base_dir / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        for name in ("llm_conversations", "god_changes", "skill_executions", "orchestrator_events"):
            path = self._run_dir / f"{name}.jsonl"
            self._fh[name] = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered

        # Write a run-start marker to each stream
        header = {"event": "run_start", "run_id": run_id, "timestamp": self._started_at,
                  "prompt_snippet": prompt[:200], "config": config}
        for name in self._fh:
            self._write_jsonl(name, header)

    def finish_run(
        self,
        final_state: str,
        iterations: int,
        duration_ms: float,
        remediation_rounds: int,
        validation_summary: dict,
    ) -> None:
        """Write run footer and append a row to run_summary.csv."""
        finished_at = _now_iso()
        footer = {
            "event": "run_end",
            "run_id": self._run_id,
            "timestamp": finished_at,
            "final_state": final_state,
            "iterations": iterations,
            "duration_ms": duration_ms,
            "remediation_rounds": remediation_rounds,
            "llm_calls": self._llm_call_count,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_response_tokens": self._total_response_tokens,
        }
        for name in self._fh:
            self._write_jsonl(name, footer)
            self._fh[name].close()
        self._fh.clear()

        # Append one row to the aggregate CSV (created if absent)
        self._append_summary_csv(
            finished_at=finished_at,
            final_state=final_state,
            iterations=iterations,
            duration_ms=duration_ms,
            remediation_rounds=remediation_rounds,
            validation_summary=validation_summary,
        )

    # ------------------------------------------------------------------
    # LLM conversation recorder
    # ------------------------------------------------------------------

    def record_llm_call(
        self,
        *,
        skill_name: str,
        iteration: int,
        call_purpose: str,
        system_prompt: str,
        user_message: str,
        raw_response: str,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """
        Record a full LLM conversation turn to llm_conversations.jsonl.

        Stores the complete system prompt, user message, and raw LLM
        completion (each truncated to 8 000 chars to keep rows manageable
        while still capturing the full context window shape).

        Fields:
          system_prompt      — the rendered system prompt sent to the LLM
          user_message       — the user-turn message (includes GOD snapshot
                               for engineer and remediation skills)
          raw_response       — the raw completion text returned by the LLM
          completion_ok      — True when the LLM returned a non-empty
                               completion without raising an exception
          prompt_tokens_est  — rough token count for prompt (len // 4)
          response_tokens_est— rough token count for response
        """
        prompt_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_message)
        response_tokens = _estimate_tokens(raw_response) if raw_response else 0
        self._llm_call_count += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_response_tokens += response_tokens

        record = {
            "event": "llm_call",
            "run_id": self._run_id,
            "timestamp": _now_iso(),
            "skill_name": skill_name,
            "iteration": iteration,
            "call_purpose": call_purpose,
            # completion result
            "completion_ok": success and bool(raw_response),
            "success": success,
            "duration_ms": round(duration_ms, 2),
            "prompt_tokens_est": prompt_tokens,
            "response_tokens_est": response_tokens,
            # full conversation — truncated to stay under JSONL row limits
            "system_prompt": _safe_truncate(system_prompt, 8000),
            "user_message": _safe_truncate(user_message, 8000),
            "raw_response": _safe_truncate(raw_response, 8000),
            "error": error,
        }
        if extra:
            record.update(extra)
        self._write_jsonl("llm_conversations", record)

    # ------------------------------------------------------------------
    # GOD change recorder
    # ------------------------------------------------------------------

    def record_god_change(
        self,
        *,
        field: str,
        before: Any,
        after: Any,
        changed_by: str,
        iteration: int,
        noop: bool = False,
    ) -> None:
        """
        Record a GOD field mutation to god_changes.jsonl.

        before_len / after_len are computed on the raw string representation
        BEFORE any truncation so they accurately reflect whether the field
        actually changed (identical lengths on identical content = no-op).

        The `noop` flag is set to True by RemediationSkill when the LLM
        returned an unchanged template, making no-op rounds immediately
        visible when grepping god_changes.jsonl.

        The `changed` boolean is True whenever before != after (by value),
        regardless of length equality.
        """
        before_str = str(before) if before is not None else ""
        after_str = str(after) if after is not None else ""

        def _repr(v: str) -> Any:
            return _safe_truncate(v, 2000)

        self._write_jsonl("god_changes", {
            "event": "god_change",
            "run_id": self._run_id,
            "timestamp": _now_iso(),
            "iteration": iteration,
            "changed_by": changed_by,
            "field": field,
            "before": _repr(before_str),
            "after": _repr(after_str),
            # lengths on the RAW strings (before truncation) for accurate diffing
            "before_len": len(before_str),
            "after_len": len(after_str),
            # convenience flags
            "changed": before_str != after_str,
            "noop": noop,
        })

    # ------------------------------------------------------------------
    # Skill execution recorder
    # ------------------------------------------------------------------

    def record_skill_execution(
        self,
        *,
        skill_name: str,
        phase: str,
        iteration: int,
        success: bool,
        duration_ms: float,
        changes_made: list[str],
        errors: list[str],
        warnings: list[str],
        inputs: Optional[dict] = None,
        outputs: Optional[dict] = None,
    ) -> None:
        self._write_jsonl("skill_executions", {
            "event": "skill_execution",
            "run_id": self._run_id,
            "timestamp": _now_iso(),
            "skill_name": skill_name,
            "phase": phase,
            "iteration": iteration,
            "success": success,
            "duration_ms": round(duration_ms, 2),
            "changes_made": changes_made,
            "errors": errors,
            "warnings": warnings,
            "inputs": {k: _safe_truncate(v) for k, v in (inputs or {}).items()},
            "outputs": {k: _safe_truncate(v) for k, v in (outputs or {}).items()},
        })

    # ------------------------------------------------------------------
    # Orchestrator event recorder
    # ------------------------------------------------------------------

    def record_orchestrator_event(
        self,
        *,
        event_type: str,
        iteration: int,
        data: dict,
    ) -> None:
        self._write_jsonl("orchestrator_events", {
            "event": "orchestrator_event",
            "run_id": self._run_id,
            "timestamp": _now_iso(),
            "iteration": iteration,
            "event_type": event_type,
            "data": data,
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_jsonl(self, stream: str, record: dict) -> None:
        fh = self._fh.get(stream)
        if fh is None:
            return
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            fh.write(line)

    def _append_summary_csv(self, *, finished_at, final_state, iterations,
                             duration_ms, remediation_rounds, validation_summary) -> None:
        csv_path = self._base_dir / "run_summary.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=self._SUMMARY_COLS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "run_id": self._run_id,
                "started_at": self._started_at,
                "finished_at": finished_at,
                "prompt_snippet": self._prompt[:120].replace("\n", " "),
                "final_state": final_state,
                "iterations": iterations,
                "duration_ms": round(duration_ms, 1),
                "remediation_rounds": remediation_rounds,
                "llm_calls": self._llm_call_count,
                "total_prompt_tokens": self._total_prompt_tokens,
                "total_response_tokens": self._total_response_tokens,
                "yaml_syntax": validation_summary.get("yaml_syntax", "N/A"),
                "cfn_lint": validation_summary.get("cfn_lint", "N/A"),
                "checkov": validation_summary.get("checkov", "N/A"),
            })

    # ------------------------------------------------------------------
    # Convenience: context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        # Close any open handles if finish_run was not called
        for fh in self._fh.values():
            try:
                fh.close()
            except Exception:
                pass
        self._fh.clear()
