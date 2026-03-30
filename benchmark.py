"""
IaC Benchmark Runner
====================

Executes the INFRA-SKILL orchestrator for every scenario in the
``iac_basic.csv`` benchmark and writes structured results to a JSONL file.

Usage
-----
    # Run all scenarios
    python benchmark.py

    # Run a single scenario by row number
    python benchmark.py --row 5

    # Resume an interrupted run (skips already-completed rows)
    python benchmark.py --resume

    # Override output directory
    python benchmark.py --output-dir results/run-01

    # Specify a different CSV path
    python benchmark.py --csv path/to/iac_basic.csv

Environment variables
---------------------
    OPENROUTER_API_KEY   – required by OpenRouterClient
    BENCHMARK_MODEL      – OpenRouter model string (optional, has a default)
    BENCHMARK_WORKERS    – parallel workers (default 1, sequential)

Output layout
-------------
    {output_dir}/
        results.jsonl          – one JSON object per scenario
        templates/             – generated YAML files (row_<N>.yaml)
        benchmark_summary.json – aggregate statistics written at the end
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from logger import setup_logging
from orchestrator import Orchestrator, OrchestratorConfig, create_default_skills
from llm_client import OpenRouterClient

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CSV = Path(__file__).parent / "Data" / "iac_basic.csv"
DEFAULT_OUTPUT_DIR = Path("benchmark_results") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
DEFAULT_MODEL = os.environ.get("BENCHMARK_MODEL", "arcee-ai/trinity-large-preview:free")
DEFAULT_WORKERS = int(os.environ.get("BENCHMARK_WORKERS", "1"))

# ---------------------------------------------------------------------------
# Per-scenario result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    row_number: int
    ground_truth_path: str
    prompt: str

    # Orchestrator output
    success: bool = False
    final_state: str = ""
    iterations: int = 0
    duration_ms: float = 0.0
    remediation_rounds: int = 0

    # Validation breakdown
    validation_summary: dict = field(default_factory=dict)
    findings_summary: dict = field(default_factory=dict)
    blocking_findings: list = field(default_factory=list)

    # Generated artefact
    template_path: Optional[str] = None   # relative path inside output_dir
    template_checksum: Optional[str] = None

    # Error capture (if orchestrator itself raised)
    error: Optional[str] = None

    # Timestamp
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """
    Iterates over every row in the benchmark CSV and executes the
    INFRA-SKILL orchestrator for each scenario's prompt.
    """

    def __init__(
        self,
        csv_path: Path = DEFAULT_CSV,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        model: str = DEFAULT_MODEL,
        workers: int = DEFAULT_WORKERS,
        resume: bool = False,
        orchestrator_config_overrides: Optional[dict] = None,
    ):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.model = model
        self.workers = workers
        self.resume = resume
        self.config_overrides = orchestrator_config_overrides or {}

        self._logger = logging.getLogger("benchmark")
        self._results_path = self.output_dir / "results.jsonl"
        self._templates_dir = self.output_dir / "templates"
        self._summary_path = self.output_dir / "benchmark_summary.json"

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _prepare_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._templates_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info(f"Output directory: {self.output_dir}")

    def _load_scenarios(self) -> list[dict]:
        """Read all rows from the benchmark CSV."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Benchmark CSV not found: {self.csv_path}")

        scenarios = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenarios.append({
                    "row_number": int(row["row_number"]),
                    "ground_truth_path": row["ground_truth_path"].strip(),
                    "prompt": row["prompt"].strip(),
                })
        self._logger.info(f"Loaded {len(scenarios)} scenarios from {self.csv_path}")
        return scenarios

    def _already_completed_rows(self) -> set[int]:
        """Return set of row numbers present in an existing results.jsonl."""
        completed: set[int] = set()
        if self.resume and self._results_path.exists():
            with open(self._results_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            completed.add(obj["row_number"])
                        except (json.JSONDecodeError, KeyError):
                            pass
            self._logger.info(f"Resuming – {len(completed)} rows already done")
        return completed

    def _build_orchestrator(self) -> Orchestrator:
        """
        Build a fully-wired Orchestrator.

        FIX: Previously this method returned an OrchestratorConfig and called
             create_orchestrator(config) — which passes the config object as the
             positional `llm_client` argument, storing the entire OrchestratorConfig
             as llm_client.  Every skill then called config.complete(...) and crashed
             with: 'OrchestratorConfig' object has no attribute 'complete'.

        The correct pattern is to build OrchestratorConfig with llm_client set to a
        real OpenRouterClient, then construct Orchestrator(config) directly.
        """
        llm = OpenRouterClient(model=self.model)

        base_kwargs = dict(
            max_remediation_rounds=3,
            max_total_iterations=30,
            verbose_logging=False,
            log_god_snapshots=False,
            enable_checkpoints=False,
            llm_client=llm,            # ← OpenRouterClient, not OrchestratorConfig
        )
        base_kwargs.update(self.config_overrides)

        config = OrchestratorConfig(**base_kwargs)
        return Orchestrator(config).with_skills(create_default_skills())

    # ------------------------------------------------------------------
    # Single scenario execution
    # ------------------------------------------------------------------

    def _run_scenario(self, scenario: dict) -> ScenarioResult:
        """Execute one benchmark scenario and return a ScenarioResult."""
        row = scenario["row_number"]
        prompt = scenario["prompt"]
        ground_truth_path = scenario["ground_truth_path"]

        result = ScenarioResult(
            row_number=row,
            ground_truth_path=ground_truth_path,
            prompt=prompt,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        self._logger.info(f"[Row {row:>4}] Starting – {prompt[:80]}...")

        try:
            orchestrator = self._build_orchestrator()
            orch_result = orchestrator.run(prompt)

            result.success = orch_result["success"]
            result.final_state = orch_result["state"]
            result.iterations = orch_result["iterations"]
            result.duration_ms = orch_result["duration_ms"]
            result.remediation_rounds = orch_result["remediation_rounds"]
            result.validation_summary = orch_result.get("validation_summary", {})
            result.findings_summary = orch_result.get("findings_summary", {})
            result.blocking_findings = orch_result.get("blocking_findings", [])
            result.template_checksum = orch_result.get("template_checksum")

            # Persist generated template
            if orch_result["success"] and orch_result.get("template"):
                template_file = self._templates_dir / f"row_{row:04d}.yaml"
                template_file.write_text(orch_result["template"], encoding="utf-8")
                result.template_path = str(template_file.relative_to(self.output_dir))

        except Exception as exc:
            result.success = False
            result.final_state = "ERROR"
            result.error = str(exc)
            self._logger.error(f"[Row {row:>4}] Unhandled exception: {exc}", exc_info=True)

        result.finished_at = datetime.now(timezone.utc).isoformat()
        status_icon = "✓" if result.success else "✗"
        self._logger.info(
            f"[Row {row:>4}] {status_icon} state={result.final_state} "
            f"iter={result.iterations} dur={result.duration_ms:.0f}ms"
        )
        return result

    # ------------------------------------------------------------------
    # Results I/O
    # ------------------------------------------------------------------

    def _append_result(self, result: ScenarioResult):
        """Append a single result as a JSON line (thread-safe write)."""
        with open(self._results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def _write_summary(self, results: list[ScenarioResult]):
        """Write aggregate statistics to benchmark_summary.json."""
        total = len(results)
        succeeded = sum(1 for r in results if r.success)
        failed = total - succeeded

        final_states: dict[str, int] = {}
        for r in results:
            final_states[r.final_state] = final_states.get(r.final_state, 0) + 1

        total_duration_ms = sum(r.duration_ms for r in results)
        avg_duration_ms = total_duration_ms / total if total else 0
        avg_iterations = sum(r.iterations for r in results) / total if total else 0

        # Validation pass-rates per validator
        validator_pass: dict[str, int] = {}
        validator_total: dict[str, int] = {}
        for r in results:
            for validator, status in r.validation_summary.items():
                validator_total[validator] = validator_total.get(validator, 0) + 1
                if status == "PASS":
                    validator_pass[validator] = validator_pass.get(validator, 0) + 1

        validator_pass_rates = {
            v: round(validator_pass.get(v, 0) / validator_total[v], 4)
            for v in validator_total
        }

        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "csv_path": str(self.csv_path),
            "model": self.model,
            "total_scenarios": total,
            "succeeded": succeeded,
            "failed": failed,
            "success_rate": round(succeeded / total, 4) if total else 0,
            "final_states_breakdown": final_states,
            "avg_duration_ms": round(avg_duration_ms, 1),
            "avg_iterations": round(avg_iterations, 2),
            "total_duration_ms": round(total_duration_ms, 1),
            "validator_pass_rates": validator_pass_rates,
        }

        self._summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._logger.info(f"Summary written to {self._summary_path}")
        return summary

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, target_row: Optional[int] = None) -> list[ScenarioResult]:
        """
        Run the full benchmark (or a single row if ``target_row`` is given).

        Returns the list of ScenarioResult objects.
        """
        self._prepare_output_dir()
        scenarios = self._load_scenarios()

        # Filter to a single row if requested
        if target_row is not None:
            scenarios = [s for s in scenarios if s["row_number"] == target_row]
            if not scenarios:
                raise ValueError(f"Row {target_row} not found in {self.csv_path}")

        # Skip already-done rows when resuming
        completed_rows = self._already_completed_rows()
        pending = [s for s in scenarios if s["row_number"] not in completed_rows]

        self._logger.info(
            f"Running {len(pending)} scenario(s) "
            f"({'sequential' if self.workers == 1 else f'{self.workers} workers'})"
        )

        all_results: list[ScenarioResult] = []
        start_wall = time.monotonic()

        if self.workers == 1:
            # Sequential – simpler, easier to debug
            for scenario in pending:
                result = self._run_scenario(scenario)
                self._append_result(result)
                all_results.append(result)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {
                    pool.submit(self._run_scenario, s): s for s in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    self._append_result(result)
                    all_results.append(result)

        elapsed = time.monotonic() - start_wall

        # Write summary
        if all_results:
            summary = self._write_summary(all_results)
            self._logger.info(
                f"\n{'=' * 60}\n"
                f"Benchmark complete in {elapsed:.1f}s\n"
                f"  Total:     {summary['total_scenarios']}\n"
                f"  Succeeded: {summary['succeeded']} "
                f"({summary['success_rate'] * 100:.1f}%)\n"
                f"  Failed:    {summary['failed']}\n"
                f"  Avg dur:   {summary['avg_duration_ms']:.0f}ms\n"
                f"{'=' * 60}"
            )

        return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the INFRA-SKILL orchestrator over the iac_basic.csv benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to the benchmark CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write results and generated templates.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenRouter model string (overrides BENCHMARK_MODEL env var).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel workers (1 = sequential; >1 requires thread-safe LLM client).",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        metavar="N",
        help="Run only row N (0-indexed, matches row_number column).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in an existing results.jsonl.",
    )
    parser.add_argument(
        "--max-remediation-rounds",
        type=int,
        default=30,
        dest="max_remediation_rounds",
        help="Max remediation rounds per scenario.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        dest="max_total_iterations",
        help="Max total iterations per scenario.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose orchestrator logging.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    config_overrides = {
        "max_remediation_rounds": args.max_remediation_rounds,
        "max_total_iterations": args.max_total_iterations,
        "verbose_logging": args.verbose,
    }

    runner = BenchmarkRunner(
        csv_path=args.csv,
        output_dir=args.output_dir,
        model=args.model,
        workers=args.workers,
        resume=args.resume,
        orchestrator_config_overrides=config_overrides,
    )

    results = runner.run(target_row=args.row)
    sys.exit(0 if all(r.success for r in results) else 1)


if __name__ == "__main__":
    main()
