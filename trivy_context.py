# infraskill/trivy_context.py
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from god import ValidationFinding

_CSV_PATH = Path(__file__).parent / "Data" / "trivy_cfn_policy_map.csv"


@lru_cache(maxsize=1)
def _load_policy_map() -> dict[str, dict]:
    """Load CSV once and return {check_id: policy_row}."""
    if not _CSV_PATH.exists():
        return {}
    with _CSV_PATH.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["check_id"]: row for row in reader if row.get("check_id")}


def get_trivy_policy_context(findings: list[ValidationFinding]) -> str:
    """
    Given failed Trivy findings, return formatted source-code blocks for matched
    check IDs suitable for remediation prompts.
    """
    policy_map = _load_policy_map()
    blocks: list[str] = []
    seen: set[str] = set()

    for finding in findings:
        check_id = (finding.check_id or finding.rule_id or "").strip()
        if not check_id or check_id in seen:
            continue
        seen.add(check_id)

        row = policy_map.get(check_id)
        if row:
            blocks.append(
                f"### [{check_id}] {row.get('check_name', '')}\n"
                f"```rego\n{row.get('source_code', '')}\n```"
            )

    return "\n\n".join(blocks)
