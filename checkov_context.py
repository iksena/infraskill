# infraskill/checkov_context.py
from __future__ import annotations
import csv
from functools import lru_cache
from pathlib import Path
from god import ValidationFinding

_CSV_PATH = Path(__file__).parent / "Data" / "checkov_cfn_policy_map.csv"

@lru_cache(maxsize=1)
def _load_policy_map() -> dict[str, dict]:
    """Load CSV once and return {check_id: {check_name, source_code}}."""
    if not _CSV_PATH.exists():
        return {}
    with _CSV_PATH.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["check_id"]: row for row in reader}

def get_checkov_policy_context(findings: list[ValidationFinding]) -> str:
    """
    Given a list of failed Checkov ValidationFindings, return a
    formatted string containing the policy source code for each,
    suitable for embedding directly into a remediation prompt.
    """
    policy_map = _load_policy_map()
    blocks: list[str] = []
    seen: set[str] = set()

    for finding in findings:
        check_id = finding.check_id or finding.rule_id
        if not check_id or check_id in seen:
            continue
        seen.add(check_id)
        row = policy_map.get(check_id)
        if row:
            blocks.append(
                f"### [{check_id}] {row['check_name']}\n"
                f"```python\n{row['source_code']}\n```"
            )

    return "\n\n".join(blocks)