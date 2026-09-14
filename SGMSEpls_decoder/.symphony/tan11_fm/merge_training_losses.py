#!/usr/bin/env python3
"""Append the validated TAN-11 FM sweep to the combined training-loss log."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
STATE = ROOT / ".symphony/tan11_fm"
COMBINED = ROOT / "training_loss.log"
FM_LOSS = STATE / "training_loss_fm.log"
PARENT = "11097462"
FIELDS = [
    "experiment", "configuration", "epoch", "duration_loss", "prior_loss",
    "diffusion_loss", "loss", "run_kind", "run_job_id", "source",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate(rows: list[dict[str, str]], label: str) -> None:
    errors = []
    for number, row in enumerate(rows, 2):
        try:
            epoch = int(row["epoch"])
            values = [float(row[key]) for key in (
                "duration_loss", "prior_loss", "diffusion_loss", "loss"
            )]
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}:row={number}:parse")
            continue
        if not 1 <= epoch <= 599 or not all(math.isfinite(value) for value in values):
            errors.append(f"{label}:row={number}:range_or_nonfinite")
        elif not math.isclose(values[3], sum(values[:3]), rel_tol=1e-9, abs_tol=1e-9):
            errors.append(f"{label}:row={number}:sum")
        if not row.get("run_job_id") or not row.get("source"):
            errors.append(f"{label}:row={number}:provenance")
    if errors:
        raise RuntimeError("invalid training losses: " + ", ".join(errors[:20]))


def main() -> None:
    baseline = [
        row for row in read_tsv(COMBINED)
        if row.get("run_job_id") != PARENT and row.get("run_kind") != "tan11_fm"
    ]
    fresh = read_tsv(FM_LOSS)
    if len(baseline) != 34 * 599:
        raise RuntimeError(f"baseline training-loss count is {len(baseline)}, not 20366")
    if len(fresh) != 5 * 599:
        raise RuntimeError(f"fresh FM training-loss count is {len(fresh)}, not 2995")
    for row in fresh:
        row["run_kind"] = "tan11_fm"
    validate(baseline, "baseline")
    validate(fresh, "tan11_fm")
    baseline_groups = Counter(
        (row["experiment"], row["configuration"], row["run_job_id"], row["run_kind"])
        for row in baseline
    )
    fresh_groups = Counter(
        (row["experiment"], row["configuration"], row["run_job_id"], row["run_kind"])
        for row in fresh
    )
    if len(baseline_groups) != 34 or set(baseline_groups.values()) != {599}:
        raise RuntimeError("baseline is not exactly 34 complete configuration-runs")
    expected_fm = {
        ("fm", f"model.masking.a:{value}", PARENT, "tan11_fm")
        for value in ("0.2", "0.4", "0.6", "0.8", "1")
    }
    if set(fresh_groups) != expected_fm or set(fresh_groups.values()) != {599}:
        raise RuntimeError(f"fresh FM loss groups are incomplete: {fresh_groups}")
    combined = baseline + fresh
    keys = {
        (row["run_job_id"], row["experiment"], row["configuration"], row["epoch"])
        for row in combined
    }
    if len(keys) != len(combined):
        raise RuntimeError("combined training losses contain duplicate provenance keys")
    combined.sort(key=lambda row: (
        row["experiment"], row["configuration"], row["run_job_id"], int(row["epoch"])
    ))
    temporary = COMBINED.with_name(COMBINED.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(combined)
    coverage = {
        "status": "complete",
        "records": len(combined),
        "baseline_records": len(baseline),
        "tan11_fm_records": len(fresh),
        "configuration_runs": len(baseline_groups) + len(fresh_groups),
        "groups": ["|".join(group) for group in sorted([*baseline_groups, *fresh_groups])],
    }
    coverage_path = STATE / "training_loss_combined_coverage.json"
    coverage_tmp = coverage_path.with_name(coverage_path.name + ".tmp")
    coverage_tmp.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    temporary.replace(COMBINED)
    coverage_tmp.replace(coverage_path)
    print(f"validated_combined_training_losses={len(combined)} groups=39")


if __name__ == "__main__":
    main()
