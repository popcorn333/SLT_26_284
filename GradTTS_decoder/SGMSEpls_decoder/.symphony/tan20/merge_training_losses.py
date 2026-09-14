#!/usr/bin/env python3
"""Atomically append validated TAN-20 losses to the combined loss ledger."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
STATE = ROOT / ".symphony/tan20"
COMBINED = ROOT / "training_loss.log"
SCOPED = STATE / "training_loss_tan20.log"
FIELDS = [
    "experiment", "configuration", "epoch", "duration_loss", "prior_loss",
    "diffusion_loss", "loss", "run_kind", "run_job_id", "source",
]
EXPECTED = {
    "levy": {"model.masking.alpha:1.8"},
    "fmDT": {
        "model.masking.a:0.2", "model.masking.a:0.4",
        "model.masking.a:0.6", "model.masking.a:0.8", "model.masking.a:1",
    },
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        if reader.fieldnames is None:
            raise RuntimeError(f"missing header: {path}")
        return rows


def validate(rows: list[dict[str, str]], label: str) -> None:
    errors: list[str] = []
    epochs: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for number, row in enumerate(rows, 2):
        try:
            epoch = int(row["epoch"])
            values = [
                float(row[field])
                for field in ("duration_loss", "prior_loss", "diffusion_loss", "loss")
            ]
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}:row={number}:parse")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{label}:row={number}:nonfinite")
        elif not math.isclose(values[3], sum(values[:3]), rel_tol=1e-9, abs_tol=1e-9):
            errors.append(f"{label}:row={number}:sum")
        if not row.get("run_kind") or not row.get("run_job_id") or not row.get("source"):
            errors.append(f"{label}:row={number}:provenance")
        key = (
            row.get("experiment", ""), row.get("configuration", ""),
            row.get("run_job_id", ""), row.get("run_kind", ""),
        )
        counts[key] += 1
        epochs[key].add(epoch)
    expected_epochs = set(range(1, 600))
    for key, count in counts.items():
        if count != 599 or epochs[key] != expected_epochs:
            errors.append(
                f"{label}:{'|'.join(key)}:count={count}:"
                f"missing={sorted(expected_epochs - epochs[key])[:10]}:"
                f"extra={sorted(epochs[key] - expected_epochs)[:10]}"
            )
    if errors:
        raise RuntimeError("invalid training-loss coverage: " + "; ".join(errors[:20]))


def main() -> None:
    baseline = [
        row for row in read_tsv(COMBINED)
        if row.get("run_kind") != "tan20"
    ]
    fresh = read_tsv(SCOPED)
    for row in fresh:
        row["run_kind"] = "tan20"
    validate(baseline, "baseline")
    validate(fresh, "tan20")

    manifest = read_tsv(STATE / "run_jobs.tsv")
    expected_groups = {
        (row["folder"], row["configuration"], row["job_id"], "tan20")
        for row in manifest
    }
    configured = {
        folder: {row["configuration"] for row in manifest if row["folder"] == folder}
        for folder in EXPECTED
    }
    if configured != EXPECTED or len(manifest) != 6:
        raise RuntimeError(f"TAN-20 run manifest coverage mismatch: {configured}")
    fresh_groups = Counter(
        (row["experiment"], row["configuration"], row["run_job_id"], row["run_kind"])
        for row in fresh
    )
    if set(fresh_groups) != expected_groups or set(fresh_groups.values()) != {599}:
        raise RuntimeError(f"TAN-20 loss groups do not match the run manifest: {fresh_groups}")

    combined = baseline + fresh
    keys = {
        (
            row["run_kind"], row["run_job_id"], row["experiment"],
            row["configuration"], row["epoch"],
        )
        for row in combined
    }
    if len(keys) != len(combined):
        raise RuntimeError("combined training-loss ledger has duplicate provenance keys")
    combined.sort(
        key=lambda row: (
            row["experiment"], row["configuration"], row["run_kind"],
            row["run_job_id"], int(row["epoch"]),
        )
    )
    output_tmp = COMBINED.with_name(COMBINED.name + ".tmp")
    with output_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(combined)
    coverage = {
        "status": "complete",
        "records": len(combined),
        "baseline_records": len(baseline),
        "tan20_records": len(fresh),
        "baseline_groups": len({
            (row["experiment"], row["configuration"], row["run_job_id"], row["run_kind"])
            for row in baseline
        }),
        "tan20_groups": len(fresh_groups),
        "tan20_run_jobs": sorted({row["job_id"] for row in manifest}),
        "tan20_configurations": sorted("|".join(group) for group in fresh_groups),
    }
    coverage_path = STATE / "training_loss_combined_coverage.json"
    coverage_tmp = coverage_path.with_name(coverage_path.name + ".tmp")
    coverage_tmp.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    output_tmp.replace(COMBINED)
    coverage_tmp.replace(coverage_path)
    print(
        f"validated_combined_training_losses={len(combined)} "
        f"tan20_records={len(fresh)} tan20_groups={len(fresh_groups)}"
    )


if __name__ == "__main__":
    main()
