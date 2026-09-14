#!/usr/bin/env python3
"""Atomically combine validated original and scorex0-redo epoch losses."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
MAIN_STATE = ROOT / ".symphony/tan7"
STATE = ROOT / ".symphony/tan7_scorex0_redo"
COMBINED = ROOT / "training_loss.log"
REDO = STATE / "training_loss_redo.log"
MAIN_STATUS = MAIN_STATE / "final_status.json"
EXPECTED = {
    "add": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6"},
    "blur": {"model.masking.b:10", "model.masking.b:20", "model.masking.b:40"},
    "fm": {"model.masking.a:1"},
    "fmDT": {"model.masking.a:1"},
    "levy": {"model.masking.alpha:1.9"},
    "mask": {"model.masking.a:1"},
    "one-shot": {"model.masking.a:0.2"},
    "product": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6"},
    "score": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6", "model.masking.a:0.8", "model.masking.a:1"},
    "scoreDT": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6", "model.masking.a:0.8", "model.masking.a:1"},
    "scorex0": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6", "model.masking.a:0.8", "model.masking.a:1"},
}
FIELDS = [
    "experiment", "configuration", "epoch", "duration_loss", "prior_loss",
    "diffusion_loss", "loss", "run_kind", "run_job_id", "source",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_numeric(rows: list[dict[str, str]], label: str) -> None:
    errors = []
    for number, row in enumerate(rows, 2):
        try:
            epoch = int(row["epoch"])
            parts = [
                float(row[key])
                for key in ("duration_loss", "prior_loss", "diffusion_loss")
            ]
            combined = float(row["loss"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}:row={number}:parse")
            continue
        if (
            not 1 <= epoch <= 599
            or not all(math.isfinite(value) for value in [*parts, combined])
        ):
            errors.append(f"{label}:row={number}:range_or_nonfinite")
        elif not math.isclose(combined, sum(parts), rel_tol=1e-9, abs_tol=1e-9):
            errors.append(f"{label}:row={number}:sum")
        if not row.get("run_job_id") or not row.get("source"):
            errors.append(f"{label}:row={number}:provenance")
    if errors:
        raise RuntimeError(
            "invalid training-loss records: " + ", ".join(errors[:20])
        )


def main() -> None:
    if not COMBINED.is_file() or not REDO.is_file():
        raise RuntimeError("main or redo training-loss log is missing")
    redo_jobs = read_tsv(STATE / "run_jobs.tsv")
    if len(redo_jobs) != 1 or redo_jobs[0]["folder"] != "scorex0":
        raise RuntimeError("scorex0 redo run manifest is not exactly one row")
    # The effective run ID is the all-task inference success barrier, while
    # the epoch losses were emitted by the original training allocation.
    redo_job = redo_jobs[0].get("training_job_id") or redo_jobs[0]["job_id"]
    effective = {
        row["folder"]: row.get("training_job_id") or row["job_id"]
        for row in read_tsv(MAIN_STATE / "run_jobs_effective.tsv")
    }
    existing = read_tsv(COMBINED)
    original = [row for row in existing if row.get("run_job_id") != redo_job]
    redo = read_tsv(REDO)
    for row in original:
        row["run_kind"] = "original"
    for row in redo:
        row["run_kind"] = "scorex0_redo"
    validate_numeric(original, "original")
    validate_numeric(redo, "redo")
    if len(original) != 29 * 599:
        raise RuntimeError(
            f"original loss count is {len(original)}, expected {29 * 599}"
        )
    if len(redo) != 5 * 599:
        raise RuntimeError(f"redo loss count is {len(redo)}, expected {5 * 599}")
    original_counts = Counter(
        (row["experiment"], row["configuration"], row["run_job_id"])
        for row in original
    )
    expected_original = {
        (folder, configuration, effective[folder])
        for folder, configurations in EXPECTED.items()
        for configuration in configurations
    }
    if (
        set(original_counts) != expected_original
        or set(original_counts.values()) != {599}
    ):
        raise RuntimeError(
            "original loss configuration/run coverage is not exactly 29 x 599"
        )
    redo_counts = Counter(
        (row["experiment"], row["configuration"], row["run_job_id"])
        for row in redo
    )
    expected_redo = {
        ("scorex0", configuration, redo_job)
        for configuration in EXPECTED["scorex0"]
    }
    if set(redo_counts) != expected_redo or set(redo_counts.values()) != {599}:
        raise RuntimeError(
            "redo loss configuration/run coverage is not exactly 5 x 599"
        )
    combined = original + redo
    keys = {
        (
            row["run_job_id"],
            row["experiment"],
            row["configuration"],
            row["epoch"],
        )
        for row in combined
    }
    if len(keys) != len(combined):
        raise RuntimeError(
            "combined training losses contain duplicate run/configuration/epoch rows"
        )
    combined.sort(
        key=lambda row: (
            row["experiment"],
            row["configuration"],
            row["run_job_id"],
            int(row["epoch"]),
        )
    )
    temporary = COMBINED.with_name(COMBINED.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(combined)
    coverage = {
        "status": "complete",
        "records": len(combined),
        "original_records": len(original),
        "scorex0_redo_records": len(redo),
        "runs": sorted(
            {
                (
                    f"{row['experiment']}|{row['configuration']}|"
                    f"{row['run_job_id']}"
                )
                for row in combined
            }
        ),
    }
    coverage_path = STATE / "training_loss_combined_coverage.json"
    coverage_tmp = coverage_path.with_name(coverage_path.name + ".tmp")
    coverage_tmp.write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )
    main_status = json.loads(MAIN_STATUS.read_text(encoding="utf-8"))
    if (
        main_status.get("status")
        not in {
            "complete",
            "scorex0_redo_loss_merged_pending_report",
            "scorex0_redo_merged_pending_final_audit",
        }
        or main_status.get("paths") != 29
        or main_status.get("training_loss_records") not in {29 * 599, 34 * 599}
    ):
        raise RuntimeError(
            f"main publication status cannot enter redo loss merge: {main_status}"
        )
    pending_status = {
        **main_status,
        "status": "scorex0_redo_loss_merged_pending_report",
        "training_loss_records": len(combined),
        "scorex0_source": "fresh_redo_pending_report",
    }
    status_tmp = MAIN_STATUS.with_name(MAIN_STATUS.name + ".tmp")
    status_tmp.write_text(
        json.dumps(pending_status, indent=2) + "\n", encoding="utf-8"
    )
    # Install the non-complete sentinel before either combined artifact.  A
    # crash can therefore leave only a conservative pending state, never an
    # authoritative-looking prior status beside a newly merged loss file.
    status_tmp.replace(MAIN_STATUS)
    temporary.replace(COMBINED)
    coverage_tmp.replace(coverage_path)
    print(
        f"validated_combined_training_losses={len(combined)} "
        f"original={len(original)} redo={len(redo)}"
    )


if __name__ == "__main__":
    main()
