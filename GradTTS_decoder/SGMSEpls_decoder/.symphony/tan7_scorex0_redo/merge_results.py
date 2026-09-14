#!/usr/bin/env python3
"""Publish the validated scorex0 redo into the final 29-path TAN-7 report."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
STATE = ROOT / ".symphony/tan7_scorex0_redo"
RESULTS = ROOT / ".symphony/results"
MAIN_CSV = RESULTS / "tan7_summary.csv"
MAIN_MD = RESULTS / "tan7_summary.md"
MAIN_COVERAGE = RESULTS / "tan7_coverage.json"
REDO_CSV = RESULTS / "tan7_scorex0_redo_summary.csv"
REDO_COVERAGE = RESULTS / "tan7_scorex0_redo_coverage.json"
MAIN_STATUS = ROOT / ".symphony/tan7/final_status.json"
REDO_STATUS = STATE / "final_status.json"
COMBINED_LOSS = ROOT / "training_loss.log"
COMBINED_LOSS_COVERAGE = STATE / "training_loss_combined_coverage.json"
FINAL_AUDIT = RESULTS / "tan7_final_audit.json"
CONFIGS = {
    "model.masking.a:0.2",
    "model.masking.a:0.4",
    "model.masking.a:0.6",
    "model.masking.a:0.8",
    "model.masking.a:1",
}
TASK_FIELDS = ["wer_task_index", "utmosv2_task_index"]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    fields, main_rows = read_csv(MAIN_CSV)
    redo_fields, redo_rows = read_csv(REDO_CSV)
    base_fields = [field for field in fields if field not in TASK_FIELDS]
    if base_fields != redo_fields:
        raise RuntimeError("main and redo CSV schemas differ")
    final_fields = base_fields + TASK_FIELDS
    if len(main_rows) != 29 or len(redo_rows) != 5:
        raise RuntimeError(
            f"summary row counts are not 29/5: "
            f"main={len(main_rows)} redo={len(redo_rows)}"
        )
    main_scorex0 = {
        row["configuration"]: row
        for row in main_rows
        if row["folder"] == "scorex0"
    }
    redo_by_config = {
        row["configuration"]: row
        for row in redo_rows
        if row["folder"] == "scorex0"
    }
    if set(main_scorex0) != CONFIGS or set(redo_by_config) != CONFIGS:
        raise RuntimeError(
            "main or redo scorex0 configuration coverage is incomplete"
        )
    merged = []
    for row in main_rows:
        if row["folder"] != "scorex0":
            retained = dict(row)
            retained["wer_task_index"] = (
                row.get("wer_task_index") or row["path_index"]
            )
            retained["utmosv2_task_index"] = (
                row.get("utmosv2_task_index") or row["path_index"]
            )
            merged.append(retained)
            continue
        replacement = dict(redo_by_config[row["configuration"]])
        redo_task_index = replacement["path_index"]
        replacement["path_index"] = row["path_index"]
        replacement["wer_task_index"] = redo_task_index
        replacement["utmosv2_task_index"] = redo_task_index
        merged.append(replacement)
    merged.sort(key=lambda row: int(row["path_index"]))
    for index, row in enumerate(merged):
        row["path_index"] = str(index)
    if [int(row["path_index"]) for row in merged] != list(range(29)):
        raise RuntimeError("merged result path indices are not exactly 0..28")
    markdown = [
        "# TAN-7 validated speech-metric summary",
        "",
        "Standard deviations are population standard deviations (ddof=0). "
        "The five scorex0 rows are from the independently requested fresh redo.",
        "",
        "| Folder | Path index | Configuration | Generated path | Metrics output path | WAV | MCD n / mean / std | logF0 n / mean / std | WER n / mean / std | UTMOSv2 n / mean / std | Jobs (training / effective run / metrics / WER[index] / UTMOSv2[index]) |",
        "|---|---:|---|---|---|---:|---|---|---|---|---|",
    ]
    for row in merged:
        def cell(metric: str) -> str:
            return (
                f"{row[f'{metric}_count']} / {row[f'{metric}_mean']} / "
                f"{row[f'{metric}_std']}"
            )

        jobs = (
            f"{row['training_job_id']} / {row['run_job_id']} / "
            f"{row['metrics_job_id']} / "
            f"{row['wer_job_id']}[{row['wer_task_index']}] / "
            f"{row['utmosv2_job_id']}[{row['utmosv2_task_index']}]"
        )
        markdown.append(
            f"| {row['folder']} | {row['path_index']} | "
            f"`{row['configuration']}` | `{row['gen_path']}` | "
            f"`{row['out_path']}` | {row['wav_count']} | "
            f"{cell('mcd')} | {cell('logf0')} | {cell('wer')} | "
            f"{cell('utmosv2')} | {jobs} |"
        )
    missing = []
    for row in merged:
        details = []
        for metric in ("mcd", "logf0", "wer", "utmosv2"):
            values = json.loads(row[f"{metric}_missing"])
            if values:
                reasons = Counter(
                    value.rsplit(":", 1)[-1] for value in values
                )
                details.append(
                    f"{metric}: {len(values)} ("
                    + ", ".join(
                        f"{key}={value}"
                        for key, value in sorted(reasons.items())
                    )
                    + ")"
                )
        if details:
            missing.append(
                f"- `{row['folder']}/{row['configuration']}`: "
                + "; ".join(details)
            )
    markdown.extend(["", "## Genuine missing values", ""])
    markdown.extend(missing or ["None."])
    markdown.extend(
        [
            "",
            "Training loss records include 17,371 original-run rows plus "
            "2,995 scorex0-redo rows in `training_loss.log`.",
        ]
    )
    main_coverage = json.loads(MAIN_COVERAGE.read_text(encoding="utf-8"))
    redo_coverage = json.loads(REDO_COVERAGE.read_text(encoding="utf-8"))
    if (
        main_coverage.get("fatal_coverage")
        or redo_coverage.get("fatal_coverage")
    ):
        raise RuntimeError("cannot merge reports with fatal coverage")
    main_paths = [
        item
        for item in main_coverage.get("paths", [])
        if item.get("folder") != "scorex0"
    ]
    redo_paths = redo_coverage.get("paths", [])
    if len(main_paths) != 24 or len(redo_paths) != 5:
        raise RuntimeError("coverage row counts are not 24 + 5")
    global_scorex0_indices = {
        row["configuration"]: row["path_index"]
        for row in main_rows
        if row["folder"] == "scorex0"
    }
    merged_redo_paths = []
    for item in redo_paths:
        configuration = item.get("configuration")
        if configuration not in global_scorex0_indices:
            raise RuntimeError(
                f"unexpected redo coverage configuration: {configuration}"
            )
        local_task_index = item.get("path_index")
        replacement = {
            **item,
            "path_index": global_scorex0_indices[configuration],
            "wer_task_index": local_task_index,
            "utmosv2_task_index": local_task_index,
        }
        merged_redo_paths.append(replacement)
    merged_paths = sorted(
        main_paths + merged_redo_paths,
        key=lambda item: int(item["path_index"]),
    )
    for index, item in enumerate(merged_paths):
        item["path_index"] = str(index)
    if [int(item["path_index"]) for item in merged_paths] != list(range(29)):
        raise RuntimeError(
            "merged coverage path indices are not exactly 0..28"
        )
    coverage_by_key = {
        (item.get("folder"), item.get("configuration")): item
        for item in merged_paths
    }
    if len(coverage_by_key) != 29:
        raise RuntimeError("merged coverage paths are not 29 unique configurations")
    coverage_fields = (
        "path_index", "gen_path", "out_path", "wav_count",
        "training_job_id", "run_job_id", "metrics_job_id", "wer_job_id",
        "wer_task_index", "utmosv2_job_id", "utmosv2_task_index",
        "finalizer_job_id",
    )
    coverage_mismatches = []
    for row in merged:
        key = (row["folder"], row["configuration"])
        item = coverage_by_key.get(key)
        if item is None:
            coverage_mismatches.append(f"{key}:missing")
            continue
        for field in coverage_fields:
            if str(item.get(field)) != str(row.get(field)):
                coverage_mismatches.append(
                    f"{key}:{field}={item.get(field)}!={row.get(field)}"
                )
        expected_missing = {
            metric: json.loads(row[f"{metric}_missing"])
            for metric in ("mcd", "logf0", "wer", "utmosv2")
        }
        if item.get("missing") != expected_missing:
            coverage_mismatches.append(f"{key}:missing_values")
    if coverage_mismatches:
        raise RuntimeError(
            "merged coverage/CSV mismatch: "
            + ", ".join(coverage_mismatches[:20])
        )
    merged_coverage = {
        "status": "complete",
        "fatal_coverage": [],
        "scorex0_source": "fresh_redo",
        "paths": merged_paths,
    }
    combined_loss_rows = 0
    with COMBINED_LOSS.open(newline="", encoding="utf-8") as handle:
        combined_loss_rows = sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
    combined_loss_coverage = json.loads(
        COMBINED_LOSS_COVERAGE.read_text(encoding="utf-8")
    )
    if (
        combined_loss_rows != 34 * 599
        or combined_loss_coverage.get("status") != "complete"
        or combined_loss_coverage.get("records") != combined_loss_rows
        or combined_loss_coverage.get("original_records") != 29 * 599
        or combined_loss_coverage.get("scorex0_redo_records") != 5 * 599
    ):
        raise RuntimeError(
            "combined training-loss publication is not exactly "
            f"20,366 validated rows: log={combined_loss_rows} "
            f"coverage={combined_loss_coverage}"
        )
    main_status = json.loads(MAIN_STATUS.read_text(encoding="utf-8"))
    redo_status = json.loads(REDO_STATUS.read_text(encoding="utf-8"))
    if (
        main_status.get("status")
        not in {
            "scorex0_redo_loss_merged_pending_report",
            "scorex0_redo_merged_pending_final_audit",
        }
        or main_status.get("paths") != 29
        or redo_status.get("status") != "complete"
        or redo_status.get("paths") != 5
    ):
        raise RuntimeError(
            f"main or redo final status is incomplete: "
            f"main={main_status} redo={redo_status}"
        )
    finalizer_job_id = os.environ.get("SLURM_JOB_ID")
    if not finalizer_job_id:
        raise RuntimeError("result merge must run inside the redo finalizer allocation")
    final_status = {
        **main_status,
        "status": "scorex0_redo_merged_pending_final_audit",
        "paths": 29,
        "training_loss_records": combined_loss_rows,
        "scorex0_source": "fresh_redo",
        "scorex0_redo_paths": 5,
        "scorex0_redo_finalizer_job_id": finalizer_job_id,
        "scorex0_redo_csv": str(REDO_CSV),
        "scorex0_redo_coverage": str(REDO_COVERAGE),
    }

    # Prepare every final artifact before publishing any of them. Install the
    # pending-audit sentinel before replacing reports, so a crash at any point
    # cannot leave the earlier main-run status falsely claiming completion.
    # Only final_audit.py is allowed to restore the completed status.
    csv_tmp = MAIN_CSV.with_name(MAIN_CSV.name + ".tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(merged)
    md_tmp = MAIN_MD.with_name(MAIN_MD.name + ".tmp")
    md_tmp.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    coverage_tmp = MAIN_COVERAGE.with_name(MAIN_COVERAGE.name + ".tmp")
    coverage_tmp.write_text(
        json.dumps(merged_coverage, indent=2) + "\n", encoding="utf-8"
    )
    status_tmp = MAIN_STATUS.with_name(MAIN_STATUS.name + ".tmp")
    status_tmp.write_text(
        json.dumps(final_status, indent=2) + "\n", encoding="utf-8"
    )
    backup = STATE / "tan7_summary_before_scorex0_redo.csv"
    if not backup.exists():
        with backup.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(main_rows)
    FINAL_AUDIT.unlink(missing_ok=True)
    status_tmp.replace(MAIN_STATUS)
    csv_tmp.replace(MAIN_CSV)
    md_tmp.replace(MAIN_MD)
    coverage_tmp.replace(MAIN_COVERAGE)
    print("published_final_tan7_paths=29 scorex0_redo_paths=5")


if __name__ == "__main__":
    main()
