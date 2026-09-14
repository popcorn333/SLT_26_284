#!/usr/bin/env python3
"""Merge the fresh five-path FM sweep into the validated 11-folder report."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
STATE = ROOT / ".symphony/tan11_fm"
RESULTS = ROOT / ".symphony/results"
BASE_CSV = RESULTS / "tan7_summary.csv"
BASE_COVERAGE = RESULTS / "tan7_coverage.json"
FM_CSV = RESULTS / "tan11_fm_summary.csv"
FM_COVERAGE = RESULTS / "tan11_fm_coverage.json"
FINAL_CSV = RESULTS / "tan11_summary.csv"
FINAL_MD = RESULTS / "tan11_summary.md"
FINAL_COVERAGE = RESULTS / "tan11_coverage.json"
TASK_FIELDS = ["wer_task_index", "utmosv2_task_index"]
FM_CONFIGS = {
    "model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6",
    "model.masking.a:0.8", "model.masking.a:1",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    finalizer = os.environ.get("SLURM_JOB_ID")
    if not finalizer:
        raise RuntimeError("TAN-11 merge must run inside the Slurm finalizer")
    base_fields, base_rows = read_csv(BASE_CSV)
    fm_fields, fm_rows = read_csv(FM_CSV)
    plain_fields = [field for field in base_fields if field not in TASK_FIELDS]
    if plain_fields != fm_fields:
        raise RuntimeError("baseline and fresh FM summary schemas differ")
    if len(base_rows) != 29 or len(fm_rows) != 5:
        raise RuntimeError(f"summary row counts are not 29/5: {len(base_rows)}/{len(fm_rows)}")
    if {row["configuration"] for row in fm_rows} != FM_CONFIGS:
        raise RuntimeError("fresh FM summary does not contain all five configurations")
    if sum(row["folder"] == "fm" for row in base_rows) != 1:
        raise RuntimeError("baseline does not contain exactly one FM row")

    fm_rows.sort(key=lambda row: float(row["configuration"].rsplit(":", 1)[1]))
    merged: list[dict[str, str]] = []
    for row in sorted(base_rows, key=lambda item: int(item["path_index"])):
        if row["folder"] != "fm":
            retained = dict(row)
            retained["wer_task_index"] = row.get("wer_task_index") or row["path_index"]
            retained["utmosv2_task_index"] = row.get("utmosv2_task_index") or row["path_index"]
            merged.append(retained)
            continue
        for fm_row in fm_rows:
            replacement = dict(fm_row)
            local_index = replacement["path_index"]
            replacement["wer_task_index"] = local_index
            replacement["utmosv2_task_index"] = local_index
            merged.append(replacement)
    for index, row in enumerate(merged):
        row["path_index"] = str(index)
    if len(merged) != 33 or [int(row["path_index"]) for row in merged] != list(range(33)):
        raise RuntimeError("merged result indices are not exactly 0..32")

    fields = plain_fields + TASK_FIELDS
    csv_tmp = FINAL_CSV.with_name(FINAL_CSV.name + ".tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)

    markdown = [
        "# TAN-11 validated 11-folder speech-metric summary",
        "",
        "Standard deviations are population standard deviations (ddof=0). The five FM rows are from Slurm array `11097462`; all other rows come from the independently validated 11-folder baseline, including the fresh scorex0 redo.",
        "",
        "| Folder | Path index | Configuration | Generated path | Metrics output path | WAV | MCD n / mean / std | logF0 n / mean / std | WER n / mean / std | UTMOSv2 n / mean / std | Jobs (training / effective run / metrics / WER[index] / UTMOSv2[index]) |",
        "|---|---:|---|---|---|---:|---|---|---|---|---|",
    ]
    for row in merged:
        def cell(metric: str) -> str:
            return f"{row[f'{metric}_count']} / {row[f'{metric}_mean']} / {row[f'{metric}_std']}"

        jobs = (
            f"{row['training_job_id']} / {row['run_job_id']} / {row['metrics_job_id']} / "
            f"{row['wer_job_id']}[{row['wer_task_index']}] / "
            f"{row['utmosv2_job_id']}[{row['utmosv2_task_index']}]"
        )
        markdown.append(
            f"| {row['folder']} | {row['path_index']} | `{row['configuration']}` | "
            f"`{row['gen_path']}` | `{row['out_path']}` | {row['wav_count']} | "
            f"{cell('mcd')} | {cell('logf0')} | {cell('wer')} | {cell('utmosv2')} | {jobs} |"
        )
    missing_lines = []
    for row in merged:
        details = []
        for metric in ("mcd", "logf0", "wer", "utmosv2"):
            missing = json.loads(row[f"{metric}_missing"])
            if missing:
                reasons = Counter(item.rsplit(":", 1)[-1] for item in missing)
                details.append(
                    f"{metric}: {len(missing)} ("
                    + ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))
                    + ")"
                )
        if details:
            missing_lines.append(f"- `{row['folder']}/{row['configuration']}`: " + "; ".join(details))
    markdown.extend(["", "## Genuine missing values", ""])
    markdown.extend(missing_lines or ["None."])
    markdown.extend([
        "",
        f"Final TAN-11 finalizer Slurm job: `{finalizer}`.",
        "Training-loss coverage: 23,361 epoch records across 39 submitted configuration-runs in `training_loss.log`.",
    ])
    md_tmp = FINAL_MD.with_name(FINAL_MD.name + ".tmp")
    md_tmp.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    base_coverage = json.loads(BASE_COVERAGE.read_text(encoding="utf-8"))
    fm_coverage = json.loads(FM_COVERAGE.read_text(encoding="utf-8"))
    if base_coverage.get("fatal_coverage") or fm_coverage.get("fatal_coverage"):
        raise RuntimeError("cannot merge reports with fatal coverage")
    base_by_key = {
        (item["folder"], item["configuration"]): item
        for item in base_coverage.get("paths", []) if item["folder"] != "fm"
    }
    fm_by_key = {
        (item["folder"], item["configuration"]): item
        for item in fm_coverage.get("paths", [])
    }
    if len(base_by_key) != 28 or len(fm_by_key) != 5:
        raise RuntimeError("coverage provenance is not exactly 28 baseline + 5 FM")
    coverage_rows = []
    for row in merged:
        key = (row["folder"], row["configuration"])
        source = fm_by_key.get(key) if row["folder"] == "fm" else base_by_key.get(key)
        if source is None:
            raise RuntimeError(f"missing coverage source for {key}")
        item = dict(source)
        item["path_index"] = row["path_index"]
        item["wer_task_index"] = row["wer_task_index"]
        item["utmosv2_task_index"] = row["utmosv2_task_index"]
        coverage_rows.append(item)
    check_fields = (
        "path_index", "gen_path", "out_path", "wav_count", "training_job_id",
        "run_job_id", "metrics_job_id", "wer_job_id", "wer_task_index",
        "utmosv2_job_id", "utmosv2_task_index", "finalizer_job_id",
    )
    mismatches = []
    for row, item in zip(merged, coverage_rows):
        for field in check_fields:
            if str(row.get(field)) != str(item.get(field)):
                mismatches.append(f"{row['folder']}/{row['configuration']}:{field}")
        expected_missing = {
            metric: json.loads(row[f"{metric}_missing"])
            for metric in ("mcd", "logf0", "wer", "utmosv2")
        }
        if item.get("missing") != expected_missing:
            mismatches.append(f"{row['folder']}/{row['configuration']}:missing")
    if mismatches:
        raise RuntimeError("summary/coverage mismatch: " + ", ".join(mismatches[:20]))
    coverage_tmp = FINAL_COVERAGE.with_name(FINAL_COVERAGE.name + ".tmp")
    coverage_tmp.write_text(
        json.dumps({
            "status": "complete",
            "fatal_coverage": [],
            "inventory_folders": 11,
            "paths": coverage_rows,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    status_path = STATE / "final_status.json"
    scoped_status = json.loads(status_path.read_text(encoding="utf-8"))
    if scoped_status.get("status") != "complete" or scoped_status.get("paths") != 5:
        raise RuntimeError(f"scoped FM finalizer is incomplete: {scoped_status}")
    pending = {
        "status": "merged_pending_final_audit",
        "paths": 33,
        "folders": 11,
        "training_loss_records": 39 * 599,
        "csv": str(FINAL_CSV),
        "markdown": str(FINAL_MD),
        "coverage": str(FINAL_COVERAGE),
        "finalizer_job_id": finalizer,
    }
    status_tmp = status_path.with_name(status_path.name + ".tmp")
    status_tmp.write_text(json.dumps(pending, indent=2) + "\n", encoding="utf-8")
    status_tmp.replace(status_path)
    csv_tmp.replace(FINAL_CSV)
    md_tmp.replace(FINAL_MD)
    coverage_tmp.replace(FINAL_COVERAGE)
    print(f"merged_tan11_paths={len(merged)} folders=11")


if __name__ == "__main__":
    main()
