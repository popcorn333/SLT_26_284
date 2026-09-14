#!/usr/bin/env python3
"""Independently audit and seal the final 29-path TAN-7 publication."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
MAIN_STATE = ROOT / ".symphony/tan7"
REDO_STATE = ROOT / ".symphony/tan7_scorex0_redo"
RESULTS = ROOT / ".symphony/results"
TRANSCRIPT = Path(
    "/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/"
    "add/resources/filelists/ljspeech/test.txt"
)
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
METRICS = {
    "mcd": "_mcd.txt",
    "logf0": "_f0.txt",
    "wer": "_wer_l.txt",
    "utmosv2": "_utmosv2.txt",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_number(path: Path) -> float:
    tokens = path.read_text(encoding="utf-8").split()
    if len(tokens) != 1:
        raise RuntimeError(f"metric sidecar is not one value: {path}")
    value = float(tokens[0])
    if not math.isfinite(value):
        raise RuntimeError(f"metric sidecar is nonfinite: {path}")
    return value


def formatted_stats(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    return (
        f"{statistics.fmean(values):.9g}",
        f"{statistics.pstdev(values) if len(values) > 1 else 0.0:.9g}",
    )


def main() -> None:
    finalizer_job_id = os.environ.get("SLURM_JOB_ID")
    if not finalizer_job_id:
        raise RuntimeError("final audit must run inside the redo finalizer")
    fresh_folders = sorted(
        path.parent.name
        for path in ROOT.glob("*/run.sh")
        if path.parent.parent == ROOT
    )
    if fresh_folders != sorted(EXPECTED):
        raise RuntimeError(f"final run.sh inventory mismatch: {fresh_folders}")
    expected_wavs = {
        Path(line.split("|", 1)[0]).name
        for line in TRANSCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(expected_wavs) != 488:
        raise RuntimeError(f"transcript inventory is {len(expected_wavs)}, not 488")

    rows = read_csv(RESULTS / "tan7_summary.csv")
    if len(rows) != 29:
        raise RuntimeError(f"final CSV has {len(rows)}/29 rows")
    if sorted(int(row["path_index"]) for row in rows) != list(range(29)):
        raise RuntimeError("final CSV path indices are not exactly 0..28")
    actual_configs = {
        folder: {
            row["configuration"] for row in rows if row["folder"] == folder
        }
        for folder in fresh_folders
    }
    if actual_configs != EXPECTED:
        raise RuntimeError(
            f"final CSV configuration coverage mismatch: {actual_configs}"
        )
    main_paths = {
        (row["folder"], row["configuration"]): row
        for row in read_tsv(MAIN_STATE / "paths.tsv")
        if row["folder"] != "scorex0"
    }
    redo_paths = {
        (row["folder"], row["configuration"]): row
        for row in read_tsv(REDO_STATE / "paths.tsv")
    }
    if len(main_paths) != 24 or len(redo_paths) != 5:
        raise RuntimeError("path provenance is not exactly 24 original + 5 redo")

    row_audits = []
    for row in rows:
        key = (row["folder"], row["configuration"])
        source = (
            redo_paths.get(key)
            if row["folder"] == "scorex0"
            else main_paths.get(key)
        )
        if source is None:
            raise RuntimeError(f"missing path provenance for {key}")
        for field in (
            "gen_path",
            "out_path",
            "training_job_id",
            "run_job_id",
            "metrics_job_id",
            "wer_job_id",
            "utmosv2_job_id",
        ):
            source_field = {
                "gen_path": "gen",
                "out_path": "out",
                "training_job_id": "freshness_anchor_job_id",
            }.get(field, field)
            if row[field] != source[source_field]:
                raise RuntimeError(
                    f"provenance mismatch {key} {field}: "
                    f"{row[field]} != {source[source_field]}"
                )
        expected_task = source["index"]
        if (
            row["wer_task_index"] != expected_task
            or row["utmosv2_task_index"] != expected_task
        ):
            raise RuntimeError(
                f"array task provenance mismatch for {key}: "
                f"WER={row['wer_task_index']} "
                f"UTMOS={row['utmosv2_task_index']} expected={expected_task}"
            )
        if (
            row["finalizer_job_id"] != finalizer_job_id
            and row["folder"] == "scorex0"
        ):
            raise RuntimeError(f"redo row has wrong finalizer provenance: {key}")
        gen = Path(row["gen_path"])
        wavs = sorted(gen.glob("*.wav"))
        if {wav.name for wav in wavs} != expected_wavs:
            raise RuntimeError(f"final WAV inventory mismatch: {gen}")
        if int(row["wav_count"]) != 488:
            raise RuntimeError(
                f"published WAV count is not 488 for {key}: {row['wav_count']}"
            )
        metric_audit = {}
        for metric, suffix in METRICS.items():
            values = []
            missing = []
            for wav in wavs:
                sidecar = wav.with_name(wav.stem + suffix)
                if sidecar.is_file():
                    values.append(read_number(sidecar))
                    continue
                if metric != "logf0":
                    missing.append(f"{wav.name}:missing")
                    continue
                marker = wav.with_name(wav.stem + "_f0_missing.json")
                try:
                    marker_data = json.loads(
                        marker.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    missing.append(f"{wav.name}:marker_invalid")
                    continue
                if (
                    marker_data.get("reason") == "no_joint_voiced_frames"
                    and marker_data.get("joint_voiced_frames") == 0
                    and marker_data.get("wav") == str(wav)
                ):
                    missing.append(
                        f"{wav.name}:genuine_undefined_no_joint_voiced_frames"
                    )
                else:
                    missing.append(f"{wav.name}:marker_unproven")
            recorded_missing = json.loads(row[f"{metric}_missing"])
            if missing != recorded_missing:
                raise RuntimeError(
                    f"missing-value record mismatch for {key}/{metric}"
                )
            if metric != "logf0" and missing:
                raise RuntimeError(f"unexpected missing {metric} values for {key}")
            if metric == "logf0" and any(
                not value.endswith(
                    ":genuine_undefined_no_joint_voiced_frames"
                )
                for value in missing
            ):
                raise RuntimeError(f"unproven logF0 missing value for {key}")
            mean, std = formatted_stats(values)
            if (
                int(row[f"{metric}_count"]) != len(values)
                or row[f"{metric}_mean"] != mean
                or row[f"{metric}_std"] != std
            ):
                raise RuntimeError(
                    f"statistic mismatch for {key}/{metric}: "
                    f"count={len(values)} mean={mean} std={std}"
                )
            if len(values) + len(missing) != 488:
                raise RuntimeError(f"incomplete {metric} coverage for {key}")
            metric_audit[metric] = {
                "count": len(values),
                "missing": len(missing),
                "mean": mean,
                "population_std": std,
            }
        row_audits.append(
            {
                "folder": row["folder"],
                "configuration": row["configuration"],
                "path_index": int(row["path_index"]),
                "jobs": {
                    "training": row["training_job_id"],
                    "effective_run": row["run_job_id"],
                    "metrics": row["metrics_job_id"],
                    "wer_array": row["wer_job_id"],
                    "wer_task": int(row["wer_task_index"]),
                    "utmosv2_array": row["utmosv2_job_id"],
                    "utmosv2_task": int(row["utmosv2_task_index"]),
                    "finalizer": row["finalizer_job_id"],
                },
                "metrics": metric_audit,
            }
        )

    coverage = json.loads(
        (RESULTS / "tan7_coverage.json").read_text(encoding="utf-8")
    )
    if (
        coverage.get("status") != "complete"
        or coverage.get("fatal_coverage")
        or coverage.get("scorex0_source") != "fresh_redo"
        or len(coverage.get("paths", [])) != 29
    ):
        raise RuntimeError(f"final coverage publication is invalid: {coverage}")
    coverage_paths = coverage["paths"]
    try:
        coverage_indices = sorted(
            int(item["path_index"]) for item in coverage_paths
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("final coverage contains an invalid path index") from error
    if coverage_indices != list(range(29)):
        raise RuntimeError(
            f"final coverage path indices are not exactly 0..28: {coverage_indices}"
        )
    coverage_by_key = {
        (item.get("folder"), item.get("configuration")): item
        for item in coverage_paths
    }
    if len(coverage_by_key) != 29:
        raise RuntimeError("final coverage paths are not 29 unique configurations")
    coverage_fields = (
        "path_index", "gen_path", "out_path", "wav_count",
        "training_job_id", "run_job_id", "metrics_job_id", "wer_job_id",
        "wer_task_index", "utmosv2_job_id", "utmosv2_task_index",
        "finalizer_job_id",
    )
    coverage_errors = []
    for row in rows:
        key = (row["folder"], row["configuration"])
        item = coverage_by_key.get(key)
        if item is None:
            coverage_errors.append(f"{key}:missing")
            continue
        for field in coverage_fields:
            if str(item.get(field)) != str(row.get(field)):
                coverage_errors.append(
                    f"{key}:{field}={item.get(field)}!={row.get(field)}"
                )
        expected_missing = {
            metric: json.loads(row[f"{metric}_missing"])
            for metric in METRICS
        }
        if item.get("missing") != expected_missing:
            coverage_errors.append(f"{key}:missing_values")
    if coverage_errors:
        raise RuntimeError(
            "final coverage/CSV mismatch: " + ", ".join(coverage_errors[:20])
        )
    combined_coverage = json.loads(
        (REDO_STATE / "training_loss_combined_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    loss_rows = read_tsv(ROOT / "training_loss.log")
    if (
        len(loss_rows) != 34 * 599
        or combined_coverage.get("status") != "complete"
        or combined_coverage.get("records") != len(loss_rows)
        or len(combined_coverage.get("runs", [])) != 34
    ):
        raise RuntimeError("final combined training-loss coverage is invalid")
    main_run_rows = read_tsv(MAIN_STATE / "run_jobs_effective.tsv")
    main_training = {
        row["folder"]: (
            row.get("training_job_id") or row["job_id"], row["stdout"]
        )
        for row in main_run_rows
    }
    redo_run_rows = read_tsv(REDO_STATE / "run_jobs.tsv")
    if (
        len(main_run_rows) != len(EXPECTED)
        or set(main_training) != set(EXPECTED)
        or len(redo_run_rows) != 1
        or redo_run_rows[0].get("folder") != "scorex0"
    ):
        raise RuntimeError("final loss run manifests are incomplete")
    # Attribute losses to the allocation that actually trained the sweep,
    # not to the later all-task inference success barrier.
    redo_job = (
        redo_run_rows[0].get("training_job_id")
        or redo_run_rows[0]["job_id"]
    )
    redo_source = (
        redo_run_rows[0].get("training_stdout")
        or redo_run_rows[0]["stdout"]
    )
    expected_loss_groups = {
        (
            folder,
            configuration,
            main_training[folder][0],
            "original",
        ): main_training[folder][1]
        for folder, configurations in EXPECTED.items()
        for configuration in configurations
    }
    expected_loss_groups.update(
        {
            ("scorex0", configuration, redo_job, "scorex0_redo"): redo_source
            for configuration in EXPECTED["scorex0"]
        }
    )
    if len(expected_loss_groups) != 34:
        raise RuntimeError("expected training-loss provenance is not 34 groups")

    loss_errors = []
    epochs_by_group: dict[tuple[str, str, str, str], set[int]] = {}
    for row_number, row in enumerate(loss_rows, start=2):
        group = (
            row.get("experiment", ""),
            row.get("configuration", ""),
            row.get("run_job_id", ""),
            row.get("run_kind", ""),
        )
        expected_source = expected_loss_groups.get(group)
        if expected_source is None:
            loss_errors.append(f"row={row_number}:unexpected_group={group}")
        elif row.get("source") != expected_source:
            loss_errors.append(f"row={row_number}:source_provenance")
        try:
            epoch = int(row["epoch"])
            duration = float(row["duration_loss"])
            prior = float(row["prior_loss"])
            diffusion = float(row["diffusion_loss"])
            combined = float(row["loss"])
        except (KeyError, TypeError, ValueError) as error:
            loss_errors.append(
                f"row={row_number}:parse={type(error).__name__}"
            )
            continue
        if not 1 <= epoch <= 599:
            loss_errors.append(f"row={row_number}:epoch={epoch}")
        if not all(
            math.isfinite(value)
            for value in (duration, prior, diffusion, combined)
        ):
            loss_errors.append(f"row={row_number}:nonfinite")
        elif not math.isclose(
            combined,
            duration + prior + diffusion,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            loss_errors.append(f"row={row_number}:combined_loss_mismatch")
        group_epochs = epochs_by_group.setdefault(group, set())
        if epoch in group_epochs:
            loss_errors.append(f"row={row_number}:duplicate_epoch={epoch}")
        group_epochs.add(epoch)
    expected_epochs = set(range(1, 600))
    if set(epochs_by_group) != set(expected_loss_groups):
        loss_errors.append("training_loss_group_coverage_mismatch")
    for group, epochs in epochs_by_group.items():
        if epochs != expected_epochs:
            loss_errors.append(
                f"group={group}:epochs={len(epochs)}/599"
            )
    missing_sources = sorted(
        source
        for source in set(expected_loss_groups.values())
        if not Path(source).is_file()
    )
    if missing_sources:
        loss_errors.append(f"missing_source_logs={missing_sources[:10]}")
    if loss_errors:
        raise RuntimeError(
            "final combined training-loss rows are invalid: "
            + ", ".join(loss_errors[:20])
        )
    loss_counts = Counter(
        (row["experiment"], row["configuration"], row["run_job_id"])
        for row in loss_rows
    )
    if len(loss_counts) != 34 or set(loss_counts.values()) != {599}:
        raise RuntimeError("final loss log is not exactly 34 run/config x 599")
    run_kind_counts = Counter(row.get("run_kind") for row in loss_rows)
    if run_kind_counts != {
        "original": 29 * 599,
        "scorex0_redo": 5 * 599,
    }:
        raise RuntimeError(
            f"final loss run-kind labels are invalid: {run_kind_counts}"
        )
    markdown = (RESULTS / "tan7_summary.md").read_text(encoding="utf-8")
    data_lines = [
        line
        for line in markdown.splitlines()
        if any(line.startswith(f"| {folder} |") for folder in EXPECTED)
    ]
    if len(data_lines) != 29:
        raise RuntimeError(f"final Markdown has {len(data_lines)}/29 table rows")
    status_path = MAIN_STATE / "final_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("status")
        != "scorex0_redo_merged_pending_final_audit"
        or status.get("paths") != 29
        or status.get("training_loss_records") != 34 * 599
        or status.get("scorex0_source") != "fresh_redo"
        or status.get("scorex0_redo_finalizer_job_id") != finalizer_job_id
    ):
        raise RuntimeError(f"pre-audit final status is invalid: {status}")

    audit_path = RESULTS / "tan7_final_audit.json"
    audit_tmp = audit_path.with_name(audit_path.name + ".tmp")
    audit = {
        "status": "complete",
        "run_folders": fresh_folders,
        "paths": len(rows),
        "wavs_per_path": 488,
        "training_loss_records": len(loss_rows),
        "training_run_configurations": len(loss_counts),
        "training_run_kind_counts": dict(run_kind_counts),
        "training_epochs_per_configuration": 599,
        "training_epoch_range": [1, 599],
        "training_loss_numeric_validation": "finite_components_and_combined_sum",
        "training_loss_source_provenance": "validated",
        "scorex0_source": "fresh_redo",
        "finalizer_job_id": finalizer_job_id,
        "rows": row_audits,
    }
    audit_tmp.write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    audit_tmp.replace(audit_path)
    completed_status = {
        **status,
        "status": "complete",
        "final_audit": str(audit_path),
    }
    status_tmp = status_path.with_name(status_path.name + ".tmp")
    status_tmp.write_text(
        json.dumps(completed_status, indent=2) + "\n", encoding="utf-8"
    )
    status_tmp.replace(status_path)
    print(
        f"final_audit_complete paths={len(rows)} "
        f"training_losses={len(loss_rows)}"
    )


if __name__ == "__main__":
    main()
