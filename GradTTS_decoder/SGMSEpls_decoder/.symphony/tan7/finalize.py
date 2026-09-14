#!/usr/bin/env python3
"""Validate TAN-7 scoring coverage and write machine/readable summaries."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
STATE = Path(os.environ.get("TAN7_STATE", ROOT / ".symphony/tan7"))
RESULTS = ROOT / ".symphony/results"
RESULT_PREFIX = os.environ.get("TAN7_RESULT_PREFIX", "tan7")
TRAINING_LOSS_PATH = Path(
    os.environ.get("TAN7_TRAINING_LOSS", ROOT / "training_loss.log")
)
FINALIZER_BARRIERS_PATH = os.environ.get("TAN7_FINALIZER_BARRIERS")
TRANSCRIPT = Path(
    "/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/"
    "add/resources/filelists/ljspeech/test.txt"
)
ALL_EXPECTED_CONFIGURATIONS = {
    "add": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6"},
    "blur": {"model.masking.b:10", "model.masking.b:20", "model.masking.b:40"},
    "fm": {"model.masking.a:1"},
    "fmDT": {"model.masking.a:1"},
    "levy": {"model.masking.alpha:1.9"},
    "mask": {"model.masking.a:1"},
    "one-shot": {"model.masking.a:0.2"},
    "product": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6"},
    "score": {
        "model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6",
        "model.masking.a:0.8", "model.masking.a:1",
    },
    "scoreDT": {
        "model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6",
        "model.masking.a:0.8", "model.masking.a:1",
    },
    "scorex0": {
        "model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6",
        "model.masking.a:0.8", "model.masking.a:1",
    },
}
if os.environ.get("TAN7_SCOPE") == "scorex0_redo":
    EXPECTED_CONFIGURATIONS = {
        "scorex0": set(ALL_EXPECTED_CONFIGURATIONS["scorex0"])
    }
elif os.environ.get("TAN7_SCOPE") == "tan11_fm":
    EXPECTED_CONFIGURATIONS = {
        "fm": {
            "model.masking.a:0.2",
            "model.masking.a:0.4",
            "model.masking.a:0.6",
            "model.masking.a:0.8",
            "model.masking.a:1",
        }
    }
elif os.environ.get("TAN7_SCOPE") == "tan12":
    EXPECTED_CONFIGURATIONS = {
        "levy": {"model.masking.alpha:1.8"},
        "fmDT": {
            "model.masking.a:0.2",
            "model.masking.a:0.4",
            "model.masking.a:0.6",
            "model.masking.a:0.8",
            "model.masking.a:1",
        },
    }
else:
    EXPECTED_CONFIGURATIONS = {
        folder: set(configurations)
        for folder, configurations in ALL_EXPECTED_CONFIGURATIONS.items()
    }
EXPECTED_PATHS = sum(len(values) for values in EXPECTED_CONFIGURATIONS.values())
EXPECTED_TRAINING_LOSS_RECORDS = EXPECTED_PATHS * 599


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sacct_rows(job_id: str) -> list[tuple[str, str, str]]:
    result = subprocess.run(
        # This Slurm cluster assigns a separate internal allocation ID to each
        # array task.  JobIDRaw therefore reports values such as ``10946831``
        # instead of the array identity ``10944210_0``, making complete tasks
        # impossible to associate with their parent array.  Display JobID
        # retains the parent/task identity used by the exact coverage checks
        # below while still identifying ordinary allocations verbatim.
        [
            "sacct", "--array", "-X", "-j", job_id,
            "--format=JobID,State,ExitCode", "-n", "-P",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 3:
            rows.append((fields[0], fields[1].rstrip("+").split()[0], fields[2]))
    return rows


def allocation_status(kind: str, job_id: str, expected_tasks: int) -> tuple[bool, bool, str]:
    rows = sacct_rows(job_id)
    exact = [row for row in rows if row[0] == job_id]
    if kind in {"wer", "utmosv2"}:
        task_pattern = re.compile(rf"^{re.escape(job_id)}_(\d+)$")
        tasks: list[tuple[int, str, str]] = []
        for row_job_id, state, exit_code in rows:
            match = task_pattern.fullmatch(row_job_id)
            if match:
                tasks.append((int(match.group(1)), state, exit_code))
        task_indices = [index for index, _, _ in tasks]
        expected_indices = list(range(expected_tasks))
        exact_ok = len(exact) <= 1 and (
            not exact
            or (exact[0][1] == "COMPLETED" and exact[0][2] == "0:0")
        )
        exact_task_coverage = sorted(task_indices) == expected_indices
        ok = exact_ok and exact_task_coverage and all(
            state == "COMPLETED" and exit_code == "0:0"
            for _, state, exit_code in tasks
        )
        terminal_states = {
            "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
            "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
        }
        exact_settled = len(exact) <= 1 and (
            not exact or exact[0][1] in terminal_states
        )
        settled = exact_settled and exact_task_coverage and all(
            state in terminal_states for _, state, _ in tasks
        )
        aggregate = f"{exact[0][1]}/{exact[0][2]}" if exact else "none"
        return ok, settled, (
            f"aggregate={aggregate} tasks={len(tasks)}/{expected_tasks} "
            f"indices={','.join(str(index) for index in sorted(task_indices)) or 'none'} "
            "states="
        ) + ",".join(
            sorted({f"{state}/{exit_code}" for _, state, exit_code in tasks}) or ["none"]
        )
    if exact:
        ok = exact[0][1] == "COMPLETED" and exact[0][2] == "0:0"
        terminal_states = {
            "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
            "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
        }
        return ok, exact[0][1] in terminal_states, f"{exact[0][1]}/{exact[0][2]}"
    return False, False, "UNKNOWN"


def allocation_ok(kind: str, job_id: str, expected_tasks: int) -> tuple[bool, str]:
    # Like the training dispatcher, tolerate only the brief accounting lag
    # immediately after an afterany dependency releases.  Persistent failures
    # and non-zero exit codes are still reported without retrying computation.
    latest = "UNKNOWN"
    for attempt in range(12):
        try:
            ok, settled, latest = allocation_status(kind, job_id, expected_tasks)
        except (OSError, subprocess.CalledProcessError) as error:
            ok, settled = False, False
            latest = f"sacct_error={type(error).__name__}"
        if settled:
            return ok, latest
        if attempt < 11:
            time.sleep(10)
    return False, latest


def validate_downstream_provenance(
    paths: list[dict[str, str]], downstream: list[dict[str, str]]
) -> None:
    """Require an exact path-index-to-scoring-job manifest."""
    metric_rows = [row for row in downstream if row.get("kind") == "metrics"]
    wer_rows = [row for row in downstream if row.get("kind") == "wer"]
    utmos_rows = [row for row in downstream if row.get("kind") == "utmosv2"]
    unexpected = [
        row for row in downstream
        if row.get("kind") not in {"metrics", "wer", "utmosv2"}
    ]
    if unexpected or len(downstream) != EXPECTED_PATHS + 2:
        raise RuntimeError(
            "downstream manifest has unexpected rows: "
            f"total={len(downstream)}/{EXPECTED_PATHS + 2} "
            f"unexpected={unexpected[:10]}"
        )
    if len(metric_rows) != EXPECTED_PATHS:
        raise RuntimeError(
            f"downstream manifest has {len(metric_rows)}/{EXPECTED_PATHS} metrics rows"
        )
    try:
        metrics_by_index = {int(row["index"]): row["job_id"] for row in metric_rows}
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("downstream metrics manifest has an invalid index/job ID") from error
    if sorted(metrics_by_index) != list(range(EXPECTED_PATHS)):
        raise RuntimeError(
            "downstream metrics indices are not exactly 0 through "
            f"{EXPECTED_PATHS - 1}: {sorted(metrics_by_index)}"
        )
    if len(set(metrics_by_index.values())) != EXPECTED_PATHS:
        raise RuntimeError("downstream metrics manifest contains duplicate job IDs")
    if len(wer_rows) != 1 or len(utmos_rows) != 1:
        raise RuntimeError(
            "downstream manifest must contain one WER and one UTMOSv2 array row: "
            f"wer={wer_rows} utmosv2={utmos_rows}"
        )
    for kind, rows in (("wer", wer_rows), ("utmosv2", utmos_rows)):
        if rows[0].get("index") != "*" or not rows[0].get("job_id"):
            raise RuntimeError(f"invalid downstream {kind} array row: {rows[0]}")
    wer_id = wer_rows[0]["job_id"]
    utmos_id = utmos_rows[0]["job_id"]
    all_scoring_job_ids = [*metrics_by_index.values(), wer_id, utmos_id]
    if len(set(all_scoring_job_ids)) != EXPECTED_PATHS + 2:
        raise RuntimeError(
            "metrics, WER, and UTMOSv2 allocations must have distinct job IDs"
        )
    mismatches = []
    for path in paths:
        try:
            index = int(path["index"])
        except (KeyError, TypeError, ValueError):
            mismatches.append(f"invalid_path_index={path.get('index', '?')}")
            continue
        expected_metric_id = metrics_by_index.get(index)
        if path.get("metrics_job_id") != expected_metric_id:
            mismatches.append(
                f"index={index}:metrics={path.get('metrics_job_id')}!={expected_metric_id}"
            )
        if path.get("wer_job_id") != wer_id:
            mismatches.append(
                f"index={index}:wer={path.get('wer_job_id')}!={wer_id}"
            )
        if path.get("utmosv2_job_id") != utmos_id:
            mismatches.append(
                f"index={index}:utmosv2={path.get('utmosv2_job_id')}!={utmos_id}"
            )
    if mismatches:
        raise RuntimeError(
            "path/downstream job provenance mismatch: " + ", ".join(mismatches[:20])
        )


def validate_inventory_and_training(paths: list[dict[str, str]]) -> int:
    inventory = read_tsv(STATE / "inventory.tsv")
    expected_folders = sorted(row["folder"] for row in inventory)
    fresh_repository_folders = sorted(
        path.parent.name
        for path in ROOT.glob("*/run.sh")
        if path.parent.parent == ROOT
    )
    if fresh_repository_folders != sorted(ALL_EXPECTED_CONFIGURATIONS):
        raise RuntimeError(
            f"final first-depth run.sh inventory changed: "
            f"expected={sorted(ALL_EXPECTED_CONFIGURATIONS)} "
            f"fresh={fresh_repository_folders}"
        )
    if sorted(EXPECTED_CONFIGURATIONS) != expected_folders:
        raise RuntimeError(
            "scoped final inventory does not match expected configurations: "
            f"inventory={expected_folders} "
            f"expected={sorted(EXPECTED_CONFIGURATIONS)}"
        )
    if "scorex0" not in fresh_repository_folders:
        raise RuntimeError("mandatory scorex0 run.sh is missing from final inventory")
    actual_configurations = {
        folder: {
            row["configuration"]
            for row in paths
            if row["folder"] == folder
        }
        for folder in expected_folders
    }
    if (
        set(expected_folders) != set(EXPECTED_CONFIGURATIONS)
        or actual_configurations != EXPECTED_CONFIGURATIONS
        or len(paths) != EXPECTED_PATHS
    ):
        raise RuntimeError(
            "final configuration coverage mismatch: "
            f"expected={EXPECTED_CONFIGURATIONS} actual={actual_configurations} "
            f"path_count={len(paths)}/{EXPECTED_PATHS}"
        )
    try:
        path_indices = [int(row["index"]) for row in paths]
    except (KeyError, ValueError) as error:
        raise RuntimeError("final path manifest contains an invalid index") from error
    if sorted(path_indices) != list(range(EXPECTED_PATHS)):
        raise RuntimeError(
            f"final path indices are not exactly 0..{EXPECTED_PATHS - 1}: "
            f"{sorted(path_indices)}"
        )
    for field in ("gen", "out"):
        values = [row.get(field, "") for row in paths]
        if any(not value for value in values) or len(set(values)) != EXPECTED_PATHS:
            raise RuntimeError(
                f"final path manifest requires {EXPECTED_PATHS} unique non-empty {field} values"
            )
    job_fields = (
        "run_job_id", "freshness_anchor_job_id", "metrics_job_id",
        "wer_job_id", "utmosv2_job_id",
    )
    missing_job_fields = [
        f"index={row.get('index', '?')}:{field}"
        for row in paths
        for field in job_fields
        if not row.get(field)
    ]
    if missing_job_fields:
        raise RuntimeError(
            "final path manifest has missing job provenance: "
            + ", ".join(missing_job_fields[:20])
        )
    invalid_freshness_anchors = []
    for row in paths:
        try:
            anchor = float(row["freshness_anchor_start_epoch"])
        except (KeyError, TypeError, ValueError):
            invalid_freshness_anchors.append(
                f"index={row.get('index', '?')}:unparseable"
            )
            continue
        if not math.isfinite(anchor) or anchor <= 0:
            invalid_freshness_anchors.append(
                f"index={row.get('index', '?')}:{anchor}"
            )
    if invalid_freshness_anchors:
        raise RuntimeError(
            "final path manifest has invalid generation freshness anchors: "
            + ", ".join(invalid_freshness_anchors[:20])
        )
    metric_job_ids = {row["metrics_job_id"] for row in paths}
    wer_job_ids = {row["wer_job_id"] for row in paths}
    utmosv2_job_ids = {row["utmosv2_job_id"] for row in paths}
    if len(metric_job_ids) != EXPECTED_PATHS:
        raise RuntimeError(
            f"expected {EXPECTED_PATHS} unique metrics jobs, found {len(metric_job_ids)}"
        )
    if len(wer_job_ids) != 1 or len(utmosv2_job_ids) != 1:
        raise RuntimeError(
            "expected exactly one WER array and one UTMOSv2 array: "
            f"wer={sorted(wer_job_ids)} utmosv2={sorted(utmosv2_job_ids)}"
        )
    original_runs = read_tsv(STATE / "run_jobs.tsv")
    if not original_runs:
        raise RuntimeError("original run manifest is empty")
    original_folders = {row.get("folder", "") for row in original_runs}
    if original_folders != set(expected_folders):
        raise RuntimeError(
            "original run manifest folder mismatch: "
            f"expected={expected_folders} actual={sorted(original_folders)}"
        )
    # Ordinary/recovery manifests have one row per folder.  Array launches may
    # instead need one row per configuration so each task has its own stdout
    # and loss provenance.  Accept both shapes, but require exact configuration
    # coverage whenever the explicit configuration column is populated.
    configured_runs = [
        (row.get("folder", ""), row.get("configuration", ""))
        for row in original_runs
        if row.get("configuration")
    ]
    if configured_runs:
        expected_run_configurations = {
            (folder, configuration)
            for folder, configurations in EXPECTED_CONFIGURATIONS.items()
            for configuration in configurations
        }
        if (
            len(configured_runs) != len(original_runs)
            or len(configured_runs) != len(expected_run_configurations)
            or len(set(configured_runs)) != len(configured_runs)
            or set(configured_runs) != expected_run_configurations
        ):
            raise RuntimeError(
                "per-configuration run manifest coverage mismatch: "
                f"expected={sorted(expected_run_configurations)} "
                f"actual={sorted(configured_runs)}"
            )
    elif len(original_runs) != len(expected_folders):
        raise RuntimeError(
            f"per-folder run manifest has {len(original_runs)}/{len(expected_folders)} rows"
        )
    configured_run_manifest = bool(configured_runs)

    def manifest_key(row: dict[str, str]) -> tuple[str, str]:
        return (
            row.get("folder", ""),
            row.get("configuration", "") if configured_run_manifest else "",
        )

    original_by_key = {manifest_key(row): row for row in original_runs}
    if len(original_by_key) != len(original_runs):
        raise RuntimeError("original run manifest contains duplicate provenance keys")
    path_keys = {
        (
            row.get("folder", ""),
            row.get("configuration", "") if configured_run_manifest else "",
        )
        for row in paths
    }
    if path_keys != set(original_by_key):
        raise RuntimeError(
            "path/run manifest provenance-key mismatch: "
            f"paths={sorted(path_keys)} runs={sorted(original_by_key)}"
        )

    path_run_jobs_by_key = {
        key: {
            row["run_job_id"]
            for row in paths
            if (
                row["folder"],
                row["configuration"] if configured_run_manifest else "",
            ) == key
        }
        for key in original_by_key
    }
    invalid_effective_paths = {
        key: sorted(job_ids)
        for key, job_ids in path_run_jobs_by_key.items()
        if job_ids != {original_by_key[key].get("job_id", "")}
    }
    if invalid_effective_paths:
        raise RuntimeError(
            "path effective allocations do not match original run jobs: "
            f"{invalid_effective_paths}"
        )
    # The effective run can be a later inference-only recovery or an all-task
    # success barrier.  Generated-WAV freshness and epoch losses must instead
    # be attributed to the allocation that actually performed training when
    # the manifest records one explicitly.
    training_job_by_key = {
        key: row.get("training_job_id") or row.get("job_id", "")
        for key, row in original_by_key.items()
    }
    invalid_training_manifests = {
        key: job_id
        for key, job_id in training_job_by_key.items()
        if not job_id
    }
    if invalid_training_manifests:
        raise RuntimeError(
            "each run provenance key must map to one non-empty training allocation: "
            f"{invalid_training_manifests}"
        )
    path_training_jobs_by_key = {
        key: {
            row["freshness_anchor_job_id"]
            for row in paths
            if (
                row["folder"],
                row["configuration"] if configured_run_manifest else "",
            ) == key
        }
        for key in original_by_key
    }
    invalid_training_jobs = {
        key: sorted(job_ids)
        for key, job_ids in path_training_jobs_by_key.items()
        if job_ids != {training_job_by_key[key]}
    }
    if invalid_training_jobs:
        raise RuntimeError(
            "path training allocations do not match original run jobs: "
            f"{invalid_training_jobs}"
        )
    run_states = read_tsv(STATE / "run_states.tsv")
    if len(run_states) != len(original_runs):
        raise RuntimeError(
            f"terminal run-state manifest has {len(run_states)}/{len(original_runs)} rows"
        )
    expected_state_allocations = Counter(
        (*manifest_key(row), row.get("job_id", ""))
        for row in original_runs
    )
    actual_state_allocations = Counter(
        (
            row.get("folder", ""),
            row.get("configuration", "") if configured_run_manifest else "",
            row.get("job_id", ""),
        )
        for row in run_states
    )
    if actual_state_allocations != expected_state_allocations:
        raise RuntimeError(
            "terminal run-state allocation coverage mismatch: "
            f"expected={expected_state_allocations} "
            f"actual={actual_state_allocations}"
        )
    invalid_run_states = []
    for state in run_states:
        key = (
            state.get("folder", ""),
            state.get("configuration", "") if configured_run_manifest else "",
        )
        expected_effective_id = original_by_key.get(key, {}).get("job_id", "")
        if (
            state.get("job_id") != expected_effective_id
            or state.get("state") != "COMPLETED"
            or state.get("exit_code") != "0:0"
        ):
            invalid_run_states.append(
                f"{'/'.join(key)}:manifest={expected_effective_id} "
                f"state={state.get('job_id')}:{state.get('state')}/{state.get('exit_code')}"
            )
    if invalid_run_states:
        raise RuntimeError(
            "effective run jobs are not proven successful: "
            + ", ".join(invalid_run_states[:20])
        )

    coverage_path = STATE / "training_loss_coverage.json"
    loss_path = TRAINING_LOSS_PATH
    if not coverage_path.is_file() or not loss_path.is_file():
        raise RuntimeError("training-loss output or coverage record is missing")
    coverage_data = json.loads(coverage_path.read_text(encoding="utf-8"))
    errors = coverage_data.get("errors", [])
    if errors:
        raise RuntimeError(f"training-loss coverage contains errors: {errors[:10]}")
    coverage = coverage_data.get("coverage", [])
    for item in coverage:
        if (
            item.get("expected_count") != item.get("actual_count")
            or item.get("missing_epochs")
            or item.get("extra_epochs")
        ):
            raise RuntimeError(f"invalid training-loss coverage row: {item}")
    expected_configurations = {
        (row["folder"], row["configuration"]) for row in paths
    }
    covered_configurations = {
        (str(item.get("experiment")), str(item.get("configuration")))
        for item in coverage
    }
    if covered_configurations != expected_configurations:
        raise RuntimeError(
            "training-loss configuration coverage mismatch: "
            f"expected={sorted(expected_configurations)} "
            f"covered={sorted(covered_configurations)}"
        )
    loss_rows = read_tsv(loss_path)
    if len(loss_rows) != coverage_data.get("records"):
        raise RuntimeError(
            f"training-loss record count mismatch: log={len(loss_rows)} "
            f"coverage={coverage_data.get('records')}"
        )
    invalid_loss_rows = []
    for row_number, row in enumerate(loss_rows, start=2):
        try:
            epoch = int(row["epoch"])
            duration = float(row["duration_loss"])
            prior = float(row["prior_loss"])
            diffusion = float(row["diffusion_loss"])
            combined = float(row["loss"])
        except (KeyError, TypeError, ValueError) as error:
            invalid_loss_rows.append(f"row={row_number}:parse={type(error).__name__}")
            continue
        components = (duration, prior, diffusion, combined)
        if not all(math.isfinite(value) for value in components):
            invalid_loss_rows.append(f"row={row_number}:nonfinite")
        elif not math.isclose(
            combined,
            duration + prior + diffusion,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            invalid_loss_rows.append(f"row={row_number}:combined_loss_mismatch")
        if not 1 <= epoch <= 599:
            invalid_loss_rows.append(f"row={row_number}:epoch={epoch}")
        if not row.get("run_job_id") or not row.get("source"):
            invalid_loss_rows.append(f"row={row_number}:missing_provenance")
        else:
            loss_key = (
                row.get("experiment", ""),
                row.get("configuration", "") if configured_run_manifest else "",
            )
            expected_training_job = training_job_by_key.get(loss_key)
            if row.get("run_job_id") != expected_training_job:
                invalid_loss_rows.append(
                    f"row={row_number}:training_job={row.get('run_job_id')}"
                    f" expected={expected_training_job}"
                )
    if invalid_loss_rows:
        raise RuntimeError(
            "training_loss.log contains invalid records: "
            + ", ".join(invalid_loss_rows[:20])
        )
    loss_keys = {
        (row["experiment"], row["configuration"], row["epoch"])
        for row in loss_rows
    }
    if len(loss_keys) != len(loss_rows):
        raise RuntimeError("training_loss.log contains duplicate experiment/configuration/epoch rows")
    if len(loss_rows) != EXPECTED_TRAINING_LOSS_RECORDS:
        raise RuntimeError(
            "training-loss total does not match the independently verified manifest: "
            f"actual={len(loss_rows)} expected={EXPECTED_TRAINING_LOSS_RECORDS}"
        )
    return len(loss_rows)


def read_score(path: Path, fresh_since: float) -> tuple[float | None, str | None]:
    if not path.is_file():
        return None, "missing"
    if path.stat().st_mtime + 2 < fresh_since:
        return None, "stale"
    try:
        tokens = path.read_text(encoding="utf-8").strip().split()
        if len(tokens) != 1:
            return None, "invalid"
        value = float(tokens[0])
    except (OSError, ValueError, IndexError):
        return None, "invalid"
    if not math.isfinite(value):
        return None, "nonfinite"
    return value, None


def read_f0_marker(wav: Path, fresh_since: float) -> tuple[bool, str]:
    marker = wav.with_name(wav.stem + "_f0_missing.json")
    if not marker.is_file():
        return False, "marker_missing"
    if marker.stat().st_mtime + 2 < fresh_since:
        return False, "marker_stale"
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "marker_invalid"
    if (
        data.get("reason") != "no_joint_voiced_frames"
        or data.get("joint_voiced_frames") != 0
        or data.get("wav") != str(wav)
    ):
        return False, "marker_unproven"
    return True, "genuine_undefined_no_joint_voiced_frames"


def stats(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.9g}", f"{std:.9g}"


def collect_all_slurm_job_ids(
    paths: list[dict[str, str]],
    downstream: list[dict[str, str]],
    finalizer_job_id: str,
) -> list[str]:
    """Collect every submitted/task Slurm identity retained by this workflow."""
    ids: set[str] = {finalizer_job_id}
    for manifest_name in (
        "training_attempts.tsv",
        "orchestration_jobs.tsv",
        "run_jobs.tsv",
        "run_states.tsv",
        "finalizer_job.tsv",
    ):
        manifest = STATE / manifest_name
        if not manifest.is_file():
            continue
        for row in read_tsv(manifest):
            for field, value in row.items():
                if field.endswith("job_id") and value:
                    ids.add(value)
    for row in [*paths, *downstream]:
        for field, value in row.items():
            if field.endswith("job_id") and value:
                ids.add(value)
    invalid = sorted(
        value for value in ids
        if not re.fullmatch(r"\d+(?:_\d+)?", value)
    )
    if invalid:
        raise RuntimeError(f"invalid Slurm job identities in workflow manifests: {invalid}")
    return sorted(
        ids,
        key=lambda value: tuple(int(part) for part in value.split("_", 1)),
    )


def main() -> None:
    finalizer_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not finalizer_job_id:
        raise RuntimeError("TAN-7 finalizer must run inside a Slurm allocation")
    # A safe manual retry after a transient accounting/filesystem failure must
    # not leave a stale failure marker beside a newly successful final report.
    (STATE / "finalizer_failure.txt").unlink(missing_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Never let a failed or interrupted retry leave authoritative-looking
    # report names behind.  The CSV and Markdown are published atomically only
    # after every downstream, inventory, freshness, and sidecar gate passes.
    csv_path = RESULTS / f"{RESULT_PREFIX}_summary.csv"
    markdown_path = RESULTS / f"{RESULT_PREFIX}_summary.md"
    coverage_path = RESULTS / f"{RESULT_PREFIX}_coverage.json"
    final_status_path = STATE / "final_status.json"
    csv_temporary = csv_path.with_name(csv_path.name + ".tmp")
    markdown_temporary = markdown_path.with_name(markdown_path.name + ".tmp")
    coverage_temporary = coverage_path.with_name(coverage_path.name + ".tmp")
    for temporary in (csv_temporary, markdown_temporary, coverage_temporary):
        temporary.unlink(missing_ok=True)
    # A retry validates the current filesystem and scheduler state, so a
    # report from an earlier successful invocation is no longer authoritative
    # until this invocation passes every gate.  Remove only TAN-7's generated
    # publication names; diagnostic coverage is republished below when the
    # audit reaches that stage.
    for published in (csv_path, markdown_path, coverage_path, final_status_path):
        published.unlink(missing_ok=True)
    paths = read_tsv(STATE / "paths.tsv")
    training_loss_records = validate_inventory_and_training(paths)
    downstream = read_tsv(STATE / "downstream_jobs.tsv")
    validate_downstream_provenance(paths, downstream)
    all_slurm_job_ids = collect_all_slurm_job_ids(
        paths, downstream, finalizer_job_id
    )
    expected_tasks = len(paths)
    state_records = []
    bad_jobs = []
    for row in downstream:
        ok, state = allocation_ok(row["kind"], row["job_id"], expected_tasks)
        state_records.append({**row, "state": state, "ok": ok})
        if not ok:
            bad_jobs.append(f"{row['kind']}:{row['job_id']}={state}")
    (STATE / "downstream_states.json").write_text(
        json.dumps(state_records, indent=2) + "\n", encoding="utf-8"
    )
    if bad_jobs:
        raise RuntimeError("downstream jobs did not all succeed: " + ", ".join(bad_jobs))

    if FINALIZER_BARRIERS_PATH:
        barrier_path = Path(FINALIZER_BARRIERS_PATH)
        barriers = read_tsv(barrier_path)
        if len(barriers) != 1:
            raise RuntimeError(
                "finalizer barrier manifest must contain exactly one row: "
                f"{barrier_path} has {len(barriers)}"
            )
        barrier = barriers[0]
        barrier_id = barrier.get("job_id", "")
        if barrier.get("kind") != "finalizer" or not barrier_id.isdigit():
            raise RuntimeError(f"invalid finalizer barrier row: {barrier}")
        barrier_ok, barrier_state = allocation_ok(
            "finalizer", barrier_id, expected_tasks
        )
        (STATE / "finalizer_barrier_states.json").write_text(
            json.dumps(
                [{**barrier, "state": barrier_state, "ok": barrier_ok}],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if not barrier_ok:
            raise RuntimeError(
                "required preceding finalizer did not succeed: "
                f"{barrier_id}={barrier_state}"
            )

    fresh_since = float((STATE / "dispatch_started_at.txt").read_text().strip())
    metric_specs = {
        "mcd": "_mcd.txt",
        "logf0": "_f0.txt",
        "wer": "_wer_l.txt",
        "utmosv2": "_utmosv2.txt",
    }
    summary_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    fatal_coverage: list[str] = []
    expected_wavs = {
        Path(line.split("|", 1)[0]).name
        for line in TRANSCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(expected_wavs) != 488:
        raise RuntimeError(
            f"unexpected transcript WAV inventory: expected 488 unique names, "
            f"found {len(expected_wavs)} in {TRANSCRIPT}"
        )
    for row in paths:
        gen = Path(row["gen"])
        wavs = sorted(gen.glob("*.wav"))
        actual_wavs = {path.name for path in wavs}
        if actual_wavs != expected_wavs:
            fatal_coverage.append(
                f"{gen}:wav_inventory expected={len(expected_wavs)} "
                f"actual={len(actual_wavs)} "
                f"missing={sorted(expected_wavs - actual_wavs)[:20]} "
                f"extra={sorted(actual_wavs - expected_wavs)[:20]}"
            )
        generation_anchor = float(row["freshness_anchor_start_epoch"])
        stale_wavs = sorted(
            wav.name for wav in wavs if wav.stat().st_mtime + 2 < generation_anchor
        )
        if stale_wavs:
            fatal_coverage.append(
                f"{gen}:generated_wavs_predate_job="
                f"{row['freshness_anchor_job_id']} count={len(stale_wavs)} "
                f"examples={stale_wavs[:20]}"
            )
        values_by_metric: dict[str, list[float]] = {}
        missing_by_metric: dict[str, list[str]] = {}
        for metric, suffix in metric_specs.items():
            values: list[float] = []
            missing: list[str] = []
            for wav in wavs:
                output = wav.with_name(wav.stem + suffix)
                # A sidecar must belong to both this dispatch and the current
                # audio file.  Comparing only with dispatch start could accept
                # a metric that predates a later WAV replacement.
                score_fresh_since = max(fresh_since, wav.stat().st_mtime)
                value, reason = read_score(output, score_fresh_since)
                if reason:
                    if metric == "logf0" and reason in {"missing", "nonfinite"}:
                        proven, marker_reason = read_f0_marker(
                            wav, score_fresh_since
                        )
                        if proven:
                            missing.append(f"{wav.name}:{marker_reason}")
                        else:
                            missing.append(f"{wav.name}:{reason}_{marker_reason}")
                    else:
                        missing.append(f"{wav.name}:{reason}")
                else:
                    values.append(value)  # type: ignore[arg-type]
            values_by_metric[metric] = values
            missing_by_metric[metric] = missing

        wav_count = len(wavs)
        if wav_count == 0:
            fatal_coverage.append(f"{gen}:no_wavs")
        for metric in ("mcd", "wer", "utmosv2"):
            if len(values_by_metric[metric]) != wav_count:
                fatal_coverage.append(
                    f"{gen}:{metric}={len(values_by_metric[metric])}/{wav_count}"
                )
        logf0_reasons = [
            item.rsplit(":", 1)[-1] for item in missing_by_metric["logf0"]
        ]
        # Missing log-F0 is accepted only when the Slurm-side validator
        # independently proved that DTW alignment had zero jointly voiced
        # frames and wrote a fresh marker beside the WAV.
        acceptable_logf0_missing = {"genuine_undefined_no_joint_voiced_frames"}
        unacceptable_logf0 = [
            reason
            for reason in logf0_reasons
            if reason not in acceptable_logf0_missing
        ]
        if unacceptable_logf0:
            fatal_coverage.append(
                f"{gen}:logf0_invalid_reasons={sorted(set(unacceptable_logf0))}"
            )
        accounted_logf0 = len(values_by_metric["logf0"]) + sum(
            reason in acceptable_logf0_missing for reason in logf0_reasons
        )
        if accounted_logf0 != wav_count:
            fatal_coverage.append(
                f"{gen}:logf0_accounted="
                f"{accounted_logf0}/{wav_count}"
            )
        summary: dict[str, object] = {
            "folder": row["folder"],
            "path_index": row["index"],
            "configuration": row["configuration"],
            "gen_path": row["gen"],
            "out_path": row["out"],
            "wav_count": wav_count,
            "training_job_id": row["freshness_anchor_job_id"],
            "run_job_id": row["run_job_id"],
            "metrics_job_id": row["metrics_job_id"],
            "wer_job_id": row["wer_job_id"],
            "utmosv2_job_id": row["utmosv2_job_id"],
            "finalizer_job_id": finalizer_job_id,
            "all_slurm_job_ids": json.dumps(all_slurm_job_ids),
        }
        for metric in metric_specs:
            mean, std = stats(values_by_metric[metric])
            summary[f"{metric}_count"] = len(values_by_metric[metric])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_missing"] = json.dumps(missing_by_metric[metric])
        summary_rows.append(summary)
        coverage_rows.append(
            {
                "folder": row["folder"],
                "path_index": row["index"],
                "configuration": row["configuration"],
                "gen_path": row["gen"],
                "out_path": row["out"],
                "wav_count": wav_count,
                "training_job_id": row["freshness_anchor_job_id"],
                "run_job_id": row["run_job_id"],
                "metrics_job_id": row["metrics_job_id"],
                "wer_job_id": row["wer_job_id"],
                "wer_task_index": row["index"],
                "utmosv2_job_id": row["utmosv2_job_id"],
                "utmosv2_task_index": row["index"],
                "finalizer_job_id": finalizer_job_id,
                "missing": missing_by_metric,
            }
        )

    fields = [
        "folder", "path_index", "configuration", "gen_path", "out_path", "wav_count",
        "mcd_count", "mcd_mean", "mcd_std", "mcd_missing",
        "logf0_count", "logf0_mean", "logf0_std", "logf0_missing",
        "wer_count", "wer_mean", "wer_std", "wer_missing",
        "utmosv2_count", "utmosv2_mean", "utmosv2_std", "utmosv2_missing",
        "training_job_id", "run_job_id", "metrics_job_id", "wer_job_id",
        "utmosv2_job_id",
        "finalizer_job_id",
        "all_slurm_job_ids",
    ]
    with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    markdown = [
        f"# {RESULT_PREFIX.upper()} validated speech-metric summary",
        "",
        "Standard deviations are population standard deviations (ddof=0). Counts are fresh per-WAV outputs.",
        "",
        "| Folder | Path index | Configuration | Generated path | Metrics output path | WAV | MCD n / mean / std | logF0 n / mean / std | WER n / mean / std | UTMOSv2 n / mean / std | Jobs (training / effective run / metrics / WER[index] / UTMOSv2[index]) |",
        "|---|---:|---|---|---|---:|---|---|---|---|---|",
    ]
    for row in summary_rows:
        cell = lambda metric: f"{row[f'{metric}_count']} / {row[f'{metric}_mean']} / {row[f'{metric}_std']}"
        jobs = (
            f"{row['training_job_id']} / {row['run_job_id']} / "
            f"{row['metrics_job_id']} / "
            f"{row['wer_job_id']}[{row['path_index']}] / "
            f"{row['utmosv2_job_id']}[{row['path_index']}]"
        )
        markdown.append(
            f"| {row['folder']} | {row['path_index']} | `{row['configuration']}` | `{row['gen_path']}` | "
            f"`{row['out_path']}` | {row['wav_count']} | "
            f"{cell('mcd')} | {cell('logf0')} | {cell('wer')} | {cell('utmosv2')} | {jobs} |"
        )
    genuine_missing = [
        item for item in coverage_rows if any(item["missing"].values())  # type: ignore[union-attr]
    ]
    markdown.extend(["", "## Genuine missing values", ""])
    if genuine_missing:
        for item in genuine_missing:
            details = []
            for metric, missing_items in item["missing"].items():  # type: ignore[union-attr]
                if not missing_items:
                    continue
                reason_counts = Counter(
                    entry.rsplit(":", 1)[-1] for entry in missing_items
                )
                reasons = ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(reason_counts.items())
                )
                details.append(f"{metric}: {len(missing_items)} ({reasons})")
            markdown.append(
                f"- `{item['folder']}/{item['configuration']}`: "
                + "; ".join(details)
            )
        markdown.append(
            "Complete per-WAV missing-value records are retained in "
            f"`{RESULT_PREFIX}_summary.csv` and `{RESULT_PREFIX}_coverage.json`."
        )
    else:
        markdown.append("None.")
    markdown.extend(
        [
            "",
            f"Finalizer Slurm job: `{finalizer_job_id}`.",
            "All Slurm job IDs (including preflights, superseded/failed attempts, "
            "dispatcher, scoring jobs, and finalizer): "
            + ", ".join(f"`{job_id}`" for job_id in all_slurm_job_ids)
            + ".",
            f"Training-loss records: {training_loss_records} in `{TRAINING_LOSS_PATH}`.",
        ]
    )
    markdown_temporary.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    coverage_temporary.write_text(
        json.dumps(
            {
                "status": "complete" if not fatal_coverage else "coverage_failed",
                "fatal_coverage": fatal_coverage,
                "all_slurm_job_ids": all_slurm_job_ids,
                "paths": coverage_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # The coverage record is diagnostic as well as authoritative, so publish
    # it even when a gate fails.  Final-looking summaries remain temporary
    # until the fatal-coverage list is empty.
    coverage_temporary.replace(coverage_path)
    if fatal_coverage:
        csv_temporary.unlink(missing_ok=True)
        markdown_temporary.unlink(missing_ok=True)
        raise RuntimeError("coverage validation failed: " + ", ".join(fatal_coverage[:20]))
    csv_temporary.replace(csv_path)
    markdown_temporary.replace(markdown_path)
    final_status_temporary = final_status_path.with_name(final_status_path.name + ".tmp")
    final_status_temporary.write_text(
        json.dumps(
            {
                "status": "complete",
                "paths": len(summary_rows),
                "training_loss_records": training_loss_records,
                "all_slurm_job_ids": all_slurm_job_ids,
                "csv": str(csv_path),
                "markdown": str(markdown_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    final_status_temporary.replace(final_status_path)
    print(f"validated_paths={len(summary_rows)} csv={csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (STATE / "finalizer_failure.txt").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        raise
