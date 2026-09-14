#!/usr/bin/env python3
"""Validate TAN-7 runs, extract all pairs, and dispatch dependent scoring."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
STATE = Path(os.environ.get("TAN7_STATE", ROOT / ".symphony/tan7"))
LOGS = STATE / "logs"
METRIC_SCRIPTS = STATE / "metrics_scripts"
RUN_JOBS_PATH = Path(os.environ.get("TAN7_RUN_JOBS", STATE / "run_jobs.tsv"))
TRAINING_LOSS_PATH = Path(
    os.environ.get("TAN7_TRAINING_LOSS", ROOT / "training_loss.log")
)
JOB_PREFIX = os.environ.get("TAN7_JOB_PREFIX", "tan7")
EXTRACT_LOSSES = Path(
    os.environ.get("TAN7_EXTRACT_LOSSES", STATE / "extract_losses.py")
)
POST_LOSS_HOOK = os.environ.get("TAN7_POST_LOSS_HOOK")
FINALIZER_BARRIER_MANIFEST = os.environ.get(
    "TAN7_FINALIZER_BARRIER_MANIFEST"
) or (
    str(ROOT / ".symphony/tan7/finalizer_job.tsv")
    if os.environ.get("TAN7_SCOPE") == "scorex0_redo"
    else None
)
METRICS_ENV_BIN = Path("/users/acp23xt/.conda/envs/Grad-TTS-EVAL/bin")
UTMOS_CACHE = Path("/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache")
UTMOS_PREFLIGHT_SCORE = (
    UTMOS_CACHE / "preflight/LJ045-0096_utmosv2.txt"
)
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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    # Some of these manifests are consumed by array tasks immediately after
    # submission (notably paths.tsv).  Write a complete sibling first and
    # atomically replace the destination so a fast-starting task can observe
    # either the old complete manifest or the new complete manifest, never a
    # truncate-in-progress file.
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def command(args: list[str], cwd: Path = ROOT) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return result.stdout


def job_state(job_id: str) -> tuple[str, str]:
    # Slurm dependencies can release a few seconds before sacct has published
    # the final allocation row.  Retry only inside this short dispatcher job;
    # this avoids falsely rejecting a completed multi-day run without creating
    # a monitoring loop on the Symphony host.
    terminal_states = {
        "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
        "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
    }
    latest = ("UNKNOWN", "")
    for attempt in range(12):
        try:
            output = command(
                # Started array tasks receive distinct internal JobIDRaw values
                # on this cluster.  Display JobID retains the submitted
                # ``array_task`` identity recorded in per-configuration
                # manifests, and is also identical to an ordinary job ID.
                ["sacct", "--array", "-X", "-j", job_id, "--format=JobID,State,ExitCode", "-n", "-P"]
            )
        except RuntimeError:
            output = ""
        for line in output.splitlines():
            fields = line.split("|")
            if len(fields) >= 3 and fields[0] == job_id:
                latest = (fields[1].rstrip("+").split()[0], fields[2])
                if latest[0] in terminal_states:
                    return latest
                break
        if attempt < 11:
            time.sleep(10)
    return latest


def job_start_time(job_id: str) -> float:
    """Return the allocation start as local epoch seconds from Slurm accounting."""
    output = command(
        ["sacct", "--array", "-X", "-j", job_id, "--format=JobID,Start", "-n", "-P"]
    )
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < 2 or fields[0] != job_id:
            continue
        start = fields[1].strip()
        if not start or start == "Unknown":
            break
        try:
            parsed = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=ZoneInfo("Europe/London")
            )
        except ValueError as error:
            raise RuntimeError(
                f"cannot parse Slurm start time for job {job_id}: {start}"
            ) from error
        return parsed.timestamp()
    raise RuntimeError(f"missing Slurm start time for job {job_id}")


def parse_pairs(text: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    gen: Path | None = None
    out: Path | None = None
    for line in [*text.splitlines(), ""]:
        if line.startswith("gen="):
            gen = Path(line[4:])
        elif line.startswith("out="):
            out = Path(line[4:])
        elif not line.strip() and gen is not None and out is not None:
            pairs.append((gen.resolve(), out.resolve()))
            gen = out = None
    return pairs


def update_metrics(path: Path, gen: Path, out: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text, gen_count = re.subn(r"(?m)^gen=.*$", f"gen={gen}", text, count=1)
    text, out_count = re.subn(r"(?m)^out=.*$", f"out={out}", text, count=1)
    if gen_count != 1 or out_count != 1:
        raise RuntimeError(f"cannot update gen/out assignments exactly once in {path}")
    path.write_text(text, encoding="utf-8")


def submit(args: list[str]) -> str:
    output = command(["sbatch", "--parsable", *args]).strip()
    if not output:
        raise RuntimeError(f"sbatch returned no job id: {args}")
    return output.split(";", 1)[0]


def validate_utmos_preflight() -> None:
    required_files = {
        UTMOS_CACHE / "utmosv2/models/fusion_stage3/fold0_s42_best_model.pth": 818531314,
        UTMOS_CACHE / (
            "huggingface/hub/models--facebook--wav2vec2-base/"
            "snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8/pytorch_model.bin"
        ): 380267417,
        UTMOS_CACHE / (
            "huggingface/hub/models--timm--tf_efficientnetv2_s.in21k_ft_in1k/"
            "snapshots/212ce86b3cc2aa27505700cd5d1a25dfd2a07dc8/model.safetensors"
        ): 86523256,
    }
    invalid = [
        f"{path}:{path.stat().st_size if path.is_file() else 'missing'}"
        for path, expected_size in required_files.items()
        if not path.is_file() or path.stat().st_size != expected_size
    ]
    if invalid:
        raise RuntimeError("UTMOSv2 staged cache validation failed: " + ", ".join(invalid))
    try:
        score_tokens = UTMOS_PREFLIGHT_SCORE.read_text(encoding="utf-8").split()
        score = float(score_tokens[0]) if len(score_tokens) == 1 else math.nan
    except (OSError, ValueError, IndexError):
        score = math.nan
    if not math.isfinite(score):
        raise RuntimeError(
            f"missing or invalid UTMOSv2 GPU preflight score: {UTMOS_PREFLIGHT_SCORE}"
        )


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("TAN-7 dispatcher must run inside a Slurm allocation")
    LOGS.mkdir(parents=True, exist_ok=True)
    METRIC_SCRIPTS.mkdir(parents=True, exist_ok=True)
    failure_path = STATE / "dispatch_failure.txt"
    status_path = STATE / "dispatch_status.json"
    journal_path = STATE / "downstream_submit_journal.tsv"
    downstream_path = STATE / "downstream_jobs.tsv"
    failure_path.unlink(missing_ok=True)

    # A Slurm requeue or an operator retry must never duplicate an already
    # recorded downstream dispatch.  A final status is an idempotent success;
    # a non-empty journal without that status is evidence of partial submission
    # and requires explicit reconciliation rather than automatic resubmission.
    if status_path.is_file():
        try:
            previous_status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid existing dispatch status: {status_path}") from error
        if previous_status.get("status") == "downstream_submitted":
            print("downstream_dispatch_already_recorded=true", flush=True)
            return
        raise RuntimeError(f"unexpected existing dispatch status: {previous_status}")
    for artifact in (journal_path, downstream_path):
        if artifact.is_file() and artifact.stat().st_size:
            raise RuntimeError(
                f"refusing duplicate submission after a possible partial dispatch: {artifact}"
            )

    inventory = read_tsv(STATE / "inventory.tsv")
    run_jobs = read_tsv(RUN_JOBS_PATH)
    expected_folders = sorted(row["folder"] for row in inventory)
    fresh_repository_folders = sorted(
        path.parent.name
        for path in ROOT.glob("*/run.sh")
        if path.parent.parent == ROOT
    )
    if fresh_repository_folders != sorted(ALL_EXPECTED_CONFIGURATIONS):
        raise RuntimeError(
            "first-depth run.sh inventory changed: "
            f"expected={sorted(ALL_EXPECTED_CONFIGURATIONS)} "
            f"fresh={fresh_repository_folders}"
        )
    if sorted(EXPECTED_CONFIGURATIONS) != expected_folders:
        raise RuntimeError(
            "scoped run manifest does not match expected configurations: "
            f"inventory={expected_folders} "
            f"expected={sorted(EXPECTED_CONFIGURATIONS)}"
        )
    if "scorex0" not in fresh_repository_folders:
        raise RuntimeError("mandatory scorex0 run.sh is missing from the final inventory")

    configured_flags = [bool(row.get("configuration")) for row in run_jobs]
    if any(configured_flags) and not all(configured_flags):
        raise RuntimeError("run manifest mixes configured and per-folder rows")
    configured_run_manifest = bool(run_jobs) and all(configured_flags)
    if configured_run_manifest:
        expected_run_keys = {
            (folder, configuration)
            for folder, configurations in EXPECTED_CONFIGURATIONS.items()
            for configuration in configurations
        }
        run_keys = [
            (row["folder"], row["configuration"])
            for row in run_jobs
        ]
    else:
        expected_run_keys = {(folder, "") for folder in expected_folders}
        run_keys = [(row["folder"], "") for row in run_jobs]
    if (
        len(run_keys) != len(expected_run_keys)
        or len(set(run_keys)) != len(run_keys)
        or set(run_keys) != expected_run_keys
    ):
        raise RuntimeError(
            "run manifest coverage mismatch: "
            f"expected={sorted(expected_run_keys)} actual={sorted(run_keys)}"
        )
    run_by_key = dict(zip(run_keys, run_jobs))

    state_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for row in run_jobs:
        state, exit_code = job_state(row["job_id"])
        state_rows.append({
            "folder": row["folder"],
            "configuration": row.get("configuration", ""),
            "job_id": row["job_id"],
            "state": state,
            "exit_code": exit_code,
        })
        if state != "COMPLETED" or exit_code != "0:0":
            failures.append(f"{row['folder']}={row['job_id']}:{state}/{exit_code}")
    write_tsv(
        STATE / "run_states.tsv",
        state_rows,
        ["folder", "configuration", "job_id", "state", "exit_code"],
    )
    if failures:
        raise RuntimeError("run jobs did not all succeed: " + ", ".join(failures))

    validate_utmos_preflight()

    # A recovered allocation can be inference-only.  In that case the
    # original training allocation is still the earliest legitimate source
    # for generated WAVs (and permits the product fallback to retain outputs
    # it independently proved were created by the failed parent).  Requiring
    # freshness against that allocation prevents a successful run from
    # silently dispatching metrics over an older, exact-name inventory.
    generation_source_job_by_key = {
        key: row.get("training_job_id") or row["job_id"]
        for key, row in run_by_key.items()
    }
    generation_started_at_by_key = {
        key: job_start_time(job_id)
        for key, job_id in generation_source_job_by_key.items()
    }
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
    path_rows: list[dict[str, object]] = []
    seen_gen: set[Path] = set()
    for inventory_row in inventory:
        folder = inventory_row["folder"]
        run_path = ROOT / inventory_row["run_path"]
        extracted = command([sys.executable, str(ROOT / "extract_metrics_paths.py"), str(run_path)])
        pairs = parse_pairs(extracted)
        if not pairs:
            raise RuntimeError(f"extract_metrics_paths.py emitted no pairs for {run_path}")
        for gen, out in pairs:
            configuration = gen.parents[2].name
            run_key = (
                (folder, configuration)
                if configured_run_manifest
                else (folder, "")
            )
            if run_key not in run_by_key:
                raise RuntimeError(
                    f"emitted path has no run manifest row: {folder}/{configuration}"
                )
            run_row = run_by_key[run_key]
            source_job = generation_source_job_by_key[run_key]
            generation_started_at = generation_started_at_by_key[run_key]
            for label, path in (("gen", gen), ("out", out)):
                try:
                    path.relative_to(ROOT)
                except ValueError as error:
                    raise RuntimeError(
                        f"emitted {label} path escapes approved repository root: {path}"
                    ) from error
            if gen in seen_gen:
                raise RuntimeError(f"duplicate emitted gen path: {gen}")
            seen_gen.add(gen)
            if not gen.is_dir():
                raise RuntimeError(f"emitted gen directory does not exist: {gen}")
            actual_wav_paths = list(gen.glob("*.wav"))
            actual_wavs = {path.name for path in actual_wav_paths}
            if actual_wavs != expected_wavs:
                raise RuntimeError(
                    f"generated WAV inventory mismatch for {gen}: "
                    f"expected={len(expected_wavs)} actual={len(actual_wavs)} "
                    f"missing={sorted(expected_wavs - actual_wavs)[:20]} "
                    f"extra={sorted(actual_wavs - expected_wavs)[:20]}"
                )
            stale_wavs = sorted(
                path.name
                for path in actual_wav_paths
                if path.stat().st_mtime + 2
                < generation_started_at
            )
            if stale_wavs:
                raise RuntimeError(
                    f"generated WAVs predate source allocation {source_job} for {gen}: "
                    f"stale_count={len(stale_wavs)} examples={stale_wavs[:20]}"
                )
            path_rows.append(
                {
                    "index": len(path_rows),
                    "folder": folder,
                    "configuration": configuration,
                    "gen": str(gen),
                    "out": str(out),
                    "run_job_id": run_row["job_id"],
                    "freshness_anchor_job_id": source_job,
                    "freshness_anchor_start_epoch": (
                        f"{generation_started_at:.6f}"
                    ),
                }
            )
    path_fields = [
        "index", "folder", "configuration", "gen", "out", "run_job_id",
        "freshness_anchor_job_id", "freshness_anchor_start_epoch",
    ]
    actual_configurations = {
        folder: {
            str(row["configuration"])
            for row in path_rows
            if row["folder"] == folder
        }
        for folder in expected_folders
    }
    if actual_configurations != EXPECTED_CONFIGURATIONS or len(path_rows) != EXPECTED_PATHS:
        raise RuntimeError(
            "emitted configuration coverage mismatch: "
            f"expected={EXPECTED_CONFIGURATIONS} actual={actual_configurations} "
            f"path_count={len(path_rows)}/{EXPECTED_PATHS}"
        )
    write_tsv(STATE / "paths.tsv", path_rows, path_fields)

    command(
        [
            sys.executable,
            str(EXTRACT_LOSSES),
            "--run-jobs",
            str(RUN_JOBS_PATH),
            "--paths",
            str(STATE / "paths.tsv"),
            "--output",
            str(TRAINING_LOSS_PATH),
            "--coverage",
            str(STATE / "training_loss_coverage.json"),
        ]
    )
    if POST_LOSS_HOOK:
        command([sys.executable, POST_LOSS_HOOK])

    # Fail before submitting any scoring work if the legacy evaluation
    # environment has disappeared or no longer provides the established
    # F0/MCD dependencies.  ``-B`` keeps this read-only environment read-only.
    command(
        [
            str(METRICS_ENV_BIN / "python"),
            "-B",
            "-c",
            (
                "import fastdtw, librosa, numpy, pysptk, pyworld, "
                "scipy, soundfile"
            ),
        ]
    )

    # Validate publication barriers before the first scoring submission.  A
    # missing, malformed, or unsuccessful predecessor must fail closed without
    # leaving a partial redo metric/array set behind.  These barriers can be
    # old enough to have left Slurm's live controller even though sacct still
    # retains their authoritative terminal state, so validate them through
    # accounting here instead of trying to create a new scheduler dependency
    # on a purged job ID later.
    barrier_rows: list[dict[str, object]] = []
    if FINALIZER_BARRIER_MANIFEST:
        barrier_manifest = Path(FINALIZER_BARRIER_MANIFEST)
        source_rows = read_tsv(barrier_manifest)
        if len(source_rows) != 1:
            raise RuntimeError(
                "finalizer barrier manifest must contain exactly one row: "
                f"{barrier_manifest} has {len(source_rows)}"
            )
        barrier = source_rows[0]
        barrier_id = barrier.get("job_id", "")
        if barrier.get("kind") != "finalizer" or not barrier_id.isdigit():
            raise RuntimeError(
                f"invalid finalizer barrier row in {barrier_manifest}: {barrier}"
            )
        barrier_state, barrier_exit_code = job_state(barrier_id)
        if barrier_state != "COMPLETED" or barrier_exit_code != "0:0":
            raise RuntimeError(
                "required preceding finalizer did not succeed: "
                f"{barrier_id}={barrier_state}/{barrier_exit_code}"
            )
        barrier_rows.append(
            {
                "kind": "finalizer",
                "job_id": barrier_id,
                "source_manifest": str(barrier_manifest),
            }
        )

    started_at = time.time()
    (STATE / "dispatch_started_at.txt").write_text(f"{started_at:.6f}\n", encoding="utf-8")
    # Empty placeholders carry no submission evidence and can be replaced by
    # the first atomic journal write below.
    journal_path.unlink(missing_ok=True)
    metric_rows: list[dict[str, object]] = []
    for row in path_rows:
        index = int(row["index"])
        folder = str(row["folder"])
        metrics_path = ROOT / folder / "metrics.sh"
        update_metrics(metrics_path, Path(str(row["gen"])), Path(str(row["out"])))
        # Keep an immutable per-pair snapshot so the exact submitted mapping is
        # preserved on disk while the folder's metrics.sh advances to later
        # sweep pairs.
        metric_script_dir = METRIC_SCRIPTS / f"{index:02d}"
        metric_script_dir.mkdir(parents=True, exist_ok=True)
        metric_script = metric_script_dir / "metrics.sh"
        metric_text = metrics_path.read_text(encoding="utf-8")
        metric_lines = metric_text.splitlines(keepends=True)
        if not metric_lines or metric_lines[0].rstrip("\r\n") != "#!/bin/bash":
            raise RuntimeError(f"unexpected metrics.sh shebang in {metrics_path}")
        # Slurm only parses #SBATCH directives up to the first executable
        # shell line.  Insert strict mode after the full directive/blank
        # preamble so the established CPU and time requests remain active.
        strict_index = 1
        while strict_index < len(metric_lines) and (
            not metric_lines[strict_index].strip()
            or metric_lines[strict_index].lstrip().startswith("#SBATCH")
        ):
            strict_index += 1
        gt_indices = [
            line_index
            for line_index, line in enumerate(metric_lines)
            if line.startswith("gt=")
        ]
        if len(gt_indices) != 1 or gt_indices[0] < strict_index:
            raise RuntimeError(f"cannot locate one established gt assignment in {metrics_path}")
        # The source metrics.sh remains untouched apart from its required
        # gen/out assignments.  Its legacy module/conda setup is not portable
        # across the current EL9 compute nodes, so replace only that setup in
        # the immutable submitted snapshot with the verified interpreter.
        del metric_lines[strict_index:gt_indices[0]]
        if not (METRICS_ENV_BIN / "python").is_file():
            raise RuntimeError(
                f"missing established metrics interpreter: {METRICS_ENV_BIN / 'python'}"
            )
        metric_lines.insert(
            strict_index,
            # Legacy ``source activate`` hooks used by the established metric
            # scripts are not reliably compatible with shell nounset.  Keep
            # fail-fast and pipeline propagation without risking an activation
            # failure on an unset, optional conda variable.
            "set -eo pipefail\n"
            "export PYTHONNOUSERSITE=1\n"
            "export PYTHONDONTWRITEBYTECODE=1\n"
            f'export PATH="{METRICS_ENV_BIN}:$PATH"\n'
            "export XDG_CACHE_HOME=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/cache\n"
            "export MPLCONFIGDIR=\"$XDG_CACHE_HOME/matplotlib\"\n"
            "export NUMBA_CACHE_DIR=\"$XDG_CACHE_HOME/numba\"\n"
            # Python's multiprocessing.Manager creates an AF_UNIX listener
            # below TMPDIR.  The repository and cache paths are long enough
            # to exceed Linux's sockaddr_un limit, so keep only this process
            # temp root short while remaining inside the approved repository.
            "export TMPDIR=\"/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.t/m-${SLURM_JOB_ID:-manual}\"\n"
            "mkdir -p \"$XDG_CACHE_HOME\" \"$MPLCONFIGDIR\" \"$NUMBA_CACHE_DIR\" \"$TMPDIR\"\n"
            "python -B -c 'import fastdtw, librosa, numpy, pysptk, pyworld, scipy, soundfile'\n",
        )
        # F0 deliberately emits no per-WAV file when a sample has no jointly
        # voiced frames.  Remove only the generated metric sidecars and the
        # proof marker for this pair before evaluation so an older value
        # cannot mask a fresh result during final coverage validation.
        out_indices = [
            line_index
            for line_index, line in enumerate(metric_lines)
            if line.startswith("out=")
        ]
        if len(out_indices) != 1:
            raise RuntimeError(f"cannot locate one out assignment in {metrics_path}")
        cleanup_index = out_indices[0] + 1
        metric_lines.insert(
            cleanup_index,
            "\n"
            "mkdir -p \"$out\"\n"
            "for wav in \"$gen\"/*.wav; do\n"
            "  [ -e \"$wav\" ] || continue\n"
            "  rm -f \"${wav%.wav}_f0.txt\" \"${wav%.wav}_f0_missing.json\" \"${wav%.wav}_mcd.txt\"\n"
            "done\n",
        )
        f0_indices = [
            line_index
            for line_index, line in enumerate(metric_lines)
            if "tools/F0/F0.py" in line and not line.lstrip().startswith("#")
        ]
        if len(f0_indices) != 1:
            raise RuntimeError(f"cannot locate one established F0 command in {metrics_path}")
        metric_lines.insert(
            f0_indices[0] + 1,
            "python /mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/validate_f0_missing.py "
            "--evaluator \"$PWD/tools/F0/F0.py\" --gt \"$gt\" --gen \"$gen\"\n",
        )
        mcd_indices = [
            line_index
            for line_index, line in enumerate(metric_lines)
            if "tools/MCD/MCD.py" in line and not line.lstrip().startswith("#")
        ]
        if len(mcd_indices) != 1:
            raise RuntimeError(f"cannot locate one established MCD command in {metrics_path}")
        metric_lines.insert(
            mcd_indices[0] + 1,
            "python /mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/validate_mcd_missing.py "
            "--evaluator \"$PWD/tools/MCD/MCD.py\" --gt \"$gt\" --gen \"$gen\"\n",
        )
        metric_text = "".join(metric_lines)
        metric_script.write_text(metric_text, encoding="utf-8")
        command(["bash", "-n", str(metric_script)])
        # Record intent before sbatch.  If the dispatcher is interrupted in
        # the narrow window after Slurm accepts the job but before its ID is
        # persisted, the non-empty SUBMITTING row makes every automatic retry
        # fail closed instead of risking a duplicate scoring allocation.
        write_tsv(
            journal_path,
            metric_rows + [{"kind": "metrics", "index": index, "job_id": "SUBMITTING"}],
            ["kind", "index", "job_id"],
        )
        job_id = submit(
            [
                f"--chdir={ROOT / folder}",
                f"--job-name={JOB_PREFIX}_m_{index:02d}",
                f"--output={LOGS / f'metrics-{index:02d}-%j.out'}",
                f"--error={LOGS / f'metrics-{index:02d}-%j.err'}",
                str(metric_script),
            ]
        )
        row["metrics_job_id"] = job_id
        metric_rows.append({"kind": "metrics", "index": index, "job_id": job_id})
        write_tsv(journal_path, metric_rows, ["kind", "index", "job_id"])

    utmos_lists = STATE / "utmos_lists"
    utmos_lists.mkdir(parents=True, exist_ok=True)
    for row in path_rows:
        (utmos_lists / f"{int(row['index']):02d}.txt").write_text(
            str(row["gen"]) + "\n", encoding="utf-8"
        )

    last_index = len(path_rows) - 1
    write_tsv(
        journal_path,
        metric_rows + [{"kind": "wer", "index": "*", "job_id": "SUBMITTING"}],
        ["kind", "index", "job_id"],
    )
    wer_id = submit([f"--array=0-{last_index}%8", str(STATE / "wer_array.sbatch")])
    journal_rows = metric_rows + [{"kind": "wer", "index": "*", "job_id": wer_id}]
    write_tsv(journal_path, journal_rows, ["kind", "index", "job_id"])
    write_tsv(
        journal_path,
        journal_rows + [{"kind": "utmosv2", "index": "*", "job_id": "SUBMITTING"}],
        ["kind", "index", "job_id"],
    )
    utmos_id = submit([f"--array=0-{last_index}%8", str(STATE / "utmos_array.sbatch")])
    journal_rows.append({"kind": "utmosv2", "index": "*", "job_id": utmos_id})
    write_tsv(journal_path, journal_rows, ["kind", "index", "job_id"])
    for row in path_rows:
        row["wer_job_id"] = wer_id
        row["utmosv2_job_id"] = utmos_id
    enriched_fields = path_fields + ["metrics_job_id", "wer_job_id", "utmosv2_job_id"]
    write_tsv(STATE / "paths.tsv", path_rows, enriched_fields)

    downstream = metric_rows + [
        {"kind": "wer", "index": "*", "job_id": wer_id},
        {"kind": "utmosv2", "index": "*", "job_id": utmos_id},
    ]
    write_tsv(journal_path, downstream, ["kind", "index", "job_id"])
    write_tsv(downstream_path, downstream, ["kind", "index", "job_id"])
    dependency_ids = [str(row["job_id"]) for row in metric_rows] + [wer_id, utmos_id]
    if barrier_rows:
        barrier_id = str(barrier_rows[0]["job_id"])
        if barrier_id in dependency_ids:
            raise RuntimeError(
                f"finalizer barrier duplicates a scoring dependency: {barrier_id}"
            )
        write_tsv(
            STATE / "finalizer_barriers.tsv",
            barrier_rows,
            ["kind", "job_id", "source_manifest"],
        )
        # The predecessor was already proven COMPLETED/0:0 above and is
        # independently revalidated by finalize.py through this manifest.  Do
        # not add its historical ID to sbatch --dependency: Slurm rejects IDs
        # after they have been purged from the live controller.
    write_tsv(
        journal_path,
        downstream + [{"kind": "finalizer", "index": "*", "job_id": "SUBMITTING"}],
        ["kind", "index", "job_id"],
    )
    finalizer_id = submit(
        [
            "--dependency=afterany:" + ":".join(dependency_ids),
            str(STATE / "finalize.sbatch"),
        ]
    )
    write_tsv(
        journal_path,
        downstream + [{"kind": "finalizer", "index": "*", "job_id": finalizer_id}],
        ["kind", "index", "job_id"],
    )
    write_tsv(
        STATE / "finalizer_job.tsv",
        [{"kind": "finalizer", "job_id": finalizer_id, "dependency": ":".join(dependency_ids)}],
        ["kind", "job_id", "dependency"],
    )
    status_path.write_text(
        json.dumps(
            {
                "status": "downstream_submitted",
                "pairs": len(path_rows),
                "metric_jobs": len(metric_rows),
                "wer_array_job_id": wer_id,
                "utmosv2_array_job_id": utmos_id,
                "finalizer_job_id": finalizer_id,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"pairs={len(path_rows)} wer={wer_id} utmosv2={utmos_id} "
        f"finalizer={finalizer_id}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / "dispatch_failure.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        raise
