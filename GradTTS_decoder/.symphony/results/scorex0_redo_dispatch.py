#!/usr/bin/env python3
"""Dispatch the scorex0 redo downstream DAG after all prerequisite jobs finish."""

import csv
import datetime as dt
import hashlib
import os
import re
import stat
import subprocess
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet")
RES = ROOT / ".symphony/results"
TRANSCRIPT = Path(
    "/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/add/resources/"
    "filelists/ljspeech/test.txt"
)
EXPECTED_FOLDERS = {
    "add", "blur", "fm", "fmDT", "levy", "mask", "one-shot", "product",
    "score", "scoreDT", "scorex0",
}
EXPECTED_PAIRS = 30
SCOREX0_RUN_ID = "10943691"
SCOREX0_EPOCH495_ARRAY_ID = "10988661"
ORIGINAL_DISPATCH_ID = "10890394"
ORIGINAL_WER_ARRAY_ID = "10935480"
ORIGINAL_UTMOS_WARMUP_ID = "10935481"
ORIGINAL_UTMOS_ARRAY_ID = "10935482"
METRICS_ENV = "/users/acp23xt/.conda/envs/Grad-TTS-EVAL"


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def job_id(output):
    return output.strip().split(";", 1)[0]


def sbatch(args):
    output = subprocess.check_output(
        ["/usr/bin/sbatch", "--parsable"] + args, universal_newlines=True)
    return job_id(output)


def assert_under_root(value, label):
    try:
        Path(value).resolve().relative_to(ROOT.resolve())
    except ValueError:
        raise RuntimeError("{} is outside the approved repository: {}".format(label, value))


def extract_pairs(folder):
    output = subprocess.check_output([
        "/usr/bin/python3", str(ROOT / "extract_metrics_paths.py"),
        str(ROOT / folder / "run.sh"),
    ], universal_newlines=True)
    pairs = []
    gen = None
    for line in output.splitlines():
        if line.startswith("gen="):
            gen = line[4:]
        elif line.startswith("out="):
            if not gen:
                raise RuntimeError("out without gen while extracting {}".format(folder))
            pairs.append((gen, line[4:]))
            gen = None
    if gen is not None or not pairs:
        raise RuntimeError("incomplete or empty mapping for {}".format(folder))
    return pairs


def accounting(job_ids):
    """Return exact terminal sacct rows, keyed by the requested stable IDs."""
    requested = sorted(set(job_ids))
    output = subprocess.check_output([
        "/usr/bin/sacct", "-X", "-j", ",".join(requested),
        "--starttime=2026-07-01",
        "--format=JobID,JobIDRaw,State,ExitCode,Start,End", "-n", "-P",
    ], universal_newlines=True)
    records = [line.split("|") for line in output.splitlines() if line.strip()]
    exact = {}
    for requested_id in requested:
        row = next(
            (parts for parts in records
             if parts[0] == requested_id or parts[1] == requested_id), None)
        if row is None:
            raise RuntimeError("missing exact accounting record for {}".format(requested_id))
        state = row[2].split()[0].rstrip("+")
        if state != "COMPLETED" or row[3] != "0:0":
            raise RuntimeError("unsuccessful prerequisite {}: {}".format(requested_id, row))
        exact[requested_id] = row
    return exact


def parse_slurm_time(value):
    if not value or value == "Unknown":
        raise RuntimeError("missing Slurm timestamp")
    # The cluster's /usr/bin/python3 is Python 3.6, which predates
    # datetime.fromisoformat().  sacct emits Start as an ISO-like local time;
    # accept its normal second/fractional-second forms without relying on a
    # newer interpreter API.
    match = re.match(
        r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?$", value)
    if match is None:
        raise RuntimeError("unexpected Slurm timestamp: {}".format(value))
    return dt.datetime.strptime(
        match.group(1), "%Y-%m-%dT%H:%M:%S").timestamp()


def replace_assignment(data, name, value):
    pattern = re.compile(rb"(?m)^" + name.encode("ascii") + rb"=[^\r\n]*(\r?\n|$)")
    data, count = pattern.subn(
        lambda match: name.encode("ascii") + b"=" + value.encode("utf-8") + match.group(1),
        data,
    )
    if count != 1:
        raise RuntimeError("expected one {} assignment, found {}".format(name, count))
    return data


def ensure_fail_fast(data):
    if re.search(rb"(?m)^set -[^\r\n]*e[^\r\n]*$", data):
        return data
    lines = data.splitlines(keepends=True)
    if not lines or not lines[0].startswith(b"#!"):
        raise RuntimeError("metrics.sh must begin with a shebang")
    insert_at = 1
    while insert_at < len(lines):
        stripped = lines[insert_at].strip()
        if stripped and not stripped.startswith(b"#"):
            break
        insert_at += 1
    lines.insert(insert_at, b"set -eo pipefail\n")
    return b"".join(lines)


def ensure_metrics_environment(data):
    pattern = re.compile(rb"(?m)^source activate [^\r\n]+(\r?\n|$)")
    replacement = ("source activate {}".format(METRICS_ENV)).encode("utf-8")
    data, count = pattern.subn(
        lambda match: replacement + match.group(1), data)
    if count != 1:
        raise RuntimeError(
            "expected one metrics environment activation, found {}".format(count))
    return data


def render_redo_scoring_scripts():
    replacements = {
        "wer_array.sbatch": (
            "scorex0_redo_wer.sbatch",
            (("mappings.tsv", "scorex0_redo_mappings.tsv"),
             ("tan6_wer", "tan6_scorex0_wer_redo"),
             ("wer-%A_%a", "scorex0-redo-wer-%A_%a")),
        ),
        "utmos_array.sbatch": (
            "scorex0_redo_utmos.sbatch",
            (("mappings.tsv", "scorex0_redo_mappings.tsv"),
             ("tan6_utmos", "tan6_scorex0_utmos_redo"),
             ("utmos-%A_%a", "scorex0-redo-utmos-%A_%a"),
             ("utmos-dir-", "scorex0-redo-utmos-dir-")),
        ),
    }
    for source_name, (target_name, pairs) in replacements.items():
        source = (RES / source_name).read_text(encoding="utf-8")
        for old, new in pairs:
            if old not in source:
                raise RuntimeError("{} lost required token {}".format(source_name, old))
            source = source.replace(old, new)
        (RES / target_name).write_text(source, encoding="utf-8")


def main():
    actual_folders = {
        path.parent.name for path in ROOT.glob("*/run.sh")
        if path.parent.parent == ROOT
    }
    if actual_folders != EXPECTED_FOLDERS:
        raise RuntimeError(
            "first-depth run.sh inventory changed: expected={} actual={}".format(
                sorted(EXPECTED_FOLDERS), sorted(actual_folders)))

    mappings = read_tsv(RES / "mappings.tsv")
    jobs = read_tsv(RES / "jobs.tsv")
    if len(mappings) != EXPECTED_PAIRS or len(jobs) != EXPECTED_PAIRS * 5:
        raise RuntimeError("stored mapping or job ledger is incomplete")
    mapping_keys = {(row["folder"], row["pair_index"]) for row in mappings}
    if len(mapping_keys) != EXPECTED_PAIRS:
        raise RuntimeError("stored mapping keys are incomplete or duplicated")

    extracted = extract_pairs("scorex0")
    stored_scorex0 = [row for row in mappings if row["folder"] == "scorex0"]
    expected_scorex0 = [
        (row["gen"], row["out"]) for row in sorted(
            stored_scorex0, key=lambda row: int(row["pair_index"]))
    ]
    if len(extracted) != 5 or extracted != expected_scorex0:
        raise RuntimeError(
            "fresh scorex0 extraction changed: stored={} extracted={}".format(
                expected_scorex0, extracted))
    for gen, out in extracted:
        assert_under_root(gen, "scorex0 gen path")
        assert_under_root(out, "scorex0 out path")

    # The afterany dependencies guarantee terminal array parents.  Credit only
    # the 25 retained paths and reject any failed/cancelled non-scorex0 task.
    retained_jobs = [
        row["job_id"] for row in jobs
        if row["folder"] != "scorex0"
        and row["stage"] in {"metrics", "wer", "utmosv2"}
    ]
    prerequisite_ids = retained_jobs + [
        SCOREX0_RUN_ID, ORIGINAL_DISPATCH_ID, ORIGINAL_UTMOS_WARMUP_ID,
    ] + [
        "{}_{}".format(SCOREX0_EPOCH495_ARRAY_ID, index)
        for index in range(5)
    ]
    records = accounting(prerequisite_ids)
    run_start = parse_slurm_time(records[SCOREX0_RUN_ID][4])
    baseline_rows = read_tsv(RES / "scorex0_redo_training_baseline.tsv")
    baseline = {row["configuration"]: row for row in baseline_rows}
    if len(baseline) != 5:
        raise RuntimeError("scorex0 training baseline must contain exactly five configurations")

    transcript_rows = [
        line for line in TRANSCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_names = {Path(line.split("|", 1)[0]).name for line in transcript_rows}
    if len(transcript_rows) != 488 or len(expected_names) != 488:
        raise RuntimeError("transcript coverage is no longer exactly 488 unique WAVs")

    redo_rows = []
    for index, (gen, out) in enumerate(extracted):
        gen_path = Path(gen)
        out_path = Path(out)
        wavs = sorted(gen_path.glob("*.wav"))
        if {wav.name for wav in wavs} != expected_names:
            raise RuntimeError("fresh scorex0 WAV coverage mismatch for pair {}".format(index))
        stale_wavs = [str(wav) for wav in wavs if wav.stat().st_mtime < run_start - 300]
        if stale_wavs:
            raise RuntimeError(
                "scorex0 redo did not refresh every WAV for pair {}: {} stale".format(
                    index, len(stale_wavs)))
        configuration = gen_path.parents[2].name
        train_logs = list((ROOT / "scorex0").glob(
            "Hydra_*/{}/tb/train.log".format(configuration)))
        if len(train_logs) != 1 or train_logs[0].stat().st_mtime < run_start - 300:
            raise RuntimeError(
                "scorex0 redo did not refresh the training log for {}".format(configuration))
        baseline_row = baseline.get(configuration)
        if baseline_row is None or Path(baseline_row["source_log"]).resolve() != \
                train_logs[0].resolve():
            raise RuntimeError("missing exact pre-run baseline for {}".format(configuration))
        log_data = train_logs[0].read_bytes()
        baseline_size = int(baseline_row["bytes"])
        prefix_matches = (
            len(log_data) >= baseline_size
            and hashlib.sha256(log_data[:baseline_size]).hexdigest()
            == baseline_row["sha256"]
        )
        if len(log_data) == baseline_size and prefix_matches:
            raise RuntimeError(
                "scorex0 redo left the training log byte-for-byte unchanged for {}".format(
                    configuration))
        # The logger may append to the prior path or replace/truncate it.  In
        # the append case, require a complete fresh 1..599 sequence strictly
        # after the captured pre-run byte boundary.  In the replace case, the
        # complete current file is necessarily post-baseline and is checked.
        fresh_log_data = (
            log_data[baseline_size:]
            if prefix_matches and len(log_data) > baseline_size
            else log_data
        )
        fresh_log = fresh_log_data.decode("utf-8", errors="replace")
        fresh_epoch_matches = re.findall(
            r"(?m)^Epoch (\d+): duration loss = ", fresh_log)
        train_epochs = {
            int(value) for value in re.findall(
                r"(?m)^Epoch (\d+): duration loss = ",
                fresh_log)
        }
        if train_epochs != set(range(1, 600)) or len(fresh_epoch_matches) < 599:
            raise RuntimeError(
                "scorex0 redo has no complete fresh training phase for {}".format(
                    configuration))

        # Remove only derived scorex0 values so a failed redo cannot pass by
        # inheriting sidecars or aggregate lines from the superseded run.
        for suffix in ("_mcd.txt", "_f0.txt", "_wer_l.txt", "_utmosv2.txt"):
            for sidecar in gen_path.glob("*" + suffix):
                sidecar.unlink()
        out_path.mkdir(parents=True, exist_ok=True)
        for aggregate in (out_path / "MCD.txt", out_path / "F0.txt"):
            if aggregate.exists():
                aggregate.unlink()
        redo_rows.append({
            "folder": "scorex0", "pair_index": str(index),
            "gen": gen, "out": out,
        })

    write_tsv(
        RES / "scorex0_redo_mappings.tsv",
        ["folder", "pair_index", "gen", "out"], redo_rows)
    render_redo_scoring_scripts()

    metric_ids = []
    metrics_script = ROOT / "scorex0/metrics.sh"
    metrics_tmp = RES / "tmp/metrics"
    metrics_cache = RES / "cache/metrics"
    metrics_config = RES / "config/metrics"
    for path in (metrics_tmp, metrics_cache, metrics_config / "matplotlib"):
        path.mkdir(parents=True, exist_ok=True)
    for row in redo_rows:
        data = ensure_metrics_environment(ensure_fail_fast(metrics_script.read_bytes()))
        data = replace_assignment(data, "gen", row["gen"])
        data = replace_assignment(data, "out", row["out"])
        metrics_script.write_bytes(data)
        # Keep an explicit per-pair script beside the orchestration records.
        # Slurm normally ingests a script at submission time, but immutable
        # snapshots make the exact payload for each allocation independently
        # inspectable and avoid relying on a shared path's later contents.
        submitted_metrics_script = RES / "scorex0_redo_metrics_{}.sh".format(
            row["pair_index"])
        submitted_metrics_script.write_bytes(data)
        submitted_metrics_script.chmod(
            stat.S_IMODE(metrics_script.stat().st_mode))
        mid = sbatch([
            "--chdir={}".format(ROOT / "scorex0"),
            "--job-name=tan6_scorex0_metrics_redo_{}".format(row["pair_index"]),
            "--output={}".format(
                RES / "scorex0-redo-metrics-{}-%j.out".format(row["pair_index"])),
            "--error={}".format(
                RES / "scorex0-redo-metrics-{}-%j.err".format(row["pair_index"])),
            "--export=ALL,TMPDIR={},XDG_CACHE_HOME={},XDG_CONFIG_HOME={},"
            "MPLCONFIGDIR={},PYTHONDONTWRITEBYTECODE=1".format(
                metrics_tmp, metrics_cache, metrics_config,
                metrics_config / "matplotlib"),
            str(submitted_metrics_script),
        ])
        metric_ids.append(mid)

    wer_id = sbatch(["--array=0-4", str(RES / "scorex0_redo_wer.sbatch")])
    utmos_id = sbatch(["--array=0-4", str(RES / "scorex0_redo_utmos.sbatch")])
    loss_id = sbatch([
        "--partition=sheffield", "--cpus-per-task=1", "--mem=4G",
        "--time=01:00:00", "--job-name=tan6_training_loss_redo",
        "--output={}".format(RES / "training-loss-redo-%j.out"),
        "--error={}".format(RES / "training-loss-redo-%j.err"),
        str(RES / "training-loss3.batch"),
    ])

    # Replace only the five scorex0 records; retain the already successful 25
    # paths and their exact task IDs.
    metric_lookup = {str(index): value for index, value in enumerate(metric_ids)}
    for row in jobs:
        if row["folder"] != "scorex0":
            continue
        index = row["pair_index"]
        if row["stage"] == "run":
            row["job_id"] = SCOREX0_RUN_ID
        elif row["stage"] == "gen_repair":
            row["job_id"] = "{}_{}".format(
                SCOREX0_EPOCH495_ARRAY_ID, index)
        elif row["stage"] == "metrics":
            row["job_id"] = metric_lookup[index]
        elif row["stage"] == "wer":
            row["job_id"] = "{}_{}".format(wer_id, index)
        elif row["stage"] == "utmosv2":
            row["job_id"] = "{}_{}".format(utmos_id, index)
    write_tsv(
        RES / "jobs.tsv",
        ["folder", "pair_index", "stage", "job_id", "gen", "out"], jobs)

    repair_rows = read_tsv(RES / "gen_repair_jobs.tsv")
    for row in repair_rows:
        if row["folder"] == "scorex0":
            row["job_id"] = "{}_{}".format(
                SCOREX0_EPOCH495_ARRAY_ID, row["pair_index"])
    write_tsv(
        RES / "gen_repair_jobs.tsv", ["folder", "pair_index", "job_id"],
        repair_rows)

    summary_dependency = ":".join(metric_ids + [wer_id, utmos_id, loss_id])
    summary_id = sbatch([
        "--dependency=afterok:{}".format(summary_dependency),
        str(RES / "validate_summary.sbatch"),
    ])
    final_id = sbatch([
        "--dependency=afterok:{}".format(summary_id),
        str(RES / "final_audit.sbatch"),
    ])

    dispatch_id = os.environ["SLURM_JOB_ID"]
    pipeline_values = [
        ("dispatch_job_id", dispatch_id),
        ("wer_array_job_id", wer_id),
        ("utmosv2_warmup_job_id", ORIGINAL_UTMOS_WARMUP_ID),
        ("utmosv2_array_job_id", utmos_id),
        ("summary_job_id", summary_id),
        ("pair_count", str(EXPECTED_PAIRS)),
        ("original_dispatch_job_id", ORIGINAL_DISPATCH_ID),
        ("original_wer_array_job_id", ORIGINAL_WER_ARRAY_ID),
        ("original_utmosv2_array_job_id", ORIGINAL_UTMOS_ARRAY_ID),
        ("scorex0_run_job_id", SCOREX0_RUN_ID),
        ("training_loss_job_id", loss_id),
    ]
    with (RES / "pipeline_job_ids.tsv").open("w", encoding="utf-8") as handle:
        for key, value in pipeline_values:
            handle.write("{}\t{}\n".format(key, value))
    with (RES / "final_audit_job_id.tsv").open("w", encoding="utf-8") as handle:
        handle.write("final_audit_job_id\t{}\n".format(final_id))
        handle.write("loss_job_id\t{}\n".format(loss_id))
    with (RES / "scorex0_redo_job_ids.tsv").open("w", encoding="utf-8") as handle:
        handle.write("stage\tjob_id\n")
        handle.write("scorex0_run\t{}\n".format(SCOREX0_RUN_ID))
        handle.write("scorex0_epoch495_array\t{}\n".format(
            SCOREX0_EPOCH495_ARRAY_ID))
        handle.write("redo_dispatch\t{}\n".format(dispatch_id))
        for index, mid in enumerate(metric_ids):
            handle.write("scorex0_metrics_{}\t{}\n".format(index, mid))
        handle.write("scorex0_wer_array\t{}\n".format(wer_id))
        handle.write("scorex0_utmosv2_array\t{}\n".format(utmos_id))
        handle.write("training_loss\t{}\n".format(loss_id))
        handle.write("summary\t{}\n".format(summary_id))
        handle.write("final_audit\t{}\n".format(final_id))
    print(
        "scorex0 redo dispatch complete run={} metrics={} wer={} utmos={} "
        "loss={} summary={} final={}".format(
            SCOREX0_RUN_ID, ",".join(metric_ids), wer_id, utmos_id,
            loss_id, summary_id, final_id),
        flush=True,
    )


if __name__ == "__main__":
    main()
