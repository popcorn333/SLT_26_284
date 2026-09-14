#!/usr/bin/env python3
"""Recover TAN-7 F0/MCD jobs after the common AF_UNIX TMPDIR failure."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
STATE = ROOT / ".symphony/tan7"
LOGS = STATE / "logs"
EXPECTED_PATHS = 29
OLD_FINALIZER = "10944212"
SHORT_TMPDIR = (
    'export TMPDIR="/mnt/parscratch/users/acp23xt/private/'
    'DTDM_sgmse/.t/m-${SLURM_JOB_ID:-manual}"'
)
TERMINAL = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"missing TSV header: {path}")
        return list(reader), list(reader.fieldnames)


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
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


def submit(args: list[str]) -> str:
    output = command(["sbatch", "--parsable", *args]).strip()
    if not output:
        raise RuntimeError(f"sbatch returned no job ID: {args}")
    job_id = output.split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid sbatch job ID: {output}")
    return job_id


def allocation_state(job_id: str) -> tuple[str, str]:
    output = command(
        [
            "sacct", "-X", "-n", "-P", "-j", job_id,
            "--format=JobIDRaw,State,ExitCode",
        ]
    )
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[0] == job_id:
            return fields[1].rstrip("+").split()[0], fields[2]
    raise RuntimeError(f"missing Slurm allocation row for {job_id}")


def main() -> None:
    dispatcher_id = os.environ.get("SLURM_JOB_ID")
    if not dispatcher_id:
        raise RuntimeError("metric recovery dispatcher must run in Slurm")

    retry_status_path = STATE / "metric_retry_status.json"
    retry_journal_path = STATE / "metric_retry_submit_journal.tsv"
    if retry_status_path.is_file():
        status = json.loads(retry_status_path.read_text(encoding="utf-8"))
        if status.get("status") == "submitted":
            print("metric_retry_already_submitted=true", flush=True)
            return
        raise RuntimeError(f"unexpected existing retry status: {status}")
    if retry_journal_path.is_file() and retry_journal_path.stat().st_size:
        raise RuntimeError(
            "refusing duplicate recovery after a possible partial retry dispatch: "
            f"{retry_journal_path}"
        )

    downstream, downstream_fields = read_tsv(STATE / "downstream_jobs.tsv")
    paths, path_fields = read_tsv(STATE / "paths.tsv")
    metrics = [row for row in downstream if row.get("kind") == "metrics"]
    wer = [row for row in downstream if row.get("kind") == "wer"]
    utmos = [row for row in downstream if row.get("kind") == "utmosv2"]
    if (
        len(metrics) != EXPECTED_PATHS
        or len(paths) != EXPECTED_PATHS
        or len(wer) != 1
        or len(utmos) != 1
        or sorted(int(row["index"]) for row in metrics)
        != list(range(EXPECTED_PATHS))
        or sorted(int(row["index"]) for row in paths)
        != list(range(EXPECTED_PATHS))
    ):
        raise RuntimeError("current 29-path downstream/path manifests are incomplete")
    metrics_by_index = {int(row["index"]): row for row in metrics}
    paths_by_index = {int(row["index"]): row for row in paths}
    mismatches = [
        index
        for index in range(EXPECTED_PATHS)
        if paths_by_index[index].get("metrics_job_id")
        != metrics_by_index[index].get("job_id")
    ]
    if mismatches:
        raise RuntimeError(f"pre-retry metric provenance mismatch: {mismatches}")

    for index in range(EXPECTED_PATHS):
        old_id = metrics_by_index[index]["job_id"]
        state, exit_code = allocation_state(old_id)
        if (state, exit_code) != ("FAILED", "1:0"):
            raise RuntimeError(
                f"metric job {index}/{old_id} is not the known safe failure: "
                f"{state}/{exit_code}"
            )
        error_log = LOGS / f"metrics-{index:02d}-{old_id}.err"
        error_text = error_log.read_text(encoding="utf-8", errors="replace")
        if "OSError: AF_UNIX path too long" not in error_text:
            raise RuntimeError(
                f"metric job {index}/{old_id} lacks the common AF_UNIX proof"
            )
        metric_script = STATE / "metrics_scripts" / f"{index:02d}" / "metrics.sh"
        script_text = metric_script.read_text(encoding="utf-8")
        if script_text.count(SHORT_TMPDIR) != 1:
            raise RuntimeError(f"metric script {index} lacks the short TMPDIR fix")
        if '$XDG_CACHE_HOME/tmp/metrics-${SLURM_JOB_ID:-manual}' in script_text:
            raise RuntimeError(f"metric script {index} retains the long TMPDIR")
        command(["bash", "-n", str(metric_script)])

    for kind, rows in (("wer", wer), ("utmosv2", utmos)):
        array_id = rows[0]["job_id"]
        state, exit_code = allocation_state(array_id)
        if state in TERMINAL and (state, exit_code) != ("COMPLETED", "0:0"):
            raise RuntimeError(
                f"cannot recover metrics while {kind} array failed: "
                f"{array_id}={state}/{exit_code}"
            )

    finalizer_rows, _ = read_tsv(STATE / "finalizer_job.tsv")
    if len(finalizer_rows) != 1 or finalizer_rows[0].get("job_id") != OLD_FINALIZER:
        raise RuntimeError(f"unexpected held finalizer manifest: {finalizer_rows}")
    old_finalizer_state, _ = allocation_state(OLD_FINALIZER)
    intent_path = STATE / "metric_retry_intent.json"
    intent_path.write_text(
        json.dumps(
            {
                "dispatcher_job_id": dispatcher_id,
                "failed_metric_jobs": [row["job_id"] for row in metrics],
                "obsolete_finalizer_job_id": OLD_FINALIZER,
                "reason": "AF_UNIX path too long in every F0 job",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if old_finalizer_state == "PENDING":
        command(["scancel", OLD_FINALIZER])
    elif old_finalizer_state != "CANCELLED":
        raise RuntimeError(
            f"obsolete finalizer is not safely cancellable: "
            f"{OLD_FINALIZER}={old_finalizer_state}"
        )

    retry_rows: list[dict[str, object]] = []
    for index in range(EXPECTED_PATHS):
        folder = paths_by_index[index]["folder"]
        script = STATE / "metrics_scripts" / f"{index:02d}" / "metrics.sh"
        write_tsv(
            retry_journal_path,
            retry_rows
            + [{"kind": "metrics", "index": index, "job_id": "SUBMITTING"}],
            ["kind", "index", "job_id"],
        )
        retry_id = submit(
            [
                f"--chdir={ROOT / folder}",
                f"--job-name=tan7_mr_{index:02d}",
                f"--output={LOGS / f'metrics-retry-{index:02d}-%j.out'}",
                f"--error={LOGS / f'metrics-retry-{index:02d}-%j.err'}",
                str(script),
            ]
        )
        retry_rows.append(
            {"kind": "metrics", "index": index, "job_id": retry_id}
        )
        write_tsv(
            retry_journal_path,
            retry_rows,
            ["kind", "index", "job_id"],
        )

    retry_by_index = {int(row["index"]): str(row["job_id"]) for row in retry_rows}
    for path in paths:
        path["metrics_job_id"] = retry_by_index[int(path["index"])]
    write_tsv(STATE / "paths.tsv", paths, path_fields)

    current_downstream = retry_rows + [wer[0], utmos[0]]
    write_tsv(STATE / "downstream_jobs.tsv", current_downstream, downstream_fields)
    dependency_ids = [str(row["job_id"]) for row in current_downstream]
    write_tsv(
        STATE / "downstream_submit_journal.tsv",
        current_downstream
        + [{"kind": "finalizer", "index": "*", "job_id": "SUBMITTING"}],
        ["kind", "index", "job_id"],
    )
    replacement_finalizer = submit(
        [
            "--dependency=afterany:" + ":".join(dependency_ids),
            str(STATE / "finalize.sbatch"),
        ]
    )
    current_journal = current_downstream + [
        {"kind": "finalizer", "index": "*", "job_id": replacement_finalizer}
    ]
    write_tsv(
        STATE / "downstream_submit_journal.tsv",
        current_journal,
        ["kind", "index", "job_id"],
    )
    write_tsv(
        STATE / "finalizer_job.tsv",
        [
            {
                "kind": "finalizer",
                "job_id": replacement_finalizer,
                "dependency": ":".join(dependency_ids),
            }
        ],
        ["kind", "job_id", "dependency"],
    )
    provenance = [
        {
            "kind": "metrics",
            "index": index,
            "failed_job_id": metrics_by_index[index]["job_id"],
            "retry_job_id": retry_by_index[index],
            "reason": "AF_UNIX path too long; retry uses short repository-local TMPDIR",
        }
        for index in range(EXPECTED_PATHS)
    ]
    provenance.append(
        {
            "kind": "finalizer",
            "index": "*",
            "failed_job_id": OLD_FINALIZER,
            "retry_job_id": replacement_finalizer,
            "reason": "obsolete held finalizer replaced with dependencies on metric retries",
        }
    )
    write_tsv(
        STATE / "metric_retry_jobs.tsv",
        provenance,
        ["kind", "index", "failed_job_id", "retry_job_id", "reason"],
    )

    dispatch_status = {
        "status": "downstream_submitted",
        "pairs": EXPECTED_PATHS,
        "metric_jobs": EXPECTED_PATHS,
        "metric_retry": True,
        "metric_retry_dispatcher_job_id": dispatcher_id,
        "wer_array_job_id": wer[0]["job_id"],
        "utmosv2_array_job_id": utmos[0]["job_id"],
        "finalizer_job_id": replacement_finalizer,
        "superseded_finalizer_job_id": OLD_FINALIZER,
    }
    (STATE / "dispatch_status.json").write_text(
        json.dumps(dispatch_status, indent=2) + "\n", encoding="utf-8"
    )
    retry_status = {
        "status": "submitted",
        "dispatcher_job_id": dispatcher_id,
        "retry_metric_job_ids": [retry_by_index[index] for index in range(EXPECTED_PATHS)],
        "wer_array_job_id": wer[0]["job_id"],
        "utmosv2_array_job_id": utmos[0]["job_id"],
        "replacement_finalizer_job_id": replacement_finalizer,
        "superseded_finalizer_job_id": OLD_FINALIZER,
    }
    retry_status_path.write_text(
        json.dumps(retry_status, indent=2) + "\n", encoding="utf-8"
    )

    corrections_path = STATE / "orchestration_corrections.tsv"
    corrections, correction_fields = read_tsv(corrections_path)
    timestamp = datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds")
    corrections.extend(
        [
            {
                "timestamp_bst": timestamp,
                "job_id": OLD_FINALIZER,
                "old_dependency": finalizer_rows[0].get("dependency", ""),
                "new_dependency": "cancelled after common metric failure",
                "reason": (
                    "All 29 metric dependencies failed with the independently "
                    "confirmed AF_UNIX TMPDIR-length error"
                ),
            },
            {
                "timestamp_bst": timestamp,
                "job_id": replacement_finalizer,
                "old_dependency": "none",
                "new_dependency": "afterany:" + ":".join(dependency_ids),
                "reason": (
                    "Replacement finalizer waits for all 29 short-TMPDIR metric "
                    "retries and the existing WER/UTMOSv2 arrays"
                ),
            },
        ]
    )
    write_tsv(corrections_path, corrections, correction_fields)
    print(
        "metric_retries="
        + ",".join(retry_by_index[index] for index in range(EXPECTED_PATHS))
        + f" wer={wer[0]['job_id']} utmosv2={utmos[0]['job_id']} "
        + f"finalizer={replacement_finalizer}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (STATE / "metric_retry_dispatch_failure.txt").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        raise
