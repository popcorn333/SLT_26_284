#!/usr/bin/env python3
"""Recover TAN-7 dispatch after a successful inference-only continuation."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
STATE = ROOT / ".symphony/tan7"
LOGS = STATE / "logs"
EFFECTIVE_JOBS = STATE / "run_jobs_effective.tsv"
TERMINAL = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    fields = [
        "folder", "run_path", "job_id", "stdout", "stderr", "training_job_id"
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def job_state(job_id: str) -> tuple[str, str]:
    # An afterany dependency can release before the terminal allocation row is
    # visible to sacct.  Keep this wait short and inside the Slurm recovery job
    # so a successful inference fallback is not rejected as UNKNOWN merely due
    # to accounting publication lag.
    latest = ("UNKNOWN", "")
    for attempt in range(12):
        result = subprocess.run(
            [
                "sacct", "-X", "-n", "-P", "-j", job_id,
                "--format=JobIDRaw,State,ExitCode",
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                fields = line.split("|")
                if len(fields) >= 3 and fields[0] == job_id:
                    latest = (fields[1].rstrip("+").split()[0], fields[2])
                    if latest[0] in TERMINAL:
                        return latest
                    break
        if attempt < 11:
            time.sleep(10)
    return latest


def succeeded(job_id: str) -> bool:
    return job_state(job_id) == ("COMPLETED", "0:0")


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("TAN-7 recovery dispatcher must run inside a Slurm allocation")
    # Keep retry diagnostics authoritative: a successful retry should not
    # coexist with a failure marker from an earlier transient attempt.
    (STATE / "recovery_dispatch_failure.txt").unlink(missing_ok=True)
    status_path = STATE / "dispatch_status.json"
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
        if status.get("status") == "downstream_submitted":
            print("normal_dispatch_already_succeeded=true", flush=True)
            return

    # A submission journal without the final status can mean the first
    # dispatcher stopped after creating some jobs. Never duplicate that work.
    for artifact in (STATE / "downstream_jobs.tsv", STATE / "downstream_submit_journal.tsv"):
        if artifact.is_file() and artifact.stat().st_size:
            raise RuntimeError(
                f"refusing automatic recovery after a possible partial dispatch: {artifact}"
            )

    original_rows = read_tsv(STATE / "run_jobs.tsv")
    fallback_rows = {
        row["folder"]: row for row in read_tsv(STATE / "inference_fallback_jobs.tsv")
    }
    effective_rows: list[dict[str, str]] = []
    selections: list[str] = []
    for row in original_rows:
        folder = row["folder"]
        original_id = row["job_id"]
        original_state = job_state(original_id)
        if original_state == ("COMPLETED", "0:0"):
            effective_rows.append({**row, "training_job_id": original_id})
            selections.append(f"{folder}=original:{original_id}")
            continue
        fallback = fallback_rows.get(folder)
        if fallback is None:
            raise RuntimeError(
                f"no safe fallback for failed run {folder}={original_id}:{original_state}"
            )
        fallback_id = fallback["fallback_job_id"]
        fallback_state = job_state(fallback_id)
        if fallback_state != ("COMPLETED", "0:0"):
            raise RuntimeError(
                f"fallback did not succeed for {folder}: original={original_id}:{original_state} "
                f"fallback={fallback_id}:{fallback_state}"
            )
        effective_rows.append(
            {
                "folder": folder,
                "run_path": row["run_path"],
                "job_id": fallback_id,
                # The fallback is deliberately inference-only: its success is
                # the effective run state, but it contains no epoch losses.
                # Keep the original run logs in the effective manifest so
                # extract_losses.py still validates the completed training
                # phase instead of reading an inference-only stdout file.
                "stdout": row["stdout"],
                "stderr": row["stderr"],
                "training_job_id": original_id,
            }
        )
        selections.append(f"{folder}=fallback:{fallback_id}(parent={original_id})")

    if len(effective_rows) != len(original_rows):
        raise RuntimeError(
            f"effective run manifest is incomplete: {len(effective_rows)}/{len(original_rows)}"
        )
    write_tsv(EFFECTIVE_JOBS, effective_rows)
    print("effective_runs=" + ",".join(selections), flush=True)

    environment = os.environ.copy()
    environment["TAN7_RUN_JOBS"] = str(EFFECTIVE_JOBS)
    subprocess.run(
        [sys.executable, str(STATE / "dispatch.py")],
        cwd=ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (STATE / "recovery_dispatch_failure.txt").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        raise
