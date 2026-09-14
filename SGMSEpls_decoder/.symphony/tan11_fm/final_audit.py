#!/usr/bin/env python3
"""Independently audit and seal the final 33-path TAN-11 publication."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
STATE = ROOT / ".symphony/tan11_fm"
RESULTS = ROOT / ".symphony/results"
CSV_PATH = RESULTS / "tan11_summary.csv"
MD_PATH = RESULTS / "tan11_summary.md"
COVERAGE_PATH = RESULTS / "tan11_coverage.json"
AUDIT_PATH = RESULTS / "tan11_final_audit.json"
TRANSCRIPT = Path(
    "/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/"
    "add/resources/filelists/ljspeech/test.txt"
)
EXPECTED = {
    "add": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6"},
    "blur": {"model.masking.b:10", "model.masking.b:20", "model.masking.b:40"},
    "fm": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6", "model.masking.a:0.8", "model.masking.a:1"},
    "fmDT": {"model.masking.a:1"},
    "levy": {"model.masking.alpha:1.9"},
    "mask": {"model.masking.a:1"},
    "one-shot": {"model.masking.a:0.2"},
    "product": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6"},
    "score": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6", "model.masking.a:0.8", "model.masking.a:1"},
    "scoreDT": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6", "model.masking.a:0.8", "model.masking.a:1"},
    "scorex0": {"model.masking.a:0.2", "model.masking.a:0.4", "model.masking.a:0.6", "model.masking.a:0.8", "model.masking.a:1"},
}
SUFFIXES = {
    "mcd": "_mcd.txt", "logf0": "_f0.txt", "wer": "_wer_l.txt",
    "utmosv2": "_utmosv2.txt",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_number(path: Path) -> float:
    tokens = path.read_text(encoding="utf-8").split()
    if len(tokens) != 1:
        raise RuntimeError(f"sidecar does not contain one number: {path}")
    value = float(tokens[0])
    if not math.isfinite(value):
        raise RuntimeError(f"sidecar is nonfinite: {path}")
    return value


def stats(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    return (
        f"{statistics.fmean(values):.9g}",
        f"{statistics.pstdev(values) if len(values) > 1 else 0.0:.9g}",
    )


def parse_pairs(text: str) -> list[tuple[str, str]]:
    pairs = []
    gen = out = None
    for line in [*text.splitlines(), ""]:
        if line.startswith("gen="):
            gen = line[4:]
        elif line.startswith("out="):
            out = line[4:]
        elif not line.strip() and gen and out:
            pairs.append((gen, out))
            gen = out = None
    return pairs


def main() -> None:
    finalizer = os.environ.get("SLURM_JOB_ID")
    if not finalizer:
        raise RuntimeError("final audit must run inside the Slurm finalizer")
    (STATE / "final_audit_failure.txt").unlink(missing_ok=True)
    folders = sorted(
        path.parent.name for path in ROOT.glob("*/run.sh")
        if path.parent.parent == ROOT
    )
    if folders != sorted(EXPECTED):
        raise RuntimeError(f"final first-depth run.sh inventory mismatch: {folders}")

    rows = read_csv(CSV_PATH)
    if len(rows) != 33 or sorted(int(row["path_index"]) for row in rows) != list(range(33)):
        raise RuntimeError("final summary is not exactly 33 indexed paths")
    actual = defaultdict(set)
    for row in rows:
        actual[row["folder"]].add(row["configuration"])
    if dict(actual) != EXPECTED:
        raise RuntimeError(f"final configuration coverage mismatch: {dict(actual)}")

    emitted = set()
    for folder in folders:
        result = subprocess.run(
            [sys.executable, str(ROOT / "extract_metrics_paths.py"), str(ROOT / folder / "run.sh")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        for gen, out in parse_pairs(result.stdout):
            configuration = Path(gen).parents[2].name
            emitted.add((folder, configuration, str(Path(gen).resolve()), str(Path(out).resolve())))
    published = {
        (row["folder"], row["configuration"], row["gen_path"], row["out_path"])
        for row in rows
    }
    if emitted != published or len(emitted) != 33:
        raise RuntimeError("current run.sh emitted paths do not exactly match the final report")

    expected_wavs = {
        Path(line.split("|", 1)[0]).name
        for line in TRANSCRIPT.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    if len(expected_wavs) != 488:
        raise RuntimeError(f"transcript inventory is {len(expected_wavs)}, not 488")
    audits = []
    for row in rows:
        gen = Path(row["gen_path"])
        wavs = sorted(gen.glob("*.wav"))
        if {wav.name for wav in wavs} != expected_wavs or int(row["wav_count"]) != 488:
            raise RuntimeError(f"WAV coverage mismatch: {gen}")
        metric_audit = {}
        for metric, suffix in SUFFIXES.items():
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
                    data = json.loads(marker.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    missing.append(f"{wav.name}:marker_invalid")
                    continue
                if (
                    data.get("reason") == "no_joint_voiced_frames"
                    and data.get("joint_voiced_frames") == 0
                    and data.get("wav") == str(wav)
                ):
                    missing.append(f"{wav.name}:genuine_undefined_no_joint_voiced_frames")
                else:
                    missing.append(f"{wav.name}:marker_unproven")
            recorded = json.loads(row[f"{metric}_missing"])
            if missing != recorded:
                raise RuntimeError(f"missing-value mismatch for {row['folder']}/{row['configuration']}/{metric}")
            if metric != "logf0" and missing:
                raise RuntimeError(f"unexpected missing {metric} outputs for {gen}")
            if metric == "logf0" and any(
                not item.endswith(":genuine_undefined_no_joint_voiced_frames") for item in missing
            ):
                raise RuntimeError(f"unproven logF0 missing value for {gen}")
            if len(values) + len(missing) != 488:
                raise RuntimeError(f"incomplete {metric} coverage for {gen}")
            mean, std = stats(values)
            if (
                int(row[f"{metric}_count"]) != len(values)
                or row[f"{metric}_mean"] != mean
                or row[f"{metric}_std"] != std
            ):
                raise RuntimeError(f"published statistics mismatch for {gen}/{metric}")
            metric_audit[metric] = {
                "count": len(values), "missing": len(missing),
                "mean": mean, "population_std": std,
            }
        audits.append({
            "folder": row["folder"], "configuration": row["configuration"],
            "path_index": int(row["path_index"]), "metrics": metric_audit,
        })

    loss_rows = read_tsv(ROOT / "training_loss.log")
    if len(loss_rows) != 39 * 599:
        raise RuntimeError(f"training_loss.log has {len(loss_rows)}, not 23361 records")
    groups = Counter()
    epochs = defaultdict(set)
    for number, row in enumerate(loss_rows, 2):
        key = (row["experiment"], row["configuration"], row["run_job_id"], row["run_kind"])
        epoch = int(row["epoch"])
        values = [float(row[name]) for name in (
            "duration_loss", "prior_loss", "diffusion_loss", "loss"
        )]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"nonfinite loss at row {number}")
        if not math.isclose(values[3], sum(values[:3]), rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(f"combined loss mismatch at row {number}")
        groups[key] += 1
        epochs[key].add(epoch)
    if len(groups) != 39 or set(groups.values()) != {599}:
        raise RuntimeError("training losses are not exactly 39 complete configuration-runs")
    if any(values != set(range(1, 600)) for values in epochs.values()):
        raise RuntimeError("training loss epoch coverage is not exactly 1..599")
    tan11_groups = {
        key for key in groups if key[2] == "11097462" and key[3] == "tan11_fm"
    }
    if tan11_groups != {
        ("fm", f"model.masking.a:{value}", "11097462", "tan11_fm")
        for value in ("0.2", "0.4", "0.6", "0.8", "1")
    }:
        raise RuntimeError("TAN-11 FM training-loss provenance is incomplete")

    coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    if coverage.get("status") != "complete" or coverage.get("fatal_coverage") or len(coverage.get("paths", [])) != 33:
        raise RuntimeError("final coverage document is incomplete")
    status_path = STATE / "final_status.json"
    pending = json.loads(status_path.read_text(encoding="utf-8"))
    if pending.get("status") != "merged_pending_final_audit" or pending.get("paths") != 33:
        raise RuntimeError(f"unexpected pre-audit status: {pending}")
    audit = {
        "status": "complete",
        "folders": folders,
        "paths": 33,
        "wavs_per_path": 488,
        "training_loss_records": len(loss_rows),
        "training_configuration_runs": len(groups),
        "finalizer_job_id": finalizer,
        "rows": audits,
    }
    audit_tmp = AUDIT_PATH.with_name(AUDIT_PATH.name + ".tmp")
    audit_tmp.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    complete = {
        **pending,
        "status": "complete",
        "audit": str(AUDIT_PATH),
    }
    status_tmp = status_path.with_name(status_path.name + ".tmp")
    status_tmp.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    audit_tmp.replace(AUDIT_PATH)
    status_tmp.replace(status_path)
    print("final_audit=complete folders=11 paths=33 training_loss_records=23361")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (STATE / "final_audit_failure.txt").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        raise
