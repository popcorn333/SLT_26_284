#!/usr/bin/env python3
"""Fail-closed terminal audit for TAN-18's fmDT and scorex0 scope."""

import csv
import io
import math
import os
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet")
RESULTS = ROOT / ".symphony" / "results"
EXPECTED_SWEEPS = tuple(
    "model.masking.a:" + value for value in ("0.2", "0.4", "0.6", "0.8", "1")
)
EXPECTED_SCOREX0_SWEEPS = tuple(
    "model.masking.a:" + value
    for value in ("0.2", "0.4", "0.6", "0.8", "1.0")
)
EXPECTED_EPOCHS = list(range(1, 600))
EXPECTED_WAVS = 488
SCOREX0_PROVENANCE = {
    "run_job_id": "10803830",
    "training_loss_job_id": "11104619",
    "dispatch_job_id": "11099159",
    "source_summary_job_id": "11104956",
    "source_final_audit_job_id": "11104981",
}
SCOREX0_EXPECTED_JOB_IDS = {
    "metric_job_id": {"11104611", "11104612", "11104613", "11104614", "11104615"},
    "wer_job_id": {"11104616_29", "11104616_30", "11104616_31", "11104616_32", "11104616_33"},
    "utmos_job_id": {"11104618_29", "11104618_30", "11104618_31", "11104618_32", "11104618_33"},
}
SIDECARS = {
    "mcd": "_mcd.txt",
    "f0": "_f0.txt",
    "wer": "_wer_l.txt",
    "utmos": "_utmosv2.txt",
}
LOSS_FIELDS = [
    "experiment", "sweep_or_configuration", "epoch", "loss",
    "loss_definition", "duration_loss", "prior_loss", "diffusion_loss",
    "source_log",
]


def read_job_id(name):
    path = RESULTS / name
    value = path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9]+", value):
        raise RuntimeError("invalid job ID in {}: {!r}".format(path, value))
    return value


def read_rows(path, delimiter=","):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def atomic_write(path, text):
    temporary = path.with_name(
        path.name + ".tan18-{}.tmp".format(os.environ.get("SLURM_JOB_ID", "manual")))
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def delimited_text(fields, rows, delimiter="\t"):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return stream.getvalue()


def validate_summary():
    mappings = read_rows(RESULTS / "tan18_fmdt_mappings.tsv", delimiter="\t")
    summary = read_rows(RESULTS / "tan18_fmdt_summary.csv")
    if len(mappings) != 5 or len(summary) != 5:
        raise RuntimeError(
            "expected five fmDT mappings and summaries; found {} and {}".format(
                len(mappings), len(summary)))
    mapping_by_sweep = {row["sweep"]: row for row in mappings}
    summary_by_sweep = {row["sweep"]: row for row in summary}
    if (set(mapping_by_sweep) != set(EXPECTED_SWEEPS)
            or set(summary_by_sweep) != set(EXPECTED_SWEEPS)):
        raise RuntimeError("fmDT sweep coverage mismatch")

    run_id = read_job_id("tan18_run_job_id.txt")
    common_ids = {
        "environment_probe_job_id": read_job_id("tan18_env_probe_job_id.txt"),
        "checkpoint_validation_job_id": read_job_id("tan18_checkpoint_job_id.txt"),
        "acceptance_gate_job_id": read_job_id("tan18_gate_job_id.txt"),
        "training_loss_job_id": read_job_id("tan18_loss_job_id.txt"),
        "dispatch_job_id": read_job_id("tan18_dispatch_job_id.txt"),
        "summary_job_id": read_job_id("tan18_fmdt_summary_job_id.txt"),
    }
    expected_tasks = {
        sweep: "{}_{}".format(run_id, index)
        for index, sweep in enumerate(EXPECTED_SWEEPS)
    }
    for sweep in EXPECTED_SWEEPS:
        mapping = mapping_by_sweep[sweep]
        row = summary_by_sweep[sweep]
        if row.get("folder") != "fmDT":
            raise RuntimeError("non-fmDT summary row: {}".format(row))
        if row.get("gen") != mapping.get("gen") or row.get("out") != mapping.get("out"):
            raise RuntimeError("mapping/summary path mismatch for {}".format(sweep))
        for field in ("wav_count", "mcd_count", "f0_count", "wer_count", "utmos_count"):
            if int(row[field]) != EXPECTED_WAVS:
                raise RuntimeError("{} {} is {}; expected {}".format(
                    sweep, field, row[field], EXPECTED_WAVS))
        for field in ("mcd_mean", "mcd_std", "f0_mean", "f0_std",
                      "wer_mean", "wer_std", "utmos_mean", "utmos_std"):
            value = float(row[field])
            if not math.isfinite(value) or (field.endswith("_std") and value < 0):
                raise RuntimeError("invalid {} for {}: {!r}".format(field, sweep, row[field]))
        if row.get("coverage_ok", "").lower() != "true":
            raise RuntimeError("coverage is not complete for {}".format(sweep))
        if row.get("genuine_missing_values"):
            raise RuntimeError("summary reports missing values for {}".format(sweep))
        if row.get("run_job_id") != run_id or row.get("run_task_job_id") != expected_tasks[sweep]:
            raise RuntimeError("run provenance mismatch for {}".format(sweep))
        for field, expected in common_ids.items():
            if row.get(field) != expected:
                raise RuntimeError("{} mismatch for {}".format(field, sweep))
        for field in ("metric_job_id", "wer_job_id", "utmos_job_id"):
            if row.get(field) != mapping.get(field):
                raise RuntimeError("{} mapping mismatch for {}".format(field, sweep))

        # Do not treat the dependency-held summary as sufficient evidence on
        # its own.  Re-read every current per-WAV value in the terminal audit,
        # recompute the population statistics, and require exact path/format
        # agreement with the metric-tool aggregate records.
        gen = Path(row["gen"])
        out = Path(row["out"])
        wavs = sorted(gen.glob("*.wav"))
        if len(wavs) != EXPECTED_WAVS:
            raise RuntimeError(
                "{} has {} WAVs at terminal audit; expected {}".format(
                    sweep, len(wavs), EXPECTED_WAVS))
        for metric, suffix in SIDECARS.items():
            values = []
            for wav in wavs:
                sidecar = wav.with_name(wav.stem + suffix)
                if not sidecar.is_file() or sidecar.stat().st_size == 0:
                    raise RuntimeError("missing fmDT sidecar {}".format(sidecar))
                try:
                    value = float(sidecar.read_text(encoding="utf-8").strip())
                except Exception as error:
                    raise RuntimeError(
                        "invalid fmDT sidecar {}: {}".format(sidecar, error))
                if not math.isfinite(value):
                    raise RuntimeError("non-finite fmDT sidecar {}".format(sidecar))
                values.append(value)
            observed_mean = sum(values) / len(values)
            observed_std = statistics.pstdev(values)
            if int(row[metric + "_count"]) != len(values):
                raise RuntimeError(
                    "{} {} terminal count mismatch".format(sweep, metric))
            for stat_name, observed in (
                    ("mean", observed_mean), ("std", observed_std)):
                reported = float(row[metric + "_" + stat_name])
                if not math.isclose(
                        reported, observed, rel_tol=1e-12, abs_tol=1e-12):
                    raise RuntimeError(
                        "{} {} {} mismatch: {} versus {}".format(
                            sweep, metric, stat_name, reported, observed))
        for field, filename in (
                ("mcd_aggregate_format", "MCD.txt"),
                ("f0_aggregate_format", "F0.txt")):
            observed = aggregate_line(out / filename, gen)
            if row.get(field) != observed:
                raise RuntimeError(
                    "{} {} terminal aggregate mismatch".format(sweep, field))
    return summary, common_ids, expected_tasks


def extract_pairs(run_path):
    output = subprocess.check_output(
        ["/usr/bin/python3", str(ROOT / "extract_metrics_paths.py"), str(run_path)],
        universal_newlines=True,
    )
    pairs = []
    current = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("gen="):
            if current:
                raise RuntimeError("incomplete extractor record before {}".format(line))
            current["gen"] = line[4:]
        elif line.startswith("out="):
            current["out"] = line[4:]
            if set(current) != {"gen", "out"}:
                raise RuntimeError("invalid extractor record {}".format(current))
            pairs.append(current)
            current = {}
    if current:
        raise RuntimeError("trailing incomplete extractor record {}".format(current))
    return pairs


def aggregate_line(path, gen):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("missing or empty aggregate file {}".format(path))
    matches = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.split("   ", 1)[0].strip() == str(gen)
    ]
    if not matches:
        raise RuntimeError("{} has no aggregate line for {}".format(path, gen))
    return matches[-1]


def validate_scorex0_summary():
    """Recompute scorex0 coverage/statistics and bind them to completed jobs."""
    legacy = [
        row for row in read_rows(RESULTS / "summary.csv")
        if row.get("folder") == "scorex0"
    ]
    pairs = extract_pairs(ROOT / "scorex0" / "run.sh")
    if len(legacy) != 5 or len(pairs) != 5:
        raise RuntimeError(
            "expected five scorex0 legacy rows and extracted pairs; found {} and {}".format(
                len(legacy), len(pairs)))
    legacy_by_pair = {(row["gen"], row["out"]): row for row in legacy}
    extracted_pairs = {(row["gen"], row["out"]) for row in pairs}
    if len(legacy_by_pair) != 5 or set(legacy_by_pair) != extracted_pairs:
        raise RuntimeError("scorex0 extracted paths do not match the validated source summary")

    observed_jobs = {field: set() for field in SCOREX0_EXPECTED_JOB_IDS}
    output = []
    observed_sweeps = set()
    legacy_prefix = {"mcd": "mcd", "f0": "logf0", "wer": "wer", "utmos": "utmosv2"}
    for pair in pairs:
        source = legacy_by_pair[(pair["gen"], pair["out"])]
        gen = Path(pair["gen"])
        out = Path(pair["out"])
        sweep = gen.parents[2].name
        observed_sweeps.add(sweep)
        wavs = sorted(gen.glob("*.wav"))
        if len(wavs) != EXPECTED_WAVS:
            raise RuntimeError(
                "scorex0 {} has {} WAVs; expected {}".format(
                    sweep, len(wavs), EXPECTED_WAVS))
        record = {
            "folder": "scorex0",
            "sweep": sweep,
            "gen": str(gen),
            "out": str(out),
            "wav_count": len(wavs),
            "coverage_ok": True,
            "genuine_missing_values": "",
            "provenance": "reused_completed",
            "run_job_id": SCOREX0_PROVENANCE["run_job_id"],
            "run_task_job_id": SCOREX0_PROVENANCE["run_job_id"],
            "gen_repair_job_id": source.get("gen_repair_job_id", ""),
            "training_loss_job_id": SCOREX0_PROVENANCE["training_loss_job_id"],
            "dispatch_job_id": SCOREX0_PROVENANCE["dispatch_job_id"],
            "summary_job_id": SCOREX0_PROVENANCE["source_summary_job_id"],
            "source_summary_job_id": SCOREX0_PROVENANCE["source_summary_job_id"],
            "source_final_audit_job_id": SCOREX0_PROVENANCE["source_final_audit_job_id"],
        }
        if source.get("run_job_id") != SCOREX0_PROVENANCE["run_job_id"]:
            raise RuntimeError("scorex0 source run provenance mismatch for {}".format(sweep))
        if not re.fullmatch(r"[0-9]+", record["gen_repair_job_id"]):
            raise RuntimeError("scorex0 generation job provenance is invalid for {}".format(sweep))
        for normalized, source_field in (
                ("metric_job_id", "metrics_job_id"),
                ("wer_job_id", "wer_job_id"),
                ("utmos_job_id", "utmosv2_job_id")):
            value = source.get(source_field, "")
            record[normalized] = value
            observed_jobs[normalized].add(value)

        for metric, suffix in SIDECARS.items():
            values = []
            for wav in wavs:
                sidecar = wav.with_name(wav.stem + suffix)
                if not sidecar.is_file() or sidecar.stat().st_size == 0:
                    raise RuntimeError("missing scorex0 sidecar {}".format(sidecar))
                try:
                    value = float(sidecar.read_text(encoding="utf-8").strip())
                except Exception as error:
                    raise RuntimeError("invalid scorex0 sidecar {}: {}".format(
                        sidecar, error))
                if not math.isfinite(value):
                    raise RuntimeError("non-finite scorex0 sidecar {}".format(sidecar))
                values.append(value)
            mean = sum(values) / len(values)
            std = statistics.pstdev(values)
            record[metric + "_count"] = len(values)
            record[metric + "_mean"] = mean
            record[metric + "_std"] = std
            prefix = legacy_prefix[metric]
            if int(source[prefix + "_count"]) != len(values):
                raise RuntimeError("scorex0 source count mismatch for {} {}".format(
                    sweep, metric))
            for stat_name, observed in (("mean", mean), ("std", std)):
                prior = float(source[prefix + "_" + stat_name])
                if not math.isclose(prior, observed, rel_tol=1e-8, abs_tol=1e-8):
                    raise RuntimeError(
                        "scorex0 source {} mismatch for {} {}: {} versus {}".format(
                            stat_name, sweep, metric, prior, observed))
            if source.get(prefix + "_missing") or source.get(prefix + "_invalid"):
                raise RuntimeError("scorex0 source reports missing/invalid {} for {}".format(
                    metric, sweep))
        record["mcd_aggregate_format"] = aggregate_line(out / "MCD.txt", gen)
        record["f0_aggregate_format"] = aggregate_line(out / "F0.txt", gen)
        output.append(record)

    if observed_sweeps != set(EXPECTED_SCOREX0_SWEEPS):
        raise RuntimeError("scorex0 sweep coverage mismatch: {}".format(
            sorted(observed_sweeps)))
    for field, expected in SCOREX0_EXPECTED_JOB_IDS.items():
        if observed_jobs[field] != expected:
            raise RuntimeError("scorex0 {} provenance mismatch: {}".format(
                field, sorted(observed_jobs[field])))
    return output


def validate_training_loss():
    path = ROOT / "training_loss.log"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != LOSS_FIELDS:
            raise RuntimeError("unexpected training_loss.log header: {}".format(reader.fieldnames))
        all_rows = list(reader)
    grouped = defaultdict(list)
    scorex0_grouped = defaultdict(list)
    for row in all_rows:
        if row.get("experiment") == "fmDT":
            grouped[row.get("sweep_or_configuration", "")].append(row)
        elif row.get("experiment") == "scorex0":
            scorex0_grouped[row.get("sweep_or_configuration", "")].append(row)
    if set(grouped) != set(EXPECTED_SWEEPS):
        raise RuntimeError("training-loss sweep coverage mismatch: {}".format(sorted(grouped)))
    for sweep in EXPECTED_SWEEPS:
        rows = grouped[sweep]
        epochs = [int(row["epoch"]) for row in rows]
        if epochs != EXPECTED_EPOCHS:
            raise RuntimeError("{} training-loss epochs are not exactly 1..599".format(sweep))
        for row in rows:
            values = [float(row[field]) for field in (
                "loss", "duration_loss", "prior_loss", "diffusion_loss")]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("non-finite loss in {} epoch {}".format(sweep, row["epoch"]))
            if not math.isclose(values[0], sum(values[1:]), rel_tol=2e-10, abs_tol=2e-10):
                raise RuntimeError("loss component mismatch in {} epoch {}".format(sweep, row["epoch"]))
    validations = read_rows(
        RESULTS / "tan18_fmdt_training_loss_validation.tsv", delimiter="\t")
    if len(validations) != 5:
        raise RuntimeError("expected five training-loss validation rows")
    for row in validations:
        if (row.get("experiment") != "fmDT"
                or row.get("sweep_or_configuration") not in EXPECTED_SWEEPS
                or int(row.get("expected_epochs", 0)) != 599
                or int(row.get("observed_final_epochs", 0)) != 599
                or row.get("status") != "complete"):
            raise RuntimeError("invalid training-loss validation row: {}".format(row))
    fm_rows = sum(len(rows) for rows in grouped.values())
    if fm_rows != 2995:
        raise RuntimeError("expected 2995 fmDT training-loss records; found {}".format(fm_rows))
    if set(scorex0_grouped) != set(EXPECTED_SCOREX0_SWEEPS):
        raise RuntimeError(
            "preserved scorex0 training-loss sweep coverage mismatch: {}".format(
                sorted(scorex0_grouped)))
    for sweep in EXPECTED_SCOREX0_SWEEPS:
        rows = scorex0_grouped[sweep]
        epochs = [int(row["epoch"]) for row in rows]
        if epochs != EXPECTED_EPOCHS:
            raise RuntimeError(
                "preserved scorex0 {} epochs are not exactly 1..599".format(sweep))
        for row in rows:
            values = [float(row[field]) for field in (
                "loss", "duration_loss", "prior_loss", "diffusion_loss")]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(
                    "non-finite preserved scorex0 loss in {} epoch {}".format(
                        sweep, row["epoch"]))
            if not math.isclose(
                    values[0], sum(values[1:]), rel_tol=2e-10, abs_tol=2e-10):
                raise RuntimeError(
                    "preserved scorex0 loss component mismatch in {} epoch {}".format(
                        sweep, row["epoch"]))
    scorex0_rows = sum(len(rows) for rows in scorex0_grouped.values())
    if scorex0_rows != 2995:
        raise RuntimeError(
            "expected 2995 preserved scorex0 training-loss records; found {}".format(
                scorex0_rows))
    return fm_rows, len(all_rows) - fm_rows, scorex0_rows


def terminal_accounting(summary, scorex0_summary, common_ids, expected_tasks):
    bridge_id = read_job_id("tan18_all_bridge_job_id.txt")
    requested = set(common_ids.values())
    requested.add(bridge_id)
    requested.update(expected_tasks.values())
    for row in summary:
        requested.update(row[field] for field in (
            "metric_job_id", "wer_job_id", "utmos_job_id"))
    for row in scorex0_summary:
        requested.update(row[field] for field in (
            "run_job_id", "gen_repair_job_id", "metric_job_id", "wer_job_id",
            "utmos_job_id", "training_loss_job_id", "dispatch_job_id",
            "source_summary_job_id", "source_final_audit_job_id"))

    query_ids = sorted({job_id.split("_", 1)[0] for job_id in requested})
    command = [
        "sacct", "-X", "-n", "-P", "-j", ",".join(query_ids),
        "--format=JobID,JobIDRaw,JobName,State,ExitCode,Start,End,Elapsed,NodeList",
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    if result.returncode:
        raise RuntimeError("sacct failed: {}".format(result.stderr.strip()))
    observed = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 9:
            continue
        job_id = fields[0].strip()
        if job_id in requested:
            observed[job_id] = {
                "requested_job_id": job_id,
                "job_id_raw": fields[1].strip(),
                "job_name": fields[2].strip(),
                "state": fields[3].strip(),
                "exit_code": fields[4].strip(),
                "start": fields[5].strip(),
                "end": fields[6].strip(),
                "elapsed": fields[7].strip(),
                "node_list": fields[8].strip(),
                "status": "complete" if (
                    fields[3].strip().startswith("COMPLETED")
                    and fields[4].strip() == "0:0") else "failed",
            }
    missing = sorted(requested - set(observed))
    failed = sorted(job_id for job_id, row in observed.items()
                    if row["status"] != "complete")
    if missing or failed:
        raise RuntimeError("terminal accounting failed; missing={} failed={}".format(
            missing, [(job_id, observed[job_id]["state"], observed[job_id]["exit_code"])
                     for job_id in failed]))
    return [observed[job_id] for job_id in sorted(observed)], bridge_id


def finalize_summary_csv(summary, bridge_id):
    """Add the dynamically assigned terminal jobs to the final CSV provenance."""
    path = RESULTS / "tan18_fmdt_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
    required = ["completion_bridge_job_id", "final_audit_job_id"]
    fields = [field for field in fields if field not in required]
    if not fields:
        raise RuntimeError("summary CSV has no base fields")
    final_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not re.fullmatch(r"[0-9]+", final_id):
        raise RuntimeError("missing or invalid final-audit Slurm job ID: {!r}".format(
            final_id))
    for row in summary:
        row["completion_bridge_job_id"] = bridge_id
        row["final_audit_job_id"] = final_id
    atomic_write(
        path,
        delimited_text(fields + required, summary, delimiter=","),
    )
    return final_id


def write_combined_summary(summary, scorex0_summary, bridge_id):
    final_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not re.fullmatch(r"[0-9]+", final_id):
        raise RuntimeError("missing or invalid final-audit Slurm job ID: {!r}".format(
            final_id))
    combined = []
    for source in summary:
        row = dict(source)
        row.update({
            "provenance": "fresh_tan18",
            "gen_repair_job_id": "",
            "source_summary_job_id": "",
            "source_final_audit_job_id": "",
        })
        combined.append(row)
    combined.extend(dict(row) for row in scorex0_summary)
    for row in combined:
        row["completion_bridge_job_id"] = bridge_id
        row["final_audit_job_id"] = final_id
    fields = [
        "folder", "sweep", "provenance", "gen", "out", "wav_count",
        "mcd_count", "mcd_mean", "mcd_std", "f0_count", "f0_mean", "f0_std",
        "wer_count", "wer_mean", "wer_std", "utmos_count", "utmos_mean",
        "utmos_std", "coverage_ok", "genuine_missing_values",
        "environment_probe_job_id", "checkpoint_validation_job_id",
        "run_job_id", "run_task_job_id", "gen_repair_job_id",
        "acceptance_gate_job_id", "training_loss_job_id", "dispatch_job_id",
        "metric_job_id", "wer_job_id", "utmos_job_id", "summary_job_id",
        "source_summary_job_id", "source_final_audit_job_id",
        "completion_bridge_job_id", "final_audit_job_id",
        "mcd_aggregate_format", "f0_aggregate_format",
    ]
    atomic_write(
        RESULTS / "tan18_summary.csv",
        delimited_text(fields, combined, delimiter=","),
    )
    return combined


def build_combined_report(combined, audit, bridge_id, final_id, fm_rows,
                          scorex0_rows, preserved_rows, inventory):
    lines = [
        "# TAN-18 fmDT and scorex0 speech-metric report",
        "",
        "TAN-18 produced five fresh fmDT configurations. The workflow-required "
        "scorex0 inventory addition is included using its already completed run "
        "and downstream graph, with current sidecars and statistics revalidated by "
        "this dependency-held terminal audit.",
        "",
        "Standard deviations are population standard deviations recomputed from "
        "the per-WAV scalar sidecars.",
        "",
        "| Folder | Sweep | Source | WAVs | MCD n / mean / sd | logF0/F0 n / mean / sd | WER n / mean / sd | UTMOSv2 n / mean / sd | Jobs (run / generation / metric / WER / UTMOS) |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in combined:
        def metric_text(metric):
            return "{} / {:.6f} / {:.6f}".format(
                row[metric + "_count"], float(row[metric + "_mean"]),
                float(row[metric + "_std"]))
        jobs = "{} / {} / {} / {} / {}".format(
            row.get("run_task_job_id", ""), row.get("gen_repair_job_id", "") or "n/a",
            row.get("metric_job_id", ""), row.get("wer_job_id", ""),
            row.get("utmos_job_id", ""))
        lines.append(
            "| `{folder}` | `{sweep}` | `{provenance}` | {wav_count} | {mcd} | "
            "{f0} | {wer} | {utmos} | `{jobs}` |".format(
                folder=row["folder"], sweep=row["sweep"],
                provenance=row["provenance"], wav_count=row["wav_count"],
                mcd=metric_text("mcd"), f0=metric_text("f0"),
                wer=metric_text("wer"), utmos=metric_text("utmos"), jobs=jobs))
    lines.extend([
        "",
        "## Paths and provenance",
        "",
    ])
    for row in combined:
        lines.extend([
            "- `{}` / `{}`: gen `{}`; out `{}`.".format(
                row["folder"], row["sweep"], row["gen"], row["out"]),
            "  - Jobs: environment probe `{}`; checkpoint validation `{}`; run "
            "array/allocation `{}`; run task `{}`; generation/repair `{}`; "
            "acceptance gate `{}`; training-loss `{}`; dispatcher `{}`; metric "
            "`{}`; WER `{}`; UTMOSv2 `{}`; summary `{}`; source summary `{}`; "
            "source final audit `{}`; completion bridge `{}`; terminal audit "
            "`{}`.".format(
                row.get("environment_probe_job_id", "") or "n/a",
                row.get("checkpoint_validation_job_id", "") or "n/a",
                row.get("run_job_id", "") or "n/a",
                row.get("run_task_job_id", "") or "n/a",
                row.get("gen_repair_job_id", "") or "n/a",
                row.get("acceptance_gate_job_id", "") or "n/a",
                row.get("training_loss_job_id", "") or "n/a",
                row.get("dispatch_job_id", "") or "n/a",
                row.get("metric_job_id", "") or "n/a",
                row.get("wer_job_id", "") or "n/a",
                row.get("utmos_job_id", "") or "n/a",
                row.get("summary_job_id", "") or "n/a",
                row.get("source_summary_job_id", "") or "n/a",
                row.get("source_final_audit_job_id", "") or "n/a",
                row.get("completion_bridge_job_id", "") or "n/a",
                row.get("final_audit_job_id", "") or "n/a"),
            "  - Aggregate formats: MCD `{}`; F0 `{}`; genuine missing values "
            "`{}`.".format(
                row.get("mcd_aggregate_format", ""),
                row.get("f0_aggregate_format", ""),
                row.get("genuine_missing_values", "") or "none"),
        ])
    lines.extend([
        "",
        "## Terminal validation",
        "",
        "- Included result sets: five fresh fmDT rows and five reused, freshly "
        "revalidated scorex0 rows.",
        "- Coverage: 488 WAVs and 488 finite MCD, logF0/F0, WER, and UTMOSv2 "
        "values for every row; genuine missing values: none.",
        "- Training loss: {} fmDT rows plus {} scorex0 rows, each covering five "
        "configurations × epochs 1–599; {} non-fmDT rows were preserved in the "
        "fmDT merge.".format(fm_rows, scorex0_rows, preserved_rows),
        "- Final first-depth rescan: {} `run.sh` folders (`{}`).".format(
            len(inventory), "`, `".join(row["folder"] for row in inventory)),
        "- Terminal accounting: {} unique allocation/task records, all "
        "`COMPLETED 0:0`.".format(len(audit)),
        "- Completion bridge: `{}`; terminal audit: `{}`.".format(
            bridge_id, final_id),
        "",
        "Machine-readable evidence: `tan18_summary.csv`; accounting evidence: "
        "`tan18_job_audit.tsv`; inventory: `tan18_fmdt_inventory.tsv`.",
    ])
    return "\n".join(lines) + "\n"


def build_report(summary, audit, bridge_id, fm_rows, preserved_rows,
                 scorex0_rows, inventory):
    base = (RESULTS / "tan18_fmdt_report.md").read_text(encoding="utf-8").rstrip()
    terminal_marker = "\n## Terminal validation\n"
    if terminal_marker in base:
        base = base.split(terminal_marker, 1)[0].rstrip()
    final_id = os.environ.get("SLURM_JOB_ID", "manual")
    lines = [
        base,
        "",
        "## Terminal validation",
        "",
        "- Fresh TAN-18 scope: fmDT, five configurations; scorex0 is carried "
        "through the workflow as a reused completed result set in the combined "
        "`tan18_summary.csv` / `tan18_report.md` artifacts.",
        "- Final first-depth rescan: {} `run.sh` folders (`{}`); fmDT is fresh and scorex0 is included from its validated completed graph.".format(
            len(inventory), "`, `".join(row["folder"] for row in inventory)),
        "- Fresh fmDT coverage: 488 WAVs and 488 finite MCD, logF0/F0, WER, "
        "and UTMOSv2 values for every configuration.",
        "- Training loss: {} fmDT records (5 configurations × 599 contiguous epochs); {} non-fmDT rows preserved.".format(
            fm_rows, preserved_rows),
        "- Included scorex0 coverage: {} training-loss records (5 configurations × 599 contiguous epochs), plus freshly revalidated speech-metric sidecars and terminal job provenance in the combined artifacts.".format(
            scorex0_rows),
        "- Predecessor accounting: {} allocation/task records, all `COMPLETED 0:0`.".format(len(audit)),
        "- Completion bridge (legacy filename): `{}`.".format(bridge_id),
        "- Final audit job: `{}` (its terminal state is inspected after this payload exits).".format(final_id),
        "- Genuine missing or invalid requested values: none.",
        "",
        "Accounting evidence: `tan18_fmdt_job_audit.tsv`.",
    ]
    return "\n".join(lines) + "\n"


def main():
    run_paths = sorted(ROOT.glob("*/run.sh"))
    inventory = [
        {
            "folder": path.parent.name,
            "run_path": str(path),
            "tan18_scope": (
                "included_fresh" if path.parent.name == "fmDT"
                else "included_reused" if path.parent.name == "scorex0"
                else "excluded"),
            "scope_reason": (
                "requested fresh experiment" if path.parent.name == "fmDT"
                else "workflow-required inventory addition with completed validated graph"
                if path.parent.name == "scorex0"
                else "outside TAN-18 fresh and carried-result scope"),
        }
        for path in run_paths
    ]
    folders = {row["folder"] for row in inventory}
    if "fmDT" not in folders or "scorex0" not in folders:
        raise RuntimeError("first-depth rescan no longer includes scorex0/run.sh")
    summary, common_ids, expected_tasks = validate_summary()
    scorex0_summary = validate_scorex0_summary()
    fm_rows, preserved_rows, scorex0_rows = validate_training_loss()
    audit, bridge_id = terminal_accounting(
        summary, scorex0_summary, common_ids, expected_tasks)
    final_id = finalize_summary_csv(summary, bridge_id)
    combined = write_combined_summary(summary, scorex0_summary, bridge_id)
    audit_fields = [
        "requested_job_id", "job_id_raw", "job_name", "state", "exit_code",
        "start", "end", "elapsed", "node_list", "status",
    ]
    atomic_write(
        RESULTS / "tan18_fmdt_job_audit.tsv",
        delimited_text(audit_fields, audit),
    )
    atomic_write(
        RESULTS / "tan18_job_audit.tsv",
        delimited_text(audit_fields, audit),
    )
    atomic_write(
        RESULTS / "tan18_fmdt_inventory.tsv",
        delimited_text(
            ["folder", "run_path", "tan18_scope", "scope_reason"], inventory),
    )
    atomic_write(
        RESULTS / "tan18_fmdt_report.md",
        build_report(
            summary, audit, bridge_id, fm_rows, preserved_rows,
            scorex0_rows, inventory),
    )
    atomic_write(
        RESULTS / "tan18_report.md",
        build_combined_report(
            combined, audit, bridge_id, final_id, fm_rows,
            scorex0_rows, preserved_rows, inventory),
    )
    status = (
        "complete\nscope=fmDT\nconfigurations=5\nwavs_per_configuration=488\n"
        "summary_rows=5\nloss_epoch_records=2995\n"
        "preserved_non_fmdt_loss_records={}\n"
        "preserved_scorex0_loss_records={}\naudited_predecessor_records={}\n"
        "first_depth_run_folders={}\nmissing_values=0\ninvalid_values=0\n"
        "final_audit_job_id={}\n"
    ).format(
        preserved_rows, scorex0_rows, len(audit), len(inventory),
        os.environ.get("SLURM_JOB_ID", "manual"))
    atomic_write(RESULTS / "tan18_fmdt_final_status.txt", status)
    combined_status = (
        "complete\nscope=fmDT+scorex0\nfresh_fmdt_configurations=5\n"
        "reused_scorex0_configurations=5\nwavs_per_configuration=488\n"
        "summary_rows=10\nfmDT_loss_epoch_records={}\n"
        "scorex0_loss_epoch_records={}\nmissing_values=0\ninvalid_values=0\n"
        "audited_predecessor_records={}\nfirst_depth_run_folders={}\n"
        "completion_bridge_job_id={}\nfinal_audit_job_id={}\n"
    ).format(
        fm_rows, scorex0_rows, len(audit), len(inventory), bridge_id, final_id)
    atomic_write(RESULTS / "tan18_final_status.txt", combined_status)
    print(
        "TAN-18 final audit passed: fmDT_configurations=5 "
        "scorex0_configurations=5 loss_records={} audited_predecessors={}".format(
            fm_rows + scorex0_rows, len(audit)), flush=True)


if __name__ == "__main__":
    main()
