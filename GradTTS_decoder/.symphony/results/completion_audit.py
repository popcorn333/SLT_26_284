#!/usr/bin/env python3
"""Final independent completion audit for TAN-6, excluding fm."""

import csv
import importlib.util
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet")
RES = ROOT / ".symphony/results"
FOLDERS = {
    "add", "blur", "fmDT", "levy", "mask", "one-shot",
    "product", "score", "scoreDT", "scorex0",
}
EXPECTED_PAIRS = 29
EXPECTED_WAVS = 488
EXPECTED_LOSS_RECORDS = 17371
METRIC_SUFFIXES = {
    "mcd": "_mcd.txt",
    "logf0": "_f0.txt",
    "wer": "_wer_l.txt",
    "utmosv2": "_utmosv2.txt",
}
TRANSCRIPT = Path(
    "/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/"
    "add/resources/filelists/ljspeech/test.txt")
LOSS_PATTERN = re.compile(
    r"^Epoch (\d+): duration loss = ([0-9eE+.-]+) \| "
    r"prior loss = ([0-9eE+.-]+) \| diffusion loss = ([0-9eE+.-]+)$",
    re.M,
)


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_key_values(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("\t", 1)
        if key in values:
            raise RuntimeError("duplicate ledger key {} in {}".format(key, path))
        values[key] = value
    return values


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
                raise RuntimeError("out without gen for {}".format(folder))
            pairs.append((gen, line[4:]))
            gen = None
    if gen is not None or not pairs:
        raise RuntimeError("invalid extracted pairs for {}".format(folder))
    return pairs


def validate_final_inventory_and_mappings():
    actual = {
        path.parent.name for path in ROOT.glob("*/run.sh")
        if path.parent.parent == ROOT and path.parent.name != "fm"
    }
    if actual != FOLDERS:
        raise RuntimeError(
            "final run.sh inventory changed: expected={} actual={}".format(
                sorted(FOLDERS), sorted(actual)))
    emitted = {}
    for folder in sorted(actual):
        for index, (gen, out) in enumerate(extract_pairs(folder)):
            emitted[(folder, str(index))] = (gen, out)
    if len(emitted) != EXPECTED_PAIRS:
        raise RuntimeError("final extraction emitted {} pairs".format(len(emitted)))

    mappings = read_tsv(RES / "mappings.tsv")
    stored = {
        (row["folder"], row["pair_index"]): (row["gen"], row["out"])
        for row in mappings
    }
    if len(mappings) != EXPECTED_PAIRS or len(stored) != EXPECTED_PAIRS:
        raise RuntimeError("mappings.tsv is incomplete or duplicated")
    if stored != emitted:
        raise RuntimeError("stored mappings differ from final run.sh extraction")
    if len({row["gen"] for row in mappings}) != EXPECTED_PAIRS or \
            len({row["out"] for row in mappings}) != EXPECTED_PAIRS:
        raise RuntimeError("generation or output paths are duplicated")
    return mappings


def validate_aggregate_file(out_dir, filename, gen):
    path = Path(out_dir) / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("missing aggregate metric output {}".format(path))
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] != gen or parts[2] != r"\pm":
            continue
        try:
            mean, std = float(parts[1]), float(parts[3])
        except ValueError:
            continue
        if math.isfinite(mean) and math.isfinite(std):
            records.append((mean, std))
    if not records:
        raise RuntimeError("{} has no finite record for {}".format(path, gen))


def validate_metrics_payload(path, expected_gen, expected_out):
    lines = path.read_text(encoding="utf-8").splitlines()
    assignments = {}
    for name in ("gt", "gen", "out"):
        values = [line.split("=", 1)[1] for line in lines
                  if line.startswith(name + "=")]
        if len(values) != 1:
            raise RuntimeError("{} has invalid {} assignment".format(path, name))
        assignments[name] = values[0]
    expected_gt = (
        "/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/ground_truth")
    if assignments != {
            "gt": expected_gt, "gen": expected_gen, "out": expected_out}:
        raise RuntimeError("{} does not match its recorded paths".format(path))
    text = "\n".join(lines)
    token_counts = {
        "tools/F0/F0.py": 1,
        "tools/MCD/MCD.py": 1,
        "--gt_wavdir_or_wavscp": 2,
        "--gen_wavdir_or_wavscp": 2,
        "--outdir": 2,
        "# TAN-10 per-WAV metric freshness gate": 1,
        "FRESHNESS_MARKER": 3,
        "sidecar was not refreshed by this job": 1,
    }
    for token, expected_count in token_counts.items():
        if text.count(token) != expected_count:
            raise RuntimeError("{} lost established token {}".format(path, token))
    if not re.search(r"(?m)^set -[^\n]*e[^\n]*$", text):
        raise RuntimeError("{} is not fail-fast".format(path))
    return assignments


def validate_metrics_scripts(mappings):
    last_by_folder = {}
    for row in mappings:
        last_by_folder[row["folder"]] = (row["gen"], row["out"])
        payload = RES / "metrics_payloads" / "metrics-{}-{}.sbatch".format(
            row["folder"], row["pair_index"])
        validate_metrics_payload(payload, row["gen"], row["out"])
        validate_aggregate_file(row["out"], "MCD.txt", row["gen"])
        validate_aggregate_file(row["out"], "F0.txt", row["gen"])
    for folder, final_pair in last_by_folder.items():
        path = ROOT / folder / "metrics.sh"
        validate_metrics_payload(path, final_pair[0], final_pair[1])


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "tan10_pipeline", RES / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_summary_stats(mappings):
    load_pipeline().summarize()
    if (RES / "validation_status.txt").read_text(
            encoding="utf-8").strip() != "complete":
        raise RuntimeError("metric sidecar validation is not complete")
    with (RES / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_PAIRS:
        raise RuntimeError("summary.csv row count is not 29")
    mapping_keys = {(row["folder"], row["pair_index"]) for row in mappings}
    summary_keys = {(row["folder"], row["pair_index"]) for row in rows}
    if summary_keys != mapping_keys:
        raise RuntimeError("summary.csv keys differ from mappings.tsv")
    mapping_by_key = {
        (row["folder"], row["pair_index"]): row for row in mappings
    }
    jobs = read_tsv(RES / "jobs.tsv")
    job_by_key = {
        (row["folder"], row["pair_index"], row["stage"]): row["job_id"]
        for row in jobs
    }
    report_text = (RES / "report.md").read_text(encoding="utf-8")
    for row in rows:
        key = (row["folder"], row["pair_index"])
        mapping = mapping_by_key[key]
        if (row["gen"], row["out"]) != (mapping["gen"], mapping["out"]):
            raise RuntimeError("summary path mismatch for {}".format(key))
        if int(row["wav_count"]) != EXPECTED_WAVS:
            raise RuntimeError("unexpected WAV count in summary row")
        for stage, column in (
                ("run", "run_job_id"),
                ("gen_repair", "gen_repair_job_id"),
                ("metrics", "metrics_job_id"),
                ("wer", "wer_job_id"),
                ("utmosv2", "utmosv2_job_id")):
            if row[column] != job_by_key[(key[0], key[1], stage)]:
                raise RuntimeError("summary job-ID mismatch for {} {}".format(
                    key, stage))
        wavs = sorted(Path(row["gen"]).glob("*.wav"))
        for metric, suffix in METRIC_SUFFIXES.items():
            if int(row[metric + "_count"]) != EXPECTED_WAVS:
                raise RuntimeError("{} sidecar count is incomplete".format(metric))
            if row[metric + "_missing"] or row[metric + "_invalid"]:
                raise RuntimeError("{} contains missing/invalid values".format(metric))
            values = []
            for wav in wavs:
                sidecar = wav.with_name(wav.stem + suffix)
                value = float(sidecar.read_text(encoding="utf-8").strip())
                if not math.isfinite(value):
                    raise RuntimeError("{} contains non-finite value {}".format(
                        metric, sidecar))
                values.append(value)
            calculated_mean = statistics.mean(values)
            calculated_std = statistics.pstdev(values)
            if row[metric + "_mean"] != "{:.8f}".format(calculated_mean) or \
                    row[metric + "_std"] != "{:.8f}".format(calculated_std):
                raise RuntimeError(
                    "{} summary statistics do not match sidecars for {}".format(
                        metric, key))
            if not math.isfinite(calculated_mean) or not math.isfinite(calculated_std):
                raise RuntimeError("{} summary is non-finite".format(metric))
        if "`{}`".format(row["gen"]) not in report_text:
            raise RuntimeError("Markdown report is missing {}".format(key))
    return rows


def validate_transcript_wavs(mappings):
    transcript_rows = [
        line for line in TRANSCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_names = {Path(line.split("|", 1)[0]).name for line in transcript_rows}
    if len(transcript_rows) != EXPECTED_WAVS or \
            len(expected_names) != EXPECTED_WAVS:
        raise RuntimeError("transcript no longer has 488 unique WAV entries")
    for row in mappings:
        observed = {path.name for path in Path(row["gen"]).glob("*.wav")}
        if observed != expected_names:
            raise RuntimeError(
                "WAV basename coverage differs for {} pair {}".format(
                    row["folder"], row["pair_index"]))


def validate_loss_values():
    validation = read_tsv(RES / "training_loss_validation.tsv")
    losses = read_tsv(ROOT / "training_loss.log")
    if len(validation) != EXPECTED_PAIRS or len(losses) != EXPECTED_LOSS_RECORDS:
        raise RuntimeError(
            "training loss coverage changed: configs={} records={}".format(
                len(validation), len(losses)))
    sources = {}
    for row in validation:
        key = (row["experiment"], row["sweep_or_configuration"])
        if row["expected_epochs"] != "599" or \
                row["observed_unique_epochs"] != "599" or \
                row["status"] != "complete":
            raise RuntimeError("incomplete training-loss validation {}".format(key))
        source = Path(row["source_log"])
        parsed = {}
        for epoch, duration, prior, diffusion in LOSS_PATTERN.findall(
                source.read_text(encoding="utf-8", errors="replace")):
            parsed[int(epoch)] = (
                float(duration), float(prior), float(diffusion))
        if set(parsed) != set(range(1, 600)):
            raise RuntimeError("source training loss is incomplete for {}".format(key))
        sources[key] = (source.resolve(), parsed)
    observed = {}
    for row in losses:
        key = (row["experiment"], row["sweep_or_configuration"])
        if key not in sources:
            raise RuntimeError("unexpected training-loss key {}".format(key))
        epoch = int(row["epoch"])
        source_path, source_values = sources[key]
        components = tuple(float(row[name]) for name in (
            "duration_loss", "prior_loss", "diffusion_loss"))
        total = float(row["loss"])
        if Path(row["source_log"]).resolve() != source_path:
            raise RuntimeError("training-loss source mismatch for {}".format(key))
        if row["loss_definition"] != "duration_loss+prior_loss+diffusion_loss":
            raise RuntimeError("training-loss definition mismatch")
        for value, source_value in zip(components, source_values[epoch]):
            if not math.isclose(value, source_value, rel_tol=1e-10, abs_tol=1e-12):
                raise RuntimeError(
                    "training-loss component mismatch for {} epoch {}".format(
                        key, epoch))
        if not math.isclose(
                total, sum(source_values[epoch]), rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError(
                "training-loss total mismatch for {} epoch {}".format(key, epoch))
        observed.setdefault(key, set()).add(epoch)
    if set(observed) != set(sources):
        raise RuntimeError("training-loss configuration coverage mismatch")
    if any(epochs != set(range(1, 600)) for epochs in observed.values()):
        raise RuntimeError("training-loss epoch coverage mismatch")
    return len(validation), len(losses)


def query_accounting(job_ids):
    requested = sorted(set(job_ids))
    pending = requested
    records_by_id = {}
    for attempt in range(6):
        output = subprocess.check_output([
            "/usr/bin/sacct", "-X", "-j", ",".join(pending),
            "--starttime=2026-07-01",
            "--format=JobID%30,JobIDRaw,JobName%40,State,ExitCode,"
            "Elapsed,Start,End,NodeList%30",
            "-n", "-P",
        ], universal_newlines=True)
        records = [line.split("|") for line in output.splitlines() if line.strip()]
        for requested_id in list(pending):
            raw = next(
                (parts for parts in records
                 if parts[0] == requested_id or parts[1] == requested_id), None)
            if raw is None:
                continue
            state = raw[3].split()[0].rstrip("+")
            if state == "COMPLETED" and raw[4] == "0:0":
                records_by_id[requested_id] = raw
        pending = [job_id for job_id in requested
                   if job_id not in records_by_id]
        if not pending:
            return records_by_id
        if attempt != 5:
            time.sleep(10)
    raise RuntimeError(
        "jobs lack exact successful terminal accounting: {}".format(pending))


def log_candidates(job_id, folders):
    pattern = re.compile(
        r"(?<![0-9]){}(?![0-9])".format(re.escape(job_id)))
    directories = [RES, ROOT] + [ROOT / folder for folder in folders]
    paths = set()
    for directory in directories:
        if directory.is_dir():
            paths.update(
                path for path in directory.iterdir()
                if path.is_file() and path.suffix in {".out", ".err"}
                and pattern.search(path.name))
    return sorted(paths)


def tail_line(path):
    last = ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                last = line.strip()
    return last.replace("\t", " ")[:1000]


def validate_jobs_and_logs():
    jobs = read_tsv(RES / "jobs.tsv")
    if len(jobs) != EXPECTED_PAIRS * 5:
        raise RuntimeError("jobs.tsv must contain exactly five stages per pair")
    keys = {
        (row["folder"], row["pair_index"], row["stage"]) for row in jobs
    }
    expected = {
        (folder, str(index), stage)
        for folder in FOLDERS
        for index in range(sum(
            1 for row in read_tsv(RES / "mappings.tsv")
            if row["folder"] == folder))
        for stage in ("run", "gen_repair", "metrics", "wer", "utmosv2")
    }
    if keys != expected:
        raise RuntimeError("jobs.tsv stage coverage is incomplete or duplicated")

    repair_job_id = os.environ.get("TAN6_SCOPE_REPAIR_JOB_ID", "")
    if not repair_job_id.isdigit():
        raise RuntimeError("missing TAN6 scope-repair job ID")
    ordinary = {repair_job_id}
    job_ids = {row["job_id"] for row in jobs} | ordinary
    records = query_accounting(job_ids)
    folders_by_job = {}
    for row in jobs:
        folders_by_job.setdefault(row["job_id"], set()).add(row["folder"])
    audit_rows = []
    log_rows = []
    for job_id in sorted(job_ids):
        raw = records[job_id]
        audit_rows.append([
            job_id, raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8],
        ])
        paths = log_candidates(job_id, folders_by_job.get(job_id, FOLDERS))
        if not paths:
            raise RuntimeError("no terminal log found for {}".format(job_id))
        for path in paths:
            log_rows.append([
                job_id, str(path), str(path.stat().st_size), tail_line(path),
            ])
    with (RES / "final_job_audit.tsv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "job_id", "job_name", "state", "exit_code", "elapsed",
            "start", "end", "node_list",
        ])
        writer.writerows(audit_rows)
    with (RES / "terminal_log_audit.tsv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["job_id", "log_path", "bytes", "last_nonempty_line"])
        writer.writerows(log_rows)
    return len(audit_rows), len(log_rows), repair_job_id


def main():
    mappings = validate_final_inventory_and_mappings()
    validate_transcript_wavs(mappings)
    validate_metrics_scripts(mappings)
    rows = validate_summary_stats(mappings)
    loss_configs, loss_records = validate_loss_values()
    audited_jobs, terminal_logs, repair_job_id = validate_jobs_and_logs()
    report = RES / "report.md"
    with report.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Completion audit\n\n"
            "- Final first-depth run.sh rescan: 10 eligible folders (`fm` excluded).\n"
            "- Final extractor rescan: 29 unique non-`fm` gen/out pairs.\n"
            "- Per-path transcript/WAV/metric coverage: 488 exact values.\n"
            "- Training-loss coverage: {} configurations and {} epoch records.\n"
            "- Successful Slurm allocations/tasks audited: {}.\n"
            "- Terminal log files inspected: {}.\n"
            "- Accounting details: final_job_audit.tsv.\n"
            "- Log details: terminal_log_audit.tsv.\n".format(
                loss_configs, loss_records, audited_jobs, terminal_logs))
    (RES / "final_status.txt").write_text(
        "complete\nfolders=10\nexcluded_folder=fm\npairs={}\nwavs_per_pair={}\n"
        "summary_rows={}\nloss_configurations={}\nloss_epoch_records={}\n"
        "audited_jobs={}\nterminal_logs={}\n".format(
            len(mappings), EXPECTED_WAVS, len(rows), loss_configs,
            loss_records, audited_jobs, terminal_logs),
        encoding="utf-8")
    print(
        "completion audit passed pairs={} losses={} jobs={} logs={}".format(
            len(mappings), loss_records, audited_jobs, terminal_logs),
        flush=True)
    with (RES / "tan6_scope_job_ids.tsv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["stage", "job_id"])
        writer.writerow(["scope_repair", repair_job_id])
        writer.writerow(["scope_audit", os.environ["SLURM_JOB_ID"]])


if __name__ == "__main__":
    main()
