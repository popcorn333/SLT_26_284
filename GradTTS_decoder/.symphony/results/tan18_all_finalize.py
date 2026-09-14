#!/usr/bin/env python3
"""TAN-18 fail-closed aggregation for every first-depth run.sh experiment."""

import csv
import io
import math
import os
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet")
RESULTS = ROOT / ".symphony" / "results"
EXTRACTOR = ROOT / "extract_metrics_paths.py"
EXPECTED_FOLDERS = {
    "add", "blur", "fm", "fmDT", "levy", "mask", "one-shot",
    "product", "score", "scoreDT", "scorex0",
}
EXPECTED_PAIR_COUNTS = {
    "add": 3,
    "blur": 3,
    "fm": 5,
    "fmDT": 5,
    "levy": 2,
    "mask": 1,
    "one-shot": 1,
    "product": 3,
    "score": 5,
    "scoreDT": 5,
    "scorex0": 5,
}
EXPECTED_WAVS = 488
EXPECTED_EPOCHS = list(range(1, 600))
EXPECTED_LOSS_RECORDS = sum(EXPECTED_PAIR_COUNTS.values()) * len(EXPECTED_EPOCHS)
RUN_JOB_IDS = {
    "add": "10800233",
    "blur": "10800211",
    "fm": "11097445",
    "fmDT": "11142109",
    "levy": "10800210",
    "mask": "10800213",
    "one-shot": "10800207",
    "product": "10800236",
    "score": "10804165",
    "scoreDT": "10884228",
    "scorex0": "10803830",
}
FM_METRIC_JOB_IDS = {
    "model.masking.a:0.2": "11104588",
    "model.masking.a:0.4": "11104589",
    "model.masking.a:0.6": "11104590",
    "model.masking.a:0.8": "11104591",
    "model.masking.a:1": "11104592",
}
FM_RUN_ARRAY_TASKS = {
    "model.masking.a:0.2": "0",
    "model.masking.a:0.4": "1",
    "model.masking.a:0.6": "2",
    "model.masking.a:0.8": "3",
    "model.masking.a:1": "4",
}
FM_DOWNSTREAM_ARRAY_TASKS = {
    "model.masking.a:0.2": "6",
    "model.masking.a:0.4": "7",
    "model.masking.a:0.6": "8",
    "model.masking.a:0.8": "9",
    "model.masking.a:1": "10",
}
FMDT_ARRAY_TASKS = {
    "model.masking.a:0.2": "0",
    "model.masking.a:0.4": "1",
    "model.masking.a:0.6": "2",
    "model.masking.a:0.8": "3",
    "model.masking.a:1": "4",
}
SIDE_CARS = {
    "mcd": "_mcd.txt",
    "logf0": "_f0.txt",
    "wer": "_wer_l.txt",
    "utmosv2": "_utmosv2.txt",
}
LOSS_FIELDS = [
    "experiment", "sweep_or_configuration", "epoch", "loss",
    "loss_definition", "duration_loss", "prior_loss", "diffusion_loss",
    "source_log",
]
LOSS_PATTERN = re.compile(
    r"^Epoch (\d+): duration loss = ([0-9eE+.-]+) \| "
    r"prior loss = ([0-9eE+.-]+) \| diffusion loss = ([0-9eE+.-]+)$",
    re.MULTILINE,
)


def atomic_write(path, text):
    temporary = path.with_name(
        path.name + ".tan18-{}.tmp".format(os.environ.get("SLURM_JOB_ID", "manual")))
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def csv_text(fields, rows, delimiter=","):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return stream.getvalue()


def extract_pairs(run_path):
    output = subprocess.check_output(
        ["/usr/bin/python3", str(EXTRACTOR), str(run_path)],
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


def configuration_from_gen(gen):
    return Path(gen).parents[2].name


def load_source_job_rows():
    old_summary = RESULTS / "summary.csv"
    with old_summary.open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    old_by_gen = {
        row["gen"]: row for row in old_rows if row["folder"] != "fmDT"
    }

    fresh_summary = RESULTS / "tan18_fmdt_summary.csv"
    if not fresh_summary.is_file():
        raise RuntimeError("fresh fmDT summary is absent: {}".format(fresh_summary))
    with fresh_summary.open(newline="", encoding="utf-8") as handle:
        fresh_rows = list(csv.DictReader(handle))
    if len(fresh_rows) != 5:
        raise RuntimeError("expected five fresh fmDT rows, found {}".format(len(fresh_rows)))
    fresh_by_gen = {row["gen"]: row for row in fresh_rows}
    if len(fresh_by_gen) != 5:
        raise RuntimeError("fresh fmDT gen paths are duplicated")
    return old_by_gen, fresh_by_gen


def aggregate_line(path, gen):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("missing or empty aggregate file {}".format(path))
    matches = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.split("   ", 1)[0].strip() == str(gen):
            matches.append(line.strip())
    if not matches:
        raise RuntimeError("{} has no aggregate line for {}".format(path, gen))
    return matches[-1]


def row_jobs(folder, configuration, gen, old_by_gen, fresh_by_gen):
    common = {
        "environment_probe_job_id": "",
        "checkpoint_validation_job_id": "",
        "run_job_id": RUN_JOB_IDS[folder],
        "run_task_job_id": RUN_JOB_IDS[folder],
        "gen_repair_job_id": "",
        "metrics_job_id": "",
        "wer_job_id": "",
        "utmosv2_job_id": "",
        "acceptance_gate_job_id": "",
        "training_loss_job_id": "",
        "dispatch_job_id": "",
        "source_summary_job_id": "",
        "source_final_audit_job_id": "",
    }
    if folder == "fm":
        run_task = FM_RUN_ARRAY_TASKS[configuration]
        downstream_task = FM_DOWNSTREAM_ARRAY_TASKS[configuration]
        common.update({
            "run_task_job_id": "{}_{}".format(
                RUN_JOB_IDS[folder], run_task),
            "metrics_job_id": FM_METRIC_JOB_IDS[configuration],
            "wer_job_id": "11104616_{}".format(downstream_task),
            "utmosv2_job_id": "11104618_{}".format(downstream_task),
            "training_loss_job_id": "11104619",
            "dispatch_job_id": "11099159",
            "source_summary_job_id": "11104956",
            "source_final_audit_job_id": "11104981",
        })
        return common
    if folder == "fmDT":
        if str(gen) not in fresh_by_gen:
            raise RuntimeError("fresh fmDT mapping absent for {}".format(gen))
        source = fresh_by_gen[str(gen)]
        common.update({
            "environment_probe_job_id": source["environment_probe_job_id"],
            "checkpoint_validation_job_id": source[
                "checkpoint_validation_job_id"],
            "run_task_job_id": source["run_task_job_id"],
            "metrics_job_id": source["metric_job_id"],
            "wer_job_id": source["wer_job_id"],
            "utmosv2_job_id": source["utmos_job_id"],
            "acceptance_gate_job_id": source["acceptance_gate_job_id"],
            "training_loss_job_id": source["training_loss_job_id"],
            "dispatch_job_id": source["dispatch_job_id"],
            "source_summary_job_id": source["summary_job_id"],
        })
        if source["run_job_id"] != common["run_job_id"]:
            raise RuntimeError("fmDT run job mismatch in fresh summary")
        expected_task = "{}_{}".format(
            RUN_JOB_IDS[folder], FMDT_ARRAY_TASKS[configuration])
        if source["run_task_job_id"] != expected_task:
            raise RuntimeError(
                "fmDT run task mismatch for {}: expected {} observed {}".format(
                    configuration, expected_task, source["run_task_job_id"]))
        return common

    if str(gen) not in old_by_gen:
        raise RuntimeError("validated legacy mapping absent for {}".format(gen))
    source = old_by_gen[str(gen)]
    common.update({
        "gen_repair_job_id": source["gen_repair_job_id"],
        "metrics_job_id": source["metrics_job_id"],
        "wer_job_id": source["wer_job_id"],
        "utmosv2_job_id": source["utmosv2_job_id"],
        "training_loss_job_id": "11104619",
        "dispatch_job_id": "11099159",
        "source_summary_job_id": "11104956",
        "source_final_audit_job_id": "11104981",
    })
    if source["run_job_id"] != common["run_job_id"]:
        raise RuntimeError("{} run job mismatch in legacy summary".format(folder))
    return common


def summarize_pairs(inventory):
    old_by_gen, fresh_by_gen = load_source_job_rows()
    rows = []
    for folder in sorted(inventory):
        for pair_index, pair in enumerate(inventory[folder]):
            gen = Path(pair["gen"])
            out = Path(pair["out"])
            configuration = configuration_from_gen(gen)
            wavs = sorted(gen.glob("*.wav"))
            record = {
                "folder": folder,
                "pair_index": pair_index,
                "sweep_or_configuration": configuration,
                "gen": str(gen),
                "out": str(out),
                "wav_count": len(wavs),
            }
            missing_all = []
            invalid_all = []
            for metric, suffix in SIDE_CARS.items():
                values = []
                missing = []
                invalid = []
                for wav in wavs:
                    sidecar = wav.with_name(wav.stem + suffix)
                    if not sidecar.is_file() or sidecar.stat().st_size == 0:
                        missing.append(sidecar.name)
                        continue
                    try:
                        value = float(sidecar.read_text(encoding="utf-8").strip())
                        if not math.isfinite(value):
                            raise ValueError("non-finite")
                    except Exception as error:
                        invalid.append("{}:{}".format(sidecar.name, error))
                        continue
                    values.append(value)
                record[metric + "_count"] = len(values)
                record[metric + "_mean"] = (
                    "{:.10f}".format(sum(values) / len(values)) if values else "")
                record[metric + "_std"] = (
                    "{:.10f}".format(statistics.pstdev(values)) if values else "")
                record[metric + "_missing"] = ";".join(missing)
                record[metric + "_invalid"] = ";".join(invalid)
                missing_all.extend(metric + ":" + value for value in missing)
                invalid_all.extend(metric + ":" + value for value in invalid)
            record["mcd_aggregate_format"] = aggregate_line(out / "MCD.txt", gen)
            record["f0_aggregate_format"] = aggregate_line(out / "F0.txt", gen)
            record["genuine_missing_values"] = ";".join(missing_all)
            record["invalid_values"] = ";".join(invalid_all)
            record["coverage_ok"] = (
                len(wavs) == EXPECTED_WAVS
                and all(record[metric + "_count"] == EXPECTED_WAVS
                        for metric in SIDE_CARS)
                and not missing_all
                and not invalid_all
            )
            record.update(row_jobs(
                folder, configuration, gen, old_by_gen, fresh_by_gen))
            record["finalizer_job_id"] = os.environ.get("SLURM_JOB_ID", "manual")
            rows.append(record)
    if len(rows) != sum(EXPECTED_PAIR_COUNTS.values()):
        raise RuntimeError("expected 38 summary rows, found {}".format(len(rows)))
    if not all(row["coverage_ok"] for row in rows):
        bad = [(row["folder"], row["sweep_or_configuration"])
               for row in rows if not row["coverage_ok"]]
        raise RuntimeError("speech metric coverage failed for {}".format(bad))
    return rows


def build_training_loss(inventory):
    fresh = []
    validations = []
    seen = set()
    for folder in sorted(inventory):
        for pair in inventory[folder]:
            configuration = configuration_from_gen(pair["gen"])
            key = (folder, configuration)
            if key in seen:
                raise RuntimeError("duplicate training configuration {}".format(key))
            seen.add(key)
            log_path = (ROOT / folder / ("Hydra_" + folder) / configuration
                        / "tb" / "train.log")
            if not log_path.is_file() or log_path.stat().st_size == 0:
                raise RuntimeError("missing or empty training log {}".format(log_path))
            matches = LOSS_PATTERN.findall(
                log_path.read_text(encoding="utf-8", errors="replace"))
            final = matches[-len(EXPECTED_EPOCHS):]
            epochs = [int(match[0]) for match in final]
            if epochs != EXPECTED_EPOCHS:
                raise RuntimeError(
                    "{} {} final losses are not epochs 1..599".format(
                        folder, configuration))
            for epoch_text, duration_text, prior_text, diffusion_text in final:
                duration = float(duration_text)
                prior = float(prior_text)
                diffusion = float(diffusion_text)
                if not all(math.isfinite(value)
                           for value in (duration, prior, diffusion)):
                    raise RuntimeError("non-finite loss for {} {} epoch {}".format(
                        folder, configuration, epoch_text))
                fresh.append({
                    "experiment": folder,
                    "sweep_or_configuration": configuration,
                    "epoch": epoch_text,
                    "loss": "{:.12g}".format(duration + prior + diffusion),
                    "loss_definition": "duration_loss+prior_loss+diffusion_loss",
                    "duration_loss": "{:.12g}".format(duration),
                    "prior_loss": "{:.12g}".format(prior),
                    "diffusion_loss": "{:.12g}".format(diffusion),
                    "source_log": str(log_path),
                })
            validations.append({
                "experiment": folder,
                "sweep_or_configuration": configuration,
                "expected_epochs": len(EXPECTED_EPOCHS),
                "observed_final_epochs": len(final),
                "earlier_records_ignored": len(matches) - len(final),
                "source_log": str(log_path),
                "status": "complete",
            })
    if len(seen) != sum(EXPECTED_PAIR_COUNTS.values()):
        raise RuntimeError("expected 38 training configurations, found {}".format(len(seen)))
    if len(fresh) != EXPECTED_LOSS_RECORDS:
        raise RuntimeError("expected {} loss rows, found {}".format(
            EXPECTED_LOSS_RECORDS, len(fresh)))

    preserved = []
    output = ROOT / "training_loss.log"
    if output.is_file():
        with output.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != LOSS_FIELDS:
                raise RuntimeError("unexpected training_loss.log header {}".format(
                    reader.fieldnames))
            preserved = [row for row in reader
                         if row.get("experiment") not in EXPECTED_FOLDERS]
    return fresh, preserved, validations


def accounting_audit(rows):
    job_fields = [
        "environment_probe_job_id", "checkpoint_validation_job_id",
        "run_job_id", "gen_repair_job_id", "metrics_job_id", "wer_job_id",
        "utmosv2_job_id", "acceptance_gate_job_id", "training_loss_job_id",
        "dispatch_job_id", "source_summary_job_id",
        "source_final_audit_job_id",
    ]
    requested = set()
    for row in rows:
        requested.update(row.get(field, "") for field in job_fields)
    bridge_path = RESULTS / "tan18_all_bridge_job_id.txt"
    if bridge_path.is_file():
        requested.add(bridge_path.read_text(encoding="utf-8").strip())
    requested.discard("")
    if not requested:
        raise RuntimeError("job audit has no requested job IDs")
    command = [
        "sacct", "-X", "-n", "-P", "-j", ",".join(sorted(requested)),
        "--format=JobID,JobIDRaw,JobName,State,ExitCode,Start,End,Elapsed,NodeList",
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    if result.returncode:
        raise RuntimeError("sacct audit failed: {}".format(result.stderr.strip()))
    observed = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 9:
            continue
        observed.append({
            "observed_job_id": fields[0].strip(),
            "job_id_raw": fields[1].strip(),
            "job_name": fields[2].strip(),
            "state": fields[3].strip(),
            "exit_code": fields[4].strip(),
            "start": fields[5].strip(),
            "end": fields[6].strip(),
            "elapsed": fields[7].strip(),
            "node_list": fields[8].strip(),
        })
    audit = []
    failures = []
    array_counts = {"11097445": 5, "11142109": 5}
    for job_id in sorted(requested):
        exact = [row for row in observed if row["observed_job_id"] == job_id]
        if job_id in array_counts:
            pattern = re.compile(re.escape(job_id) + r"_[0-4]$")
            selected = [row for row in observed
                        if pattern.fullmatch(row["observed_job_id"])]
            if len(selected) != array_counts[job_id]:
                failures.append("{} expected {} array tasks, found {}".format(
                    job_id, array_counts[job_id], len(selected)))
        elif exact:
            selected = exact
        else:
            selected = []
            failures.append("{} absent from sacct".format(job_id))
        for row in selected:
            status = ("complete" if row["state"].startswith("COMPLETED")
                      and row["exit_code"] == "0:0" else "failed")
            record = dict(row)
            record["requested_job_id"] = job_id
            record["status"] = status
            audit.append(record)
            if status != "complete":
                failures.append("{} observed {} {} {}".format(
                    job_id, row["observed_job_id"], row["state"], row["exit_code"]))
    if failures:
        raise RuntimeError("job accounting audit failed: {}".format(failures))
    return audit


def markdown_report(rows, inventory, audit, loss_rows, preserved_rows):
    finalizer = os.environ.get("SLURM_JOB_ID", "manual")
    bridge_path = RESULTS / "tan18_all_bridge_job_id.txt"
    bridge = bridge_path.read_text(encoding="utf-8").strip()
    fmdt_summary = rows[[row["folder"] for row in rows].index("fmDT")][
        "source_summary_job_id"]
    lines = [
        "# TAN-18 all-folder speech-metric report",
        "",
        "This report covers every first-depth `run.sh` discovered at finalization: "
        "11 folders and 38 generated paths. Arithmetic means and population standard "
        "deviations (`ddof=0`) are recomputed from scalar files beside each WAV.",
        "",
        "Training loss contains {} current-scope records (38 configurations x 599 "
        "epochs) plus {} preserved out-of-scope rows.".format(
            len(loss_rows), len(preserved_rows)),
        "",
        "Common orchestration: legacy all-folder dispatcher `11099159`, fmDT "
        "environment probe `11141986`, checkpoint validation `11142108`, fresh "
        "fmDT run `11142109_[0-4]`, gate `11142110`, training-loss job "
        "`11142111`, dispatcher `11142112`, fmDT summary `{}`, all-folder bridge "
        "`{}`, and all-folder finalizer `{}`.".format(
            fmdt_summary, bridge, finalizer),
        "",
        "## Final run.sh inventory",
        "",
        "| Folder | run.sh | Credited run job | Extracted paths |",
        "|---|---|---:|---:|",
    ]
    for folder in sorted(inventory):
        lines.append("| `{}` | `{}/run.sh` | `{}` | {} |".format(
            folder, folder, RUN_JOB_IDS[folder], len(inventory[folder])))
    lines.extend([
        "",
        "## Per-path results",
        "",
        "| Folder | Sweep/configuration | WAVs | MCD n / mean / std | logF0/F0 n / mean / std | WER n / mean / std | UTMOSv2 n / mean / std | Jobs: run task / metric / WER / UTMOSv2 | Coverage |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in rows:
        def stats(metric):
            return "{} / {} / {}".format(
                row[metric + "_count"], row[metric + "_mean"],
                row[metric + "_std"])
        lines.append(
            "| `{folder}` | `{sweep_or_configuration}` | {wav_count} | {mcd} | "
            "{logf0} | {wer} | {utmosv2} | `{run_task_job_id}` / `{metrics_job_id}` / "
            "`{wer_job_id}` / `{utmosv2_job_id}` | OK |".format(
                mcd=stats("mcd"), logf0=stats("logf0"), wer=stats("wer"),
                utmosv2=stats("utmosv2"), **row))
        lines.extend([
            "",
            "- Gen: `{}`".format(row["gen"]),
            "- Out: `{}`".format(row["out"]),
            "- MCD aggregate: `{}`".format(row["mcd_aggregate_format"]),
            "- F0 aggregate: `{}`".format(row["f0_aggregate_format"]),
            "- Genuine missing values: `{}`".format(
                row["genuine_missing_values"] or "none"),
            "- Invalid values: `{}`".format(row["invalid_values"] or "none"),
            "",
        ])
    lines.extend([
        "## Validation",
        "",
        "- Final first-depth rescan: 11 `run.sh` folders, including `fm` and `scorex0`.",
        "- Extractor coverage: 38 unique gen/out pairs.",
        "- Per-path coverage: 488 WAVs and 488 finite values for each of MCD, "
        "logF0/F0, WER, and UTMOSv2.",
        "- Training loss: 38 configurations, epochs 1-599, {} records.".format(
            len(loss_rows)),
        "- Terminal accounting: {} successful allocation/task records across every "
        "referenced run, repair, metrics, WER, UTMOSv2, gate, loss, dispatcher, "
        "probe, checkpoint, summary, bridge, and prior audit job.".format(
            len(audit)),
        "- Failed fmDT attempts `11141050`, `11141599`, `11141635` and "
        "`11141980` remain explicitly uncredited; their cancelled dependency "
        "children are not used in any result.",
        "- Genuine missing or invalid metric values: none.",
        "",
        "Machine-readable results: `tan18_all_summary.csv`.",
        "Training-loss validation: `tan18_all_training_loss_validation.tsv`.",
        "Accounting evidence: `tan18_all_job_audit.tsv`.",
    ])
    return "\n".join(lines) + "\n"


def main():
    run_paths = sorted(ROOT.glob("*/run.sh"))
    folders = {path.parent.name for path in run_paths}
    if folders != EXPECTED_FOLDERS:
        raise RuntimeError("final run.sh inventory changed: expected={} observed={}".format(
            sorted(EXPECTED_FOLDERS), sorted(folders)))
    inventory = {}
    for run_path in run_paths:
        folder = run_path.parent.name
        inventory[folder] = extract_pairs(run_path)
    counts = {folder: len(pairs) for folder, pairs in inventory.items()}
    if counts != EXPECTED_PAIR_COUNTS:
        raise RuntimeError("extractor pair counts changed: {}".format(counts))
    gen_paths = [pair["gen"] for pairs in inventory.values() for pair in pairs]
    out_paths = [pair["out"] for pairs in inventory.values() for pair in pairs]
    if len(gen_paths) != len(set(gen_paths)) or len(out_paths) != len(set(out_paths)):
        raise RuntimeError("duplicate gen or out paths in final inventory")

    rows = summarize_pairs(inventory)
    loss_rows, preserved_rows, loss_validation = build_training_loss(inventory)
    audit = accounting_audit(rows)

    summary_fields = [
        "folder", "pair_index", "sweep_or_configuration", "gen", "out",
        "wav_count", "mcd_count", "mcd_mean", "mcd_std", "mcd_missing",
        "mcd_invalid", "logf0_count", "logf0_mean", "logf0_std",
        "logf0_missing", "logf0_invalid", "wer_count", "wer_mean", "wer_std",
        "wer_missing", "wer_invalid", "utmosv2_count", "utmosv2_mean",
        "utmosv2_std", "utmosv2_missing", "utmosv2_invalid", "coverage_ok",
        "genuine_missing_values", "invalid_values", "environment_probe_job_id",
        "checkpoint_validation_job_id", "run_job_id", "run_task_job_id",
        "gen_repair_job_id", "metrics_job_id", "wer_job_id", "utmosv2_job_id",
        "acceptance_gate_job_id", "training_loss_job_id", "dispatch_job_id",
        "source_summary_job_id", "source_final_audit_job_id", "finalizer_job_id",
        "mcd_aggregate_format", "f0_aggregate_format",
    ]
    validation_fields = [
        "experiment", "sweep_or_configuration", "expected_epochs",
        "observed_final_epochs", "earlier_records_ignored", "source_log", "status",
    ]
    audit_fields = [
        "requested_job_id", "observed_job_id", "job_id_raw", "job_name", "state",
        "exit_code", "start", "end", "elapsed", "node_list", "status",
    ]
    inventory_rows = [
        {
            "folder": folder,
            "run_path": str(ROOT / folder / "run.sh"),
            "run_job_id": RUN_JOB_IDS[folder],
            "pair_count": len(inventory[folder]),
            "status": "complete",
        }
        for folder in sorted(inventory)
    ]

    atomic_write(
        ROOT / "training_loss.log",
        csv_text(LOSS_FIELDS, preserved_rows + loss_rows, delimiter="\t"),
    )
    atomic_write(
        RESULTS / "tan18_all_training_loss_validation.tsv",
        csv_text(validation_fields, loss_validation, delimiter="\t"),
    )
    atomic_write(
        RESULTS / "tan18_all_summary.csv",
        csv_text(summary_fields, rows),
    )
    atomic_write(
        RESULTS / "tan18_all_job_audit.tsv",
        csv_text(audit_fields, audit, delimiter="\t"),
    )
    atomic_write(
        RESULTS / "tan18_all_inventory.tsv",
        csv_text(
            ["folder", "run_path", "run_job_id", "pair_count", "status"],
            inventory_rows, delimiter="\t"),
    )
    atomic_write(
        RESULTS / "tan18_all_report.md",
        markdown_report(rows, inventory, audit, loss_rows, preserved_rows),
    )
    status = (
        "complete\nfolders=11\npairs=38\nwavs_per_pair=488\n"
        "summary_rows=38\nloss_configurations=38\nloss_epoch_records={}\n"
        "preserved_out_of_scope_loss_records={}\naudited_job_records={}\n"
        "missing_values=0\ninvalid_values=0\n"
    ).format(EXPECTED_LOSS_RECORDS, len(preserved_rows), len(audit))
    atomic_write(RESULTS / "tan18_all_final_status.txt", status)
    print(
        "TAN-18 all-folder finalization complete: folders=11 pairs=38 "
        "loss_records={} audited_job_records={}".format(
            EXPECTED_LOSS_RECORDS, len(audit)),
        flush=True,
    )


if __name__ == "__main__":
    main()
