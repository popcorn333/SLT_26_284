#!/usr/bin/env python3
import csv
import math
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet")
RES = ROOT / ".symphony/results"
EXPECTED_FOLDERS = {"add", "blur", "fm", "fmDT", "levy", "mask",
                    "one-shot", "product", "score", "scoreDT", "scorex0"}
EXPECTED_PAIRS = 30
EXPECTED_LOSS_CONFIGS = 30
EXPECTED_LOSS_RECORDS = 17970
LOSS_PATTERN = re.compile(
    r"^Epoch (\d+): duration loss = ([0-9eE+.-]+) \| "
    r"prior loss = ([0-9eE+.-]+) \| diffusion loss = ([0-9eE+.-]+)$",
    re.M,
)
TRANSCRIPT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/add/resources/filelists/ljspeech/test.txt")
GROUND_TRUTH = "/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/ground_truth"
METRICS_ENV = "/users/acp23xt/.conda/envs/Grad-TTS-EVAL"
METRICS = ("mcd", "logf0", "wer", "utmosv2")
METRIC_SUFFIXES = {
    "mcd": "_mcd.txt",
    "logf0": "_f0.txt",
    "wer": "_wer_l.txt",
    "utmosv2": "_utmosv2.txt",
}
STAGES = {"run", "gen_repair", "metrics", "wer", "utmosv2"}
EXPECTED_RUN_IDS = {
    "add": "10800233", "blur": "10800211", "fm": "10803966",
    "fmDT": "10800235", "levy": "10800210", "mask": "10800213",
    "one-shot": "10800207", "product": "10800236",
    "score": "10804165", "scoreDT": "10920857", "scorex0": "10943691",
}
RUN_LINEAGE = (
    ("add", "10800204", {"FAILED"}),
    ("fmDT", "10800205", {"FAILED"}),
    ("product", "10800206", {"FAILED"}),
    ("fm", "10800208", {"FAILED"}),
    ("scoreDT", "10800209", {"FAILED"}),
    ("score", "10800212", {"TIMEOUT"}),
    # This replacement allocation was cancelled before it started.  Retain it
    # in the submitted-run ledger, but never credit it as completed work.
    ("scoreDT", "10803834", {"CANCELLED"}),
    ("scorex0", "10803830", {"COMPLETED"}),
    # The former scorex0 rescue completed successfully but was superseded by
    # the explicitly requested fresh scorex0 training/inference run.
    ("scorex0", "10804166", {"COMPLETED"}),
    # The long parent cannot complete its remaining sequential sweep within
    # its wall-time.  It is retained as lineage only; the dependency-gated
    # continuation tasks and their validation gate are the authoritative phase.
    ("scoreDT", "10884228", {"COMPLETED", "TIMEOUT"}),
    # The original GPU continuation never started and was cancelled after the
    # completed parent was independently shown to contain every epoch and
    # checkpoint.  A CPU-only array now records that validation without
    # consuming GPUs or mutating training outputs.
    ("scoreDT", "10920856", {"CANCELLED"}),
    ("scoreDT", "10932976_0", {"COMPLETED"}),
    ("scoreDT", "10932976_1", {"COMPLETED"}),
    ("scoreDT", "10932976_2", {"COMPLETED"}),
    ("scoreDT", "10932976_3", {"COMPLETED"}),
    ("scoreDT", "10932976_4", {"COMPLETED"}),
)

# These unsuccessful historical allocations have no surviving Slurm stdout or
# stderr file.  Their exact terminal accounting remains mandatory and the
# absence is reported explicitly; successful allocations may never use this
# exception.
HISTORICAL_RUNS_WITHOUT_TERMINAL_LOG = {
    "10800209", "10803834", "10920856",
}


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_key_values(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("\t", 1)
        result[key] = value
    return result


def assert_under_root(value, label):
    try:
        Path(value).resolve().relative_to(ROOT.resolve())
    except ValueError:
        raise RuntimeError("{} path is outside approved repository tree: {}".format(label, value))


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
                raise RuntimeError("out without gen while re-extracting {}".format(folder))
            pairs.append((gen, line[4:]))
            gen = None
    if gen is not None or not pairs:
        raise RuntimeError("incomplete or empty re-extracted mapping for {}".format(folder))
    return pairs


def validate_metrics_script(folder, final_pair):
    path = ROOT / folder / "metrics.sh"
    lines = path.read_text(encoding="utf-8").splitlines()
    assignments = {}
    for name in ("gt", "gen", "out"):
        matches = [line.split("=", 1)[1] for line in lines
                   if line.startswith(name + "=")]
        if len(matches) != 1:
            raise RuntimeError("{} must contain exactly one {} assignment".format(path, name))
        assignments[name] = matches[0]
    if assignments["gt"] != GROUND_TRUTH:
        raise RuntimeError("{} ground-truth assignment changed".format(path))
    if (assignments["gen"], assignments["out"]) != final_pair:
        raise RuntimeError("{} does not retain its final emitted gen/out pair".format(path))

    commands = [" ".join(line.split()) for line in lines]
    for tool in ("tools/F0/F0.py", "tools/MCD/MCD.py"):
        matching = [line for line in commands if tool in line]
        if len(matching) != 1:
            raise RuntimeError("{} must retain exactly one {} command".format(path, tool))
        command = matching[0]
        for argument in ("--gt_wavdir_or_wavscp", '"$gt"',
                         "--gen_wavdir_or_wavscp", '"$gen"',
                         "--outdir", '"$out"'):
            if argument not in command:
                raise RuntimeError("{} {} command lost {}".format(path, tool, argument))
    if not any(re.match(r"^set\s+-[^\s]*e", line.strip()) for line in lines):
        raise RuntimeError("{} does not fail fast".format(path))
    activation = "source activate {}".format(METRICS_ENV)
    if lines.count(activation) != 1:
        raise RuntimeError("{} does not activate the established metrics environment by absolute path".format(
            path))


def validate_aggregate_metric(out_dir, filename, gen):
    path = Path(out_dir) / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("missing aggregate metric output {}".format(path))
    matching = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(gen):
            continue
        parts = line.split()
        if len(parts) != 4 or parts[0] != gen or parts[2] != r"\pm":
            continue
        try:
            mean, std = float(parts[1]), float(parts[3])
        except ValueError:
            continue
        if math.isfinite(mean) and math.isfinite(std):
            matching.append((mean, std))
    if not matching:
        raise RuntimeError("{} has no finite aggregate record for {}".format(path, gen))


def validate_rendered_scoring_scripts():
    requirements = {
        RES / "wer_array.sbatch": (
            "#SBATCH --gres=gpu:a100:1",
            "batch_wer_epoch495.py",
            "--model large-v3",
            "--output-suffix _wer_l.txt",
            "--overwrite",
            str(TRANSCRIPT),
        ),
        RES / "utmos_array.sbatch": (
            "#SBATCH --gres=gpu:1",
            "score_ourdir_utmosv2.py",
            "--device auto",
            "--overwrite",
            "UTMOSV2_CHACHE=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/utmosv2",
        ),
        RES / "utmos_warmup.sbatch": (
            "#SBATCH --gres=gpu:1",
            "utmosv2.create_model(pretrained=True, device=\"cuda:0\")",
            "UTMOSV2_CHACHE=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/utmosv2",
        ),
        RES / "scorex0_redo_wer.sbatch": (
            "#SBATCH --gres=gpu:a100:1",
            "scorex0_redo_mappings.tsv",
            "batch_wer_epoch495.py",
            "--model large-v3",
            "--output-suffix _wer_l.txt",
            "--overwrite",
            str(TRANSCRIPT),
        ),
        RES / "scorex0_redo_utmos.sbatch": (
            "#SBATCH --gres=gpu:1",
            "scorex0_redo_mappings.tsv",
            "score_ourdir_utmosv2.py",
            "--device auto",
            "--overwrite",
            "UTMOSV2_CHACHE=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/utmosv2",
        ),
    }
    for path, tokens in requirements.items():
        if not path.is_file():
            raise RuntimeError("missing rendered scoring script {}".format(path))
        source = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in source:
                raise RuntimeError("{} lost required token {}".format(path, token))


def accounting_record(job_id):
    terminal_states = {
        "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
        "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED",
        "SPECIAL_EXIT", "TIMEOUT",
    }
    last_error = None
    record = None
    # Dependencies can release a few seconds before sacct exposes the final
    # exact task record.  Retry only inside this short Slurm audit allocation;
    # never accept a nonterminal state as completed evidence.
    for attempt in range(6):
        try:
            output = subprocess.check_output([
                "/usr/bin/sacct", "-X", "-j", job_id,
                "--starttime=2026-07-01",
                # This cluster assigns each array task a distinct numeric
                # JobIDRaw (for example, alias 10932976_0 has raw ID
                # 10932977).  Successful task ledgers retain Slurm's stable
                # array aliases, while a never-started compact array has only
                # its raw master ID available to query.  Request and match
                # both fields, then normalize the record back to the stable
                # eight-column audit schema below.
                "--format=JobID,JobIDRaw,JobName%40,State,ExitCode,Elapsed,Start,End,NodeList%30",
                "-n", "-P",
            ], universal_newlines=True)
            records = [line.split("|") for line in output.splitlines() if line.strip()]
            raw_record = next(
                (parts for parts in records
                 if parts[0] == job_id or parts[1] == job_id), None)
            record = ([raw_record[0]] + raw_record[2:]) if raw_record else None
            if record is not None:
                state = record[2].split()[0].rstrip("+")
                if state in terminal_states:
                    return record
                last_error = "nonterminal accounting state {}".format(record[2])
            else:
                last_error = "exact accounting record is not visible"
        except subprocess.CalledProcessError as error:
            last_error = "sacct failed with exit code {}".format(error.returncode)
        if attempt != 5:
            time.sleep(10)
    raise RuntimeError("no terminal exact accounting record for {}: {}".format(
        job_id, last_error))


def accounting(job_id):
    record = accounting_record(job_id)
    if record[2].rstrip("+") != "COMPLETED" or record[3] != "0:0":
        raise RuntimeError("unsuccessful accounting for {}: {}".format(job_id, record))
    return record


def log_candidates(job_id, folders):
    # Match numeric boundaries explicitly.  A plain substring glob for array
    # task ``123_1`` also matches ``123_10`` through ``123_19``, which would
    # misattribute terminal logs even though every task has its own file.
    job_pattern = re.compile(r"(?<![0-9]){}(?![0-9])".format(
        re.escape(job_id)))
    directories = [RES, ROOT]
    directories.extend(ROOT / folder for folder in folders)
    candidates = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        candidates.update(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix in {".out", ".err"}
            and job_pattern.search(path.name))
    return sorted(candidates)


def tail_line(path):
    last = ""
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.strip():
                last = line.strip()
    return last.replace("\t", " ")[:1000]


def main():
    if "10884228" in (ROOT / "scoreDT/inference.py").read_text(encoding="utf-8"):
        raise RuntimeError("temporary TAN-6 scoreDT inference guard was not removed")

    actual = {path.parent.name for path in ROOT.glob("*/run.sh")
              if path.parent.parent == ROOT}
    if actual != EXPECTED_FOLDERS:
        raise RuntimeError("final run.sh rescan changed: expected={} actual={}".format(
            sorted(EXPECTED_FOLDERS), sorted(actual)))

    emitted = {}
    emitted_by_folder = {}
    emitted_loss_keys = set()
    for folder in sorted(actual):
        pairs = extract_pairs(folder)
        emitted_by_folder[folder] = pairs
        for index, (gen, out) in enumerate(pairs):
            key = (folder, str(index))
            emitted[key] = {"gen": gen, "out": out}
            assert_under_root(gen, "re-extracted gen")
            assert_under_root(out, "re-extracted out")
            gen_path = Path(gen)
            if gen_path.name != "Epoch_495" or gen_path.parent.name != "converted" or \
                    gen_path.parents[1].name != "test":
                raise RuntimeError("unexpected generated path layout for {}: {}".format(key, gen))
            emitted_loss_keys.add((folder, gen_path.parents[2].name))
    if len(emitted) != EXPECTED_PAIRS:
        raise RuntimeError("final run.sh extraction emitted {} pairs, expected {}".format(
            len(emitted), EXPECTED_PAIRS))
    if len(emitted_loss_keys) != EXPECTED_LOSS_CONFIGS:
        raise RuntimeError("final run.sh emissions identify {} training configurations, expected {}".format(
            len(emitted_loss_keys), EXPECTED_LOSS_CONFIGS))

    if (RES / "validation_status.txt").read_text(encoding="utf-8").strip() != "complete":
        raise RuntimeError("metric coverage validation is not complete")
    validate_rendered_scoring_scripts()
    loss_status = (RES / "training_loss_status.txt").read_text(encoding="utf-8")
    if not loss_status.startswith("complete\n"):
        raise RuntimeError("training-loss validation is not complete")

    mappings = read_tsv(RES / "mappings.tsv")
    jobs = read_tsv(RES / "jobs.tsv")
    job_lookup = {(row["folder"], row["pair_index"], row["stage"]): row["job_id"]
                  for row in jobs}
    with (RES / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    loss_validation = read_tsv(RES / "training_loss_validation.tsv")
    if len(mappings) != EXPECTED_PAIRS or len(summary) != EXPECTED_PAIRS:
        raise RuntimeError("mapping/summary rows must both equal {}".format(EXPECTED_PAIRS))
    mapping_keys = {(row["folder"], row["pair_index"]) for row in mappings}
    summary_keys = {(row["folder"], row["pair_index"]) for row in summary}
    if len(mapping_keys) != EXPECTED_PAIRS or mapping_keys != summary_keys:
        raise RuntimeError("mapping/summary keys are incomplete or duplicated")
    if mapping_keys != set(emitted):
        raise RuntimeError("stored mappings do not cover the final run.sh emissions")
    if {row["folder"] for row in mappings} != EXPECTED_FOLDERS:
        raise RuntimeError("mapping folder coverage mismatch")
    for field in ("gen", "out"):
        values = [row[field] for row in mappings]
        if len(set(values)) != EXPECTED_PAIRS:
            raise RuntimeError("mapping {} paths are incomplete or duplicated".format(field))
    mapping_lookup = {(row["folder"], row["pair_index"]): row for row in mappings}
    for row in mappings:
        assert_under_root(row["gen"], "gen")
        assert_under_root(row["out"], "out")
        key = (row["folder"], row["pair_index"])
        if row["gen"] != emitted[key]["gen"] or row["out"] != emitted[key]["out"]:
            raise RuntimeError("stored mapping differs from final run.sh emission for {}".format(key))
        validate_aggregate_metric(row["out"], "MCD.txt", row["gen"])
        validate_aggregate_metric(row["out"], "F0.txt", row["gen"])
    for folder, pairs in emitted_by_folder.items():
        validate_metrics_script(folder, pairs[-1])

    transcript_lines = [line for line in TRANSCRIPT.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    expected_wav_names = {Path(line.split("|", 1)[0]).name for line in transcript_lines}
    expected_wavs = len(transcript_lines)
    if expected_wavs != 488:
        raise RuntimeError("transcript-derived expected WAV count changed: {}".format(expected_wavs))
    if len(expected_wav_names) != expected_wavs:
        raise RuntimeError("transcript contains duplicate WAV basenames")
    for row in summary:
        mapping = mapping_lookup[(row["folder"], row["pair_index"])]
        if row["gen"] != mapping["gen"] or row["out"] != mapping["out"]:
            raise RuntimeError("summary path mismatch for {} pair {}".format(
                row["folder"], row["pair_index"]))
        for field, stage in (("run_job_id", "run"),
                             ("gen_repair_job_id", "gen_repair"),
                             ("metrics_job_id", "metrics"),
                             ("wer_job_id", "wer"),
                             ("utmosv2_job_id", "utmosv2")):
            expected_job_id = job_lookup.get((row["folder"], row["pair_index"], stage))
            if row[field] != expected_job_id:
                raise RuntimeError("summary {} mismatch for {} pair {}".format(
                    field, row["folder"], row["pair_index"]))
        wav_count = int(row["wav_count"])
        if wav_count != expected_wavs:
            raise RuntimeError("{} pair {} WAV count {} != {}".format(
                row["folder"], row["pair_index"], wav_count, expected_wavs))
        wavs = sorted(Path(row["gen"]).glob("*.wav"))
        if len(wavs) != wav_count:
            raise RuntimeError("summary/filesystem WAV count mismatch for {} pair {}".format(
                row["folder"], row["pair_index"]))
        observed_wav_names = {wav.name for wav in wavs}
        if observed_wav_names != expected_wav_names:
            raise RuntimeError("WAV-name coverage mismatch for {} pair {}".format(
                row["folder"], row["pair_index"]))
        for metric in METRICS:
            if int(row[metric + "_count"]) != wav_count:
                raise RuntimeError("{} coverage mismatch for {} pair {}".format(
                    metric, row["folder"], row["pair_index"]))
            if row[metric + "_missing"] or row[metric + "_invalid"]:
                raise RuntimeError("{} missing/invalid values for {} pair {}".format(
                    metric, row["folder"], row["pair_index"]))
            values = []
            for wav in wavs:
                sidecar = wav.with_name(wav.stem + METRIC_SUFFIXES[metric])
                try:
                    value = float(sidecar.read_text(encoding="utf-8").strip())
                except (OSError, ValueError) as error:
                    raise RuntimeError("invalid {} sidecar {}: {}".format(
                        metric, sidecar, error))
                if not math.isfinite(value):
                    raise RuntimeError("non-finite {} sidecar {}".format(metric, sidecar))
                values.append(value)
            recomputed = {
                metric + "_mean": statistics.mean(values),
                metric + "_std": statistics.pstdev(values),
            }
            for field in (metric + "_mean", metric + "_std"):
                if not math.isfinite(float(row[field])):
                    raise RuntimeError("non-finite {} for {} pair {}".format(
                        field, row["folder"], row["pair_index"]))
                expected_field = "{:.8f}".format(recomputed[field])
                if row[field] != expected_field:
                    raise RuntimeError(
                        "{} mismatch for {} pair {}: stored={} recomputed={}".format(
                            field, row["folder"], row["pair_index"],
                            row[field], expected_field))

    job_keys = {(row["folder"], row["pair_index"], row["stage"]) for row in jobs}
    expected_job_keys = {(folder, pair, stage)
                         for folder, pair in mapping_keys for stage in STAGES}
    if len(jobs) != EXPECTED_PAIRS * len(STAGES) or job_keys != expected_job_keys:
        raise RuntimeError("jobs.tsv does not cover every mapping/stage exactly once")
    repair_rows = read_tsv(RES / "gen_repair_jobs.tsv")
    repair_lookup = {(row["folder"], row["pair_index"]): row["job_id"]
                     for row in repair_rows}
    if len(repair_rows) != EXPECTED_PAIRS or set(repair_lookup) != mapping_keys:
        raise RuntimeError("gen-repair job ledger does not exactly cover final mappings")
    for row in jobs:
        mapping = mapping_lookup[(row["folder"], row["pair_index"])]
        if row["gen"] != mapping["gen"] or row["out"] != mapping["out"]:
            raise RuntimeError("job path mismatch for {} pair {} stage {}".format(
                row["folder"], row["pair_index"], row["stage"]))
        if row["stage"] == "run" and row["job_id"] != EXPECTED_RUN_IDS[row["folder"]]:
            raise RuntimeError("non-authoritative run job for {} pair {}".format(
                row["folder"], row["pair_index"]))
        if row["stage"] == "gen_repair" and row["job_id"] != repair_lookup[
                (row["folder"], row["pair_index"])]:
            raise RuntimeError("non-authoritative generation repair for {} pair {}".format(
                row["folder"], row["pair_index"]))

    if len(loss_validation) != EXPECTED_LOSS_CONFIGS:
        raise RuntimeError("training-loss configuration count mismatch: {}".format(
            len(loss_validation)))
    if any(row["status"] != "complete" or
           row["expected_epochs"] != row["observed_unique_epochs"]
           for row in loss_validation):
        raise RuntimeError("incomplete training-loss validation row")
    expected_loss_records = sum(int(row["expected_epochs"]) for row in loss_validation)
    if expected_loss_records != EXPECTED_LOSS_RECORDS:
        raise RuntimeError("training-loss expected records mismatch: {}".format(
            expected_loss_records))
    with (ROOT / "training_loss.log").open(newline="", encoding="utf-8") as handle:
        loss_reader = csv.DictReader(handle, delimiter="\t")
        required_loss_fields = {
            "experiment", "sweep_or_configuration", "epoch", "loss",
            "loss_definition", "duration_loss", "prior_loss",
            "diffusion_loss", "source_log",
        }
        if loss_reader.fieldnames is None or not required_loss_fields.issubset(
                set(loss_reader.fieldnames)):
            raise RuntimeError("training_loss.log is missing required labelled fields")
        loss_records = list(loss_reader)
    observed_loss_records = len(loss_records)
    if observed_loss_records != EXPECTED_LOSS_RECORDS:
        raise RuntimeError("training-loss row mismatch: {}".format(observed_loss_records))
    validation_epochs = {}
    validation_sources = {}
    source_loss_records = {}
    for row in loss_validation:
        key = (row["experiment"], row["sweep_or_configuration"])
        if key in validation_epochs:
            raise RuntimeError("duplicate training-loss validation configuration: {}".format(key))
        source_candidates = list(
            (ROOT / row["experiment"]).glob(
                "Hydra_*/{}/tb/train.log".format(row["sweep_or_configuration"])))
        if len(source_candidates) != 1 or \
                source_candidates[0].resolve() != Path(row["source_log"]).resolve():
            raise RuntimeError("training-loss source log does not match the final sweep output for {}".format(key))
        validation_epochs[key] = int(row["expected_epochs"])
        validation_sources[key] = source_candidates[0].resolve()
        parsed = {}
        for epoch, duration, prior, diffusion in LOSS_PATTERN.findall(
                source_candidates[0].read_text(encoding="utf-8", errors="replace")):
            parsed[int(epoch)] = (float(duration), float(prior), float(diffusion))
        source_loss_records[key] = parsed
    if set(validation_epochs) != emitted_loss_keys:
        raise RuntimeError("training-loss configurations do not match final run.sh sweep emissions")
    observed_epochs = {}
    record_keys = set()
    for row in loss_records:
        key = (row["experiment"], row["sweep_or_configuration"])
        if key not in validation_epochs:
            raise RuntimeError("unexpected training-loss configuration: {}".format(key))
        try:
            epoch = int(row["epoch"])
            loss = float(row["loss"])
            components = tuple(float(row[field]) for field in (
                "duration_loss", "prior_loss", "diffusion_loss"))
        except (TypeError, ValueError):
            raise RuntimeError("invalid training-loss record: {}".format(row))
        if not math.isfinite(loss) or not all(math.isfinite(value) for value in components):
            raise RuntimeError("non-finite training loss for {} epoch {}".format(key, epoch))
        if row["loss_definition"] != "duration_loss+prior_loss+diffusion_loss":
            raise RuntimeError("unexpected training-loss definition for {} epoch {}".format(
                key, epoch))
        if Path(row["source_log"]).resolve() != validation_sources[key]:
            raise RuntimeError("training-loss source label mismatch for {} epoch {}".format(
                key, epoch))
        source_components = source_loss_records[key].get(epoch)
        if source_components is None:
            raise RuntimeError("training-loss epoch is absent from its source log for {} epoch {}".format(
                key, epoch))
        for observed, source in zip(components, source_components):
            if not math.isclose(observed, source, rel_tol=1e-10, abs_tol=1e-12):
                raise RuntimeError("training-loss component mismatch for {} epoch {}".format(
                    key, epoch))
        if not math.isclose(loss, sum(source_components), rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError("training-loss total mismatch for {} epoch {}".format(
                key, epoch))
        record_key = key + (epoch,)
        if record_key in record_keys:
            raise RuntimeError("duplicate training-loss epoch record: {}".format(record_key))
        record_keys.add(record_key)
        observed_epochs.setdefault(key, set()).add(epoch)
    if set(observed_epochs) != set(validation_epochs):
        raise RuntimeError("training-loss record/validation configuration mismatch")
    for key, expected in validation_epochs.items():
        expected_set = set(range(1, expected + 1))
        if observed_epochs[key] != expected_set:
            raise RuntimeError("training-loss epoch set mismatch for {}".format(key))

    pipeline_ids = read_key_values(RES / "pipeline_job_ids.tsv")
    finalizer_ids = read_key_values(RES / "final_audit_job_id.tsv")
    expected_pipeline_keys = {
        "dispatch_job_id", "wer_array_job_id", "utmosv2_warmup_job_id",
        "utmosv2_array_job_id", "summary_job_id", "pair_count",
        "original_dispatch_job_id", "original_wer_array_job_id",
        "original_utmosv2_array_job_id", "scorex0_run_job_id",
        "training_loss_job_id",
    }
    expected_finalizer_keys = {"final_audit_job_id", "loss_job_id"}
    if set(pipeline_ids) != expected_pipeline_keys or pipeline_ids["pair_count"] != str(
            EXPECTED_PAIRS):
        raise RuntimeError("pipeline job-id ledger is incomplete or stale")
    if set(finalizer_ids) != expected_finalizer_keys:
        raise RuntimeError("finalizer job-id ledger is incomplete or stale")
    if pipeline_ids["original_dispatch_job_id"] != "10890394":
        raise RuntimeError("unexpected original dispatcher allocation")
    if pipeline_ids["original_wer_array_job_id"] != "10935480" or \
            pipeline_ids["original_utmosv2_array_job_id"] != "10935482":
        raise RuntimeError("unexpected retained scoring-array allocation")
    if pipeline_ids["scorex0_run_job_id"] != EXPECTED_RUN_IDS["scorex0"]:
        raise RuntimeError("unexpected fresh scorex0 run allocation")
    if finalizer_ids["loss_job_id"] != pipeline_ids["training_loss_job_id"]:
        raise RuntimeError("training-loss ledgers disagree")
    if finalizer_ids["final_audit_job_id"] != __import__("os").environ["SLURM_JOB_ID"]:
        raise RuntimeError("final-audit allocation does not match its submitted ledger")
    mapping_tasks = {(row["folder"], row["pair_index"]): task
                     for task, row in enumerate(mappings)}
    for row in jobs:
        key = (row["folder"], row["pair_index"])
        if row["stage"] == "wer":
            if row["folder"] == "scorex0":
                expected_job_id = "{}_{}".format(
                    pipeline_ids["wer_array_job_id"], row["pair_index"])
            else:
                expected_job_id = "{}_{}".format(
                    pipeline_ids["original_wer_array_job_id"], mapping_tasks[key])
            if row["job_id"] != expected_job_id:
                raise RuntimeError("WER array task/path mapping mismatch for {}".format(key))
        if row["stage"] == "utmosv2":
            if row["folder"] == "scorex0":
                expected_job_id = "{}_{}".format(
                    pipeline_ids["utmosv2_array_job_id"], row["pair_index"])
            else:
                expected_job_id = "{}_{}".format(
                    pipeline_ids["original_utmosv2_array_job_id"], mapping_tasks[key])
            if row["job_id"] != expected_job_id:
                raise RuntimeError("UTMOSv2 array task/path mapping mismatch for {}".format(key))
    lineage_job_ids = {job_id for unused_folder, job_id, unused_states in RUN_LINEAGE}
    audited_ids = {row["job_id"] for row in jobs
                   if row["job_id"] not in lineage_job_ids}
    # Array accounting is represented by the concrete task IDs recorded in
    # jobs.tsv (for example, 12345_0), and Slurm does not guarantee a separate
    # exact accounting row for the array's submission/master ID.  Audit every
    # task above, plus the two ordinary allocations in this ledger.
    for key in ("dispatch_job_id", "original_dispatch_job_id",
                "utmosv2_warmup_job_id", "summary_job_id"):
        audited_ids.add(pipeline_ids[key])
    audited_ids.update(value for key, value in finalizer_ids.items()
                       if key != "final_audit_job_id")
    audit_rows = []
    lineage_rows = []
    log_rows = []
    missing_log_rows = []
    job_folders = {}
    for row in jobs:
        job_folders.setdefault(row["job_id"], set()).add(row["folder"])
    for job_id in sorted(audited_ids):
        record = accounting(job_id)
        audit_rows.append(record)
        candidates = log_candidates(job_id, job_folders.get(job_id, EXPECTED_FOLDERS))
        if not candidates:
            raise RuntimeError("no terminal log found for successful job {}".format(job_id))
        for path in candidates:
            log_rows.append([job_id, str(path), str(path.stat().st_size), tail_line(path)])

    for folder, job_id, allowed_states in RUN_LINEAGE:
        record = accounting_record(job_id)
        state = record[2].split()[0].rstrip("+")
        if state not in allowed_states or (state == "COMPLETED" and record[3] != "0:0"):
            raise RuntimeError("unexpected parent run state for {} {}: {}".format(
                folder, job_id, record))
        lineage_rows.append([folder] + record)
        candidates = log_candidates(job_id, {folder})
        if not candidates:
            if (job_id not in HISTORICAL_RUNS_WITHOUT_TERMINAL_LOG or
                    state == "COMPLETED"):
                raise RuntimeError("no terminal parent-run log found for {}".format(job_id))
            missing_log_rows.append([
                job_id, folder, state,
                "historical unsuccessful allocation has no surviving terminal log",
            ])
        for path in candidates:
            log_rows.append([job_id, str(path), str(path.stat().st_size), tail_line(path)])

    with (RES / "final_job_audit.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["job_id", "job_name", "state", "exit_code", "elapsed",
                         "start", "end", "node_list"])
        writer.writerows(audit_rows)
    with (RES / "run_lineage_audit.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["folder", "parent_job_id", "job_name", "state", "exit_code",
                         "elapsed", "start", "end", "node_list"])
        writer.writerows(lineage_rows)
    with (RES / "terminal_log_audit.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["job_id", "log_path", "bytes", "last_nonempty_line"])
        writer.writerows(log_rows)
    with (RES / "missing_historical_terminal_logs.tsv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["job_id", "folder", "terminal_state", "reason"])
        writer.writerows(missing_log_rows)

    metric_report = (RES / "report.md").read_text(encoding="utf-8").rstrip()
    lines = [metric_report, "", "## Training-loss coverage", "",
             "Combined labelled file: `{}` ({} epoch records across {} configurations).".format(
                 ROOT / "training_loss.log", observed_loss_records, len(loss_validation)),
             "Each `loss` is the logged duration loss + prior loss + diffusion loss; the three components and exact source log are retained in every record.", "",
             "| Experiment | Sweep/configuration | Expected epochs | Observed unique epochs | Duplicate retry records ignored |",
             "|---|---|---:|---:|---:|"]
    for row in loss_validation:
        lines.append("| {} | {} | {} | {} | {} |".format(
            row["experiment"], row["sweep_or_configuration"], row["expected_epochs"],
            row["observed_unique_epochs"], row["duplicate_records_ignored"]))
    lines += ["", "## Pipeline orchestration", "",
              "| Stage | Slurm job ID |",
              "|---|---:|",
              "| Original 30-path dispatcher | {} |".format(
                  pipeline_ids["original_dispatch_job_id"]),
              "| scorex0 redo dispatcher | {} |".format(
                  pipeline_ids["dispatch_job_id"]),
              "| Retained 25-path Whisper large-v3 WER array | {} tasks 0-24 |".format(
                  pipeline_ids["original_wer_array_job_id"]),
              "| scorex0 redo Whisper large-v3 WER array | {} |".format(
                  pipeline_ids["wer_array_job_id"]),
              "| UTMOSv2 cache warmup | {} |".format(
                  pipeline_ids["utmosv2_warmup_job_id"]),
              "| Retained 25-path UTMOSv2 scoring array | {} tasks 0-24 |".format(
                  pipeline_ids["original_utmosv2_array_job_id"]),
              "| scorex0 redo UTMOSv2 scoring array | {} |".format(
                  pipeline_ids["utmosv2_array_job_id"]),
              "| Metric coverage/summary | {} |".format(
                  pipeline_ids["summary_job_id"]),
              "| Training-loss aggregation | {} |".format(
                  finalizer_ids["loss_job_id"]),
              "| Final audit | {} |".format(finalizer_ids["final_audit_job_id"]),
              "", "## Submitted run lineage", "",
              "Unsuccessful or superseded parent attempts are recorded explicitly and are not credited as successful phases.", "",
              "| Experiment | Job ID | Job name | State | Exit code | Elapsed |",
              "|---|---:|---|---|---|---:|"]
    for row in lineage_rows:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            row[0], row[1], row[2], row[3], row[4], row[5]))
    lines += ["", "## Final validation", "",
              "- First-depth `run.sh` folders: {}".format(", ".join(sorted(actual))),
              "- Final `run.sh` re-extraction: {} exact, unique gen/out pairs.".format(len(emitted)),
              "- Mapping/summary rows: {}".format(len(mappings)),
              "- Transcript-derived WAVs per path: {}".format(expected_wavs),
              "- WAV basename coverage: exact transcript match for every generated path.",
              "- Established `metrics.sh` ground-truth, F0, and MCD commands: preserved for every folder.",
              "- Rendered GPU scoring scripts: Whisper large-v3/_wer_l and established UTMOSv2 scorer validated.",
              "- Aggregate `MCD.txt` and `F0.txt` records: finite and present for every gen path.",
              "- UTMOSv2 cache-warmup allocation: {} (successful and terminal).".format(
                  pipeline_ids["utmosv2_warmup_job_id"]),
              "- Slurm allocations/tasks audited successful: {}".format(len(audit_rows)),
              "- Parent run allocations audited as explicit rescue lineage: {}".format(len(lineage_rows)),
              "- Terminal log files inspected: {}".format(len(log_rows)),
              "- Missing historical terminal logs (unsuccessful allocations only): {}".format(
                  len(missing_log_rows)),
              "- Metric sidecar coverage: complete for every generated WAV.",
              "- Metric count/mean/population-standard-deviation fields: recomputed from sidecars and exact.",
              "- Training-loss sweep/source-log mapping: exact for every final `run.sh` emission.",
              "- Training-loss epoch coverage: complete.",
              "- Training-loss totals and components: independently matched to every labelled source-log epoch.", ""]
    (RES / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    (RES / "final_status.txt").write_text(
        "complete\nfolders={}\npairs={}\nexpected_wavs_per_pair={}\n"
        "loss_configurations={}\nloss_epoch_records={}\naudited_jobs={}\n"
        "run_lineage_jobs={}\nterminal_logs={}\n"
        "missing_historical_terminal_logs={}\n".format(
            len(actual), len(mappings), expected_wavs, len(loss_validation),
            observed_loss_records, len(audit_rows), len(lineage_rows), len(log_rows),
            len(missing_log_rows)), encoding="utf-8")
    print("final audit complete folders={} pairs={} loss_records={} jobs={} lineage={} logs={} missing_historical_logs={}".format(
        len(actual), len(mappings), observed_loss_records, len(audit_rows),
        len(lineage_rows), len(log_rows), len(missing_log_rows)))


if __name__ == "__main__":
    main()
