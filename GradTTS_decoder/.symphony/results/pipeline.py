#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet")
RES = ROOT / ".symphony/results"
FOLDERS = ["add", "blur", "fmDT", "levy", "mask", "one-shot", "product", "score", "scoreDT", "scorex0"]
EXPECTED_PAIR_COUNT = 29
TRANSCRIPT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/add/resources/filelists/ljspeech/test.txt")
RUN_IDS = {
    "add": "10800233", "blur": "10800211",
    "fmDT": "10800235", "levy": "10800210", "mask": "10800213",
    "one-shot": "10800207", "product": "10800236",
    "score": "10804165", "scoreDT": "10884228", "scorex0": "10803830",
}
METRICS = [("mcd", "_mcd.txt"), ("logf0", "_f0.txt"),
           ("wer", "_wer_l.txt"), ("utmosv2", "_utmosv2.txt")]
METRICS_ENV = "/users/acp23xt/.conda/envs/Grad-TTS-EVAL"

WER_SCRIPT = r"""#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=tan10_wer
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/wer-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/wer-%A_%a.err
set -euo pipefail
ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
RES="$ROOT/.symphony/results"
GEN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$RES/mappings.tsv" | cut -f3)
test -n "$GEN"
test -d "$GEN"
EPOCH=$(basename "$GEN")
case "$EPOCH" in
    Epoch_[0-9]*) ;;
    *) echo "mapped generation directory has an invalid epoch basename: $GEN" >&2; exit 1 ;;
esac
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XDG_CACHE_HOME=/mnt/parscratch/users/acp23xt/private/codex_whisper/.cache
export XDG_CONFIG_HOME=/mnt/parscratch/users/acp23xt/private/codex_whisper/.config
export XDG_DATA_HOME=/mnt/parscratch/users/acp23xt/private/codex_whisper/.local/share
export TMPDIR=/mnt/parscratch/users/acp23xt/private/codex_whisper/.tmp
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
module purge
module load Anaconda3/2022.10
module load CUDA/12.4.0
module load FFmpeg/6.0-GCCcore-12.3.0
cd /mnt/parscratch/users/acp23xt/private/codex_whisper
FRESHNESS_MARKER="$RES/wer-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.started"
touch "$FRESHNESS_MARKER"
./.conda/envs/codex_whisper/bin/python batch_wer_epoch495.py   --root "$GEN"   --transcript /mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/add/resources/filelists/ljspeech/test.txt   --epoch "$EPOCH"   --model large-v3   --output-suffix _wer_l.txt   --overwrite   --progress-every 25
/usr/bin/python3 - "$GEN" "$FRESHNESS_MARKER" _wer_l.txt <<'PY'
import math
import sys
from pathlib import Path

gen = Path(sys.argv[1])
marker = Path(sys.argv[2])
suffixes = sys.argv[3:]
wavs = sorted(gen.glob("*.wav"))
if len(wavs) != 488:
    raise RuntimeError("expected 488 WAVs; found {} in {}".format(len(wavs), gen))
for wav in wavs:
    for suffix in suffixes:
        sidecar = wav.with_name(wav.stem + suffix)
        if not sidecar.is_file() or sidecar.stat().st_size == 0:
            raise RuntimeError("missing or empty sidecar {}".format(sidecar))
        if sidecar.stat().st_mtime_ns < marker.stat().st_mtime_ns:
            raise RuntimeError("sidecar was not refreshed by this job: {}".format(sidecar))
        value = float(sidecar.read_text(encoding="utf-8").strip())
        if not math.isfinite(value):
            raise RuntimeError("non-finite sidecar {}".format(sidecar))
print("fresh WER sidecars validated: {}".format(len(wavs)), flush=True)
PY
"""

UTMOS_SCRIPT = r"""#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=tan10_utmos
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/utmos-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/utmos-%A_%a.err
set -euo pipefail
ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
RES="$ROOT/.symphony/results"
GEN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$RES/mappings.tsv" | cut -f3)
test -n "$GEN"
test -d "$GEN"
DIR_LIST="$RES/utmos-dir-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
printf '%s\n' "$GEN" > "$DIR_LIST"
module purge
module load Anaconda3/2022.10
module load CUDA/12.4.0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XDG_CACHE_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache
export XDG_CONFIG_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.config
export XDG_DATA_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.local/share
export HF_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/huggingface
export TORCH_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/torch
export UTMOSV2_CHACHE=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/utmosv2
export TRITON_CACHE_DIR=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/triton
export NUMBA_CACHE_DIR=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/numba
export MPLCONFIGDIR=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.config/matplotlib
export TMPDIR=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.tmp
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$UTMOSV2_CHACHE" "$TRITON_CACHE_DIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
cd /mnt/parscratch/users/acp23xt/private/codex_utmosv2
FRESHNESS_MARKER="$RES/utmos-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.started"
touch "$FRESHNESS_MARKER"
/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python   score_ourdir_utmosv2.py   --dir-list "$DIR_LIST"   --batch-size 16   --num-workers 4   --device auto   --overwrite
/usr/bin/python3 - "$GEN" "$FRESHNESS_MARKER" _utmosv2.txt <<'PY'
import math
import sys
from pathlib import Path

gen = Path(sys.argv[1])
marker = Path(sys.argv[2])
suffixes = sys.argv[3:]
wavs = sorted(gen.glob("*.wav"))
if len(wavs) != 488:
    raise RuntimeError("expected 488 WAVs; found {} in {}".format(len(wavs), gen))
for wav in wavs:
    for suffix in suffixes:
        sidecar = wav.with_name(wav.stem + suffix)
        if not sidecar.is_file() or sidecar.stat().st_size == 0:
            raise RuntimeError("missing or empty sidecar {}".format(sidecar))
        if sidecar.stat().st_mtime_ns < marker.stat().st_mtime_ns:
            raise RuntimeError("sidecar was not refreshed by this job: {}".format(sidecar))
        value = float(sidecar.read_text(encoding="utf-8").strip())
        if not math.isfinite(value):
            raise RuntimeError("non-finite sidecar {}".format(sidecar))
print("fresh UTMOSv2 sidecars validated: {}".format(len(wavs)), flush=True)
PY
"""

UTMOS_WARMUP_SCRIPT = r"""#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=tan10_utmos_cache
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/utmos-cache-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/utmos-cache-%j.err
set -euo pipefail
module purge
module load Anaconda3/2022.10
module load CUDA/12.4.0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache
export XDG_CONFIG_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.config
export XDG_DATA_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.local/share
export HF_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/huggingface
export TORCH_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/torch
export UTMOSV2_CHACHE=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/utmosv2
export TMPDIR=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.tmp
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$UTMOSV2_CHACHE"
cd /mnt/parscratch/users/acp23xt/private/codex_utmosv2
/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python - <<'PY'
import utmosv2
utmosv2.create_model(pretrained=True, device="cuda:0")
print("UTMOSv2 model and dependencies cached successfully", flush=True)
PY
"""

LOSS_SCRIPT = r"""#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=tan10_training_loss
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/training-loss-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/training-loss-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/aggregate_training_loss.py
"""

SUMMARY_SCRIPT = r"""#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --job-name=tan10_summary
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/summary-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/summary-%j.err
set -euo pipefail
/usr/bin/python3 /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/pipeline.py summarize
"""

def job_id(output):
    return output.strip().split(";")[0]

def sbatch(args):
    return job_id(subprocess.check_output(["sbatch", "--parsable"] + args, universal_newlines=True))

def replace_assignment(data, name, value):
    pattern = re.compile(rb"(?m)^" + name.encode("ascii") + rb"=[^\r\n]*(\r?\n|$)")
    replacement = value.encode("utf-8")
    data, count = pattern.subn(
        lambda match: name.encode("ascii") + b"=" + replacement + match.group(1), data)
    if count != 1:
        raise RuntimeError("expected exactly one {} assignment; found {}".format(name, count))
    return data

def ensure_fail_fast(data):
    if re.search(rb"(?m)^set -[^\r\n]*e[^\r\n]*$", data):
        return data
    lines = data.splitlines(keepends=True)
    if not lines or not lines[0].startswith(b"#!"):
        raise RuntimeError("metrics.sh must begin with a shebang")
    # Slurm parses #SBATCH directives only until the first executable line.
    # Keep the complete leading comment/blank directive block intact and add
    # fail-fast behavior immediately before the script's first command.
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
        raise RuntimeError("expected exactly one metrics environment activation; found {}".format(
            count))
    return data

def ensure_metrics_freshness_validation(data):
    """Make a metrics allocation fail if any per-WAV result is stale/missing."""
    sentinel = b"# TAN-10 per-WAV metric freshness gate"
    if sentinel in data:
        return data
    command = b'python "tools/F0/F0.py"'
    if data.count(command) != 1:
        raise RuntimeError("expected exactly one F0 metric command")
    marker = (
        b'# TAN-10 per-WAV metric freshness gate\n'
        b'FRESHNESS_MARKER="/mnt/parscratch/users/acp23xt/private/DTDM_Unet/'
        b'.symphony/results/metrics-${SLURM_JOB_ID}.started"\n'
        b'touch "$FRESHNESS_MARKER"\n\n'
    )
    data = data.replace(command, marker + command, 1)
    validator = rb'''

/usr/bin/python3 - "$gen" "$FRESHNESS_MARKER" _f0.txt _mcd.txt <<'PY'
import math
import sys
from pathlib import Path

gen = Path(sys.argv[1])
marker = Path(sys.argv[2])
suffixes = sys.argv[3:]
wavs = sorted(gen.glob("*.wav"))
if len(wavs) != 488:
    raise RuntimeError("expected 488 WAVs; found {} in {}".format(len(wavs), gen))
for wav in wavs:
    for suffix in suffixes:
        sidecar = wav.with_name(wav.stem + suffix)
        if not sidecar.is_file() or sidecar.stat().st_size == 0:
            raise RuntimeError("missing or empty sidecar {}".format(sidecar))
        if sidecar.stat().st_mtime_ns < marker.stat().st_mtime_ns:
            raise RuntimeError("sidecar was not refreshed by this job: {}".format(sidecar))
        value = float(sidecar.read_text(encoding="utf-8").strip())
        if not math.isfinite(value):
            raise RuntimeError("non-finite sidecar {}".format(sidecar))
print("fresh MCD/logF0 sidecars validated: {}".format(len(wavs)), flush=True)
PY
'''
    return data.rstrip() + b"\n" + validator

def extract_pairs(folder):
    output = subprocess.check_output(
        ["/usr/bin/python3", str(ROOT / "extract_metrics_paths.py"),
         str(ROOT / folder / "run.sh")], universal_newlines=True)
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
    if not pairs:
        raise RuntimeError("no metric pairs for {}".format(folder))
    return pairs

def assert_current_inventory():
    actual = sorted(path.parent.name for path in ROOT.glob("*/run.sh")
                    if path.parent.parent == ROOT and path.parent.name != "fm")
    if actual != sorted(FOLDERS):
        raise RuntimeError("first-depth run.sh inventory changed: expected={} actual={}".format(
            sorted(FOLDERS), actual))

def assert_under_root(path, label):
    try:
        Path(path).resolve().relative_to(ROOT.resolve())
    except ValueError:
        raise RuntimeError("{} path is outside approved repository tree: {}".format(label, path))

def dispatch():
    assert_current_inventory()
    dispatcher_id = os.environ.get("SLURM_JOB_ID")
    if not dispatcher_id or not dispatcher_id.isdigit():
        raise RuntimeError("dispatch must run inside its recorded Slurm allocation")
    dispatcher_dependency = "afterok:{}".format(dispatcher_id)
    transcript_lines = [line for line in TRANSCRIPT.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    expected_wav_names = {Path(line.split("|", 1)[0]).name
                          for line in transcript_lines}
    if len(transcript_lines) != 488 or len(expected_wav_names) != 488:
        raise RuntimeError(
            "transcript coverage changed: rows={} unique_wavs={}".format(
                len(transcript_lines), len(expected_wav_names)))
    mappings = []
    for folder in FOLDERS:
        for index, pair in enumerate(extract_pairs(folder)):
            assert_under_root(pair[0], "gen")
            assert_under_root(pair[1], "out")
            gen_path = Path(pair[0])
            if not gen_path.is_dir():
                raise RuntimeError("missing generated directory {}".format(gen_path))
            observed_wav_names = {path.name for path in gen_path.glob("*.wav")}
            if observed_wav_names != expected_wav_names:
                raise RuntimeError(
                    "generated WAV coverage mismatch for {} pair {}: "
                    "expected={} observed={} missing={} unexpected={}".format(
                        folder, index, len(expected_wav_names),
                        len(observed_wav_names),
                        len(expected_wav_names - observed_wav_names),
                        len(observed_wav_names - expected_wav_names)))
            mappings.append((folder, str(index), pair[0], pair[1]))
    if len(mappings) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("expected {} emitted pairs; found {}".format(
            EXPECTED_PAIR_COUNT, len(mappings)))
    for field, position in (("gen", 2), ("out", 3)):
        values = [row[position] for row in mappings]
        if len(set(values)) != len(values):
            raise RuntimeError("emitted {} paths are not unique".format(field))
    emitted_configurations = {
        (folder, configuration_from_gen(folder, gen))
        for folder, unused_index, gen, unused_out in mappings
    }
    discovered_configurations = {
        (folder, path.parent.parent.name)
        for folder in FOLDERS
        for path in (ROOT / folder).glob("Hydra_*/*/.hydra/config.yaml")
    }
    if len(emitted_configurations) != EXPECTED_PAIR_COUNT or \
            emitted_configurations != discovered_configurations:
        raise RuntimeError(
            "emitted and trained Hydra configurations differ: emitted={} discovered={}".format(
                sorted(emitted_configurations), sorted(discovered_configurations)))
    with (RES / "mappings.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["folder", "pair_index", "gen", "out"])
        writer.writerows(mappings)

    repair_rows = read_tsv(RES / "gen_repair_jobs.tsv")
    repair_lookup = {(row["folder"], row["pair_index"]): row["job_id"] for row in repair_rows}
    expected_repair_keys = {(folder, index) for folder, index, unused_gen, unused_out in mappings}
    if set(repair_lookup) != expected_repair_keys:
        raise RuntimeError("gen repair mapping keys do not match emitted pairs")
    jobs = []
    metric_ids = []
    metrics_tmp = RES / "tmp/metrics"
    metrics_cache = RES / "cache/metrics"
    metrics_config = RES / "config/metrics"
    metrics_data = RES / "data/metrics"
    metrics_numba = metrics_cache / "numba"
    metrics_joblib = metrics_tmp / "joblib"
    metrics_payloads = RES / "metrics_payloads"
    for path in (
            metrics_tmp, metrics_cache, metrics_config, metrics_data,
            metrics_numba, metrics_joblib, metrics_config / "matplotlib",
            metrics_payloads):
        path.mkdir(parents=True, exist_ok=True)
    for folder, index, gen, out in mappings:
        # The established metric programs append aggregate records beneath
        # ``out`` but do not create that directory themselves.  Generation
        # produces the converted WAV tree only, so create each already-
        # validated in-repository metrics directory before submitting its job.
        Path(out).mkdir(parents=True, exist_ok=True)
        metrics_sh = ROOT / folder / "metrics.sh"
        data = metrics_sh.read_bytes()
        data = ensure_fail_fast(data)
        # Retain the established environment but activate it by its absolute,
        # read-only path so job execution does not depend on user-relative
        # Conda environment discovery.
        data = ensure_metrics_environment(data)
        data = ensure_metrics_freshness_validation(data)
        data = replace_assignment(data, "gen", gen)
        data = replace_assignment(data, "out", out)
        metrics_sh.write_bytes(data)
        # ``sbatch`` reads a script when the job is submitted, but the exact
        # timing of that read relative to the next loop iteration is not an
        # orchestration contract we should rely on.  More importantly, the
        # job may start only after this shared per-folder script has been
        # rewritten for another sweep.  Submit an immutable per-pair payload
        # so each allocation is permanently tied to its recorded gen/out
        # mapping while the established experiment script remains updated to
        # the final emitted pair as required.
        metrics_payload = metrics_payloads / (
            "metrics-{}-{}.sbatch".format(folder, index))
        metrics_payload.write_bytes(data)
        mid = sbatch([
            "--dependency={}".format(dispatcher_dependency),
            "--chdir={}".format(ROOT / folder),
            "--job-name=tan10_m_{}_{}".format(folder, index),
            "--output={}".format(RES / ("metrics-{}-{}-%j.out".format(folder, index))),
            "--error={}".format(RES / ("metrics-{}-{}-%j.err".format(folder, index))),
            "--export=ALL,TMPDIR={},XDG_CACHE_HOME={},XDG_CONFIG_HOME={},XDG_DATA_HOME={},MPLCONFIGDIR={},NUMBA_CACHE_DIR={},JOBLIB_TEMP_FOLDER={},PYTHONDONTWRITEBYTECODE=1".format(
                metrics_tmp, metrics_cache, metrics_config, metrics_data,
                metrics_config / "matplotlib", metrics_numba,
                metrics_joblib),
            str(metrics_payload),
        ])
        metric_ids.append(mid)
        run_id = RUN_IDS[folder]
        jobs.append([folder, index, "run", run_id, gen, out])
        jobs.append([folder, index, "gen_repair", repair_lookup[(folder, index)], gen, out])
        jobs.append([folder, index, "metrics", mid, gen, out])

    (RES / "wer_array.sbatch").write_text(WER_SCRIPT, encoding="utf-8")
    (RES / "utmos_array.sbatch").write_text(UTMOS_SCRIPT, encoding="utf-8")
    (RES / "utmos_warmup.sbatch").write_text(UTMOS_WARMUP_SCRIPT, encoding="utf-8")
    (RES / "training_loss.sbatch").write_text(LOSS_SCRIPT, encoding="utf-8")
    (RES / "validate_summary.sbatch").write_text(SUMMARY_SCRIPT, encoding="utf-8")
    last = len(mappings) - 1
    wid = sbatch(["--dependency={}".format(dispatcher_dependency),
                  "--array=0-{}".format(last), str(RES / "wer_array.sbatch")])
    uwid = sbatch(["--dependency={}".format(dispatcher_dependency),
                   str(RES / "utmos_warmup.sbatch")])
    uid = sbatch(["--dependency=afterok:{}".format(uwid),
                  "--array=0-{}".format(last), str(RES / "utmos_array.sbatch")])
    lid = sbatch(["--dependency={}".format(dispatcher_dependency),
                  str(RES / "training_loss.sbatch")])
    for task, (folder, index, gen, out) in enumerate(mappings):
        jobs.append([folder, index, "wer", "{}_{}".format(wid, task), gen, out])
        jobs.append([folder, index, "utmosv2", "{}_{}".format(uid, task), gen, out])

    with (RES / "jobs.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["folder", "pair_index", "stage", "job_id", "gen", "out"])
        writer.writerows(jobs)

    dependency = ":".join(metric_ids + [wid, uid, lid])
    sid = sbatch(["--dependency=afterok:{}".format(dependency),
                  str(RES / "validate_summary.sbatch")])
    aid = sbatch(["--dependency=afterok:{}".format(sid),
                  str(RES / "completion_audit.sbatch")])
    with (RES / "pipeline_job_ids.tsv").open("w", encoding="utf-8") as handle:
        handle.write("dispatch_job_id\t{}\n".format(dispatcher_id))
        handle.write("wer_array_job_id\t{}\n".format(wid))
        handle.write("utmosv2_warmup_job_id\t{}\n".format(uwid))
        handle.write("utmosv2_array_job_id\t{}\n".format(uid))
        handle.write("training_loss_job_id\t{}\n".format(lid))
        handle.write("summary_job_id\t{}\n".format(sid))
        handle.write("final_audit_job_id\t{}\n".format(aid))
        handle.write("pair_count\t{}\n".format(len(mappings)))
    print("dispatcher complete pairs={} wer={} utmos_warmup={} utmos={} training_loss={} summary={} final_audit={}".format(
        len(mappings), wid, uwid, uid, lid, sid, aid))

def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))

def read_value(path):
    text = path.read_text(encoding="utf-8").strip()
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite")
    return value

def fmt(value):
    return "" if value is None else "{:.8f}".format(value)

def configuration_from_gen(folder, gen):
    """Return the Hydra configuration represented by an emitted gen path."""
    path = Path(gen).resolve()
    experiment_root = (ROOT / folder).resolve()
    try:
        relative = path.relative_to(experiment_root)
    except ValueError:
        raise RuntimeError("generated path is outside {}: {}".format(
            experiment_root, path))
    parts = relative.parts
    if len(parts) != 5 or parts[2:4] != ("test", "converted") or \
            not re.fullmatch(r"Epoch_[0-9]+", parts[4]):
        raise RuntimeError("unexpected generated path layout for {}: {}".format(
            folder, path))
    return parts[1]

def validate_training_loss(mappings):
    status_lines = (RES / "training_loss_status.txt").read_text(
        encoding="utf-8").splitlines()
    if status_lines != ["complete", "configurations=29", "epoch_records=17371"]:
        raise RuntimeError("training-loss status is incomplete or malformed")

    expected_keys = {
        (row["folder"], configuration_from_gen(row["folder"], row["gen"]))
        for row in mappings
    }
    if len(expected_keys) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("training-loss mapping keys are incomplete or duplicated")

    validation = read_tsv(RES / "training_loss_validation.tsv")
    validation_keys = {
        (row["experiment"], row["sweep_or_configuration"])
        for row in validation
    }
    if len(validation) != EXPECTED_PAIR_COUNT or validation_keys != expected_keys:
        raise RuntimeError("training-loss validation does not match emitted configurations")
    source_logs = {}
    for row in validation:
        key = (row["experiment"], row["sweep_or_configuration"])
        if (row["expected_epochs"], row["observed_unique_epochs"], row["status"]) != \
                ("599", "599", "complete"):
            raise RuntimeError("incomplete training-loss validation for {}".format(key))
        source = Path(row["source_log"])
        if not source.is_file():
            raise RuntimeError("missing validated training log {}".format(source))
        source_logs[key] = source

    loss_rows = read_tsv(ROOT / "training_loss.log")
    if len(loss_rows) != 17371:
        raise RuntimeError("expected 17371 training-loss records; found {}".format(
            len(loss_rows)))
    observed = {}
    for row in loss_rows:
        key = (row["experiment"], row["sweep_or_configuration"])
        if key not in expected_keys:
            raise RuntimeError("unexpected training-loss configuration {}".format(key))
        epoch = int(row["epoch"])
        values = [float(row[name]) for name in (
            "loss", "duration_loss", "prior_loss", "diffusion_loss")]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("non-finite training loss for {} epoch {}".format(
                key, epoch))
        if row["source_log"] != str(source_logs[key]):
            raise RuntimeError("training-loss source mismatch for {}".format(key))
        observed.setdefault(key, set()).add(epoch)
    expected_epochs = set(range(1, 600))
    if set(observed) != expected_keys:
        raise RuntimeError("training-loss record configuration coverage mismatch")
    for key, epochs in observed.items():
        if epochs != expected_epochs:
            raise RuntimeError("training-loss epoch coverage mismatch for {}".format(key))

    aggregate_mtime = (ROOT / "training_loss.log").stat().st_mtime_ns
    if any(source.stat().st_mtime_ns > aggregate_mtime for source in source_logs.values()):
        raise RuntimeError("training_loss.log predates a source training log")
    return len(observed), len(loss_rows)

def summarize():
    assert_current_inventory()
    mappings = read_tsv(RES / "mappings.tsv")
    if len(mappings) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("expected {} mapping rows; found {}".format(
            EXPECTED_PAIR_COUNT, len(mappings)))
    loss_configurations, loss_records = validate_training_loss(mappings)
    transcript_lines = [line for line in TRANSCRIPT.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    expected_wav_names = {Path(line.split("|", 1)[0]).name for line in transcript_lines}
    expected_wavs = len(transcript_lines)
    if len(expected_wav_names) != expected_wavs:
        raise RuntimeError("transcript contains duplicate WAV basenames")
    jobs = read_tsv(RES / "jobs.tsv")
    lookup = {}
    for job in jobs:
        lookup[(job["folder"], job["pair_index"], job["stage"])] = job["job_id"]
    rows = []
    problems = []
    for mapping in mappings:
        folder = mapping["folder"]
        index = mapping["pair_index"]
        gen = Path(mapping["gen"])
        wavs = sorted(gen.glob("*.wav"))
        stems = [wav.stem for wav in wavs]
        row = {
            "folder": folder, "pair_index": index, "gen": str(gen),
            "out": mapping["out"], "wav_count": str(len(wavs)),
            "run_job_id": lookup.get((folder, index, "run"), ""),
            "gen_repair_job_id": lookup.get((folder, index, "gen_repair"), ""),
            "metrics_job_id": lookup.get((folder, index, "metrics"), ""),
            "wer_job_id": lookup.get((folder, index, "wer"), ""),
            "utmosv2_job_id": lookup.get((folder, index, "utmosv2"), ""),
        }
        if not wavs:
            problems.append("{} pair {} has no WAV files".format(folder, index))
        if len(wavs) != expected_wavs:
            problems.append("{} pair {} WAV coverage: expected={}, observed={}".format(
                folder, index, expected_wavs, len(wavs)))
        observed_wav_names = {wav.name for wav in wavs}
        if observed_wav_names != expected_wav_names:
            problems.append(
                "{} pair {} WAV-name coverage: missing={}, unexpected={}".format(
                    folder, index,
                    len(expected_wav_names - observed_wav_names),
                    len(observed_wav_names - expected_wav_names)))
        for metric, suffix in METRICS:
            values, missing, invalid = [], [], []
            for stem in stems:
                path = gen / (stem + suffix)
                if not path.is_file():
                    missing.append(stem)
                    continue
                try:
                    values.append(read_value(path))
                except (OSError, ValueError) as error:
                    invalid.append("{} ({})".format(stem, error))
            row[metric + "_count"] = str(len(values))
            row[metric + "_mean"] = fmt(statistics.mean(values) if values else None)
            row[metric + "_std"] = fmt(statistics.pstdev(values) if values else None)
            row[metric + "_missing"] = ";".join(missing)
            row[metric + "_invalid"] = ";".join(invalid)
            if missing or invalid:
                problems.append("{} pair {} {}: missing={}, invalid={}".format(
                    folder, index, metric, len(missing), len(invalid)))
        rows.append(row)

    columns = ["folder", "pair_index", "gen", "out", "wav_count"]
    for metric, unused in METRICS:
        columns += [metric + "_count", metric + "_mean", metric + "_std",
                    metric + "_missing", metric + "_invalid"]
    columns += ["run_job_id", "gen_repair_job_id", "metrics_job_id",
                "wer_job_id", "utmosv2_job_id"]
    with (RES / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# TAN-6 speech metric summary (fm excluded)", "",
        "Arithmetic means and population standard deviations (ddof=0) are computed from the per-WAV files beside each generated WAV.", "",
        "Training loss: {} configurations and {} labelled epoch records; exact epoch 1-599 coverage is validated in `training_loss_validation.tsv` and the combined records are in `../../training_loss.log`.".format(
            loss_configurations, loss_records), "",
        "| Folder | Pair | WAV n | MCD n / mean / std | logF0 n / mean / std | WER n / mean / std | UTMOSv2 n / mean / std | Slurm run / repair / metrics / WER / UTMOSv2 | Gen path |",
        "|---|---:|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        def cell(name):
            return "{} / {} / {}".format(row[name + "_count"], row[name + "_mean"], row[name + "_std"])
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} / {} / {} / {} / {} | `{}` |".format(
            row["folder"], row["pair_index"], row["wav_count"], cell("mcd"),
            cell("logf0"), cell("wer"), cell("utmosv2"), row["run_job_id"],
            row["gen_repair_job_id"], row["metrics_job_id"], row["wer_job_id"],
            row["utmosv2_job_id"], row["gen"]))
    lines += ["", "## Missing or invalid values", ""]
    if problems:
        lines += ["- " + problem for problem in problems]
    else:
        lines.append("None. Every metric has one valid value per generated WAV.")
    lines += ["", "Machine-readable data: `summary.csv`", ""]
    (RES / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (RES / "validation_status.txt").write_text(
        "complete_with_missing\n" if problems else "complete\n", encoding="utf-8")
    print("rows={} missing_or_invalid_groups={}".format(len(rows), len(problems)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dispatch", "summarize"])
    args = parser.parse_args()
    dispatch() if args.mode == "dispatch" else summarize()

if __name__ == "__main__":
    main()
