#!/bin/bash -l
#SBATCH --cpus-per-task=4
#SBATCH --time=0-10:00:00
set -eo pipefail
ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
export TMPDIR="$ROOT/.symphony/tmp/fmdt-metric-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$ROOT/.symphony/cache"
export XDG_CONFIG_HOME="$ROOT/.symphony/config"
export XDG_DATA_HOME="$ROOT/.symphony/data"
export MPLCONFIGDIR="$ROOT/.symphony/config/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$MPLCONFIGDIR"
module load Anaconda3/2025.06-1
source activate /users/acp23xt/.conda/envs/Grad-TTS-EVAL
gt=/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/ground_truth
gen="${GEN_PATH:-/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:1/test/converted/Epoch_595}"
out="${OUT_PATH:-/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:1/test/metrics}"

# TAN-10 per-WAV metric freshness gate
FRESHNESS_MARKER="/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/metrics-${SLURM_JOB_ID}.started"
touch "$FRESHNESS_MARKER"

python "tools/F0/F0.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"
python "tools/MCD/MCD.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"


/usr/bin/python3 - "$gen" "$out" "$FRESHNESS_MARKER" _f0.txt _mcd.txt <<'PY'
import math
import re
import sys
from pathlib import Path

gen = Path(sys.argv[1])
out = Path(sys.argv[2])
marker = Path(sys.argv[3])
suffixes = sys.argv[4:]
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
aggregate_pattern = re.compile(
    r"^(?P<path>.+)   (?P<mean>[0-9.eE+-]+) \\pm (?P<std>[0-9.eE+-]+)$")
for name in ("MCD.txt", "F0.txt"):
    aggregate = out / name
    if not aggregate.is_file() or aggregate.stat().st_size == 0:
        raise RuntimeError("missing or empty aggregate file {}".format(aggregate))
    if aggregate.stat().st_mtime_ns < marker.stat().st_mtime_ns:
        raise RuntimeError("aggregate file was not refreshed by this job: {}".format(aggregate))
    matches = [line.strip() for line in aggregate.read_text(errors="replace").splitlines()
               if line.split("   ", 1)[0].strip() == str(gen)]
    if not matches:
        raise RuntimeError("fresh aggregate file omits generation path: {}".format(gen))
    match = aggregate_pattern.fullmatch(matches[-1])
    if not match or match.group("path") != str(gen):
        raise RuntimeError("invalid latest aggregate line in {}: {}".format(
            aggregate, matches[-1]))
    values = (float(match.group("mean")), float(match.group("std")))
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite latest aggregate line in {}".format(aggregate))
print("fresh MCD/logF0 sidecars and aggregates validated: {}".format(len(wavs)), flush=True)
PY
