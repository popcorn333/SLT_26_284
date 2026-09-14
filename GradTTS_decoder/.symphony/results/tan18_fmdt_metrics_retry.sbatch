#!/bin/bash -l
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --job-name=tan18_fmdt_metric_r3
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-metric-r3-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-metric-r3-%j.err
#SBATCH --chdir=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT
set -euo pipefail
export PATH=/usr/bin:/bin

ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
RES="$ROOT/.symphony/results"
MAP="$RES/tan18_fmdt_mappings.tsv"
TASK_INDEX=${TAN18_TASK_INDEX:-${SLURM_ARRAY_TASK_ID:?TAN18_TASK_INDEX or SLURM_ARRAY_TASK_ID is required}}
ROW=$((TASK_INDEX + 2))
GEN=$(sed -n "${ROW}p" "$MAP" | cut -f3)
OUT=$(sed -n "${ROW}p" "$MAP" | cut -f4)
test -n "$GEN"
test -n "$OUT"
test -d "$GEN"

# Python's multiprocessing manager creates a Unix-domain socket below TMPDIR.
# Keep this path short enough for the 108-byte AF_UNIX limit on every node.
export TMPDIR="$ROOT/.symphony/t"
export XDG_CACHE_HOME="$ROOT/.symphony/cache"
export XDG_CONFIG_HOME="$ROOT/.symphony/config"
export XDG_DATA_HOME="$ROOT/.symphony/data"
export MPLCONFIGDIR="$ROOT/.symphony/config/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$MPLCONFIGDIR"

module purge
module load Anaconda3/2022.10
source activate /users/acp23xt/.conda/envs/Grad-TTS-EVAL

GT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/ground_truth
MARKER="$RES/tan18-fmdt-metric-r3-${SLURM_JOB_ID}.started"
touch "$MARKER"

# Preserve fmDT/metrics.sh's established ground-truth and metric commands.
python tools/F0/F0.py --gt_wavdir_or_wavscp "$GT" --gen_wavdir_or_wavscp "$GEN" --outdir "$OUT"
python tools/MCD/MCD.py --gt_wavdir_or_wavscp "$GT" --gen_wavdir_or_wavscp "$GEN" --outdir "$OUT"

/usr/bin/python3 - "$GEN" "$OUT" "$MARKER" <<'PY'
import math
import re
import sys
from pathlib import Path

gen = Path(sys.argv[1])
out = Path(sys.argv[2])
marker = Path(sys.argv[3])
wavs = sorted(gen.glob("*.wav"))
if len(wavs) != 488:
    raise RuntimeError("expected 488 WAVs; found {} in {}".format(len(wavs), gen))
for wav in wavs:
    for suffix in ("_f0.txt", "_mcd.txt"):
        sidecar = wav.with_name(wav.stem + suffix)
        if not sidecar.is_file() or sidecar.stat().st_size == 0:
            raise RuntimeError("missing or empty sidecar {}".format(sidecar))
        if sidecar.stat().st_mtime_ns < marker.stat().st_mtime_ns:
            raise RuntimeError("sidecar was not refreshed by this retry: {}".format(sidecar))
        value = float(sidecar.read_text(encoding="utf-8").strip())
        if not math.isfinite(value):
            raise RuntimeError("non-finite sidecar {}".format(sidecar))

pattern = re.compile(
    r"^(?P<path>.+)   (?P<mean>[0-9.eE+-]+) \\pm (?P<std>[0-9.eE+-]+)$")
for filename in ("MCD.txt", "F0.txt"):
    aggregate = out / filename
    if not aggregate.is_file() or aggregate.stat().st_size == 0:
        raise RuntimeError("missing or empty aggregate {}".format(aggregate))
    if aggregate.stat().st_mtime_ns < marker.stat().st_mtime_ns:
        raise RuntimeError("aggregate was not refreshed by this retry: {}".format(aggregate))
    matches = [
        line.strip()
        for line in aggregate.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.split("   ", 1)[0].strip() == str(gen)
    ]
    match = pattern.fullmatch(matches[-1]) if matches else None
    if not match or match.group("path") != str(gen):
        raise RuntimeError("invalid latest aggregate line in {}".format(aggregate))
    values = (float(match.group("mean")), float(match.group("std")))
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite aggregate line in {}".format(aggregate))
print("fresh retry MCD/logF0 outputs validated: {}".format(len(wavs)), flush=True)
PY
