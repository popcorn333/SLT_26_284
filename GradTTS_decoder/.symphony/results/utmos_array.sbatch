#!/bin/bash
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
