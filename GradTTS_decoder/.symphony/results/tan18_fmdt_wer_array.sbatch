#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=tan18_fmdt_wer_r1
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-wer-r1-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-wer-r1-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
RES="$ROOT/.symphony/results"
TASK_INDEX=${TAN18_TASK_INDEX:-${SLURM_ARRAY_TASK_ID:?TAN18_TASK_INDEX or SLURM_ARRAY_TASK_ID is required}}
GEN=$(sed -n "$((TASK_INDEX + 2))p" "$RES/tan18_fmdt_mappings.tsv" | cut -f3)
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
WHISPER_ENV=/mnt/parscratch/users/acp23xt/private/codex_whisper/.conda/envs/codex_whisper
export PATH="$WHISPER_ENV/bin:/usr/bin:/bin"
cd /mnt/parscratch/users/acp23xt/private/codex_whisper
FRESHNESS_MARKER="$RES/tan18-fmdt-wer-r1-${SLURM_JOB_ID}.started"
touch "$FRESHNESS_MARKER"
"$WHISPER_ENV/bin/python" batch_wer_epoch495.py   --root "$GEN"   --transcript /mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/add/resources/filelists/ljspeech/test.txt   --epoch "$EPOCH"   --model large-v3   --output-suffix _wer_l.txt   --overwrite   --progress-every 25
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
