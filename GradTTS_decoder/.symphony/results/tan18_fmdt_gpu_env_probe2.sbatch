#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=tan18_fmdt_gpu_probe2
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-gpu-probe2-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-gpu-probe2-%j.err
#SBATCH --chdir=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
set -euo pipefail
export PATH=/usr/bin:/bin

ROOT=/mnt/parscratch/users/acp23xt/private
WHISPER_ENV="$ROOT/codex_whisper/.conda/envs/codex_whisper"
UTMOS_ENV="$ROOT/conda/envs/codex_utmosv2"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}

test -x "$WHISPER_ENV/bin/python"
test -x "$WHISPER_ENV/bin/ffmpeg"
test -s "$ROOT/codex_whisper/.cache/whisper/large-v3.pt"
"$WHISPER_ENV/bin/ffmpeg" -version | head -1
"$WHISPER_ENV/bin/python" - <<'PY'
import torch
import whisper
assert torch.cuda.is_available(), "Whisper environment cannot see CUDA"
assert torch.cuda.device_count() >= 1
print("whisper_probe_ok torch={} cuda={} device={}".format(
    torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)), flush=True)
PY

test -x "$UTMOS_ENV/bin/python"
test -s "$ROOT/codex_utmosv2/.cache/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth"
"$UTMOS_ENV/bin/python" - <<'PY'
import torch
import utmosv2
assert torch.cuda.is_available(), "UTMOSv2 environment cannot see CUDA"
assert torch.cuda.device_count() >= 1
print("utmos_probe_ok torch={} cuda={} device={} module={}".format(
    torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), utmosv2.__file__), flush=True)
PY
