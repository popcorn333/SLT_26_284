#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=tan20_preflight
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan20/logs/preflight-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan20/logs/preflight-%j.err

set -euo pipefail
root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
state="$root/.symphony/tan20"
sgmse_env=/mnt/parscratch/users/acp23xt/private/conda/envs/CUDA124_sgmsetts
sgmse="$sgmse_env/bin/python"
export CUDA_HOME=/opt/apps/testapps/el7-znver3/software/staging/CUDA/12.4.0
whisper=/mnt/parscratch/users/acp23xt/private/codex_whisper/.conda/envs/codex_whisper/bin/python
utmos=/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python
metrics=/users/acp23xt/.conda/envs/Grad-TTS-EVAL/bin/python
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$sgmse_env/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$sgmse_env/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$state/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export HF_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/huggingface
export TORCH_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/torch
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TMPDIR="$root/.t/t20p-${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$TMPDIR" "$state/logs"

test -x "$CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" --version
"$sgmse" -B -c 'import torch; assert torch.cuda.is_available(); import hydra, numpy, scipy; print("sgmse_cuda=" + torch.cuda.get_device_name(0), flush=True)'
(cd "$root/levy" && "$sgmse" -B -c 'import train, inference; print("levy_imports_ok", flush=True)')
(cd "$root/fmDT" && "$sgmse" -B -c 'import train, inference; print("fmDT_imports_ok", flush=True)')
"$metrics" -B -c 'import fastdtw, librosa, numpy, pysptk, pyworld, scipy, soundfile; print("metrics_imports_ok", flush=True)'
(cd /mnt/parscratch/users/acp23xt/private/codex_whisper && "$whisper" -B -c 'import torch, whisper, jiwer; assert torch.cuda.is_available(); print("whisper_cuda=" + torch.cuda.get_device_name(0), flush=True)')
test -x /mnt/parscratch/users/acp23xt/private/codex_whisper/.conda/envs/codex_whisper/bin/ffmpeg
(cd /mnt/parscratch/users/acp23xt/private/codex_utmosv2 && "$utmos" -B -c 'import torch, utmosv2; assert torch.cuda.is_available(); print("utmos_cuda=" + torch.cuda.get_device_name(0), flush=True)')
test "$(wc -l < /mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/add/resources/filelists/ljspeech/test.txt)" -eq 488
test -s /mnt/parscratch/users/acp23xt/private/codex_whisper/.cache/whisper/large-v3.pt
test -s /mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth
printf 'tan20_preflight_ok=true\n'
