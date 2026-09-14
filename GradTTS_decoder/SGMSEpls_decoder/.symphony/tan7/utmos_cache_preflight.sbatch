#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=tan7_utmos_cache
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/logs/utmos-cache-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/logs/utmos-cache-%j.err

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cache_root=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache
export UTMOSV2_CHACHE="$cache_root/utmosv2"
export HF_HOME="$cache_root/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$cache_root/torch"
export XDG_CACHE_HOME="$cache_root"
export CUDA_CACHE_PATH="$cache_root/cuda"
export TMPDIR="$cache_root/tmp/preflight-${SLURM_JOB_ID}"

mkdir -p \
  "$UTMOSV2_CHACHE/models/fusion_stage3" \
  "$HF_HOME/hub/models--facebook--wav2vec2-base" \
  "$HF_HOME/hub/models--timm--tf_efficientnetv2_s.in21k_ft_in1k" \
  "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TMPDIR" "$cache_root/preflight"

cp -a --reflink=auto \
  /users/acp23xt/.cache/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth \
  "$UTMOSV2_CHACHE/models/fusion_stage3/fold0_s42_best_model.pth"
cp -a --reflink=auto \
  /users/acp23xt/.cache/huggingface/hub/models--facebook--wav2vec2-base/. \
  "$HF_HOME/hub/models--facebook--wav2vec2-base/"
cp -a --reflink=auto \
  /users/acp23xt/.cache/huggingface/hub/models--timm--tf_efficientnetv2_s.in21k_ft_in1k/. \
  "$HF_HOME/hub/models--timm--tf_efficientnetv2_s.in21k_ft_in1k/"

test "$(stat -c %s "$UTMOSV2_CHACHE/models/fusion_stage3/fold0_s42_best_model.pth")" -eq 818531314
test -s "$HF_HOME/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8/config.json"
test -s "$HF_HOME/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8/preprocessor_config.json"
test "$(stat -L -c %s "$HF_HOME/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8/pytorch_model.bin")" -eq 380267417
test "$(stat -L -c %s "$HF_HOME/hub/models--timm--tf_efficientnetv2_s.in21k_ft_in1k/snapshots/212ce86b3cc2aa27505700cd5d1a25dfd2a07dc8/model.safetensors")" -eq 86523256

cp -a --reflink=auto \
  /mnt/parscratch/users/acp23xt/public/LJSpeech-1.1/wavs/LJ045-0096.wav \
  "$cache_root/preflight/LJ045-0096.wav"
rm -f "$cache_root/preflight/LJ045-0096_utmosv2.txt"

module purge
module load Anaconda3/2022.10
module load CUDA/12.4.0
cd /mnt/parscratch/users/acp23xt/private/codex_utmosv2
/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python \
  score_ourdir_utmosv2.py \
  --dir-list /mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/utmos_preflight_list.txt \
  --batch-size 1 \
  --num-workers 0 \
  --device auto \
  --overwrite

/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python -B -c \
  'import math, pathlib; p=pathlib.Path("/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/preflight/LJ045-0096_utmosv2.txt"); v=float(p.read_text().strip()); assert math.isfinite(v); print(f"validated_utmos_preflight={v:.8f}")'
