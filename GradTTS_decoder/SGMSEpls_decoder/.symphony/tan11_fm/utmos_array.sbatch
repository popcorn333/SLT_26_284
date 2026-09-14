#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=tan11_fm_utmos
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan11_fm/logs/utmos-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan11_fm/logs/utmos-%A_%a.err

set -euo pipefail
module purge
module load Anaconda3/2022.10
module load CUDA/12.4.0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export UTMOSV2_CHACHE=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/utmosv2
export HF_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/torch
export XDG_CACHE_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
state=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan11_fm
export TMPDIR="/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.t/f${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TMPDIR"

cd /mnt/parscratch/users/acp23xt/private/codex_utmosv2
/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python -B -c \
  'import torch; assert torch.cuda.is_available(), "UTMOSv2 GPU allocation has no visible CUDA device"; print("utmosv2_cuda=" + torch.cuda.get_device_name(0), flush=True)'

list=$(printf '%s/utmos_lists/%02d.txt' "$state" "$SLURM_ARRAY_TASK_ID")
test -s "$list"
gen=$(sed -n '1p' "$list")
test -d "$gen"
case "$gen" in
  /mnt/parscratch/users/acp23xt/private/DTDM_sgmse/fm/*) ;;
  *) printf 'refusing out-of-scope gen path: %s\n' "$gen" >&2; exit 1 ;;
esac
for wav in "$gen"/*.wav; do
  [ -e "$wav" ] || continue
  rm -f "${wav%.wav}_utmosv2.txt"
done
/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python \
  score_ourdir_utmosv2.py \
  --dir-list "$list" \
  --batch-size 16 \
  --num-workers 4 \
  --device auto \
  --overwrite

wav_count=$(find "$gen" -maxdepth 1 -type f -name '*.wav' | wc -l)
utmos_count=$(find "$gen" -maxdepth 1 -type f -name '*_utmosv2.txt' | wc -l)
test "$wav_count" -eq 488
test "$utmos_count" -eq "$wav_count"
printf 'validated_utmosv2_coverage=%s/%s\n' "$utmos_count" "$wav_count"
