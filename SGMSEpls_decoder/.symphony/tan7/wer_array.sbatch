#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=tan7_wer_lgv3
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/logs/wer-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/logs/wer-%A_%a.err

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XDG_CACHE_HOME=/mnt/parscratch/users/acp23xt/private/codex_whisper/.cache
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TMPDIR="/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/tmp/wer-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TMPDIR"
module purge
module load Anaconda3/2022.10
module load CUDA/12.4.0
module load FFmpeg/6.0-GCCcore-12.3.0

cd /mnt/parscratch/users/acp23xt/private/codex_whisper
./.conda/envs/codex_whisper/bin/python -B -c \
  'import torch; assert torch.cuda.is_available(), "Whisper GPU allocation has no visible CUDA device"; print("whisper_cuda=" + torch.cuda.get_device_name(0), flush=True)'

paths=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/paths.tsv
gen=$(awk -F '\t' -v row="$((SLURM_ARRAY_TASK_ID + 2))" 'NR == row {print $4}' "$paths")
test -n "$gen"
case "$gen" in
  /mnt/parscratch/users/acp23xt/private/DTDM_sgmse/*) ;;
  *) printf 'refusing out-of-scope gen path: %s\n' "$gen" >&2; exit 1 ;;
esac
epoch=$(basename "$gen")
for wav in "$gen"/*.wav; do
  [ -e "$wav" ] || continue
  rm -f "${wav%.wav}_wer_l.txt"
done
./.conda/envs/codex_whisper/bin/python batch_wer_epoch495.py \
  --root "$gen" \
  --transcript /mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/add/resources/filelists/ljspeech/test.txt \
  --epoch "$epoch" \
  --model large-v3 \
  --output-suffix _wer_l.txt \
  --overwrite \
  --progress-every 25

wav_count=$(find "$gen" -maxdepth 1 -type f -name '*.wav' | wc -l)
wer_count=$(find "$gen" -maxdepth 1 -type f -name '*_wer_l.txt' | wc -l)
test "$wav_count" -eq 488
test "$wer_count" -eq "$wav_count"
printf 'validated_wer_coverage=%s/%s\n' "$wer_count" "$wav_count"
