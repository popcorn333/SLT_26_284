#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=gpu33
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=tan20_wer
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan20/logs/wer-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan20/logs/wer-%A_%a.err

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
WHISPER_ENV=/mnt/parscratch/users/acp23xt/private/codex_whisper/.conda/envs/codex_whisper
export PATH="$WHISPER_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$WHISPER_ENV/lib:${LD_LIBRARY_PATH:-}"
root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
state=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan20
export TMPDIR="$root/.t/t20w-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TMPDIR"

cd /mnt/parscratch/users/acp23xt/private/codex_whisper
"$WHISPER_ENV/bin/python" -B -c \
  'import torch; assert torch.cuda.is_available(), "Whisper GPU allocation has no visible CUDA device"; print("whisper_cuda=" + torch.cuda.get_device_name(0), flush=True)'
test -x "$WHISPER_ENV/bin/ffmpeg"

paths="$state/paths.tsv"
gen=$(awk -F '\t' -v row="$((SLURM_ARRAY_TASK_ID + 2))" 'NR == row {print $4}' "$paths")
test -n "$gen"
case "$gen" in
  /mnt/parscratch/users/acp23xt/private/DTDM_sgmse/fmDT/*) ;;
  /mnt/parscratch/users/acp23xt/private/DTDM_sgmse/levy/*) ;;
  *) printf 'refusing out-of-scope gen path: %s\n' "$gen" >&2; exit 1 ;;
esac
epoch=$(basename "$gen")
"$WHISPER_ENV/bin/python" batch_wer_epoch495.py \
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
