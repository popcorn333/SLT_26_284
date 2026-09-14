#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=0-20:00:00
set -eo pipefail
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/cache
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export TMPDIR="/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.t/m-${SLURM_JOB_ID:-manual}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TMPDIR"
module load Anaconda3/2022.10
source activate Grad-TTS-EVAL
export PATH="/users/acp23xt/.conda/envs/Grad-TTS-EVAL/bin:$PATH"
python -B -c 'import fastdtw, librosa, numpy, pysptk, pyworld, scipy, soundfile'
gt=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse_stanage/ground_truth
gen=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/add/add_inf/model.masking.a:0.4/test/converted/Epoch_595
out=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/add/add_inf/model.masking.a:0.4/test/metrics

mkdir -p "$out"
for wav in "$gen"/*.wav; do
  [ -e "$wav" ] || continue
  rm -f "${wav%.wav}_f0.txt" "${wav%.wav}_f0_missing.json" "${wav%.wav}_mcd.txt"
done

python "tools/F0/F0.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"
python /mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/validate_f0_missing.py --evaluator "$PWD/tools/F0/F0.py" --gt "$gt" --gen "$gen"
python "tools/MCD/MCD.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"
python /mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/validate_mcd_missing.py --evaluator "$PWD/tools/MCD/MCD.py" --gt "$gt" --gen "$gen"

