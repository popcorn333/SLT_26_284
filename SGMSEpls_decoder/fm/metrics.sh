#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=0-05:00:00
if module is-avail Anaconda3/2025.06-1; then
    module load Anaconda3/2025.06-1
fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate Grad-TTS-EVAL
else
    echo "Grad-TTS-EVAL requires Anaconda3/2025.06-1 (or another conda installation)" >&2
    exit 1
fi
gt=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse_stanage/ground_truth
gen=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/fm/fm_inf/model.masking.a:1/test/converted/Epoch_595
out=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/fm/fm_inf/model.masking.a:1/test/metrics

python "tools/F0/F0.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"
python "tools/MCD/MCD.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"

