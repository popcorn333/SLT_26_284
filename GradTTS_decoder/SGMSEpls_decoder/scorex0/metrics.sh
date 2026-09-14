#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=0-20:00:00
module load Anaconda3/2022.10
source activate Grad-TTS-EVAL
gt=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse_stanage/ground_truth
gen=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/scorex0/scorex0_inf/model.masking.a:0.2/test/converted/Epoch_595
out=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/scorex0/scorex0_inf/model.masking.a:0.2/test/metrics

python "tools/F0/F0.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"
python "tools/MCD/MCD.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"

