#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=0-10:00:00
module load Anaconda3/2025.06-1
source activate Grad-TTS-EVAL
gt=/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/ground_truth
gen=/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/add/add_inf/model.masking.a:0.6/test/converted/Epoch_595
out=/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/add/add_inf/model.masking.a:0.6/test/metrics

python "tools/F0/F0.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"
python "tools/MCD/MCD.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"

