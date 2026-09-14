# input -> root_directory. It is the dir containing cvt and gt folders for one model parameter choice
# output -> in cvt/epoch_idx folder, it outputs mcd.txt and f0.txt
source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate CUDA122_Grad-TTS_eval

root_directory="/exp/exp4/acp23xt/DTDM_sgmse/add/add_inf/model.masking.a:1.5/test"

./tools4epochs/MCD/MCD.sh ${root_directory}
./tools4epochs/F0/F0.sh ${root_directory}

