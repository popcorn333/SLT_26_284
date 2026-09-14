#!/bin/bash
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export MKL_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

export CUDA_VISIBLE_DEVICES=1

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate CUDA122_Grad-TTS 
HYDRA_FULL_ERROR=1 \
bash -c 'exec -a score_eval python inference.py -m --config-name=config_eval_swp +data=data_swp model.masking.a=0.2,0.4,0.6,0.8,1&>25th_April_eval.log&'

