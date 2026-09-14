#!/bin/bash
# Request 10 gigabytes of real memory (mem)
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1         # Number of GPUs
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --job-name=Parametric_Masking_3rd_Loss
#SBATCH --time=2-00:00:00
export SLURM_EXPORT_ENV=ALL
module load Anaconda3/2022.05
source activate grad-tts-masking

HYDRA_FULL_ERROR=1 python eval_all.py -m --config-name=config_eval eval.evaluation_mode=LOSSES +data=data training.n_timesteps=1,2,3,4,5,6,7,8,9,10 model.Masking.d=0.1,0.5 
