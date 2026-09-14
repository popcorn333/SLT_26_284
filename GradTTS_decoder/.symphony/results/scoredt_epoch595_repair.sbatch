#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --job-name=tan10_scoredt_595
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scoredt-595-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scoredt-595-%j.err
set -euo pipefail

ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
RESULTS="$ROOT/.symphony/results"

module purge
module load Anaconda3/2022.10
module load GCC/12.3.0
module load CUDA/12.4.0
source activate grad-tts-masking

export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONDONTWRITEBYTECODE=1

cd "$ROOT/scoreDT"
for value in 1.0 0.8 0.6 0.4 0.2; do
    checkpoint="$ROOT/scoreDT/Hydra_scoreDT/model.masking.a:$value/checkpoint/grad_595.pt"
    test -s "$checkpoint"
done

HYDRA_FULL_ERROR=1 python "$RESULTS/run_scoredt_epoch595.py" -m \
    --config-name=config_eval_swp +data=data_swp \
    model.masking.a=1.0,0.8,0.6,0.4,0.2

for value in 1.0 0.8 0.6 0.4 0.2; do
    gen="$ROOT/scoreDT/scoreDT_inf/model.masking.a:$value/test/converted/Epoch_595"
    test "$(find "$gen" -maxdepth 1 -type f -name '*.wav' | wc -l)" -eq 488
done
