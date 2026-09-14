#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --job-name=tan11_fm_infer_recovery
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan11_fm/logs/recovery-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan11_fm/logs/recovery-%A_%a.err

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
state="$root/.symphony/tan11_fm"
parent=11097462
values=(0.2 0.6)
value="${values[${SLURM_ARRAY_TASK_ID}]}"

accounting=$(sacct -X -j "$parent" --array --format=JobID,State,ExitCode -n -P)
check_state() {
  local task="$1" expected_state="$2" expected_exit="$3" line
  line=$(awk -F '|' -v id="${parent}_${task}" '$1 == id {print $2 "|" $3}' <<<"$accounting")
  test "$line" = "${expected_state}|${expected_exit}"
}
check_state 0 FAILED 7:0
check_state 1 COMPLETED 0:0
check_state 2 FAILED 7:0
check_state 3 COMPLETED 0:0
check_state 4 COMPLETED 0:0
for task in 0 2; do
  grep -q 'Epoch 599:' "$state/logs/run-${parent}_${task}.out"
  grep -q 'Bus error' "$state/logs/run-${parent}_${task}.err"
done

checkpoint="$root/fm/Hydra_fm/model.masking.a:${value}/checkpoint"
for epoch in $(seq 450 5 595); do
  test -s "$checkpoint/grad_${epoch}.pt"
done

cd "$root/fm"
module purge
module load Anaconda3/2022.10
module load GCC/12.3.0
module load CUDA/12.4.0
source activate CUDA124_sgmsetts
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=$(which gcc)
export CXX=$(which g++)
HYDRA_FULL_ERROR=1 python inference.py -m --config-name=config_eval_swp +data=data_swp model.masking.a="$value"

for epoch in $(seq 450 5 595); do
  generated="$root/fm/fm_inf/model.masking.a:${value}/test/converted/Epoch_${epoch}"
  test -d "$generated"
  test "$(find "$generated" -maxdepth 1 -type f -name '*.wav' | wc -l)" -eq 488
done
printf 'validated_recovery_configuration=%s epochs=30 wavs_per_epoch=488\n' "$value"
