#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=tan7_inf_fallback
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/logs/inference-fallback-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/logs/inference-fallback-%j.err

set -eo pipefail

folder=${1:?experiment folder is required}
parent_job=${2:?parent Slurm job is required}
case "$folder" in
  score|scoreDT|scorex0)
    override_key=model.masking.a
    configurations='1 0.8 0.6 0.4 0.2'
    configuration_csv=1,0.8,0.6,0.4,0.2
    ;;
  add)
    override_key=model.masking.a
    configurations='0.2 0.4 0.6'
    configuration_csv=0.2,0.4,0.6
    ;;
  blur)
    override_key=model.masking.b
    configurations='10 20 40'
    configuration_csv=10,20,40
    ;;
  product)
    override_key=model.masking.a
    configurations='0.2 0.4 0.6'
    configuration_csv=0.2,0.4,0.6
    ;;
  *) printf 'unsupported fallback folder: %s\n' "$folder" >&2; exit 2 ;;
esac

root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
state_dir="$root/.symphony/tan7"
parent_stdout="$state_dir/logs/run-${folder}-${parent_job}.out"
test -s "$parent_stdout"

# Dependency release and accounting publication are not atomic on this
# cluster.  Wait for only the terminal allocation row inside this Slurm job;
# this is bounded to two minutes and never polls from the Symphony host.
parent_record=
for attempt in $(seq 1 12); do
  parent_record=$(sacct -X -n -P -j "$parent_job" --format=JobIDRaw,State%32,ExitCode 2>/dev/null | awk -F '|' -v id="$parent_job" '$1 == id {print $2 "|" $3; exit}') || parent_record=
  if [ -n "$parent_record" ]; then
    observed_state=${parent_record%%|*}
    observed_state=${observed_state%% *}
    observed_state=${observed_state%+}
    case "$observed_state" in
      BOOT_FAIL|CANCELLED|COMPLETED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|TIMEOUT) break ;;
    esac
  fi
  if [ "$attempt" -lt 12 ]; then
    sleep 10
  fi
done
test -n "$parent_record"
parent_state=${parent_record%%|*}
parent_state=${parent_state%% *}
parent_state=${parent_state%+}
parent_exit=${parent_record##*|}
if [ "$parent_state" = COMPLETED ] && [ "$parent_exit" = 0:0 ]; then
  printf 'parent_already_completed=%s state=%s/%s\n' "$parent_job" "$parent_state" "$parent_exit"
  exit 0
fi
case "$parent_state" in
  BOOT_FAIL|CANCELLED*|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|TIMEOUT) ;;
  *) printf 'parent is not a settled failure: %s=%s/%s\n' "$parent_job" "$parent_state" "$parent_exit" >&2; exit 1 ;;
esac

parent_start=$(sacct -X -n -P -j "$parent_job" --format=JobIDRaw,Start | awk -F '|' -v id="$parent_job" '$1 == id {print $2; exit}')
test -n "$parent_start"
test "$parent_start" != Unknown

cd "$root/$folder"
for configuration in $configurations; do
  checkpoint="Hydra_${folder}/${override_key}:${configuration}/checkpoint/grad_595.pt"
  if [ ! -s "$checkpoint" ]; then
    printf 'cannot safely continue inference; missing checkpoint: %s\n' "$checkpoint" >&2
    exit 1
  fi
  if find "$checkpoint" -maxdepth 0 ! -newermt "$parent_start" -print -quit | grep -q .; then
    printf 'cannot safely continue inference; checkpoint predates parent allocation %s (%s): %s\n' \
      "$parent_job" "$parent_start" "$checkpoint" >&2
    exit 1
  fi
done

# A parent can finish one or more Epoch_595 inference targets and then linger
# during interpreter/process teardown until its allocation expires.  Retain a
# target only after independently proving exact transcript-name coverage and
# freshness against the parent allocation.  Rerun only the configurations
# that fail that proof; this keeps a recovery allocation bounded and avoids
# replacing already-valid generated audio.
transcript=/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/add/resources/filelists/ljspeech/test.txt
mkdir -p "$state_dir/tmp"
expected_names="$state_dir/tmp/expected_wav_names-${SLURM_JOB_ID}.txt"
trap 'rm -f "$expected_names"' EXIT
awk -F '|' 'NF {name=$1; sub(/^.*\//, "", name); print name}' "$transcript" | sort -u > "$expected_names"
expected_count=$(wc -l < "$expected_names")
test "$expected_count" -eq 488

pending_configurations=
validated_existing_count=0
for configuration in $configurations; do
  gen="${folder}_inf/${override_key}:${configuration}/test/converted/Epoch_595"
  target_complete=true
  if [ ! -d "$gen" ] || [ "$(find "$gen" -maxdepth 1 -type f -name '*.wav' | wc -l)" -ne "$expected_count" ]; then
    target_complete=false
  elif ! comm -3 \
    "$expected_names" \
    <(find "$gen" -maxdepth 1 -type f -name '*.wav' -printf '%f\n' | sort -u) \
    | awk 'END {exit(NR == 0 ? 0 : 1)}'; then
    target_complete=false
  elif find "$gen" -maxdepth 1 -type f -name '*.wav' ! -newermt "$parent_start" -print -quit | grep -q .; then
    target_complete=false
  fi
  if [ "$target_complete" = true ]; then
    validated_existing_count=$((validated_existing_count + 1))
  else
    pending_configurations="${pending_configurations}${pending_configurations:+ }${configuration}"
  fi
done

if [ -z "$pending_configurations" ]; then
  printf 'validated_existing_inference_fallback=%s parent=%s state=%s/%s paths=%s wavs_per_path=%s\n' \
    "$folder" "$parent_job" "$parent_state" "$parent_exit" "$validated_existing_count" "$expected_count"
  exit 0
fi

configuration_csv=
for configuration in $pending_configurations; do
  configuration_csv="${configuration_csv}${configuration_csv:+,}${configuration}"
done
printf 'inference_fallback_plan=%s retained_paths=%s pending_configurations=%s\n' \
  "$folder" "$validated_existing_count" "$configuration_csv"

# Preserve the completed training phase in this successful continuation's log.
# If a timed-out parent must be replaced in run_jobs.tsv, extract_losses.py can
# still validate every Hydra marker and epoch from the replacement allocation.
printf 'parent_training_log_begin=%s\n' "$parent_stdout"
cat "$parent_stdout"
printf 'parent_training_log_end=%s\n' "$parent_stdout"

export SLURM_EXPORT_ENV=ALL
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$state_dir/cache"
export TMPDIR="$state_dir/tmp/inference-fallback-${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
module purge
module load Anaconda3/2022.10
module load GCC/12.3.0
module load CUDA/12.4.0
source activate CUDA124_sgmsetts

export CUDA_HOME
CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC
CC=$(which gcc)
export CXX
CXX=$(which g++)

export TAN7_EPOCH595_ONLY=1
HYDRA_FULL_ERROR=1 python inference.py -m --config-name=config_eval_swp +data=data_swp "${override_key}=${configuration_csv}"
printf 'validated_inference_fallback=%s parent=%s state=%s/%s\n' "$folder" "$parent_job" "$parent_state" "$parent_exit"
