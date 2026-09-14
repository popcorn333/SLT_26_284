#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=tan7_validate_existing
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/logs/validate-existing-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/logs/validate-existing-%j.err

set -euo pipefail

folder=${1:?experiment folder is required}
parent_job=${2:?parent Slurm job is required}
case "$folder" in
  score|scoreDT|scorex0)
    override_key=model.masking.a
    configurations='1 0.8 0.6 0.4 0.2'
    ;;
  *) printf 'unsupported validation folder: %s\n' "$folder" >&2; exit 2 ;;
esac

root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
state_dir="$root/.symphony/tan7"
transcript=/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/add/resources/filelists/ljspeech/test.txt
cd "$root/$folder"

# The job is submitted afterany so it can diagnose the allocation itself.  A
# successful parent needs no replacement; this validator is only legitimate
# for a settled failure whose training checkpoints and final inference output
# can both be tied to that allocation.
parent_record=
for attempt in $(seq 1 12); do
  parent_record=$(sacct -X -n -P -j "$parent_job" --format=JobIDRaw,State%32,ExitCode 2>/dev/null | awk -F '|' -v id="$parent_job" '$1 == id {print $2 "|" $3; exit}') || parent_record=
  if [ -n "$parent_record" ]; then
    parent_state=${parent_record%%|*}
    parent_state=${parent_state%% *}
    parent_state=${parent_state%+}
    case "$parent_state" in
      BOOT_FAIL|CANCELLED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|TIMEOUT) break ;;
    esac
  fi
  if [ "$attempt" -lt 12 ]; then sleep 10; fi
done
test -n "$parent_record"
parent_state=${parent_record%%|*}
parent_state=${parent_state%% *}
parent_state=${parent_state%+}
parent_exit=${parent_record##*|}
case "$parent_state" in
  BOOT_FAIL|CANCELLED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|TIMEOUT) ;;
  *) printf 'parent is not a settled failure: %s=%s/%s\n' "$parent_job" "$parent_state" "$parent_exit" >&2; exit 1 ;;
esac

parent_start=$(sacct -X -n -P -j "$parent_job" --format=JobIDRaw,Start | awk -F '|' -v id="$parent_job" '$1 == id {print $2; exit}')
test -n "$parent_start"
test "$parent_start" != Unknown

mkdir -p "$state_dir/tmp"
expected_names="$state_dir/tmp/expected_wav_names-${SLURM_JOB_ID}.txt"
trap 'rm -f "$expected_names"' EXIT
awk -F '|' 'NF {name=$1; sub(/^.*\//, "", name); print name}' "$transcript" | sort -u > "$expected_names"
expected_count=$(wc -l < "$expected_names")
test "$expected_count" -eq 488

validated=0
for configuration in $configurations; do
  checkpoint="Hydra_${folder}/${override_key}:${configuration}/checkpoint/grad_595.pt"
  gen="${folder}_inf/${override_key}:${configuration}/test/converted/Epoch_595"
  test -s "$checkpoint"
  if find "$checkpoint" -maxdepth 0 ! -newermt "$parent_start" -print -quit | grep -q .; then
    printf 'checkpoint predates parent allocation %s (%s): %s\n' "$parent_job" "$parent_start" "$checkpoint" >&2
    exit 1
  fi
  test -d "$gen"
  actual_count=$(find "$gen" -maxdepth 1 -type f -name '*.wav' | wc -l)
  test "$actual_count" -eq "$expected_count"
  if ! comm -3 "$expected_names" <(find "$gen" -maxdepth 1 -type f -name '*.wav' -printf '%f\n' | sort -u) | awk 'END {exit(NR == 0 ? 0 : 1)}'; then
    printf 'generated WAV names do not match transcript: %s\n' "$gen" >&2
    exit 1
  fi
  if find "$gen" -maxdepth 1 -type f -name '*.wav' ! -newermt "$parent_start" -print -quit | grep -q .; then
    printf 'generated WAV predates parent allocation %s (%s): %s\n' "$parent_job" "$parent_start" "$gen" >&2
    exit 1
  fi
  validated=$((validated + 1))
  printf 'validated_configuration=%s/%s wavs=%s checkpoint=%s\n' "$folder" "$configuration" "$actual_count" "$checkpoint"
done

test "$validated" -eq 5
printf 'validated_existing_inference=%s parent=%s state=%s/%s paths=%s wavs_per_path=%s\n' \
  "$folder" "$parent_job" "$parent_state" "$parent_exit" "$validated" "$expected_count"
