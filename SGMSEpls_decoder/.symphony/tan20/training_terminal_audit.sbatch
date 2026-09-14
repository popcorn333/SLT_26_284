#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --job-name=tan20_train_audit
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan20/logs/training-audit-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan20/logs/training-audit-%j.err

set -euo pipefail

root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
state="$root/.symphony/tan20"
output="$state/training_terminal_states.tsv"
temporary="$output.tmp.${SLURM_JOB_ID}"
expected=(
  11142818_0
  11142819_0
  11142819_1
  11142819_2
  11142819_3
  11142819_4
)

mkdir -p "$state/logs"
printf 'job_id\tstate\texit_code\n' > "$temporary"

for attempt in $(seq 1 12); do
  accounting=$(sacct --array -X -j 11142818,11142819 \
    --format=JobID,State,ExitCode -n -P)
  complete=1
  : > "$temporary.rows"
  for job_id in "${expected[@]}"; do
    row=$(printf '%s\n' "$accounting" | awk -F '|' -v wanted="$job_id" '$1 == wanted {print $1 "\t" $2 "\t" $3; exit}')
    if [ -z "$row" ]; then
      complete=0
      break
    fi
    printf '%s\n' "$row" >> "$temporary.rows"
  done
  if [ "$complete" -eq 1 ]; then
    cat "$temporary.rows" >> "$temporary"
    break
  fi
  if [ "$attempt" -eq 12 ]; then
    printf 'accounting did not expose every expected array task\n' >&2
    exit 1
  fi
  sleep 10
done

mv "$temporary" "$output"

failures=$(awk -F '\t' 'NR > 1 && ($2 != "COMPLETED" || $3 != "0:0") {print $1 "=" $2 "/" $3}' "$output")
if [ -n "$failures" ]; then
  printf 'training terminal audit failed:\n%s\n' "$failures" >&2
  exit 1
fi

printf 'validated_training_tasks=6/6\n'
