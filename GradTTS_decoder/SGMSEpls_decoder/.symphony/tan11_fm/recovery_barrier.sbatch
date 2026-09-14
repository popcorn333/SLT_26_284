#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --job-name=tan11_fm_recovery_barrier
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan11_fm/logs/recovery-barrier-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan11_fm/logs/recovery-barrier-%j.err

set -euo pipefail
root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
state="$root/.symphony/tan11_fm"
parent=11097462
test -n "${RECOVERY_ARRAY_JOB_ID:-}"
accounting=$(sacct -X -j "$RECOVERY_ARRAY_JOB_ID" --array --format=JobID,State,ExitCode -n -P)
for task in 0 1; do
  line=$(awk -F '|' -v id="${RECOVERY_ARRAY_JOB_ID}_${task}" '$1 == id {print $2 "|" $3}' <<<"$accounting")
  test "$line" = 'COMPLETED|0:0'
done

for value in 0.2 0.4 0.6 0.8 1; do
  grep -q 'Epoch 599:' "$state/logs/run-${parent}_$(case "$value" in 0.2) echo 0;; 0.4) echo 1;; 0.6) echo 2;; 0.8) echo 3;; 1) echo 4;; esac).out"
  generated="$root/fm/fm_inf/model.masking.a:${value}/test/converted/Epoch_595"
  test -d "$generated"
  test "$(find "$generated" -maxdepth 1 -type f -name '*.wav' | wc -l)" -eq 488
done

python3 - "$state/run_jobs.tsv" "$SLURM_JOB_ID" "$RECOVERY_ARRAY_JOB_ID" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
barrier = sys.argv[2]
recovery = sys.argv[3]
temporary = path.with_name(path.name + ".tmp")
fields = [
    "folder", "run_path", "job_id", "stdout", "stderr",
    "training_job_id", "recovery_array_job_id",
]
row = {
    "folder": "fm",
    "run_path": "fm/run.sh",
    "job_id": barrier,
    "stdout": str(path.parent / "logs" / f"recovery-barrier-{barrier}.out"),
    "stderr": str(path.parent / "logs" / f"recovery-barrier-{barrier}.err"),
    "training_job_id": "11097462",
    "recovery_array_job_id": recovery,
}
with temporary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow(row)
temporary.replace(path)
PY
printf 'validated_recovery_array=%s barrier=%s fm_paths=5\n' "$RECOVERY_ARRAY_JOB_ID" "$SLURM_JOB_ID"
