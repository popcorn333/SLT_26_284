#!/bin/bash -l
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --job-name=tan18_fmdt_retry2_dispatch
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-retry2-dispatch-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-retry2-dispatch-%j.err
#SBATCH --chdir=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
set -euo pipefail

ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
RES="$ROOT/.symphony/results"
MAP="$RES/tan18_fmdt_mappings.tsv"
RETRY_JOBS="$RES/tan18_fmdt_retry2_jobs.tsv"
WER_ARRAY=11149440
UTMOS_ARRAY=11149441
LOSS_JOB=11142111
BRIDGE_JOB=11143257
FAILED_DISPATCH=11149572
FAILED_METRIC_ARRAY=11149575

METRIC_ARRAY=""
SUMMARY=""
FINAL=""

# If any child-submission or bookkeeping step fails, cancel every child created
# here. The held metric array is released only after the full graph is recorded.
cleanup_on_failure() {
    status=$?
    trap - EXIT
    if (( status != 0 )); then
        ids=()
        test -z "$FINAL" || ids+=("$FINAL")
        test -z "$SUMMARY" || ids+=("$SUMMARY")
        test -z "$METRIC_ARRAY" || ids+=("$METRIC_ARRAY")
        if (( ${#ids[@]} > 0 )); then
            scancel "${ids[@]}" || true
        fi
    fi
    exit "$status"
}
trap cleanup_on_failure EXIT

METRIC_ARRAY=$(sbatch --parsable --hold "$RES/tan18_fmdt_metrics_retry.sbatch")
if [[ ! "$METRIC_ARRAY" =~ ^[0-9]+$ ]]; then
    echo "invalid retry2 metric array ID: $METRIC_ARRAY" >&2
    exit 1
fi

/usr/bin/python3 - "$MAP" "$METRIC_ARRAY" <<'PY'
import csv
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
array_id = sys.argv[2]
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fields = list(reader.fieldnames or [])
    rows = list(reader)
expected_fields = ["folder", "sweep", "gen", "out", "metric_job_id", "wer_job_id", "utmos_job_id"]
if fields != expected_fields:
    raise RuntimeError("unexpected mappings header: {}".format(fields))
expected = ["model.masking.a:" + value for value in ("0.2", "0.4", "0.6", "0.8", "1")]
if len(rows) != 5 or [row["sweep"] for row in rows] != expected:
    raise RuntimeError("unexpected retry2 mapping coverage")
for index, row in enumerate(rows):
    row["metric_job_id"] = "{}_{}".format(array_id, index)
temporary = path.with_name(path.name + ".retry2-{}.tmp".format(array_id))
with temporary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
os.replace(str(temporary), str(path))
PY

SUMMARY=$(sbatch --parsable --export=NIL \
    --dependency="afterok:$METRIC_ARRAY:$WER_ARRAY:$UTMOS_ARRAY:$LOSS_JOB" \
    "$RES/tan18_fmdt_summary.sbatch")
if [[ ! "$SUMMARY" =~ ^[0-9]+$ ]]; then
    echo "invalid retry2 summary ID: $SUMMARY" >&2
    exit 1
fi

FINAL=$(sbatch --parsable \
    --dependency="afterok:$SUMMARY:$BRIDGE_JOB" \
    "$RES/tan18_all_finalize.sbatch")
if [[ ! "$FINAL" =~ ^[0-9]+$ ]]; then
    echo "invalid retry2 final ID: $FINAL" >&2
    exit 1
fi

printf '%s\n' "$SUMMARY" > "$RES/tan18_fmdt_summary_job_id.txt"
printf '%s\n' "$FINAL" > "$RES/tan18_all_finalize_job_id.txt"
printf 'stage\tjob_id\tdetail\n' > "$RETRY_JOBS"
printf 'failed_retry_dispatch\t%s\tdependency submission raced the first metric retry failure\n' "$FAILED_DISPATCH" >> "$RETRY_JOBS"
printf 'failed_metric_retry_array\t%s\tAF_UNIX path too long before any sidecar was written\n' "$FAILED_METRIC_ARRAY" >> "$RETRY_JOBS"
printf 'metric_retry2_array\t%s\tfive tasks held until the complete downstream graph was recorded\n' "$METRIC_ARRAY" >> "$RETRY_JOBS"
printf 'retained_wer_array\t%s\tfive original pending tasks\n' "$WER_ARRAY" >> "$RETRY_JOBS"
printf 'retained_utmos_array\t%s\tfive original pending tasks\n' "$UTMOS_ARRAY" >> "$RETRY_JOBS"
printf 'summary_retry2\t%s\tafter retry2 metrics, retained WER/UTMOSv2, and completed loss\n' "$SUMMARY" >> "$RETRY_JOBS"
printf 'final_audit_retry2\t%s\tafter replacement summary and completed bridge\n' "$FINAL" >> "$RETRY_JOBS"
printf 'fmDT_metric_retry_array2\t%s\t\tSUBMITTED_HELD\tshort approved-tree TMPDIR; released only after summary/final dependency submission\n' "$METRIC_ARRAY" >> "$RES/tan18_active_graph.tsv"
printf 'fmDT_summary_retry2\t%s\tafterok:%s:%s:%s:%s\tSUBMITTED\treplacement summary after every retained/retry metric task and completed loss\n' "$SUMMARY" "$METRIC_ARRAY" "$WER_ARRAY" "$UTMOS_ARRAY" "$LOSS_JOB" >> "$RES/tan18_active_graph.tsv"
printf 'fmDT_final_audit_retry2\t%s\tafterok:%s:%s\tSUBMITTED\tterminal fmDT plus scorex0 audit and final report\n' "$FINAL" "$SUMMARY" "$BRIDGE_JOB" >> "$RES/tan18_active_graph.tsv"

scontrol release "$METRIC_ARRAY"
printf 'metric_retry2=%s summary_retry2=%s final_retry2=%s retained_wer=%s retained_utmos=%s\n' \
    "$METRIC_ARRAY" "$SUMMARY" "$FINAL" "$WER_ARRAY" "$UTMOS_ARRAY"
