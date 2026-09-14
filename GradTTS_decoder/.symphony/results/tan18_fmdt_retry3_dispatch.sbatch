#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --job-name=tan18_fmdt_retry3_dispatch
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-retry3-dispatch-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-retry3-dispatch-%j.err
#SBATCH --chdir=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
set -euo pipefail

ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
RES="$ROOT/.symphony/results"
MAP="$RES/tan18_fmdt_mappings.tsv"
RETRY_JOBS="$RES/tan18_fmdt_retry3_jobs.tsv"
LOSS_JOB=11142111
BRIDGE_JOB=11143257

metric_ids=()
wer_ids=()
utmos_ids=()
SUMMARY=""
FINAL=""

cleanup_on_failure() {
    status=$?
    trap - EXIT
    if (( status != 0 )); then
        ids=()
        test -z "$FINAL" || ids+=("$FINAL")
        test -z "$SUMMARY" || ids+=("$SUMMARY")
        ids+=("${metric_ids[@]}" "${wer_ids[@]}" "${utmos_ids[@]}")
        if (( ${#ids[@]} > 0 )); then
            scancel "${ids[@]}" || true
        fi
    fi
    exit "$status"
}
trap cleanup_on_failure EXIT

for index in 0 1 2 3 4; do
    id=$(sbatch --parsable --hold --export="ALL,TAN18_TASK_INDEX=$index" "$RES/tan18_fmdt_metrics_retry.sbatch")
    [[ "$id" =~ ^[0-9]+$ ]]
    metric_ids+=("$id")

    id=$(sbatch --parsable --hold --export="TAN18_TASK_INDEX=$index" "$RES/tan18_fmdt_wer_array.sbatch")
    [[ "$id" =~ ^[0-9]+$ ]]
    wer_ids+=("$id")

    id=$(sbatch --parsable --hold --export="TAN18_TASK_INDEX=$index" "$RES/tan18_fmdt_utmos_array.sbatch")
    [[ "$id" =~ ^[0-9]+$ ]]
    utmos_ids+=("$id")
done

metric_csv=$(IFS=,; echo "${metric_ids[*]}")
wer_csv=$(IFS=,; echo "${wer_ids[*]}")
utmos_csv=$(IFS=,; echo "${utmos_ids[*]}")

/usr/bin/python3 - "$MAP" "$metric_csv" "$wer_csv" "$utmos_csv" <<'PY'
import csv
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
job_columns = [value.split(",") for value in sys.argv[2:]]
if any(len(values) != 5 for values in job_columns):
    raise RuntimeError("retry3 did not submit exactly five jobs per stage")
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fields = list(reader.fieldnames or [])
    rows = list(reader)
expected_fields = ["folder", "sweep", "gen", "out", "metric_job_id", "wer_job_id", "utmos_job_id"]
if fields != expected_fields:
    raise RuntimeError("unexpected mappings header: {}".format(fields))
expected = ["model.masking.a:" + value for value in ("0.2", "0.4", "0.6", "0.8", "1")]
if len(rows) != 5 or [row["sweep"] for row in rows] != expected:
    raise RuntimeError("unexpected retry3 mapping coverage")
for row, metric, wer, utmos in zip(rows, *job_columns):
    row["metric_job_id"] = metric
    row["wer_job_id"] = wer
    row["utmos_job_id"] = utmos
temporary = path.with_name(path.name + ".retry3.tmp")
with temporary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
os.replace(str(temporary), str(path))
PY

all_payload_ids=("${metric_ids[@]}" "${wer_ids[@]}" "${utmos_ids[@]}")
deps=$(IFS=:; echo "${all_payload_ids[*]}")
SUMMARY=$(sbatch --parsable --export=NIL \
    --dependency="afterok:$deps:$LOSS_JOB" \
    "$RES/tan18_fmdt_summary.sbatch")
[[ "$SUMMARY" =~ ^[0-9]+$ ]]
FINAL=$(sbatch --parsable \
    --dependency="afterok:$SUMMARY:$BRIDGE_JOB" \
    "$RES/tan18_all_finalize.sbatch")
[[ "$FINAL" =~ ^[0-9]+$ ]]

printf '%s\n' "$SUMMARY" > "$RES/tan18_fmdt_summary_job_id.txt"
printf '%s\n' "$FINAL" > "$RES/tan18_all_finalize_job_id.txt"
printf 'stage\tjob_id\tdetail\n' > "$RETRY_JOBS"
printf 'failed_retry2_dispatch\t11149614\told WER array failed during dependency submission\n' >> "$RETRY_JOBS"
printf 'cancelled_retry2_metric_array\t11149621\theld array cancelled by fail-closed dispatcher; never ran\n' >> "$RETRY_JOBS"
for index in 0 1 2 3 4; do
    printf 'metric_retry3\t%s\tconfiguration %s\n' "${metric_ids[$index]}" "$index" >> "$RETRY_JOBS"
    printf 'wer_retry1\t%s\tconfiguration %s, Whisper large-v3\n' "${wer_ids[$index]}" "$index" >> "$RETRY_JOBS"
    printf 'utmos_retry1\t%s\tconfiguration %s\n' "${utmos_ids[$index]}" "$index" >> "$RETRY_JOBS"
done
printf 'summary_retry3\t%s\tafter all 15 explicit payload jobs and completed loss\n' "$SUMMARY" >> "$RETRY_JOBS"
printf 'final_audit_retry3\t%s\tafter replacement summary and completed bridge\n' "$FINAL" >> "$RETRY_JOBS"
printf 'fmDT_payload_retry3\t%s\t\tSUBMITTED_HELD\tfive explicit metric, five WER, and five UTMOSv2 jobs\n' "$deps" >> "$RES/tan18_active_graph.tsv"
printf 'fmDT_summary_retry3\t%s\tafterok:%s:%s\tSUBMITTED\treplacement summary after all explicit payload jobs and completed loss\n' "$SUMMARY" "$deps" "$LOSS_JOB" >> "$RES/tan18_active_graph.tsv"
printf 'fmDT_final_audit_retry3\t%s\tafterok:%s:%s\tSUBMITTED\tterminal fmDT plus scorex0 audit and final report\n' "$FINAL" "$SUMMARY" "$BRIDGE_JOB" >> "$RES/tan18_active_graph.tsv"

for id in "${all_payload_ids[@]}"; do
    scontrol release "$id"
done
printf 'metrics=%s wer=%s utmos=%s summary=%s final=%s\n' \
    "$metric_csv" "$wer_csv" "$utmos_csv" "$SUMMARY" "$FINAL"
