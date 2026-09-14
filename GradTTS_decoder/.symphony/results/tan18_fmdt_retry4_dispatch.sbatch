#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --job-name=tan18_fmdt_retry4_dispatch
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-retry4-dispatch-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-retry4-dispatch-%j.err
#SBATCH --chdir=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
set -euo pipefail

ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
RES="$ROOT/.symphony/results"
MAP="$RES/tan18_fmdt_mappings.tsv"
RETRY_JOBS="$RES/tan18_fmdt_retry4_jobs.tsv"

METRIC=""
WER=""
UTMOS=""
SUMMARY=""
FINAL=""

cleanup_on_failure() {
    status=$?
    trap - EXIT
    if (( status != 0 )); then
        ids=()
        test -z "$FINAL" || ids+=("$FINAL")
        test -z "$SUMMARY" || ids+=("$SUMMARY")
        test -z "$METRIC" || ids+=("$METRIC")
        test -z "$WER" || ids+=("$WER")
        test -z "$UTMOS" || ids+=("$UTMOS")
        if (( ${#ids[@]} > 0 )); then
            scancel "${ids[@]}" || true
        fi
    fi
    exit "$status"
}
trap cleanup_on_failure EXIT

# Hold every payload until its arrays, replacement summary, terminal audit,
# mappings, and provenance artifacts have all been created successfully.
METRIC=$(sbatch --parsable --hold --export=NIL --array=0-4 \
    "$RES/tan18_fmdt_metrics_retry.sbatch")
[[ "$METRIC" =~ ^[0-9]+$ ]]
WER=$(sbatch --parsable --hold --export=NIL --array=0-4 \
    "$RES/tan18_fmdt_wer_array.sbatch")
[[ "$WER" =~ ^[0-9]+$ ]]
UTMOS=$(sbatch --parsable --hold --export=NIL --array=0-4 \
    "$RES/tan18_fmdt_utmos_array.sbatch")
[[ "$UTMOS" =~ ^[0-9]+$ ]]

# Jobs 11142111 (training-loss merge) and 11143257 (completion bridge) are
# already terminal COMPLETED 0:0 and are revalidated by the terminal audit.
# Do not attach new controller dependencies to aged-out completed jobs.
SUMMARY=$(sbatch --parsable --export=NIL \
    --dependency="afterok:$METRIC:$WER:$UTMOS" \
    "$RES/tan18_fmdt_summary.sbatch")
[[ "$SUMMARY" =~ ^[0-9]+$ ]]
FINAL=$(sbatch --parsable --export=NIL \
    --dependency="afterok:$SUMMARY" \
    "$RES/tan18_all_finalize.sbatch")
[[ "$FINAL" =~ ^[0-9]+$ ]]

/usr/bin/python3 - "$MAP" "$METRIC" "$WER" "$UTMOS" <<'PY'
import csv
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
parents = sys.argv[2:]
if len(parents) != 3 or any(not value.isdigit() for value in parents):
    raise RuntimeError("retry4 did not receive three valid array parent IDs")
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fields = list(reader.fieldnames or [])
    rows = list(reader)
expected_fields = ["folder", "sweep", "gen", "out", "metric_job_id", "wer_job_id", "utmos_job_id"]
if fields != expected_fields:
    raise RuntimeError("unexpected mappings header: {}".format(fields))
expected = ["model.masking.a:" + value for value in ("0.2", "0.4", "0.6", "0.8", "1")]
if len(rows) != 5 or [row["sweep"] for row in rows] != expected:
    raise RuntimeError("unexpected retry4 mapping coverage")
for index, row in enumerate(rows):
    row["metric_job_id"] = "{}_{}".format(parents[0], index)
    row["wer_job_id"] = "{}_{}".format(parents[1], index)
    row["utmos_job_id"] = "{}_{}".format(parents[2], index)
temporary = path.with_name(path.name + ".retry4.tmp")
with temporary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
os.replace(str(temporary), str(path))
PY

printf '%s\n' "$SUMMARY" > "$RES/tan18_fmdt_summary_job_id.txt"
printf '%s\n' "$FINAL" > "$RES/tan18_all_finalize_job_id.txt"
printf 'stage\tjob_id\tdetail\n' > "$RETRY_JOBS"
printf 'failed_retry3_dispatch\t11149644\tcontroller dependency on aged completed predecessor rejected\n' >> "$RETRY_JOBS"
printf 'cancelled_retry3_payload_range\t11149783-11149797\tfifteen held jobs cancelled before start by fail-closed cleanup\n' >> "$RETRY_JOBS"
printf 'metric_retry4_array\t%s\tfive fmDT configurations\n' "$METRIC" >> "$RETRY_JOBS"
printf 'wer_retry2_array\t%s\tfive fmDT configurations, Whisper large-v3\n' "$WER" >> "$RETRY_JOBS"
printf 'utmos_retry2_array\t%s\tfive fmDT configurations\n' "$UTMOS" >> "$RETRY_JOBS"
for index in 0 1 2 3 4; do
    printf 'metric_retry4_task\t%s_%s\tconfiguration %s\n' "$METRIC" "$index" "$index" >> "$RETRY_JOBS"
    printf 'wer_retry2_task\t%s_%s\tconfiguration %s\n' "$WER" "$index" "$index" >> "$RETRY_JOBS"
    printf 'utmos_retry2_task\t%s_%s\tconfiguration %s\n' "$UTMOS" "$index" "$index" >> "$RETRY_JOBS"
done
printf 'summary_retry4\t%s\tafter all three five-task arrays\n' "$SUMMARY" >> "$RETRY_JOBS"
printf 'final_audit_retry4\t%s\tafter replacement summary; revalidates completed loss and bridge\n' "$FINAL" >> "$RETRY_JOBS"
printf 'fmDT_metric_retry4\t%s\t\tSUBMITTED_HELD\tfive-task CPU metric array\n' "$METRIC" >> "$RES/tan18_active_graph.tsv"
printf 'fmDT_wer_retry2\t%s\t\tSUBMITTED_HELD\tfive-task Whisper large-v3 array\n' "$WER" >> "$RES/tan18_active_graph.tsv"
printf 'fmDT_utmos_retry2\t%s\t\tSUBMITTED_HELD\tfive-task UTMOSv2 array\n' "$UTMOS" >> "$RES/tan18_active_graph.tsv"
printf 'fmDT_summary_retry4\t%s\tafterok:%s:%s:%s\tSUBMITTED\treplacement summary\n' "$SUMMARY" "$METRIC" "$WER" "$UTMOS" >> "$RES/tan18_active_graph.tsv"
printf 'fmDT_final_audit_retry4\t%s\tafterok:%s\tSUBMITTED\tterminal fmDT plus scorex0 audit and final report\n' "$FINAL" "$SUMMARY" >> "$RES/tan18_active_graph.tsv"

scontrol release "$METRIC" "$WER" "$UTMOS"
printf 'metric=%s wer=%s utmos=%s summary=%s final=%s\n' \
    "$METRIC" "$WER" "$UTMOS" "$SUMMARY" "$FINAL"
