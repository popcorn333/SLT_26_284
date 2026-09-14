#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --job-name=tan18_fmdt_dispatch
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-dispatch-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-dispatch-%j.err
set -euo pipefail
ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
RES="$ROOT/.symphony/results"
RUN="$ROOT/fmDT/run.sh"
: "${RUN_ARRAY_ID:?RUN_ARRAY_ID is required}"
: "${ACCEPTANCE_GATE_JOB_ID:?ACCEPTANCE_GATE_JOB_ID is required}"
: "${TRAINING_LOSS_JOB_ID:?TRAINING_LOSS_JOB_ID is required}"
METRICS="$ROOT/fmDT/metrics.sh"
PAIRS="$RES/tan18_fmdt_pairs.txt"
MAP="$RES/tan18_fmdt_mappings.tsv"
JOBS="$RES/tan18_fmdt_jobs.tsv"
/usr/bin/python3 "$ROOT/extract_metrics_paths.py" "$RUN" > "$PAIRS"
printf 'folder\tsweep\tgen\tout\tmetric_job_id\twer_job_id\tutmos_job_id\n' > "$MAP"
printf 'stage\tjob_id\tdetail\nrun_array\t%s\tfmDT a=0.2,0.4,0.6,0.8,1\nacceptance_gate\t%s\tafterok run array; validates sacct plus fresh training and inference outputs\ntraining_loss\t%s\tafterok acceptance gate\ndispatch\t%s\tafterok acceptance gate\n' "$RUN_ARRAY_ID" "$ACCEPTANCE_GATE_JOB_ID" "$TRAINING_LOSS_JOB_ID" "$SLURM_JOB_ID" > "$JOBS"
mapfile -t gens < <(sed -n 's/^gen=//p' "$PAIRS")
mapfile -t outs < <(sed -n 's/^out=//p' "$PAIRS")
expected_sweeps=(
  model.masking.a:0.2
  model.masking.a:0.4
  model.masking.a:0.6
  model.masking.a:0.8
  model.masking.a:1
)
test "${#gens[@]}" -eq "${#expected_sweeps[@]}"
test "${#gens[@]}" -eq "${#outs[@]}"

# Validate the entire extracted mapping before submitting the first child.  A
# partial or duplicated extractor result must fail without launching a partial
# metrics/WER/UTMOSv2 graph.
declare -A seen_gens=()
declare -A seen_outs=()
for i in "${!expected_sweeps[@]}"; do
  sweep="${expected_sweeps[$i]}"
  expected_gen="$ROOT/fmDT/fmDT_inf/$sweep/test/converted/Epoch_595"
  expected_out="$ROOT/fmDT/fmDT_inf/$sweep/test/metrics"
  test "${gens[$i]}" = "$expected_gen"
  test "${outs[$i]}" = "$expected_out"
  test -d "${gens[$i]}"
  test -z "${seen_gens[${gens[$i]}]+present}"
  test -z "${seen_outs[${outs[$i]}]+present}"
  seen_gens["${gens[$i]}"]=1
  seen_outs["${outs[$i]}"]=1
done
metric_ids=()
for i in "${!gens[@]}"; do
  gen="${gens[$i]}"
  out="${outs[$i]}"
  sweep=$(basename "$(dirname "$(dirname "$(dirname "$gen")")")")
  job=$(sbatch --parsable --chdir="$ROOT/fmDT" \
    --job-name="tan18_fmdt_m_$i" \
    --output="$RES/tan18-fmdt-metric-$i-%j.out" \
    --error="$RES/tan18-fmdt-metric-$i-%j.err" \
    --export="GEN_PATH=$gen,OUT_PATH=$out" \
    "$METRICS")
  metric_ids+=("$job")
  printf 'fmDT\t%s\t%s\t%s\t%s\t\t\n' "$sweep" "$gen" "$out" "$job" >> "$MAP"
  printf 'metric\t%s\t%s\n' "$job" "$gen" >> "$JOBS"
done
last=$((${#gens[@]} - 1))
wer=$(sbatch --parsable --export=NIL --exclude=gpu33,gpu34 --array="0-$last" "$RES/tan18_fmdt_wer_array.sbatch")
utmos=$(sbatch --parsable --export=NIL --exclude=gpu33,gpu34 --array="0-$last" "$RES/tan18_fmdt_utmos_array.sbatch")
awk -F '\t' -v OFS='\t' -v w="$wer" -v u="$utmos" 'NR==1 {print; next} {$6=w "_" (NR-2); $7=u "_" (NR-2); print}' "$MAP" > "$MAP.tmp"
mv "$MAP.tmp" "$MAP"
printf 'wer_array\t%s\t%s paths\nutmos_array\t%s\t%s paths\n' "$wer" "${#gens[@]}" "$utmos" "${#gens[@]}" >> "$JOBS"
deps=$(IFS=:; echo "${metric_ids[*]}")
summary=$(sbatch --parsable --export=NIL --dependency="afterok:${deps}:$wer:$utmos:$TRAINING_LOSS_JOB_ID" "$RES/tan18_fmdt_summary.sbatch")
printf 'summary\t%s\tafter all fmDT metrics\n' "$summary" >> "$JOBS"
printf '%s\n' "$summary" > "$RES/tan18_fmdt_summary_job_id.txt"
echo "pairs=${#gens[@]} metric_jobs=${metric_ids[*]} wer=$wer utmos=$utmos summary=$summary"
