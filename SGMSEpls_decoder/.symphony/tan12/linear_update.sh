#!/bin/bash
set -euo pipefail
mode=${1:?mode required}
detail=${2:-}
: "${LINEAR_API_KEY:?LINEAR_API_KEY is required}"
issue_id=24217a23-96bf-456a-991e-0deb26688434
comment_id=19a5f9a1-2c17-42e4-b3d9-2cf9b7c23934
state=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan12
results=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/results
if [[ "$mode" == dispatch ]]; then
  metrics=$(jq -r ".metric_jobs" "$state/dispatch_status.json")
  wer=$(jq -r ".wer_array_job_id" "$state/dispatch_status.json")
  utmos=$(jq -r ".utmosv2_array_job_id" "$state/dispatch_status.json")
  finalizer=$(jq -r ".finalizer_job_id" "$state/dispatch_status.json")
  body=$(printf "## Codex Workpad — Slurm-monitored\n\nThe standalone 4 GB CPU dispatcher completed after run arrays 11108873 and 11108874.\n\n- Metrics jobs: %s submissions (see paths.tsv)\n- Whisper large-v3 array: %s\n- UTMOSv2 array: %s\n- Finalizer: %s\n\nThe finalizer will validate sacct states, logs, per-WAV coverage, statistics, and reports before closing TAN-12.\n" "$metrics" "$wer" "$utmos" "$finalizer")
elif [[ "$mode" == complete ]]; then
  report=$(<"$results/tan12_summary.md")
  body=$(printf "## Codex Workpad — Complete\n\nStandalone Slurm monitoring completed successfully. Run arrays: Levy 11108873; fmDT 11108874. Finalizer: %s.\n\n%s" "${SLURM_JOB_ID:-unknown}" "$report")
elif [[ "$mode" == failure ]]; then
  body=$(printf "## Codex Workpad — Persistent failure\n\nStandalone Slurm orchestration stopped safely. Job: %s. Stage: %s. Inspect %s/logs; TAN-12 remains In Progress.\n" "${SLURM_JOB_ID:-unknown}" "$detail" "$state")
else
  printf "unknown mode: %s\n" "$mode" >&2
  exit 2
fi
query="mutation UpdateComment(\$id:String!,\$body:String!){commentUpdate(id:\$id,input:{body:\$body}){success}}"
payload=$(jq -nc --arg q "$query" --arg id "$comment_id" --arg body "$body" "{query:\$q,variables:{id:\$id,body:\$body}}")
response=$(curl -fsS https://api.linear.app/graphql -H "Content-Type: application/json" -H "Authorization: $LINEAR_API_KEY" --data-binary "$payload")
jq -e ".data.commentUpdate.success == true and (.errors == null)" <<<"$response" >/dev/null
if [[ "$mode" == complete ]]; then
  query="mutation Done(\$id:String!,\$stateId:String!){issueUpdate(id:\$id,input:{stateId:\$stateId}){success}}"
  payload=$(jq -nc --arg q "$query" --arg id "$issue_id" --arg stateId ad81390f-ea4b-47a5-b4ba-682cd008ca90 "{query:\$q,variables:{id:\$id,stateId:\$stateId}}")
  response=$(curl -fsS https://api.linear.app/graphql -H "Content-Type: application/json" -H "Authorization: $LINEAR_API_KEY" --data-binary "$payload")
  jq -e ".data.issueUpdate.success == true and (.errors == null)" <<<"$response" >/dev/null
fi
