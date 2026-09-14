#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --job-name=tan18_fmdt_gate
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-gate-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-gate-%j.err
set -euo pipefail
ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
RES="$ROOT/.symphony/results"
: "${RUN_ARRAY_ID:?RUN_ARRAY_ID is required}"
AUDIT="$RES/fmdt-run-${RUN_ARRAY_ID}-gate-sacct.tsv"
sacct -X -n -P -j "$RUN_ARRAY_ID" --format=JobID,State,ExitCode > "$AUDIT"
/usr/bin/python3 - "$AUDIT" "$RUN_ARRAY_ID" <<'PY'
import re
import sys
from pathlib import Path
rows = []
for line in Path(sys.argv[1]).read_text().splitlines():
    fields = line.split('|')
    if (len(fields) >= 3
            and re.fullmatch(re.escape(sys.argv[2]) + r'_[0-4]', fields[0].strip())):
        rows.append(tuple(field.strip() for field in fields[:3]))
if len(rows) != 5:
    raise RuntimeError('expected fmDT array tasks _0 through _4, observed {}'.format(rows))
bad = [row for row in rows if not row[1].startswith('COMPLETED') or row[2] != '0:0']
if bad:
    raise RuntimeError('fmDT array accounting failure: {}'.format(bad))
print('fmDT sacct gate passed: {}'.format(rows), flush=True)
PY
/usr/bin/python3 "$RES/tan18_fmdt_acceptance_gate.py"
