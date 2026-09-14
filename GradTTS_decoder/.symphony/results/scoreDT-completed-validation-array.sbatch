#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --array=0-4
#SBATCH --job-name=tan6_scoreDT_complete_check
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scoreDT-completed-validation-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scoreDT-completed-validation-%A_%a.err
set -euo pipefail

ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
values=(1.0 0.8 0.6 0.4 0.2)
value=${values[$SLURM_ARRAY_TASK_ID]}
log_path="$ROOT/scoreDT/Hydra_scoreDT/model.masking.a:$value/tb/train.log"
checkpoint="$ROOT/scoreDT/Hydra_scoreDT/model.masking.a:$value/checkpoint/grad_495.pt"

/usr/bin/python3 - "$log_path" "$checkpoint" "$value" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
value = sys.argv[3]
pattern = re.compile(
    r"^Epoch (\d+): duration loss = ([0-9eE+.-]+) \| "
    r"prior loss = ([0-9eE+.-]+) \| diffusion loss = ([0-9eE+.-]+)$",
    re.M,
)
if not log_path.is_file():
    raise RuntimeError("missing scoreDT training log {}".format(log_path))
matches = pattern.findall(log_path.read_text(encoding="utf-8", errors="replace"))
epochs = {int(match[0]) for match in matches}
expected = set(range(1, 600))
if epochs != expected:
    raise RuntimeError(
        "scoreDT {} incomplete: observed={} missing={} extra={}".format(
            value, len(epochs), sorted(expected - epochs)[:20],
            sorted(epochs - expected)[:20]))
if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
    raise RuntimeError("missing nonempty scoreDT checkpoint {}".format(checkpoint))
print(
    "scoreDT {} completed-parent validation: epochs=599 duplicate_records={} checkpoint_bytes={}".format(
        value, len(matches) - len(epochs), checkpoint.stat().st_size),
    flush=True,
)
PY
