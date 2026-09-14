#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-4
#SBATCH --job-name=tan6_scoreDT_continue
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scoreDT-training-continuation-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scoreDT-training-continuation-%A_%a.err
set -euo pipefail

ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
cd "$ROOT/scoreDT"

values=(1.0 0.8 0.6 0.4 0.2)
value=${values[$SLURM_ARRAY_TASK_ID]}
run_dir="Hydra_scoreDT/model.masking.a:$value"
log_path="$run_dir/tb/train.log"
checkpoint_dir="$run_dir/checkpoint"

validate_training() {
    /usr/bin/python3 - "$log_path" "$checkpoint_dir" "$value" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
checkpoint_dir = Path(sys.argv[2])
value = sys.argv[3]
pattern = re.compile(
    r"^Epoch (\d+): duration loss = ([0-9eE+.-]+) \| "
    r"prior loss = ([0-9eE+.-]+) \| diffusion loss = ([0-9eE+.-]+)$",
    re.M,
)
if not log_path.is_file():
    raise SystemExit(1)
epochs = {int(match.group(1)) for match in pattern.finditer(
    log_path.read_text(encoding="utf-8", errors="replace"))}
expected = set(range(1, 600))
if epochs != expected:
    missing = sorted(expected - epochs)
    extra = sorted(epochs - expected)
    print("scoreDT {} incomplete: observed={} missing={} extra={}".format(
        value, len(epochs), missing[:20], extra[:20]), flush=True)
    raise SystemExit(1)
checkpoint = checkpoint_dir / "grad_495.pt"
if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
    print("scoreDT {} missing nonempty {}".format(value, checkpoint), flush=True)
    raise SystemExit(1)
print("scoreDT {} training complete: epochs=599 checkpoint={}".format(
    value, checkpoint), flush=True)
PY
}

if validate_training; then
    echo "configuration $value already complete; no training rerun required"
    exit 0
fi

export SLURM_EXPORT_ENV=ALL
module purge
module load Anaconda3/2022.10
module load GCC/12.3.0
module load CUDA/12.4.0
source activate grad-tts-masking

# The in-memory source injection keeps the checked-in training source and its
# Hydra sweep paths unchanged.  Once Hydra enters the existing configuration
# directory it selects that directory's newest checkpoint, loads the complete
# model, and resumes at the following epoch.  A configuration without a local
# checkpoint follows the original run.sh path from the shared pretrained model.
HYDRA_FULL_ERROR=1 python -c '
from pathlib import Path
path = Path("train.py")
source = path.read_text()
marker = "    checkpoint = torch.load(cfg.training.pre_checkpoint)\n"
if source.count(marker) != 1:
    raise RuntimeError("expected exactly one scoreDT checkpoint-load marker")
injected = """    local_checkpoints = sorted(
        __import__(\"pathlib\").Path(cfg.training.checkpoint_save).glob(\"grad_*.pt\"),
        key=lambda item: int(re.search(r\"_(\\d+)\\.pt$\", item.name).group(1)),
    )
    if local_checkpoints:
        cfg.training.pre_checkpoint = str(local_checkpoints[-1])
        cfg.training.load_encoder = True
        cfg.training.load_decoder = True
        print(\"TAN-6 resuming from {}\".format(cfg.training.pre_checkpoint), flush=True)
    checkpoint = torch.load(cfg.training.pre_checkpoint)
"""
source = source.replace(marker, injected)
namespace = {"__name__": "__main__", "__file__": str(path.resolve())}
exec(compile(source, str(path), "exec"), namespace)
' -m --config-name=config_swp +data=data_swp "model.masking.a=$value"

validate_training
