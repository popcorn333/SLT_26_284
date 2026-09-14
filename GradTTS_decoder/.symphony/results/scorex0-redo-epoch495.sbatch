#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --job-name=tan6_scorex0_epoch495_redo
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scorex0-redo-gen-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scorex0-redo-gen-%A_%a.err
set -euo pipefail

ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
TRANSCRIPT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/add/resources/filelists/ljspeech/test.txt
cd "$ROOT/scorex0"

values=(1.0 0.8 0.6 0.4 0.2)
value=${values[$SLURM_ARRAY_TASK_ID]}
run_dir="Hydra_scorex0/model.masking.a:$value"
checkpoint="$run_dir/checkpoint/grad_495.pt"
train_log="$run_dir/tb/train.log"
gen="scorex0_inf/model.masking.a:$value/test/converted/Epoch_495"

test -s "$checkpoint"
test -s "$train_log"
grep -q '^Epoch 599:' "$train_log"
repair_start=$(date +%s)

export SLURM_EXPORT_ENV=ALL
export TMPDIR="$ROOT/.symphony/results/tmp/scorex0-redo-gen"
export XDG_CACHE_HOME="$ROOT/.symphony/results/cache/scorex0-redo-gen"
export XDG_CONFIG_HOME="$ROOT/.symphony/results/config/scorex0-redo-gen"
export MPLCONFIGDIR="$XDG_CONFIG_HOME/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$MPLCONFIGDIR"
module purge
module load Anaconda3/2022.10
module load GCC/12.3.0
module load CUDA/12.4.0
source activate grad-tts-masking

# Execute the checked-in inference program in memory with only its checkpoint
# iteration narrowed to the exact epoch emitted by extract_metrics_paths.py.
# The repository source and the normal run.sh payload remain untouched.
HYDRA_FULL_ERROR=1 python -c '
from pathlib import Path

path = Path("inference.py")
source = path.read_text()
variants = [
    "        for i in range(ending_epoch - 1, ending_epoch - 21, -1):",
    "        for i in range(ending_epoch - 1, ending_epoch - 31, -1):",
    "        for i in range(ending_epoch - 1, ending_epoch - 41, -1):",
]
matches = [old for old in variants if source.count(old) == 1]
if len(matches) != 1:
    raise RuntimeError(
        "expected exactly one supported scorex0 inference loop; found {}".format(matches))
quote = chr(34)
replacement = (
    "        for i in [next(j for j, checkpoint_path in enumerate(checkpoint_files) "
    "if _get_basename(checkpoint_path) == " + quote + "grad_495" + quote + ")]:"
)
source = source.replace(matches[0], replacement)
namespace = {"__name__": "__main__", "__file__": str(path.resolve())}
exec(compile(source, str(path), "exec"), namespace)
' -m --config-name=config_eval_swp +data=data_swp "model.masking.a=$value"

/usr/bin/python3 - "$TRANSCRIPT" "$gen" "$repair_start" "$value" <<'PY'
import sys
from pathlib import Path

transcript = Path(sys.argv[1])
gen = Path(sys.argv[2])
repair_start = int(sys.argv[3])
value = sys.argv[4]
expected = {
    Path(line.split("|", 1)[0]).name
    for line in transcript.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
wavs = list(gen.glob("*.wav"))
observed = {wav.name for wav in wavs}
if len(expected) != 488 or observed != expected:
    raise RuntimeError(
        "scorex0 {} epoch-495 WAV coverage mismatch: expected={} observed={} "
        "missing={} unexpected={}".format(
            value, len(expected), len(observed), len(expected - observed),
            len(observed - expected)))
stale = [wav.name for wav in wavs if wav.stat().st_mtime < repair_start - 300]
if stale:
    raise RuntimeError(
        "scorex0 {} epoch-495 repair left {} stale WAVs".format(
            value, len(stale)))
print(
    "scorex0 {} fresh epoch-495 generation complete: wavs={}".format(
        value, len(wavs)),
    flush=True,
)
PY
