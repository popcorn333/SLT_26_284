#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=0-10:00:00
set -eo pipefail
module load Anaconda3/2025.06-1
source activate /users/acp23xt/.conda/envs/Grad-TTS-EVAL
gt=/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/ground_truth
gen=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scoreDT/scoreDT_inf/model.masking.a:0.2/test/converted/Epoch_595
out=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scoreDT/scoreDT_inf/model.masking.a:0.2/test/metrics

# TAN-10 per-WAV metric freshness gate
FRESHNESS_MARKER="/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/metrics-${SLURM_JOB_ID}.started"
touch "$FRESHNESS_MARKER"

python "tools/F0/F0.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"
python "tools/MCD/MCD.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"


/usr/bin/python3 - "$gen" "$FRESHNESS_MARKER" _f0.txt _mcd.txt <<'PY'
import math
import sys
from pathlib import Path

gen = Path(sys.argv[1])
marker = Path(sys.argv[2])
suffixes = sys.argv[3:]
wavs = sorted(gen.glob("*.wav"))
if len(wavs) != 488:
    raise RuntimeError("expected 488 WAVs; found {} in {}".format(len(wavs), gen))
for wav in wavs:
    for suffix in suffixes:
        sidecar = wav.with_name(wav.stem + suffix)
        if not sidecar.is_file() or sidecar.stat().st_size == 0:
            raise RuntimeError("missing or empty sidecar {}".format(sidecar))
        if sidecar.stat().st_mtime_ns < marker.stat().st_mtime_ns:
            raise RuntimeError("sidecar was not refreshed by this job: {}".format(sidecar))
        value = float(sidecar.read_text(encoding="utf-8").strip())
        if not math.isfinite(value):
            raise RuntimeError("non-finite sidecar {}".format(sidecar))
print("fresh MCD/logF0 sidecars validated: {}".format(len(wavs)), flush=True)
PY
