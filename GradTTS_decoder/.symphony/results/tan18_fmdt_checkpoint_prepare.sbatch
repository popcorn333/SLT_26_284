#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --job-name=tan18_fmdt_checkpoint
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-checkpoint-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-checkpoint-%j.err
set -euo pipefail

ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
SOURCE="$ROOT/one-shot/checkpts/grad-tts-new_0.pt"
DESTINATION="$ROOT/fmDT/checkpts/grad-tts-new_0.pt"
EXPECTED=e83b05852ddf36c6249afcb2a02a62c6aecb67fd09b5bb2dde4ba9228d823b29

test -s "$SOURCE"
source_digest=$(sha256sum "$SOURCE" | awk '{print $1}')
test "$source_digest" = "$EXPECTED"

if test -e "$DESTINATION"; then
    test -f "$DESTINATION"
    test -s "$DESTINATION"
    destination_digest=$(sha256sum "$DESTINATION" | awk '{print $1}')
    test "$destination_digest" = "$EXPECTED"
    echo "fmDT base checkpoint already valid: $DESTINATION"
    exit 0
fi

temporary="$DESTINATION.tan18-${SLURM_JOB_ID}.tmp"
trap 'rm -f "$temporary"' EXIT
cp --reflink=auto --preserve=mode,timestamps "$SOURCE" "$temporary"
test "$(sha256sum "$temporary" | awk '{print $1}')" = "$EXPECTED"
mv "$temporary" "$DESTINATION"
trap - EXIT
echo "prepared fmDT base checkpoint: $DESTINATION sha256=$EXPECTED"
