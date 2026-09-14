#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --job-name=tan6_cadence_reconcile
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/cadence-reconcile-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/cadence-reconcile-%j.err
set -euo pipefail

date --iso-8601=seconds
/usr/bin/squeue -j 10935482,10943691,10988661,10944226 \
  -o '%.24i|%.16T|%.12M|%.24R|%.32E|%.20S'
/usr/bin/squeue --start -j 10935482,10943691,10988661,10944226 \
  -o '%.24i|%.16T|%.12M|%.24R|%.32E|%.20S'
/usr/bin/sacct -X -j 10935482,10943691,10988661,10944226 \
  --starttime=2026-07-01 \
  --format=JobID,JobIDRaw,State,ExitCode,Elapsed,Start,End,NodeList -n -P
