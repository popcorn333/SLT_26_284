#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --job-name=tan10_reconcile
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/cadence-reconcile-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/cadence-reconcile-%j.err
set -euo pipefail

date '+snapshot_time=%Y-%m-%dT%H:%M:%S%z'
/usr/bin/sacct -X -j 11097445,11099158,11099159 \
  --starttime=2026-07-31 \
  --format=JobID%30,JobIDRaw,JobName%32,State,ExitCode,Elapsed,Start,End,NodeList%24 \
  -n -P
/usr/bin/squeue -h -j 11097445,11099158,11099159 \
  -o 'queue|%i|%T|%r|%M|%S|%N' || true
