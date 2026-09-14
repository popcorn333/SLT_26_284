#!/usr/bin/env python3
"""TAN-18: fail closed unless the submitted fmDT array produced fully fresh outputs."""
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path('/mnt/parscratch/users/acp23xt/private/DTDM_Unet')
RES = ROOT / '.symphony' / 'results'
PATTERN = re.compile(
    r'^Epoch (\d+): duration loss = ([0-9.eE+-]+) '
    r'\| prior loss = ([0-9.eE+-]+) '
    r'\| diffusion loss = ([0-9.eE+-]+)$')
EXPECTED_EPOCHS = list(range(1, 600))
EXPECTED_CHECKPOINTS = {'grad_{}.pt'.format(epoch) for epoch in range(5, 600, 5)}

# The five array tasks can begin a few minutes apart, and this gate was created
# while the earliest task was already running.  Use each task's authoritative
# Slurm start time rather than this file's mtime so valid early checkpoints are
# not rejected while outputs predating the current task still fail closed.
run_array_id = os.environ.get('RUN_ARRAY_ID', '').strip()
if not re.fullmatch(r'[0-9]+', run_array_id):
    raise RuntimeError('RUN_ARRAY_ID is missing or invalid: {!r}'.format(run_array_id))
result = subprocess.run(
    ['sacct', '-X', '-n', '-P', '-j', run_array_id,
     '--format=JobID,Start'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
if result.returncode:
    raise RuntimeError('unable to read fmDT task start times: {}'.format(
        result.stderr.strip()))
task_start_ns = {}
for line in result.stdout.splitlines():
    fields = [field.strip() for field in line.split('|')]
    match = re.fullmatch(re.escape(run_array_id) + r'_([0-4])', fields[0]) \
        if len(fields) >= 2 else None
    if not match or not fields[1] or fields[1] == 'Unknown':
        continue
    # Slurm and repository mtimes use the cluster's local timezone.  A naive
    # datetime therefore converts against the same host timezone as stat().
    start = datetime.strptime(fields[1], '%Y-%m-%dT%H:%M:%S')
    task_start_ns[int(match.group(1))] = int(start.timestamp() * 1000000000)
if set(task_start_ns) != set(range(5)):
    raise RuntimeError('expected start times for fmDT tasks 0..4, observed {}'.format(
        sorted(task_start_ns)))

for task_index, value in enumerate(('0.2', '0.4', '0.6', '0.8', '1')):
    fresh_after_ns = task_start_ns[task_index]
    config = 'model.masking.a:' + value
    run_dir = ROOT / 'fmDT' / 'Hydra_fmDT' / config
    train_log = run_dir / 'tb' / 'train.log'
    if (not train_log.is_file()
            or train_log.stat().st_size == 0
            or train_log.stat().st_mtime_ns <= fresh_after_ns):
        raise RuntimeError('{} training log is missing, empty, or stale: {}'.format(
            config, train_log))
    matches = []
    for line in train_log.read_text(errors='replace').splitlines():
        match = PATTERN.match(line.strip())
        if match:
            matches.append(int(match.group(1)))
    if matches[-599:] != EXPECTED_EPOCHS:
        raise RuntimeError('{} final training records are not contiguous epochs 1..599'.format(config))

    checkpoint_dir = run_dir / 'checkpoint'
    checkpoints = {path.name: path for path in checkpoint_dir.glob('grad_*.pt')}
    if set(checkpoints) != EXPECTED_CHECKPOINTS:
        raise RuntimeError('{} checkpoint coverage mismatch: observed={}'.format(config, len(checkpoints)))
    stale_checkpoints = [path.name for path in checkpoints.values()
                         if path.stat().st_size == 0
                         or path.stat().st_mtime_ns <= fresh_after_ns]
    if stale_checkpoints:
        raise RuntimeError('{} stale/empty checkpoints: {}'.format(config, stale_checkpoints[:10]))

    gen = ROOT / 'fmDT' / 'fmDT_inf' / config / 'test' / 'converted' / 'Epoch_595'
    wavs = sorted(gen.glob('*.wav'))
    if len(wavs) != 488:
        raise RuntimeError('{} expected 488 WAVs, found {}'.format(config, len(wavs)))
    stale_wavs = [path.name for path in wavs
                  if path.stat().st_size == 0
                  or path.stat().st_mtime_ns <= fresh_after_ns]
    if stale_wavs:
        raise RuntimeError('{} stale/empty WAVs: {}'.format(config, stale_wavs[:10]))
    print('{} fresh: epochs=599 checkpoints=119 wavs=488'.format(config), flush=True)

print('fmDT acceptance gate passed for all five configurations', flush=True)
