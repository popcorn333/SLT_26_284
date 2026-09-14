#!/usr/bin/env python3
"""Merge and validate fresh TAN-18 fmDT epoch losses without touching peers."""

import csv
import math
import os
import re
from pathlib import Path


ROOT = Path('/mnt/parscratch/users/acp23xt/private/DTDM_Unet')
RESULTS = ROOT / '.symphony' / 'results'
OUTPUT = ROOT / 'training_loss.log'
CONFIGURATIONS = tuple('model.masking.a:' + value for value in (
    '0.2', '0.4', '0.6', '0.8', '1'))
EXPECTED_EPOCHS = list(range(1, 600))
FIELDS = [
    'experiment', 'sweep_or_configuration', 'epoch', 'loss',
    'loss_definition', 'duration_loss', 'prior_loss', 'diffusion_loss',
    'source_log',
]
LOSS_PATTERN = re.compile(
    r'^Epoch (\d+): duration loss = ([0-9eE+.-]+) \| '
    r'prior loss = ([0-9eE+.-]+) \| diffusion loss = ([0-9eE+.-]+)$',
    re.M,
)


def main():
    preserved = []
    if OUTPUT.is_file():
        with OUTPUT.open('r', newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            if reader.fieldnames != FIELDS:
                raise RuntimeError(
                    'unexpected training_loss.log header: {}'.format(
                        reader.fieldnames))
            preserved = [row for row in reader
                         if row.get('experiment') != 'fmDT']

    fresh = []
    validation = []
    for configuration in CONFIGURATIONS:
        log_path = (ROOT / 'fmDT' / 'Hydra_fmDT' / configuration
                    / 'tb' / 'train.log')
        if not log_path.is_file() or log_path.stat().st_size == 0:
            raise RuntimeError('missing or empty training log {}'.format(log_path))
        matches = LOSS_PATTERN.findall(
            log_path.read_text(encoding='utf-8', errors='replace'))
        final = matches[-len(EXPECTED_EPOCHS):]
        epochs = [int(match[0]) for match in final]
        if epochs != EXPECTED_EPOCHS:
            raise RuntimeError(
                '{} final loss records are not contiguous epochs 1..599'.format(
                    configuration))
        for epoch_text, duration_text, prior_text, diffusion_text in final:
            duration = float(duration_text)
            prior = float(prior_text)
            diffusion = float(diffusion_text)
            if not all(math.isfinite(value) for value in
                       (duration, prior, diffusion)):
                raise RuntimeError(
                    'non-finite loss for {} epoch {}'.format(
                        configuration, epoch_text))
            fresh.append({
                'experiment': 'fmDT',
                'sweep_or_configuration': configuration,
                'epoch': epoch_text,
                'loss': '{:.12g}'.format(duration + prior + diffusion),
                'loss_definition':
                    'duration_loss+prior_loss+diffusion_loss',
                'duration_loss': '{:.12g}'.format(duration),
                'prior_loss': '{:.12g}'.format(prior),
                'diffusion_loss': '{:.12g}'.format(diffusion),
                'source_log': str(log_path),
            })
        validation.append({
            'experiment': 'fmDT',
            'sweep_or_configuration': configuration,
            'expected_epochs': str(len(EXPECTED_EPOCHS)),
            'observed_final_epochs': str(len(final)),
            'earlier_records_ignored': str(len(matches) - len(final)),
            'source_log': str(log_path),
            'status': 'complete',
        })

    expected_records = len(CONFIGURATIONS) * len(EXPECTED_EPOCHS)
    if len(fresh) != expected_records:
        raise RuntimeError(
            'expected {} fresh fmDT records, observed {}'.format(
                expected_records, len(fresh)))

    job_id = os.environ.get('SLURM_JOB_ID', 'manual')
    temporary = OUTPUT.with_name(
        OUTPUT.name + '.tan18-{}.tmp'.format(job_id))
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter='\t')
        writer.writeheader()
        writer.writerows(preserved)
        writer.writerows(fresh)
    temporary.replace(OUTPUT)

    validation_path = RESULTS / 'tan18_fmdt_training_loss_validation.tsv'
    validation_fields = [
        'experiment', 'sweep_or_configuration', 'expected_epochs',
        'observed_final_epochs', 'earlier_records_ignored', 'source_log',
        'status',
    ]
    with validation_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=validation_fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(validation)
    (RESULTS / 'tan18_fmdt_training_loss_status.txt').write_text(
        'complete\nconfigurations={}\nepoch_records={}\n'
        'non_fmdt_records_preserved={}\n'.format(
            len(CONFIGURATIONS), len(fresh), len(preserved)),
        encoding='utf-8')
    print(
        'fmDT training loss validated: configurations={} epochs={} '
        'non_fmDT_rows_preserved={}'.format(
            len(CONFIGURATIONS), len(fresh), len(preserved)),
        flush=True)


if __name__ == '__main__':
    main()
