#!/usr/bin/env python3
import csv, math, os, re, statistics
from pathlib import Path

ROOT = Path('/mnt/parscratch/users/acp23xt/private/DTDM_Unet')
RES = ROOT / '.symphony/results'
MAP = RES / 'tan18_fmdt_mappings.tsv'
JOBS = RES / 'tan18_fmdt_jobs.tsv'
CSV = RES / 'tan18_fmdt_summary.csv'
REPORT = RES / 'tan18_fmdt_report.md'
metrics = [('MCD', '_mcd.txt'), ('logF0/F0', '_f0.txt'), ('WER', '_wer_l.txt'), ('UTMOSv2', '_utmosv2.txt')]

def read_job_id(name):
    path = RES / name
    value = path.read_text(encoding='utf-8').strip()
    if not re.fullmatch(r'[0-9]+', value):
        raise RuntimeError('invalid job ID in {}: {!r}'.format(path, value))
    return value

def aggregate_line(path, gen):
    if not path.is_file():
        return 'missing aggregate file'
    matches = [
        line.strip() for line in path.read_text(errors='replace').splitlines()
        if line.split('   ', 1)[0].strip() == str(gen)
    ]
    if matches:
        # The established MCD/F0 tools append aggregate records.  The
        # dependency-held metric job is the newest producer, so report its
        # final matching record rather than a stale line from an older run.
        return matches[-1]
    return 'gen path absent from aggregate file'

with MAP.open(newline='') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
expected_sweeps = {
    'model.masking.a:' + value for value in ('0.2', '0.4', '0.6', '0.8', '1')
}
if len(rows) != 5 or {row['sweep'] for row in rows} != expected_sweeps:
    raise RuntimeError('fmDT mapping coverage mismatch: {}'.format(
        [row.get('sweep', '') for row in rows]))
with JOBS.open(newline='') as f:
    job_rows = list(csv.DictReader(f, delimiter='\t'))
job_ids = {row['stage']: row['job_id'] for row in job_rows}
environment_probe_job_id = read_job_id('tan18_env_probe_job_id.txt')
checkpoint_validation_job_id = read_job_id('tan18_checkpoint_job_id.txt')
run_job_id = job_ids.get('run_array', '')
if not re.fullmatch(r'[0-9]+', run_job_id):
    raise RuntimeError('missing or invalid active run-array job ID: {!r}'.format(
        run_job_id))
task_index_by_sweep = {
    'model.masking.a:' + value: index
    for index, value in enumerate(('0.2', '0.4', '0.6', '0.8', '1'))
}
output = []
for row in rows:
    gen = Path(row['gen'])
    wavs = sorted(gen.glob('*.wav'))
    record = dict(row)
    record['wav_count'] = len(wavs)
    missing_all = []
    for label, suffix in metrics:
        vals, missing = [], []
        for wav in wavs:
            sidecar = wav.with_name(wav.stem + suffix)
            try:
                value = float(sidecar.read_text().strip())
                if not math.isfinite(value):
                    raise ValueError('non-finite')
                vals.append(value)
            except Exception:
                missing.append(sidecar.name)
        key = {'MCD':'mcd','logF0/F0':'f0','WER':'wer','UTMOSv2':'utmos'}[label]
        record[key + '_count'] = len(vals)
        record[key + '_mean'] = (sum(vals) / len(vals)) if vals else ''
        record[key + '_std'] = statistics.pstdev(vals) if vals else ''
        record[key + '_missing'] = ';'.join(missing)
        missing_all.extend(missing)
    out = Path(row['out'])
    record['mcd_aggregate_format'] = aggregate_line(out / 'MCD.txt', gen)
    record['f0_aggregate_format'] = aggregate_line(out / 'F0.txt', gen)
    record['coverage_ok'] = (len(wavs) == 488 and
                             all(record[k + '_count'] == 488
                                 for k in ('mcd','f0','wer','utmos')))
    record['genuine_missing_values'] = ';'.join(missing_all)
    record['environment_probe_job_id'] = environment_probe_job_id
    record['checkpoint_validation_job_id'] = checkpoint_validation_job_id
    record['run_job_id'] = run_job_id
    record['run_task_job_id'] = '{}_{}'.format(
        run_job_id, task_index_by_sweep[row['sweep']])
    record['acceptance_gate_job_id'] = job_ids.get('acceptance_gate', '')
    record['training_loss_job_id'] = job_ids.get('training_loss', '')
    record['dispatch_job_id'] = job_ids.get('dispatch', '')
    record['summary_job_id'] = os.environ.get('SLURM_JOB_ID', job_ids.get('summary', ''))
    output.append(record)
fields = ['folder','sweep','gen','out','wav_count','mcd_count','mcd_mean','mcd_std','f0_count','f0_mean','f0_std','wer_count','wer_mean','wer_std','utmos_count','utmos_mean','utmos_std','coverage_ok','genuine_missing_values','environment_probe_job_id','checkpoint_validation_job_id','run_job_id','run_task_job_id','acceptance_gate_job_id','training_loss_job_id','dispatch_job_id','metric_job_id','wer_job_id','utmos_job_id','summary_job_id','mcd_aggregate_format','f0_aggregate_format']
with CSV.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k:r.get(k,'') for k in fields} for r in output)
run_task_ids = ', '.join(
    '`{}_{}`'.format(run_job_id, index) for index in range(5))
lines = ['# TAN-18 fmDT speech-metric report', '', 'Standard deviations are population standard deviations over per-WAV scalar sidecars.', '', 'The metric tools emitted aggregate lines in `path   mean \\pm std` format and per-WAV scalar `_mcd.txt`/`_f0.txt` sidecars; summaries below are recomputed from the sidecars.', '', 'Common jobs: environment probe `{}`, checkpoint validation `{}`, run array `{}`, acceptance gate `{}`, training-loss `{}`, dispatcher `{}`, summary `{}`.'.format(environment_probe_job_id, checkpoint_validation_job_id, run_job_id, job_ids.get('acceptance_gate',''), job_ids.get('training_loss',''), job_ids.get('dispatch',''), os.environ.get('SLURM_JOB_ID','')), '', 'Run task IDs: {}.'.format(run_task_ids), '', '| Sweep | WAVs | MCD n / mean / sd | logF0/F0 n / mean / sd | WER n / mean / sd | UTMOSv2 n / mean / sd | Jobs (run task / metric / WER / UTMOS) | Coverage |', '|---|---:|---:|---:|---:|---:|---|---|']
for r in output:
    fmt = lambda key: f"{r[key+'_count']} / {r[key+'_mean']:.6f} / {r[key+'_std']:.6f}" if r[key+'_count'] else '0 / NA / NA'
    lines.append(f"| `{r['sweep']}` | {r['wav_count']} | {fmt('mcd')} | {fmt('f0')} | {fmt('wer')} | {fmt('utmos')} | `{r['run_task_job_id']}` / `{r['metric_job_id']}` / `{r['wer_job_id']}` / `{r['utmos_job_id']}` | {'OK' if r['coverage_ok'] else 'MISSING'} |")
    lines.extend(['', f"- Gen: `{r['gen']}`", f"- Out: `{r['out']}`", f"- MCD aggregate: `{r['mcd_aggregate_format']}`", f"- F0 aggregate: `{r['f0_aggregate_format']}`", f"- Genuine missing values: `{r['genuine_missing_values'] or 'none'}`", ''])
REPORT.write_text('\n'.join(lines) + '\n')
if not output or not all(r['coverage_ok'] for r in output):
    raise SystemExit('coverage validation failed; see tan18_fmdt_summary.csv')
print(f'wrote {CSV} and {REPORT}; paths={len(output)}; all coverage OK')
