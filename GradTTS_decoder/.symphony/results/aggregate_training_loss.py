#!/usr/bin/env python3
"""Aggregate and validate every first-depth DTDM_Unet training loss."""

import csv
import math
import re
import subprocess
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet")
RES = ROOT / ".symphony/results"
FOLDERS = [
    "add", "blur", "fm", "fmDT", "levy", "mask", "one-shot",
    "product", "score", "scoreDT", "scorex0",
]
LOSS_PATTERN = re.compile(
    r"^Epoch (\d+): duration loss = ([0-9eE+.-]+) \| "
    r"prior loss = ([0-9eE+.-]+) \| diffusion loss = ([0-9eE+.-]+)$",
    re.M,
)


def emitted_pair_count(folder):
    output = subprocess.check_output([
        "/usr/bin/python3", str(ROOT / "extract_metrics_paths.py"),
        str(ROOT / folder / "run.sh"),
    ], universal_newlines=True)
    gen_count = sum(line.startswith("gen=") for line in output.splitlines())
    out_count = sum(line.startswith("out=") for line in output.splitlines())
    if not gen_count or gen_count != out_count:
        raise RuntimeError("invalid emitted pair count for {}".format(folder))
    return gen_count


def main():
    actual = sorted(
        path.parent.name for path in ROOT.glob("*/run.sh")
        if path.parent.parent == ROOT
    )
    if actual != sorted(FOLDERS):
        raise RuntimeError(
            "first-depth run.sh inventory changed: expected={} actual={}".format(
                sorted(FOLDERS), actual))

    rows = []
    validation = []
    for experiment in FOLDERS:
        expected_configurations = emitted_pair_count(experiment)
        config_paths = sorted(
            (ROOT / experiment).glob("Hydra_*/*/.hydra/config.yaml"))
        if len(config_paths) != expected_configurations:
            raise RuntimeError(
                "{} configurations={} emitted_pairs={}".format(
                    experiment, len(config_paths), expected_configurations))

        for config_path in config_paths:
            run_dir = config_path.parent.parent
            configuration = run_dir.name
            log_path = run_dir / "tb/train.log"
            if not log_path.is_file():
                raise RuntimeError("missing training log {}".format(log_path))

            config_text = config_path.read_text(encoding="utf-8", errors="replace")
            epoch_match = re.search(
                r"(?m)^\s*n_epochs:\s*(\d+)\s*$", config_text)
            if not epoch_match:
                raise RuntimeError("missing n_epochs in {}".format(config_path))
            expected_epochs = int(epoch_match.group(1)) - 1

            matches = LOSS_PATTERN.findall(
                log_path.read_text(encoding="utf-8", errors="replace"))
            epoch_values = {}
            for epoch, duration, prior, diffusion in matches:
                values = (float(duration), float(prior), float(diffusion))
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError(
                        "non-finite loss for {} {} epoch {}".format(
                            experiment, configuration, epoch))
                # Training retries append repeated epochs.  The final record
                # is the authoritative value from the most recent phase.
                epoch_values[int(epoch)] = values

            expected_set = set(range(1, expected_epochs + 1))
            actual_set = set(epoch_values)
            if actual_set != expected_set:
                raise RuntimeError(
                    "{} {} epoch coverage missing={} extra={}".format(
                        experiment, configuration,
                        sorted(expected_set - actual_set)[:20],
                        sorted(actual_set - expected_set)[:20]))

            validation.append([
                experiment, configuration, str(expected_epochs),
                str(len(epoch_values)), str(len(matches) - len(epoch_values)),
                str(log_path), "complete",
            ])
            for epoch in range(1, expected_epochs + 1):
                duration, prior, diffusion = epoch_values[epoch]
                rows.append([
                    experiment, configuration, str(epoch),
                    "{:.12g}".format(duration + prior + diffusion),
                    "duration_loss+prior_loss+diffusion_loss",
                    "{:.12g}".format(duration), "{:.12g}".format(prior),
                    "{:.12g}".format(diffusion), str(log_path),
                ])

    if len(validation) != 38 or len(rows) != 22762:
        raise RuntimeError(
            "unexpected aggregate size configurations={} records={}".format(
                len(validation), len(rows)))

    temp_path = ROOT / "training_loss.log.tmp"
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "experiment", "sweep_or_configuration", "epoch", "loss",
            "loss_definition", "duration_loss", "prior_loss",
            "diffusion_loss", "source_log",
        ])
        writer.writerows(rows)
    temp_path.replace(ROOT / "training_loss.log")

    with (RES / "training_loss_validation.tsv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "experiment", "sweep_or_configuration", "expected_epochs",
            "observed_unique_epochs", "duplicate_records_ignored",
            "source_log", "status",
        ])
        writer.writerows(validation)
    (RES / "training_loss_status.txt").write_text(
        "complete\nconfigurations=38\nepoch_records=22762\n",
        encoding="utf-8")
    print(
        "training loss complete configurations={} epoch_records={}".format(
            len(validation), len(rows)), flush=True)


if __name__ == "__main__":
    main()
