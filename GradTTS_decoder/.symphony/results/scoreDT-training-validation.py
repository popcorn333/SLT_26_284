#!/usr/bin/env python3
import csv
import re
from pathlib import Path

ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet")
RESULTS = ROOT / ".symphony/results"
EXPECTED_FOLDERS = {
    "add", "blur", "fm", "fmDT", "levy", "mask", "one-shot",
    "product", "score", "scoreDT", "scorex0",
}
VALUES = ("1.0", "0.8", "0.6", "0.4", "0.2")
LOSS_PATTERN = re.compile(
    r"^Epoch (\d+): duration loss = ([0-9eE+.-]+) \| "
    r"prior loss = ([0-9eE+.-]+) \| diffusion loss = ([0-9eE+.-]+)$",
    re.M,
)


def main():
    actual = {
        path.parent.name for path in ROOT.glob("*/run.sh")
        if path.parent.parent == ROOT
    }
    if actual != EXPECTED_FOLDERS:
        raise RuntimeError(
            "first-depth run.sh inventory changed: expected={} actual={}".format(
                sorted(EXPECTED_FOLDERS), sorted(actual)))

    expected_epochs = set(range(1, 600))
    rows = []
    for value in VALUES:
        run_dir = ROOT / "scoreDT/Hydra_scoreDT/model.masking.a:{}".format(value)
        log_path = run_dir / "tb/train.log"
        if not log_path.is_file():
            raise RuntimeError("missing scoreDT training log {}".format(log_path))
        matches = LOSS_PATTERN.findall(
            log_path.read_text(encoding="utf-8", errors="replace"))
        observed_epochs = {int(match[0]) for match in matches}
        if observed_epochs != expected_epochs:
            raise RuntimeError(
                "scoreDT {} epoch coverage mismatch: observed={} missing={} extra={}".format(
                    value, len(observed_epochs),
                    sorted(expected_epochs - observed_epochs)[:20],
                    sorted(observed_epochs - expected_epochs)[:20]))
        checkpoint = run_dir / "checkpoint/grad_495.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError("missing nonempty checkpoint {}".format(checkpoint))
        rows.append([
            "scoreDT", "model.masking.a:{}".format(value), "599",
            str(len(observed_epochs)), str(len(matches) - len(observed_epochs)),
            str(log_path), str(checkpoint), "complete",
        ])

    output = RESULTS / "scoreDT_training_validation.tsv"
    temporary = output.with_suffix(".tsv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "experiment", "sweep_or_configuration", "expected_epochs",
            "observed_unique_epochs", "duplicate_records_ignored",
            "source_log", "epoch495_checkpoint", "status",
        ])
        writer.writerows(rows)
    temporary.replace(output)
    print("scoreDT training validation complete: configurations=5 epochs=2995")


if __name__ == "__main__":
    main()

