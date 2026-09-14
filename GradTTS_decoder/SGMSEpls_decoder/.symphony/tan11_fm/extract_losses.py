#!/usr/bin/env python3
"""Extract the five TAN-11 FM array training logs with exact epoch coverage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
STATE = ROOT / ".symphony/tan11_fm"
PARENT = "11097462"
CONFIG_TO_TASK = {
    "model.masking.a:0.2": 0,
    "model.masking.a:0.4": 1,
    "model.masking.a:0.6": 2,
    "model.masking.a:0.8": 3,
    "model.masking.a:1": 4,
}
LOSS_RE = re.compile(
    r"Epoch\s+(\d+):\s+duration loss\s*=\s*([-+0-9.eE]+)\s*"
    r"\|\s*prior loss\s*=\s*([-+0-9.eE]+)\s*"
    r"\|\s*diffusion loss\s*=\s*([-+0-9.eE]+)"
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-jobs", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    args = parser.parse_args()

    jobs = read_tsv(args.run_jobs)
    if len(jobs) != 1 or jobs[0].get("training_job_id") != PARENT:
        raise RuntimeError(f"unexpected TAN-11 run manifest: {jobs}")
    paths = read_tsv(args.paths)
    configurations = {row["configuration"] for row in paths}
    if configurations != set(CONFIG_TO_TASK) or len(paths) != 5:
        raise RuntimeError(f"unexpected FM path coverage: {configurations}")

    rows = []
    coverage = []
    errors = []
    for configuration, task in sorted(CONFIG_TO_TASK.items()):
        source = STATE / "logs" / f"run-{PARENT}_{task}.out"
        seen: dict[int, tuple[float, float, float]] = {}
        for raw in source.read_text(errors="replace").splitlines():
            match = LOSS_RE.search(raw)
            if not match:
                continue
            epoch = int(match.group(1))
            values = tuple(float(match.group(index)) for index in range(2, 5))
            if epoch in seen:
                errors.append(f"{configuration}: duplicate epoch {epoch}")
            elif not all(math.isfinite(value) for value in values):
                errors.append(f"{configuration}: nonfinite epoch {epoch}")
            else:
                seen[epoch] = values
        expected = set(range(1, 600))
        missing = sorted(expected - set(seen))
        extra = sorted(set(seen) - expected)
        if missing or extra:
            errors.append(
                f"{configuration}: actual={len(seen)} missing={missing[:20]} extra={extra[:20]}"
            )
        for epoch, values in sorted(seen.items()):
            duration, prior, diffusion = values
            rows.append(
                {
                    "experiment": "fm",
                    "configuration": configuration,
                    "epoch": epoch,
                    "duration_loss": repr(duration),
                    "prior_loss": repr(prior),
                    "diffusion_loss": repr(diffusion),
                    "loss": repr(duration + prior + diffusion),
                    "run_job_id": PARENT,
                    "source": str(source),
                }
            )
        coverage.append(
            {
                "experiment": "fm",
                "configuration": configuration,
                "config_path": str(
                    ROOT / "fm" / "Hydra_fm" / configuration / ".hydra/config.yaml"
                ),
                "first_epoch": 1,
                "last_epoch": 599,
                "expected_count": 599,
                "actual_count": len(seen),
                "missing_epochs": missing,
                "extra_epochs": extra,
                "array_task": task,
                "source": str(source),
            }
        )

    coverage_tmp = args.coverage.with_name(args.coverage.name + ".tmp")
    coverage_tmp.write_text(
        json.dumps({"records": len(rows), "coverage": coverage, "errors": errors}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    coverage_tmp.replace(args.coverage)
    if errors or len(rows) != 5 * 599:
        raise RuntimeError("training loss validation failed: " + "; ".join(errors[:10]))
    fields = [
        "experiment", "configuration", "epoch", "duration_loss", "prior_loss",
        "diffusion_loss", "loss", "run_job_id", "source",
    ]
    output_tmp = args.output.with_name(args.output.name + ".tmp")
    with output_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    output_tmp.replace(args.output)
    print(f"validated_training_loss_records={len(rows)} configurations=5")


if __name__ == "__main__":
    main()
