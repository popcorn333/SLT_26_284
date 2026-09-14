#!/usr/bin/env python3
"""Extract and validate per-epoch TAN-7 training losses from fresh Slurm logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_sgmse")
MARKER_RE = re.compile(r"#\d+\s*:\s*(.*)$")
LOSS_RE = re.compile(
    r"Epoch\s+(\d+):\s+duration loss\s*=\s*([-+0-9.eE]+)\s*"
    r"\|\s*prior loss\s*=\s*([-+0-9.eE]+)\s*"
    r"\|\s*diffusion loss\s*=\s*([-+0-9.eE]+)"
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def training_config(folder: str, configuration: str) -> tuple[Path, int, int]:
    candidates = sorted(
        (ROOT / folder).glob(f"Hydra_*/{configuration}/.hydra/config.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"missing Hydra training config for {folder}/{configuration}")
    path = candidates[0]
    values: dict[str, str] = {}
    in_training = False
    training_indent = -1
    for raw in path.read_text(errors="replace").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if stripped == "training:":
            in_training = True
            training_indent = indent
            continue
        if in_training and indent <= training_indent:
            in_training = False
        if in_training and ":" in stripped:
            key, value = stripped.split(":", 1)
            if key in {"n_epochs", "pre_checkpoint"}:
                values[key] = value.strip().strip("'\"")
    if "n_epochs" not in values or "pre_checkpoint" not in values:
        raise RuntimeError(f"incomplete training settings in {path}")
    checkpoint_match = re.search(r"_(\d+)\.pt$", values["pre_checkpoint"])
    if not checkpoint_match:
        raise RuntimeError(f"cannot infer starting epoch from {values['pre_checkpoint']}")
    return path, int(checkpoint_match.group(1)) + 1, int(values["n_epochs"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-jobs", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "training_loss.log")
    parser.add_argument("--coverage", type=Path, required=True)
    args = parser.parse_args()

    jobs = read_tsv(args.run_jobs)
    paths = read_tsv(args.paths)
    expected_by_folder: dict[str, list[str]] = defaultdict(list)
    for row in paths:
        expected_by_folder[row["folder"]].append(row["configuration"])
    expected_by_folder = {
        folder: sorted(set(configurations))
        for folder, configurations in expected_by_folder.items()
    }

    records: dict[tuple[str, str, int], tuple[float, float, float, str, str]] = {}
    errors: list[str] = []
    for job in jobs:
        folder = job["folder"]
        configurations = expected_by_folder.get(folder, [])
        # Hydra emits ``#N : override=value`` markers for multiruns, but the
        # single-configuration run.sh files log epochs directly with no sweep
        # marker.  Their folder has exactly one independently manifested
        # configuration, so it is safe and necessary to select it up front.
        current: str | None = configurations[0] if len(configurations) == 1 else None
        # Inference-only recovery manifests retain the effective run's log in
        # ``stdout`` and identify the allocation that actually emitted epoch
        # losses separately.  Older and ordinary manifests need no new field.
        output_path = Path(job.get("training_stdout") or job["stdout"])
        if not output_path.is_file():
            errors.append(f"missing training stdout: {output_path}")
            continue
        with output_path.open(errors="replace") as handle:
            for raw in handle:
                marker = MARKER_RE.search(raw)
                if marker:
                    marker_text = marker.group(1)
                    matches = [
                        configuration
                        for configuration in configurations
                        if configuration.replace(":", "=", 1) in marker_text
                    ]
                    current = matches[0] if len(matches) == 1 else None
                match = LOSS_RE.search(raw)
                if not match:
                    continue
                if current is None:
                    errors.append(f"unmapped epoch loss in {output_path}: {raw.strip()[:200]}")
                    continue
                epoch = int(match.group(1))
                components = tuple(float(match.group(i)) for i in range(2, 5))
                if not all(math.isfinite(value) for value in components):
                    errors.append(f"non-finite loss for {folder}/{current} epoch {epoch}")
                    continue
                key = (folder, current, epoch)
                if key in records:
                    errors.append(f"duplicate loss record for {folder}/{current} epoch {epoch}")
                    continue
                # An inference-only recovery can have a different effective
                # job ID from the allocation that emitted the training loss.
                # Label each epoch with its actual training allocation when
                # the effective manifest provides that provenance field.
                training_job_id = job.get("training_job_id") or job["job_id"]
                records[key] = (*components, training_job_id, str(output_path))

    coverage: list[dict[str, object]] = []
    for folder, configurations in sorted(expected_by_folder.items()):
        for configuration in configurations:
            try:
                config_path, first_epoch, n_epochs = training_config(folder, configuration)
            except RuntimeError as error:
                errors.append(str(error))
                continue
            actual = sorted(
                epoch
                for record_folder, record_config, epoch in records
                if record_folder == folder and record_config == configuration
            )
            expected = list(range(first_epoch, n_epochs))
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            if missing or extra:
                errors.append(
                    f"loss coverage mismatch for {folder}/{configuration}: "
                    f"expected={len(expected)} actual={len(actual)} "
                    f"missing={missing[:20]} extra={extra[:20]}"
                )
            coverage.append(
                {
                    "experiment": folder,
                    "configuration": configuration,
                    "config_path": str(config_path),
                    "first_epoch": first_epoch,
                    "last_epoch": n_epochs - 1,
                    "expected_count": len(expected),
                    "actual_count": len(actual),
                    "missing_epochs": missing,
                    "extra_epochs": extra,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_temporary = args.output.with_name(args.output.name + ".tmp")
    with output_temporary.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "experiment",
            "configuration",
            "epoch",
            "duration_loss",
            "prior_loss",
            "diffusion_loss",
            "loss",
            "run_job_id",
            "source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for (folder, configuration, epoch), values in sorted(records.items()):
            duration, prior, diffusion, job_id, source = values
            writer.writerow(
                {
                    "experiment": folder,
                    "configuration": configuration,
                    "epoch": epoch,
                    # Preserve round-trip float precision.  The finalizer
                    # independently checks that the combined loss equals the
                    # sum of these serialized components; shortening each
                    # value to nine significant digits can manufacture a
                    # mismatch even when the source log is internally exact.
                    "duration_loss": repr(duration),
                    "prior_loss": repr(prior),
                    "diffusion_loss": repr(diffusion),
                    "loss": repr(duration + prior + diffusion),
                    "run_job_id": job_id,
                    "source": source,
                }
            )
    coverage_temporary = args.coverage.with_name(args.coverage.name + ".tmp")
    coverage_temporary.write_text(
        json.dumps({"records": len(records), "coverage": coverage, "errors": errors}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    coverage_temporary.replace(args.coverage)
    if errors:
        output_temporary.unlink(missing_ok=True)
        raise SystemExit("training loss validation failed: " + "; ".join(errors[:10]))
    output_temporary.replace(args.output)
    print(f"validated_training_loss_records={len(records)} configurations={len(coverage)}")


if __name__ == "__main__":
    main()
