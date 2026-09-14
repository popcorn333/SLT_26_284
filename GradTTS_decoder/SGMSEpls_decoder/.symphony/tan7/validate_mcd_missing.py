#!/usr/bin/env python3
"""Repair missing or invalid per-WAV MCD sidecars after the established evaluator."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("tan7_mcd_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MCD evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--gen", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    args = parser.parse_args()

    evaluator = load_evaluator(args.evaluator.resolve())
    gen_wavs = sorted(args.gen.glob("*.wav"))
    gt_wavs = sorted(Path(path) for path in evaluator.find_files(str(args.gt)))
    gt_by_stem: dict[str, list[Path]] = {}
    for path in gt_wavs:
        gt_by_stem.setdefault(path.stem, []).append(path)

    repaired = 0
    for gen_path in gen_wavs:
        score_path = gen_path.with_name(gen_path.stem + "_mcd.txt")
        if score_path.is_file():
            try:
                tokens = score_path.read_text(encoding="utf-8").split()
                value = float(tokens[0]) if len(tokens) == 1 else math.nan
            except (OSError, ValueError, IndexError):
                value = math.nan
            if math.isfinite(value):
                continue
            score_path.unlink(missing_ok=True)

        matches = gt_by_stem.get(gen_path.stem, [])
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one ground-truth WAV for {gen_path.name}, found {len(matches)}"
            )
        gen_audio, gen_rate = evaluator.sf.read(gen_path, dtype="float64")
        gt_audio, gt_rate = evaluator.sf.read(matches[0], dtype="float64")
        if gen_rate != gt_rate:
            raise RuntimeError(
                f"sample-rate mismatch for {gen_path}: generated={gen_rate} gt={gt_rate}"
            )

        feature_args = {
            "n_fft": 1024,
            "n_shift": 256,
            "mcep_dim": None,
            "mcep_alpha": None,
        }
        gen_mcep = evaluator.sptk_extract(gen_audio, gen_rate, **feature_args)
        gt_mcep = evaluator.sptk_extract(gt_audio, gt_rate, **feature_args)
        _, path = evaluator.fastdtw(
            gen_mcep, gt_mcep, dist=evaluator.spatial.distance.euclidean
        )
        time_warp = np.asarray(path).T
        difference = gen_mcep[time_warp[0]] - gt_mcep[time_warp[1]]
        squared_sum = np.sum(difference ** 2, axis=1)
        value = float(
            np.mean(10.0 / np.log(10.0) * np.sqrt(2.0 * squared_sum))
        )
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite repaired MCD value for {gen_path}")
        score_path.write_text(f"{value:.4f}\n", encoding="utf-8")
        repaired += 1

    print(f"mcd_wavs={len(gen_wavs)} repaired={repaired}", flush=True)


if __name__ == "__main__":
    main()
