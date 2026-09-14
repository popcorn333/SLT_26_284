#!/usr/bin/env python3
"""Repair skipped log-F0 sidecars and prove genuinely undefined values."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("tan7_f0_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load F0 evaluator: {path}")
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
    undefined = 0
    for gen_path in gen_wavs:
        score_path = gen_path.with_name(gen_path.stem + "_f0.txt")
        marker_path = gen_path.with_name(gen_path.stem + "_f0_missing.json")
        marker_path.unlink(missing_ok=True)
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
            gt_audio = evaluator.librosa.resample(
                gt_audio, orig_sr=gt_rate, target_sr=gen_rate
            )

        feature_args = {
            "f0min": 40,
            "f0max": 800,
            "n_fft": 1024,
            "n_shift": 256,
            "mcep_dim": None,
            "mcep_alpha": None,
        }
        gen_mcep, gen_f0 = evaluator.world_extract(
            gen_audio, gen_rate, **feature_args
        )
        gt_mcep, gt_f0 = evaluator.world_extract(
            gt_audio, gen_rate, **feature_args
        )
        _, path = evaluator.fastdtw(
            gen_mcep, gt_mcep, dist=evaluator.spatial.distance.euclidean
        )
        time_warp = np.asarray(path).T
        gen_aligned = gen_f0[time_warp[0]]
        gt_aligned = gt_f0[time_warp[1]]
        jointly_voiced = np.flatnonzero((gen_aligned != 0) & (gt_aligned != 0))
        if jointly_voiced.size:
            error = np.log(gen_aligned[jointly_voiced]) - np.log(
                gt_aligned[jointly_voiced]
            )
            value = float(np.sqrt(np.mean(error ** 2)))
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite repaired log-F0 value for {gen_path}")
            score_path.write_text(f"{value:.4f}\n", encoding="utf-8")
            repaired += 1
        else:
            marker_path.write_text(
                json.dumps(
                    {
                        "reason": "no_joint_voiced_frames",
                        "joint_voiced_frames": 0,
                        "wav": str(gen_path),
                        "ground_truth": str(matches[0]),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            undefined += 1

    print(
        f"f0_wavs={len(gen_wavs)} repaired={repaired} genuinely_undefined={undefined}",
        flush=True,
    )


if __name__ == "__main__":
    main()
