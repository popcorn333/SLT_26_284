#!/usr/bin/env python3
"""Print metrics paths for every Python inference command in a Hydra run.sh."""

import argparse
import copy
from pathlib import Path
import re
import shlex

from extract_metrics_paths import (
    expand_sweep_overrides,
    extract_python_command,
    interpolate,
    join_shell_lines,
    override_key,
    override_value,
    parse_run_args,
    read_config_info,
)


def extract_inference_commands(run_text):
    """Return the script and arguments from all Python inference commands."""
    commands = []
    for line in join_shell_lines(run_text):
        if "python" not in line or "inference" not in line:
            continue
        try:
            commands.append(extract_python_command(line))
        except RuntimeError:
            # A line can contain both words in an assignment or comment-like text
            # without actually being an executable Python command.
            continue
    if not commands:
        raise RuntimeError(
            "cannot find a command containing both 'python' and 'inference' in run.sh"
        )
    return commands


def command_metrics_paths(experiment_dir, run_text, script, args):
    """Resolve all metrics paths implied by one Hydra inference command."""
    config_name, cli_job_name, overrides = parse_run_args(args)

    marker = re.search(
        r"(?m)^#\s*metrics-sweep-overrides:\s*(.+?)\s*$", run_text
    )
    if marker:
        declared = shlex.split(marker.group(1))
        declared_keys = {override_key(token) for token in declared}
        overrides = [
            token for token in overrides if override_key(token) not in declared_keys
        ]
        overrides.extend(declared)

    config_name = config_name or "config_eval"
    yaml_path = experiment_dir / "config" / f"{config_name}.yaml"
    if not yaml_path.exists():
        fallback = experiment_dir / "config" / "config_eval.yaml"
        if fallback.exists():
            yaml_path = fallback
        else:
            raise RuntimeError(
                f"cannot find config/{config_name}.yaml or config/config_eval.yaml"
            )

    base_info = read_config_info(yaml_path)
    results = []
    for one_override_set in expand_sweep_overrides(overrides):
        info = copy.deepcopy(base_info)
        for token in one_override_set:
            if override_key(token) == "basename" and "=" in token:
                info["basename"] = override_value(token)

        dirname_items = []
        for token in one_override_set:
            key = override_key(token)
            if key.startswith("hydra.") or key in info["exclude_keys"]:
                continue
            if "=" in token:
                dirname_items.append(
                    f"{key}{info['kv_sep']}{override_value(token)}"
                )
            else:
                dirname_items.append(token)
        override_dirname = info["item_sep"].join(dirname_items)

        if cli_job_name:
            job_name = cli_job_name
        elif info["hydra_job_name"]:
            job_name = interpolate(
                info["hydra_job_name"], info["basename"], "", override_dirname
            )
        else:
            job_name = Path(script).stem

        sweep_dir = interpolate(
            info["sweep_dir"], info["basename"], job_name, override_dirname
        )
        sweep_subdir = interpolate(
            info["sweep_subdir"], info["basename"], job_name, override_dirname
        )
        working_dir = Path(sweep_dir) / sweep_subdir
        if not Path(sweep_dir).is_absolute():
            working_dir = experiment_dir / working_dir
        working_dir = working_dir.resolve()
        results.append(
            (
                working_dir / "test" / info["cvt_dir"] / "Epoch_595",
                working_dir / "test" / "metrics",
            )
        )

    return yaml_path.resolve(), results


def all_metrics_paths(run_sh):
    """Resolve paths for every inference command in run_sh, in file order."""
    run_sh = run_sh.expanduser().resolve()
    if not run_sh.is_file():
        raise RuntimeError(f"run.sh does not exist: {run_sh}")

    experiment_dir = run_sh.parent
    run_text = run_sh.read_text(errors="ignore")
    results = []
    for script, args in extract_inference_commands(run_text):
        yaml_path, paths = command_metrics_paths(
            experiment_dir, run_text, script, args
        )
        results.append((script, yaml_path, paths))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_sh", type=Path, help="path to an experiment's run.sh")
    args = parser.parse_args()
    try:
        commands = all_metrics_paths(args.run_sh)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    first = True
    for script, yaml_path, paths in commands:
        for gen, out in paths:
            if not first:
                print()
            first = False
            print(f"script={script}")
            print(f"config={yaml_path}")
            print(f"gen={gen}")
            print(f"out={out}")


if __name__ == "__main__":
    main()
