# -------------------------------------------------------------------
# Updated path logic
#
# This script is now assumed to live in a directory that is at the same
# level as the experiment subdirectories.
#
# Example:
#   DTDM_sgmse_stanage/
#   ├── add/
#   ├── blur/
#   ├── levy/
#   └── scripts/
#       └── submit_metrics.py
#
# Therefore:
#   SCRIPT_DIR = DTDM_sgmse_stanage/scripts
#   ROOT       = DTDM_sgmse_stanage
'''
Args: no args
Usage: This script scans all subdirectories in the same directory as itself, looking for run.sh and metrics.sh files. For each run.sh, it extracts the Hydra config information and generates the corresponding gen and out paths for metrics.sh. It then modifies metrics.sh to set the gt, gen, and out variables, and submits the job using sbatch.
'''

# -------------------------------------------------------------------

from pathlib import Path
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from itertools import product
import copy

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# Do not scan the script/helper directory itself as an experiment folder.
EXCLUDE_DIRS = {SCRIPT_DIR.resolve()}

GT = "/mnt/parscratch/users/acp23xt/private/DTDM_Unet_stanage/ground_truth"


def join_shell_lines(text):
    lines = []
    buf = ""

    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue

        if s.endswith("\\"):
            buf += s[:-1] + " "
        else:
            lines.append(buf + s)
            buf = ""

    if buf:
        lines.append(buf)

    return lines


def tokenize_python_line(line):
    toks = shlex.split(line)

    # Handle:
    # bash -c 'exec -a add python inference.py ...'
    for tok in toks:
        if "python" in tok and ".py" in tok and " " in tok:
            return shlex.split(tok)

    return toks


def extract_python_command(run_text):
    lines = join_shell_lines(run_text)

    candidates = [
        line for line in lines
        if "python" in line and ".py" in line
    ]

    if not candidates:
        raise RuntimeError("cannot find python command in run.sh")

    toks = tokenize_python_line(candidates[-1])

    py_i = None
    for i, tok in enumerate(toks):
        if re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", Path(tok).name):
            py_i = i
            break

    if py_i is None:
        raise RuntimeError("cannot locate python token in run.sh")

    script_i = None
    for i in range(py_i + 1, len(toks)):
        if toks[i].endswith(".py"):
            script_i = i
            break

    if script_i is None:
        raise RuntimeError("cannot locate Python script token in run.sh")

    return toks[script_i], toks[script_i + 1:]


def parse_run_args(args):
    config_name = None
    task_overrides = []
    cli_job_name = None

    i = 0
    while i < len(args):
        tok = args[i]

        if tok in ("-m", "--multirun"):
            i += 1
            continue

        if tok == "--config-name":
            config_name = args[i + 1] if i + 1 < len(args) else None
            i += 2
            continue

        if tok.startswith("--config-name="):
            config_name = tok.split("=", 1)[1]
            i += 1
            continue

        if tok in ("--config-path", "--config-dir"):
            i += 2
            continue

        if tok.startswith("--config-path=") or tok.startswith("--config-dir="):
            i += 1
            continue

        if tok.startswith("-"):
            i += 1
            continue

        if "=" in tok or tok.startswith("~"):
            key = tok.split("=", 1)[0].lstrip("+~")

            if key == "hydra.job.name":
                cli_job_name = tok.split("=", 1)[1]
            else:
                task_overrides.append(tok)

        i += 1

    return config_name, cli_job_name, task_overrides


def strip_comment(line):
    return re.sub(r"\s+#.*$", "", line).rstrip()


def clean_value(v):
    v = v.strip()
    if len(v) >= 2 and v[0] in ("'", '"') and v[-1] == v[0]:
        v = v[1:-1]
    return v


def read_config_info(yaml_path):
    lines = yaml_path.read_text(errors="ignore").splitlines()

    basename = None
    hydra_job_name = None
    sweep_dir = None
    sweep_subdir = None

    kv_sep = "="
    item_sep = ","
    exclude_keys = []

    in_hydra = False
    in_job = False
    in_sweep = False
    in_config = False
    in_override = False
    in_exclude = False

    hydra_indent = job_indent = sweep_indent = config_indent = override_indent = exclude_indent = None

    for raw in lines:
        no_comment = strip_comment(raw)
        if not no_comment.strip():
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        s = no_comment.strip()

        # IMPORTANT: read basename for this specific subdir/config
        if indent == 0:
            m = re.match(r"^basename\s*:\s*(.+)$", s)
            if m:
                basename = clean_value(m.group(1))

        if re.fullmatch(r"hydra\s*:", s):
            in_hydra = True
            in_job = in_sweep = in_config = in_override = in_exclude = False
            hydra_indent = indent
            continue

        if not in_hydra:
            continue

        if indent <= hydra_indent and not re.fullmatch(r"hydra\s*:", s):
            in_hydra = False
            in_job = in_sweep = in_config = in_override = in_exclude = False
            continue

        if re.fullmatch(r"job\s*:", s):
            in_job = True
            in_sweep = False
            in_config = in_override = in_exclude = False
            job_indent = indent
            continue

        if re.fullmatch(r"sweep\s*:", s):
            in_sweep = True
            in_job = False
            in_config = in_override = in_exclude = False
            sweep_indent = indent
            continue

        if in_job:
            if indent <= job_indent:
                in_job = False
                in_config = in_override = in_exclude = False
            else:
                m = re.match(r"^name\s*:\s*(.+)$", s)
                if m and not in_config and not in_override:
                    hydra_job_name = clean_value(m.group(1))
                    continue

                if re.fullmatch(r"config\s*:", s):
                    in_config = True
                    in_override = in_exclude = False
                    config_indent = indent
                    continue

        if in_config:
            if indent <= config_indent:
                in_config = False
                in_override = in_exclude = False
            elif re.fullmatch(r"override_dirname\s*:", s):
                in_override = True
                in_exclude = False
                override_indent = indent
                continue

        if in_override:
            if indent <= override_indent:
                in_override = False
                in_exclude = False
            else:
                m = re.match(r"^kv_sep\s*:\s*(.+)$", s)
                if m:
                    kv_sep = clean_value(m.group(1))
                    continue

                m = re.match(r"^item_sep\s*:\s*(.+)$", s)
                if m:
                    item_sep = clean_value(m.group(1))
                    continue

                if re.fullmatch(r"exclude_keys\s*:", s):
                    in_exclude = True
                    exclude_indent = indent
                    continue

        if in_exclude:
            if indent <= exclude_indent:
                in_exclude = False
            else:
                m = re.match(r"^-\s*(.+)$", s)
                if m:
                    exclude_keys.append(clean_value(m.group(1)))
                    continue

        if in_sweep:
            if indent <= sweep_indent:
                in_sweep = False
            else:
                m = re.match(r"^(dir|subdir)\s*:\s*(.+)$", s)
                if m:
                    key = m.group(1)
                    val = clean_value(m.group(2))
                    if key == "dir":
                        sweep_dir = val
                    elif key == "subdir":
                        sweep_subdir = val

    if basename is None:
        raise RuntimeError(f"cannot find top-level basename in {yaml_path}")

    if sweep_dir is None:
        raise RuntimeError(f"cannot find hydra.sweep.dir in {yaml_path}")

    if sweep_subdir is None:
        raise RuntimeError(f"cannot find hydra.sweep.subdir in {yaml_path}")

    return {
        "basename": basename,
        "hydra_job_name": hydra_job_name,
        "sweep_dir": sweep_dir,
        "sweep_subdir": sweep_subdir,
        "kv_sep": kv_sep,
        "item_sep": item_sep,
        "exclude_keys": exclude_keys,
    }


def override_key(tok):
    return tok.split("=", 1)[0].lstrip("+~")


def override_value(tok):
    return tok.split("=", 1)[1] if "=" in tok else ""


def split_top_level_commas(value):
    """
    Split Hydra sweep values like:
        0.2,0.4,0.6

    but avoid splitting commas inside:
        [a,b]
        {a:b,c:d}
        func(a,b)
        quoted strings
    """
    parts = []
    buf = []

    depth_square = 0
    depth_curly = 0
    depth_paren = 0
    quote = None
    escape = False

    for ch in value:
        if escape:
            buf.append(ch)
            escape = False
            continue

        if ch == "\\":
            buf.append(ch)
            escape = True
            continue

        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            buf.append(ch)
            quote = ch
            continue

        if ch == "[":
            depth_square += 1
            buf.append(ch)
            continue

        if ch == "]":
            depth_square = max(0, depth_square - 1)
            buf.append(ch)
            continue

        if ch == "{":
            depth_curly += 1
            buf.append(ch)
            continue

        if ch == "}":
            depth_curly = max(0, depth_curly - 1)
            buf.append(ch)
            continue

        if ch == "(":
            depth_paren += 1
            buf.append(ch)
            continue

        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            buf.append(ch)
            continue

        if (
            ch == ","
            and depth_square == 0
            and depth_curly == 0
            and depth_paren == 0
        ):
            parts.append("".join(buf).strip())
            buf = []
            continue

        buf.append(ch)

    parts.append("".join(buf).strip())

    return parts


def expand_sweep_overrides(task_overrides):
    """
    Convert Hydra comma sweeps into separate override lists.

    Example:
        ["+data=data_swp", "model.masking.a=0.2,0.4,0.6"]

    becomes:
        ["+data=data_swp", "model.masking.a=0.2"]
        ["+data=data_swp", "model.masking.a=0.4"]
        ["+data=data_swp", "model.masking.a=0.6"]

    If there are multiple sweep overrides, it creates the Cartesian product,
    matching Hydra multirun behavior.
    """
    choices_per_override = []

    for tok in task_overrides:
        if "=" not in tok:
            choices_per_override.append([tok])
            continue

        key_part, value_part = tok.split("=", 1)
        values = split_top_level_commas(value_part)

        if len(values) <= 1:
            choices_per_override.append([tok])
        else:
            choices_per_override.append([
                f"{key_part}={v}" for v in values
            ])

    expanded = []
    for combo in product(*choices_per_override):
        expanded.append(list(combo))

    if not expanded:
        expanded = [[]]

    return expanded


def apply_basename_override(info, task_overrides):
    for tok in task_overrides:
        if override_key(tok) == "basename" and "=" in tok:
            info["basename"] = override_value(tok)
    return info


def build_override_dirname(task_overrides, kv_sep, item_sep, exclude_keys):
    items = []

    for tok in task_overrides:
        key = override_key(tok)

        if key.startswith("hydra."):
            continue

        # Example: +data=data_swp -> key=data, excluded
        if key in exclude_keys:
            continue

        if "=" in tok:
            items.append(f"{key}{kv_sep}{override_value(tok)}")
        else:
            items.append(tok)

    return item_sep.join(items)


def resolve_interpolation(s, basename, job_name, override_dirname):
    s = s.replace("${basename}", basename)
    s = s.replace("${hydra.job.name}", job_name)
    s = s.replace("${hydra:job.name}", job_name)
    s = s.replace("${hydra.job.override_dirname}", override_dirname)
    s = s.replace("${hydra:job.override_dirname}", override_dirname)
    return s


def replace_var(text, name, value):
    new_line = f"{name}={value}"
    pat = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=.*$", re.M)

    if pat.search(text):
        return pat.sub(new_line, text, count=1)

    return new_line + "\n" + text


metrics_paths = []
gen_out_pairs = []

print(f"script_dir={SCRIPT_DIR}")
print(f"search_root={ROOT}")

for subdir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    if subdir.resolve() in EXCLUDE_DIRS:
        print(f"[{subdir.name}] skip: script/helper directory")
        continue

    run_sh = subdir / "run.sh"
    metrics_sh = subdir / "metrics.sh"

    if not run_sh.exists():
        print(f"[{subdir.name}] skip: no run.sh")
        continue

    if not metrics_sh.exists():
        print(f"[{subdir.name}] skip: no metrics.sh")
        continue

    try:
        script, args = extract_python_command(run_sh.read_text(errors="ignore"))
        config_name, cli_job_name, task_overrides = parse_run_args(args)

        if config_name is None:
            config_name = "config_eval"

        yaml_file = subdir / "config" / f"{config_name}.yaml"

        if not yaml_file.exists():
            fallback = subdir / "config" / "config_eval.yaml"
            if fallback.exists():
                yaml_file = fallback
            else:
                raise RuntimeError(f"cannot find config/{config_name}.yaml or config/config_eval.yaml")

        base_info = read_config_info(yaml_file)
        expanded_task_overrides = expand_sweep_overrides(task_overrides)

    except Exception as e:
        print(f"[{subdir.name}] skip: {e}")
        continue

    backup = metrics_sh.with_name(
        f"metrics.sh.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(metrics_sh, backup)
    original_metrics_text = metrics_sh.read_text(errors="ignore")

    print(f"[{subdir.name}]")
    print(f"  config={yaml_file}")
    print(f"  number_of_metric_jobs={len(expanded_task_overrides)}")
    print(f"  backup={backup}")

    for job_idx, one_task_overrides in enumerate(expanded_task_overrides, start=1):
        try:
            info = copy.deepcopy(base_info)
            info = apply_basename_override(info, one_task_overrides)

            override_dirname = build_override_dirname(
                one_task_overrides,
                info["kv_sep"],
                info["item_sep"],
                info["exclude_keys"],
            )

            if cli_job_name:
                job_name = cli_job_name
            elif info["hydra_job_name"]:
                job_name = resolve_interpolation(
                    info["hydra_job_name"],
                    info["basename"],
                    "",
                    override_dirname,
                )
            else:
                job_name = Path(script).stem

            sweep_dir = resolve_interpolation(
                info["sweep_dir"],
                info["basename"],
                job_name,
                override_dirname,
            )

            sweep_subdir = resolve_interpolation(
                info["sweep_subdir"],
                info["basename"],
                job_name,
                override_dirname,
            )

            if Path(sweep_dir).is_absolute():
                workingdir = Path(sweep_dir) / sweep_subdir
            else:
                workingdir = subdir.resolve() / sweep_dir / sweep_subdir

            workingdir = workingdir.resolve()

            gen = workingdir / "test" / "converted" / "Epoch_495"
            out = workingdir / "test" / "metrics"

            metrics_paths.append(str(out))
            gen_out_pairs.append((str(gen), str(out)))

            print(f"  job {job_idx}/{len(expanded_task_overrides)}")
            print(f"    overrides={one_task_overrides}")
            print(f"    basename={info['basename']}")
            print(f"    hydra.job.name={job_name}")
            print(f"    hydra.job.override_dirname={override_dirname}")
            print(f"    workingdir={workingdir}")
            print(f"    gt={GT}")
            print(f"    {{gen: {gen}, out: {out}}}")

            if not gen.exists():
                print(f"    WARNING: gen path does not exist: {gen}")

            text = original_metrics_text
            text = replace_var(text, "gt", GT)
            text = replace_var(text, "gen", str(gen))
            text = replace_var(text, "out", str(out))
            metrics_sh.write_text(text)

            subprocess.run(["sbatch", "metrics.sh"], cwd=subdir, check=False)

        except Exception as e:
            print(f"  job {job_idx}/{len(expanded_task_overrides)} skip: {e}")
            continue


# Save metrics_path.txt beside this script, not inside an experiment subdir.
metrics_path_txt = SCRIPT_DIR / "metrics_path.txt"
metrics_path_txt.write_text("\n".join(metrics_paths) + "\n")

print("\nAll {gen, out} pairs:")
for gen, out in gen_out_pairs:
    print(f"{{gen: {gen}, out: {out}}}")

print(f"\nSaved {len(metrics_paths)} metrics output paths to: {metrics_path_txt}")
