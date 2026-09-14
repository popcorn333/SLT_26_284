'''
This script reads file paths from "metrics_path.txt" generated from submit_metrics_all_hydra.py in the same directory, extracts F0 and MCD metrics from the corresponding files, and saves a summary in "summary_metrics.txt". Each line in the output file contains the relative file path (with a specified prefix removed) followed by the extracted F0 and MCD metrics. If a metric file is missing, it will indicate "MISSING". If the expected pattern for metrics is not found, it will return the entire content of the file as a fallback.
'''
from pathlib import Path
import re

metrics_path_file = Path("metrics_path.txt")
output_file = Path("summary_metrics.txt")

prefix_to_remove = "/mnt/parscratch/users/acp23xt/private/DTDM_Unet/"

def extract_metric(path: Path) -> str:
    if not path.exists():
        return "MISSING"

    content = " ".join(path.read_text(encoding="utf-8").split())

    # Extract pattern like: 0.3576 \pm 0.0835
    match = re.search(
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\\pm\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        content
    )

    if match:
        return f"{match.group(1)} \\pm {match.group(2)}"

    # Fallback: if no \pm pattern is found, return the whole compressed content
    return content

with metrics_path_file.open("r", encoding="utf-8") as f_in, \
     output_file.open("w", encoding="utf-8") as f_out:

    for line in f_in:
        file_path = line.strip()

        if not file_path:
            continue

        metric_dir = Path(file_path)

        f0_metric = extract_metric(metric_dir / "F0.txt")
        mcd_metric = extract_metric(metric_dir / "MCD.txt")

        # Remove long prefix from filepath
        saved_path = file_path.removeprefix(prefix_to_remove)

        # Save format:
        # 'relative/filepath F0_metric MCD_metric',
        f_out.write(f"'{saved_path} {f0_metric} {mcd_metric}',\n")

print(f"Saved summary to {output_file}")
