import numpy as np
import pandas as pd

input_list = "mos_files.txt"
output_file = "mos_summary.txt"

lines_out = []

with open(input_list) as f:
    paths = [l.strip() for l in f if l.strip()]

for p in paths:
    # header=0 -> first line is column name
    df = pd.read_csv(p, header=0)

    mos = df.iloc[:, 1].astype(float).values   # second column
    mean = mos.mean()
    std = mos.std()

    # experiment name
    exp_name = p.split("/test_set/")[-1]
    exp_name = exp_name.split("/test/converted")[0]

    lines_out.append(f"{exp_name}  {mean:.4f} ± {std:.4f}")

with open(output_file, "w") as f:
    f.write("\n".join(lines_out))

print("Saved:", output_file)

