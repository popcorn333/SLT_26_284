#!/usr/bin/env python3
"""Remove the issue-excluded fm rows from TAN-6 canonical ledgers."""

import csv
from pathlib import Path


ROOT = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet")
RES = ROOT / ".symphony/results"
EXPECTED_FOLDERS = {
    "add", "blur", "fmDT", "levy", "mask", "one-shot", "product",
    "score", "scoreDT", "scorex0",
}


def filter_tsv(path, expected_rows=None):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row.get("folder") != "fm"]
    if not fieldnames or "folder" not in fieldnames:
        raise RuntimeError("{} is not a folder-scoped ledger".format(path))
    if any(row["folder"] == "fm" or "/fm/" in "\t".join(row.values())
           for row in rows):
        raise RuntimeError("fm leaked into {}".format(path))
    if expected_rows is not None and len(rows) != expected_rows:
        raise RuntimeError(
            "{} has {} non-fm rows; expected {}".format(
                path, len(rows), expected_rows))
    temp = path.with_name(path.name + ".fm-exclusion.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fieldnames,
            lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    return rows


def main():
    mappings = filter_tsv(RES / "mappings.tsv", 29)
    if {row["folder"] for row in mappings} != EXPECTED_FOLDERS:
        raise RuntimeError("non-fm mapping folder coverage is incomplete")
    mapping_keys = {(row["folder"], row["pair_index"]) for row in mappings}
    if len(mapping_keys) != 29:
        raise RuntimeError("non-fm mapping keys are duplicated")

    jobs = filter_tsv(RES / "jobs.tsv", 145)
    stages = {"run", "gen_repair", "metrics", "wer", "utmosv2"}
    job_keys = {
        (row["folder"], row["pair_index"], row["stage"]) for row in jobs
    }
    expected_job_keys = {
        (folder, pair, stage)
        for folder, pair in mapping_keys for stage in stages
    }
    if job_keys != expected_job_keys:
        raise RuntimeError("non-fm jobs ledger lacks exact five-stage coverage")

    repairs = filter_tsv(RES / "gen_repair_jobs.tsv", 29)
    if {(row["folder"], row["pair_index"]) for row in repairs} != mapping_keys:
        raise RuntimeError("non-fm repair ledger differs from mappings")
    print("fm exclusion applied: folders=10 pairs=29 jobs=145", flush=True)


if __name__ == "__main__":
    main()
