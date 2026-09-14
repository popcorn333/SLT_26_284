#!/usr/bin/env python3
import subprocess
from pathlib import Path

RES = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results")
SCOREDT_INFERENCE = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scoreDT/inference.py")
LOSS_JOB_ID = "10884248"
TEMPORARY_GUARD = """            # TAN-6 allocation 10884228 skips the normal last-20 checkpoint pass.
            # Epoch_495 lies immediately before that 500..595 window and is
            # produced by the dependency-gated exact-checkpoint repair array.
            # Keep the original loop intact for that repair wrapper.
            if os.environ.get('SLURM_JOB_ID') == '10884228' and checkpoint_name != 'grad_495':
                continue
"""


def read_ids():
    ids = {}
    for line in (RES / "pipeline_job_ids.tsv").read_text(encoding="utf-8").splitlines():
        key, value = line.split("\t", 1)
        ids[key] = value
    required = {"dispatch_job_id", "wer_array_job_id", "utmosv2_warmup_job_id",
                "utmosv2_array_job_id", "summary_job_id", "pair_count"}
    if set(ids) != required:
        raise RuntimeError("unexpected pipeline job-id keys: {}".format(sorted(ids)))
    if ids["pair_count"] != "30":
        raise RuntimeError("dispatcher did not emit 30 pairs")
    return ids


def remove_temporary_guard():
    source = SCOREDT_INFERENCE.read_text(encoding="utf-8")
    guard_count = source.count(TEMPORARY_GUARD)
    if guard_count == 0:
        if "10884228" in source:
            raise RuntimeError("unexpected TAN-6 scoreDT guard text remains")
        return
    if guard_count != 1:
        raise RuntimeError("expected at most one TAN-6 scoreDT inference guard")
    temporary = SCOREDT_INFERENCE.with_suffix(".py.tan6-cleanup.tmp")
    temporary.write_text(source.replace(TEMPORARY_GUARD, ""), encoding="utf-8")
    temporary.replace(SCOREDT_INFERENCE)
    if "10884228" in SCOREDT_INFERENCE.read_text(encoding="utf-8"):
        raise RuntimeError("TAN-6 scoreDT inference guard cleanup did not complete")


def main():
    ids = read_ids()
    remove_temporary_guard()
    finalizer_job_id = __import__("os").environ["SLURM_JOB_ID"]
    output = subprocess.check_output([
        "/usr/bin/sbatch", "--parsable",
        # The successful training-loss allocation is old enough that this
        # cluster has purged it from the live controller.  Depending on that
        # ID makes sbatch reject the otherwise valid submission.  The final
        # audit still requires and audits its terminal sacct record, status
        # file, validation table, and combined output; only the live summary
        # allocation belongs in the scheduler dependency expression.
        "--dependency=afterok:{}".format(ids["summary_job_id"]),
        str(RES / "final_audit.sbatch"),
    ], universal_newlines=True)
    audit_job_id = output.strip().split(";")[0]
    (RES / "final_audit_job_id.tsv").write_text(
        "final_audit_job_id\t{}\nfinalizer_job_id\t{}\nloss_job_id\t{}\n".format(
            audit_job_id, finalizer_job_id, LOSS_JOB_ID), encoding="utf-8")
    print("final audit submitted {} after summary {}; it will audit completed loss {} and finalizer {}".format(
        audit_job_id, ids["summary_job_id"], LOSS_JOB_ID, finalizer_job_id))


if __name__ == "__main__":
    main()
