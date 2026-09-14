#!/usr/bin/env python3
"""Run scoreDT inference for the epoch-595 checkpoint only.

The experiment's inference loop unconditionally indexes the latest 30
checkpoints.  TAN-10 needs the emitted Epoch_595 path, so run an in-memory copy
that filters the checkpoint list to epoch 595 and bounds the loop to that one
entry.  The experiment source itself remains untouched.
"""

import sys
from pathlib import Path


SOURCE = Path("/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scoreDT/inference.py")
OLD_LOOP = "for i in range(ending_epoch - 1, ending_epoch - 31, -1):"
NEW_LOOP = "for i in range(ending_epoch - 1, -1, -1):"
CHECKPOINT_LIST = "checkpoint_files = sorted(checkpoint_files, key=get_integer_part)"
FILTERED_CHECKPOINT_LIST = (
    CHECKPOINT_LIST
    + "\n    checkpoint_files = [path for path in checkpoint_files "
      "if int(os.path.splitext(os.path.basename(path))[0].rsplit('_', 1)[-1]) == 595]"
    + "\n    if len(checkpoint_files) != 1:"
    + "\n        raise RuntimeError(f'expected one epoch-595 checkpoint; "
      "found {len(checkpoint_files)}')"
)


def main():
    source = SOURCE.read_text(encoding="utf-8")
    if source.count(OLD_LOOP) != 1:
        raise RuntimeError("scoreDT inference loop no longer matches the audited source")
    if source.count(CHECKPOINT_LIST) != 1:
        raise RuntimeError("scoreDT checkpoint discovery no longer matches the audited source")
    source = source.replace(CHECKPOINT_LIST, FILTERED_CHECKPOINT_LIST)
    source = source.replace(OLD_LOOP, NEW_LOOP)
    sys.path.insert(0, str(SOURCE.parent))
    namespace = {
        "__file__": str(SOURCE),
        "__name__": "__main__",
        "__package__": None,
    }
    exec(compile(source, str(SOURCE), "exec"), namespace)


if __name__ == "__main__":
    main()
