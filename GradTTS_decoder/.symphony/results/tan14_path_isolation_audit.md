# TAN-14 path-isolation audit

Checkpoint: 2026-08-03 18:50 BST.

- TAN-12 array `11108874_[0-4]` completed `0:0` and ran from `/mnt/parscratch/users/acp23xt/private/DTDM_sgmse`.
- Its `fmDT/run.sh` explicitly changes directory to `/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/fmDT`.
- TAN-12 Hydra metadata records training output under `/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/fmDT/Hydra_fmDT/...` and inference output under `/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/fmDT/fmDT_inf/...`.
- TAN-14 targets `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT`; the two repository and `fmDT` paths have distinct filesystem inodes and are not symlinks.
- TAN-12 dispatcher `11110474` failed `1:0` on a literal `$config` path before submitting any metric, WER, UTMOSv2, loss, or finalizer jobs.
- Failed TAN-14 recovery `11115675` detected the absent finalizer ledger and failed closed before submitting any work.

Therefore TAN-12 does not write to TAN-14's fixed paths, and no TAN-12 finalizer dependency is required for the fresh TAN-14 graph submitted as `11116063` -> `11116064` -> (`11116065`, `11116066`).
