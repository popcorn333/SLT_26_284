# TAN-18 fmDT speech-metric report

Standard deviations are population standard deviations over per-WAV scalar sidecars.

The metric tools emitted aggregate lines in `path   mean \pm std` format and per-WAV scalar `_mcd.txt`/`_f0.txt` sidecars; summaries below are recomputed from the sidecars.

Common jobs: environment probe `11141986`, checkpoint validation `11142108`, run array `11142109`, acceptance gate `11142110`, training-loss `11142111`, dispatcher `11142112`, summary `11149889`.

Run task IDs: `11142109_0`, `11142109_1`, `11142109_2`, `11142109_3`, `11142109_4`.

| Sweep | WAVs | MCD n / mean / sd | logF0/F0 n / mean / sd | WER n / mean / sd | UTMOSv2 n / mean / sd | Jobs (run task / metric / WER / UTMOS) | Coverage |
|---|---:|---:|---:|---:|---:|---|---|
| `model.masking.a:0.2` | 488 | 488 / 5.665403 / 0.499919 | 488 / 0.330340 / 0.081164 | 488 / 0.054129 / 0.109149 | 488 / 3.762079 / 0.235018 | `11142109_0` / `11149886_0` / `11149887_0` / `11149888_0` | OK |

- Gen: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.2/test/converted/Epoch_595`
- Out: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.2/test/metrics`
- MCD aggregate: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.2/test/converted/Epoch_595   5.6654 \pm 0.4999`
- F0 aggregate: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.2/test/converted/Epoch_595   0.3303 \pm 0.0812`
- Genuine missing values: `none`

| `model.masking.a:0.4` | 488 | 488 / 5.773281 / 0.508966 | 488 / 0.335068 / 0.080567 | 488 / 0.053814 / 0.110859 | 488 / 3.808242 / 0.238333 | `11142109_1` / `11149886_1` / `11149887_1` / `11149888_1` | OK |

- Gen: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.4/test/converted/Epoch_595`
- Out: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.4/test/metrics`
- MCD aggregate: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.4/test/converted/Epoch_595   5.7733 \pm 0.5090`
- F0 aggregate: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.4/test/converted/Epoch_595   0.3351 \pm 0.0806`
- Genuine missing values: `none`

| `model.masking.a:0.6` | 488 | 488 / 5.788932 / 0.511998 | 488 / 0.332448 / 0.081240 | 488 / 0.053693 / 0.120439 | 488 / 3.832532 / 0.232794 | `11142109_2` / `11149886_2` / `11149887_2` / `11149888_2` | OK |

- Gen: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.6/test/converted/Epoch_595`
- Out: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.6/test/metrics`
- MCD aggregate: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.6/test/converted/Epoch_595   5.7889 \pm 0.5120`
- F0 aggregate: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.6/test/converted/Epoch_595   0.3324 \pm 0.0812`
- Genuine missing values: `none`

| `model.masking.a:0.8` | 488 | 488 / 5.816632 / 0.521664 | 488 / 0.332258 / 0.080956 | 488 / 0.057279 / 0.112703 | 488 / 3.831799 / 0.230267 | `11142109_3` / `11149886_3` / `11149887_3` / `11149888_3` | OK |

- Gen: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.8/test/converted/Epoch_595`
- Out: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.8/test/metrics`
- MCD aggregate: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.8/test/converted/Epoch_595   5.8166 \pm 0.5217`
- F0 aggregate: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:0.8/test/converted/Epoch_595   0.3323 \pm 0.0810`
- Genuine missing values: `none`

| `model.masking.a:1` | 488 | 488 / 5.801117 / 0.506703 | 488 / 0.328161 / 0.079523 | 488 / 0.051362 / 0.107068 | 488 / 3.855028 / 0.229430 | `11142109_4` / `11149886_4` / `11149887_4` / `11149888_4` | OK |

- Gen: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:1/test/converted/Epoch_595`
- Out: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:1/test/metrics`
- MCD aggregate: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:1/test/converted/Epoch_595   5.8011 \pm 0.5067`
- F0 aggregate: `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:1/test/converted/Epoch_595   0.3282 \pm 0.0795`
- Genuine missing values: `none`

## Terminal validation

- Fresh TAN-18 scope: fmDT, five configurations; scorex0 is carried through the workflow as a reused completed result set in the combined `tan18_summary.csv` / `tan18_report.md` artifacts.
- Final first-depth rescan: 11 `run.sh` folders (`add`, `blur`, `fm`, `fmDT`, `levy`, `mask`, `one-shot`, `product`, `score`, `scoreDT`, `scorex0`); fmDT is fresh and scorex0 is included from its validated completed graph.
- Fresh fmDT coverage: 488 WAVs and 488 finite MCD, logF0/F0, WER, and UTMOSv2 values for every configuration.
- Training loss: 2995 fmDT records (5 configurations × 599 contiguous epochs); 16772 non-fmDT rows preserved.
- Included scorex0 coverage: 2995 training-loss records (5 configurations × 599 contiguous epochs), plus freshly revalidated speech-metric sidecars and terminal job provenance in the combined artifacts.
- Predecessor accounting: 52 allocation/task records, all `COMPLETED 0:0`.
- Completion bridge (legacy filename): `11143257`.
- Final audit job: `11149890` (its terminal state is inspected after this payload exits).
- Genuine missing or invalid requested values: none.

Accounting evidence: `tan18_fmdt_job_audit.tsv`.
