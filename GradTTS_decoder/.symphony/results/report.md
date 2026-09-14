# TAN-6 speech metric summary (fm excluded)

Arithmetic means and population standard deviations (ddof=0) are computed from the per-WAV files beside each generated WAV.

Training loss: 29 configurations and 17371 labelled epoch records; exact epoch 1-599 coverage is validated in `training_loss_validation.tsv` and the combined records are in `../../training_loss.log`.

| Folder | Pair | WAV n | MCD n / mean / std | logF0 n / mean / std | WER n / mean / std | UTMOSv2 n / mean / std | Slurm run / repair / metrics / WER / UTMOSv2 | Gen path |
|---|---:|---:|---|---|---|---|---|---|
| add | 0 | 488 | 488 / 5.22860246 / 0.51571136 | 488 / 0.33235779 / 0.08137479 | 488 / 0.05160835 / 0.11042808 | 488 / 3.56612609 / 0.26523322 | 10800233 / 10804134 / 11104582 / 11104616_0 / 11104618_0 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/add/add_inf/model.masking.a:0.2/test/converted/Epoch_595` |
| add | 1 | 488 | 488 / 5.22674262 / 0.51670631 | 488 / 0.32626967 / 0.07822411 | 488 / 0.05179954 / 0.11461558 | 488 / 3.58227939 / 0.28205235 | 10800233 / 10804135 / 11104583 / 11104616_1 / 11104618_1 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/add/add_inf/model.masking.a:0.4/test/converted/Epoch_595` |
| add | 2 | 488 | 488 / 5.22283217 / 0.51874660 | 488 / 0.33155164 / 0.07978093 | 488 / 0.05005060 / 0.10480525 | 488 / 3.57937372 / 0.27753496 | 10800233 / 10804137 / 11104584 / 11104616_2 / 11104618_2 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/add/add_inf/model.masking.a:0.6/test/converted/Epoch_595` |
| blur | 0 | 488 | 488 / 5.40901004 / 0.51311332 | 488 / 0.30576947 / 0.07140201 | 488 / 0.05498463 / 0.10887069 | 488 / 3.73261399 / 0.24874814 | 10800211 / 10804136 / 11104585 / 11104616_3 / 11104618_3 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/blur/blur_inf/model.masking.b:10/test/converted/Epoch_595` |
| blur | 1 | 488 | 488 / 5.60845676 / 0.52043833 | 488 / 0.32150410 / 0.07490203 | 488 / 0.05607166 / 0.11054607 | 488 / 3.83865507 / 0.23086501 | 10800211 / 10804138 / 11104586 / 11104616_4 / 11104618_4 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/blur/blur_inf/model.masking.b:20/test/converted/Epoch_595` |
| blur | 2 | 488 | 488 / 5.46489344 / 0.50992404 | 488 / 0.35211865 / 0.08365415 | 488 / 0.05320790 / 0.10657854 | 488 / 3.63619445 / 0.29456768 | 10800211 / 10804139 / 11104587 / 11104616_5 / 11104618_5 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/blur/blur_inf/model.masking.b:40/test/converted/Epoch_595` |
| fmDT | 0 | 488 | 488 / 5.84532357 / 0.50491024 | 488 / 0.33252828 / 0.07949468 | 488 / 0.05439327 / 0.10912589 | 488 / 3.84439037 / 0.22496189 | 10800235 / 10804142 / 11104593 / 11104616_11 / 11104618_11 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/fmDT_inf/model.masking.a:1/test/converted/Epoch_595` |
| levy | 0 | 488 | 488 / 5.76548709 / 0.52431700 | 488 / 0.32590738 / 0.08261754 | 488 / 0.05325422 / 0.11184210 | 488 / 4.00260550 / 0.22064146 | 10800210 / 10804144 / 11104594 / 11104616_12 / 11104618_12 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/levy/levy_inf/model.masking.alpha:1.8/test/converted/Epoch_595` |
| levy | 1 | 488 | 488 / 5.77868955 / 0.51376140 | 488 / 0.33052664 / 0.07982598 | 488 / 0.05280606 / 0.10676572 | 488 / 4.01091429 / 0.21640917 | 10800210 / 10804145 / 11104595 / 11104616_13 / 11104618_13 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/levy/levy_inf/model.masking.alpha:1.9/test/converted/Epoch_595` |
| mask | 0 | 488 | 488 / 5.49075225 / 0.52841419 | 488 / 0.34473914 / 0.08439456 | 488 / 0.05404698 / 0.11470202 | 488 / 3.89570393 / 0.22955245 | 10800213 / 10804146 / 11104596 / 11104616_14 / 11104618_14 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/mask/mask_inf/model.masking.a:1/test/converted/Epoch_595` |
| one-shot | 0 | 488 | 488 / 5.27667418 / 0.53103026 | 488 / 0.32418217 / 0.07987556 | 488 / 0.05065978 / 0.10625309 | 488 / 3.60536069 / 0.26957869 | 10800207 / 10804150 / 11104597 / 11104616_15 / 11104618_15 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/one-shot/one-shot_inf/model.masking.a:0.2/test/converted/Epoch_595` |
| product | 0 | 488 | 488 / 5.64939262 / 0.50976131 | 488 / 0.33159283 / 0.08714212 | 488 / 0.05239957 / 0.10526128 | 488 / 3.94954294 / 0.22671243 | 10800236 / 10804151 / 11104598 / 11104616_16 / 11104618_16 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/product/product_inf/model.masking.a:0.2/test/converted/Epoch_595` |
| product | 1 | 488 | 488 / 5.69468586 / 0.50891401 | 488 / 0.32764467 / 0.07961460 | 488 / 0.05356989 / 0.11002689 | 488 / 4.00026415 / 0.21276610 | 10800236 / 10804149 / 11104599 / 11104616_17 / 11104618_17 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/product/product_inf/model.masking.a:0.4/test/converted/Epoch_595` |
| product | 2 | 488 | 488 / 5.69284898 / 0.52766855 | 488 / 0.32554221 / 0.07727282 | 488 / 0.05430426 / 0.11011195 | 488 / 3.98408283 / 0.22830094 | 10800236 / 10804152 / 11104600 / 11104616_18 / 11104618_18 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/product/product_inf/model.masking.a:0.6/test/converted/Epoch_595` |
| score | 0 | 488 | 488 / 6.28183689 / 0.47247031 | 488 / 0.34126824 / 0.07930899 | 488 / 0.06510301 / 0.12566701 | 488 / 3.05820553 / 0.30229199 | 10804165 / 10804157 / 11104601 / 11104616_19 / 11104618_19 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/score/score_inf/model.masking.a:1.0/test/converted/Epoch_595` |
| score | 1 | 488 | 488 / 6.13280287 / 0.46770960 | 488 / 0.33905963 / 0.07425300 | 488 / 0.06136248 / 0.11584412 | 488 / 3.23907171 / 0.29926707 | 10804165 / 10804155 / 11104602 / 11104616_20 / 11104618_20 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/score/score_inf/model.masking.a:0.8/test/converted/Epoch_595` |
| score | 2 | 488 | 488 / 6.01774303 / 0.48859014 | 488 / 0.33464303 / 0.07857781 | 488 / 0.06310115 / 0.11404477 | 488 / 3.47262423 / 0.26561464 | 10804165 / 10804153 / 11104603 / 11104616_21 / 11104618_21 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/score/score_inf/model.masking.a:0.6/test/converted/Epoch_595` |
| score | 3 | 488 | 488 / 5.89721967 / 0.47966454 | 488 / 0.34083115 / 0.08173405 | 488 / 0.05877341 / 0.11651047 | 488 / 3.58372423 / 0.25723459 | 10804165 / 10804148 / 11104604 / 11104616_22 / 11104618_22 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/score/score_inf/model.masking.a:0.4/test/converted/Epoch_595` |
| score | 4 | 488 | 488 / 5.73468914 / 0.48675747 | 488 / 0.33363832 / 0.08233665 | 488 / 0.05617589 / 0.11353524 | 488 / 3.61826412 / 0.25680372 | 10804165 / 10804154 / 11104605 / 11104616_23 / 11104618_23 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/score/score_inf/model.masking.a:0.2/test/converted/Epoch_595` |
| scoreDT | 0 | 488 | 488 / 6.19149570 / 0.47136661 | 488 / 0.33120635 / 0.07658406 | 488 / 0.06258472 / 0.11914875 | 488 / 3.06268411 / 0.30925898 | 10884228 / 11099158 / 11104606 / 11104616_24 / 11104618_24 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scoreDT/scoreDT_inf/model.masking.a:1.0/test/converted/Epoch_595` |
| scoreDT | 1 | 488 | 488 / 6.05169037 / 0.49294013 | 488 / 0.33264980 / 0.07905274 | 488 / 0.05874388 / 0.11387912 | 488 / 3.50828477 / 0.26698456 | 10884228 / 11099158 / 11104607 / 11104616_25 / 11104618_25 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scoreDT/scoreDT_inf/model.masking.a:0.8/test/converted/Epoch_595` |
| scoreDT | 2 | 488 | 488 / 5.94943852 / 0.48820458 | 488 / 0.33457746 / 0.07429764 | 488 / 0.06007011 / 0.12432020 | 488 / 3.56682649 / 0.24271926 | 10884228 / 11099158 / 11104608 / 11104616_26 / 11104618_26 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scoreDT/scoreDT_inf/model.masking.a:0.6/test/converted/Epoch_595` |
| scoreDT | 3 | 488 | 488 / 5.89685635 / 0.48559181 | 488 / 0.33847049 / 0.08202807 | 488 / 0.06021608 / 0.12825520 | 488 / 3.62801774 / 0.24429371 | 10884228 / 11099158 / 11104609 / 11104616_27 / 11104618_27 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scoreDT/scoreDT_inf/model.masking.a:0.4/test/converted/Epoch_595` |
| scoreDT | 4 | 488 | 488 / 5.70867725 / 0.49046287 | 488 / 0.33227316 / 0.08328667 | 488 / 0.05779842 / 0.11261058 | 488 / 3.58821481 / 0.25068953 | 10884228 / 11099158 / 11104610 / 11104616_28 / 11104618_28 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scoreDT/scoreDT_inf/model.masking.a:0.2/test/converted/Epoch_595` |
| scorex0 | 0 | 488 | 488 / 5.59351680 / 0.54258174 | 488 / 0.32171270 / 0.07739887 | 488 / 0.05336866 / 0.11412884 | 488 / 3.94323130 / 0.22310756 | 10803830 / 10804141 / 11104611 / 11104616_29 / 11104618_29 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scorex0/scorex0_inf/model.masking.a:1.0/test/converted/Epoch_595` |
| scorex0 | 1 | 488 | 488 / 5.60267193 / 0.52464453 | 488 / 0.32009344 / 0.07955848 | 488 / 0.05106252 / 0.10757316 | 488 / 3.94520844 / 0.22678629 | 10803830 / 10804159 / 11104612 / 11104616_30 / 11104618_30 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scorex0/scorex0_inf/model.masking.a:0.8/test/converted/Epoch_595` |
| scorex0 | 2 | 488 | 488 / 5.67035348 / 0.52792594 | 488 / 0.32445840 / 0.07877262 | 488 / 0.05534674 / 0.11295522 | 488 / 3.96841781 / 0.21894019 | 10803830 / 10804156 / 11104613 / 11104616_31 / 11104618_31 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scorex0/scorex0_inf/model.masking.a:0.6/test/converted/Epoch_595` |
| scorex0 | 3 | 488 | 488 / 5.86461045 / 0.51695780 | 488 / 0.32806189 / 0.07797515 | 488 / 0.05639191 / 0.11823432 | 488 / 4.00253346 / 0.20479155 | 10803830 / 10804158 / 11104614 / 11104616_32 / 11104618_32 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scorex0/scorex0_inf/model.masking.a:0.4/test/converted/Epoch_595` |
| scorex0 | 4 | 488 | 488 / 5.94049775 / 0.52058474 | 488 / 0.33081906 / 0.08338351 | 488 / 0.05351163 / 0.10780143 | 488 / 4.00238137 / 0.22638577 | 10803830 / 10804140 / 11104615 / 11104616_33 / 11104618_33 | `/mnt/parscratch/users/acp23xt/private/DTDM_Unet/scorex0/scorex0_inf/model.masking.a:0.2/test/converted/Epoch_595` |

## Missing or invalid values

None. Every metric has one valid value per generated WAV.

Machine-readable data: `summary.csv`

## Completion audit

- Final first-depth run.sh rescan: 10 eligible folders (`fm` excluded).
- Final extractor rescan: 29 unique non-`fm` gen/out pairs.
- Per-path transcript/WAV/metric coverage: 488 exact values.
- Training-loss coverage: 29 configurations and 17371 epoch records.
- Successful Slurm allocations/tasks audited: 123.
- Terminal log files inspected: 238.
- Accounting details: final_job_audit.tsv.
- Log details: terminal_log_audit.tsv.
