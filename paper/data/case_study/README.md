# Case-study inputs (vendor study)

Result tables of the HCM-vs-normal vendor study on M&Ms-2, which used features from SORAT's nnFormer masks and from the manual contours. The study code is on this repository's `eece568-hcm-vendor` branch (`hcm_vendor/`). `manifest.json` records the producing commit and package versions.

| File | Content |
|---|---|
| `s1_e3_probe.csv` | Vendor probe on normal hearts: balanced accuracy, 95% CI and permutation p by feature family, nnFormer masks vs manual contours |
| `s1_gap_nnformer_vs_gt.csv` | Cross-vendor AUC and specificity loss per classifier and direction, nnFormer vs manual, paired-bootstrap difference with Holm p |
| `table1_cohort.csv` | Cohort: M&Ms-2 normal and HCM subjects by vendor |

Features were computed from the raw (not post-processed) masks.
