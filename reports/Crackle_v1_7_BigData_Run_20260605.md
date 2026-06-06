# Crackle v1.7 Big Data Run

Date: 2026-06-05

## Scope

This run continues the crack branch with larger real crack/DIC data. L-bracket FEM data is not used here. The useful external crack data is kept on `G:` to avoid filling `C:`.

## Downloaded / Audited Data

| dataset | local path | status | useful role |
|---|---|---|---|
| CrackMNIST 64 L | `<workspace>\external_datasets\real_fracture_20260605\crackmnist_cache\crackmnist_64_L.h5` | verified MD5 `5963d788af31f4a24ebe904cb3ad43db` | DIC displacement fields + crack-tip / crack-path masks for real tip localization validation/training |
| CrackMNIST 128 L | `<workspace>\external_datasets\real_fracture_20260605\crackmnist_cache\crackmnist_128_L.h5` | verified MD5 `5527a54cc382623edcb2022e95a9ed2d`; first interrupted copy quarantined and removed after MD5 mismatch | higher-resolution DIC/mask training |
| DLR DIC S_160_2.0 | `<workspace>\external_datasets\real_fracture_20260605\dlr_dic_zenodo_5740216\S_160_2.0.zip` | zip valid; MD5 `108B0B6C9BF100BD44415199CC8D4261`; 1413 entries | DIC nodemap / force / crack-length style covariates; not mask-supervised in this archive |
| DLR DIC S_160_4.7 | `<workspace>\external_datasets\real_fracture_20260605\dlr_dic_zenodo_5740216\S_160_4.7.zip` | previously verified; contains GroundTruth and Nodemaps | DIC crack mask / tip validation |
| NASA PHM 2019 fatigue curves | `<workspace>\external_datasets\real_fracture_20260605\nasa_phm_2019\PHMDC2019_Data.try.zip` | verified zip | 1D curve forecasting / Paris competitor head-to-head |

Important caution: the first `crackmnist_128_L.h5` download had the correct byte length and could be opened by HDF5, but its MD5 was `acecab9e02e9c2d7fc27cc2f7bc726fe`, not the Zenodo/API checksum `5527a54cc382623edcb2022e95a9ed2d`. It has been moved to a `suspect_md5` filename and must not be used for claims.
After re-download with range chunks, the final `128_L` file passed the Zenodo/API MD5 and the suspect copy plus chunk files were removed.

## CrackMNIST 64L Training Result

Model: `xgboost_dic_pixel_cuda_v1`

Input features per pixel:

- `ux`, `uy`
- displacement magnitude
- `ux/uy` spatial gradients
- gradient magnitude
- normalized pixel coordinates

Training:

| item | value |
|---|---:|
| train images | 42,056 |
| sampled rows | 30,514,963 |
| positive rows | 376,611 |
| negative rows | 30,138,352 |
| backend | `xgboost_cuda` |
| train time | 54.91 s |
| validation-calibrated threshold | 0.992785 |

Results:

| split | images | pixel F1 | pixel IoU | top-k mask IoU | top-1 hit rate | tip loc err px | eval images/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 11,736 | 0.6500 | 0.4815 | 0.5099 | 0.9188 | 1.1107 | 772.8 |
| test | 16,560 | 0.5272 | 0.3579 | 0.3875 | 0.8145 | 1.9792 | 707.4 |

Three-seed test aggregate:

| metric | mean | std | seeds |
|---|---:|---:|---:|
| pixel F1 | 0.5268 | 0.0005 | 3 |
| pixel IoU | 0.3576 | 0.0005 | 3 |
| top-k mask IoU | 0.3873 | 0.0006 | 3 |
| top-1 hit rate | 0.8151 | 0.0008 | 3 |
| tip loc err px | 1.9754 | 0.0095 | 3 |
| train seconds | 54.80 | 0.10 | 3 |

Interpretation: after validation-threshold calibration, the pixel mask metrics are usable and the top-k / tip localization metrics are strong enough to justify this as a real DIC-to-tip localization baseline. This is still not a full physical hazard validation because it is spatial mask supervision, not true event-time data.

## Traditional DIC Rule Baselines

These are non-learning baselines using only hand-written DIC magnitude/gradient scores.

Test split:

| model | pixel F1 | top-k mask IoU | top-1 hit rate | tip loc err px |
|---|---:|---:|---:|---:|
| `xgboost_dic_pixel_cuda_v1_calibrated` | 0.5272 | 0.3875 | 0.8145 | 1.9792 |
| `dic_displacement_magnitude_rule` | 0.0000 | 0.0023 | 0.0045 | 37.5005 |
| `dic_gradient_magnitude_rule` | 0.0011 | 0.0030 | 0.0031 | 29.7144 |
| `dic_energy_gradient_proxy_rule` | 0.0092 | 0.0036 | 0.0051 | 29.9591 |
| `dic_max_abs_gradient_rule` | 0.0008 | 0.0026 | 0.0036 | 29.5526 |

Conclusion: on CrackMNIST 64L, the learned XGBoost baseline is not merely reproducing a simple DIC magnitude or gradient threshold. The learned model improves test top-1 hit rate from under 1% to 81.45%, raises pixel F1 to 0.5272, and reduces tip localization error from roughly 29-38 px to 1.98 px.

## CrackMNIST 128L Trial

The 128L file was verified and trained after the 64L run. It did not improve the result under the current pixel-feature XGBoost setup.

| model | train rows | train seconds | test pixel F1 | test pixel IoU | test top-k IoU | test top-1 hit | test loc err px |
|---|---:|---:|---:|---:|---:|---:|---:|
| `xgboost_dic_pixel_cuda_128L_np40` | 42,912,282 | 75.87 | 0.4301 | 0.2740 | 0.2896 | 0.7009 | 5.2314 |
| `xgboost_dic_pixel_cuda_128L_np80` | 84,768,682 | 145.14 | 0.4298 | 0.2737 | 0.2887 | 0.6930 | 5.2718 |

Decision: keep 128L as a useful high-resolution dataset, but use 64L as the current best baseline for this XGBoost/DIC feature family. The 128L result likely needs a convolutional/local-context model rather than independent per-pixel tree features.

## Existing Real Curve Head-to-Head

NASA PHM curve validation remains a different task: 1D crack-length forecasting. It is useful for Paris-style competitor comparison, but it does not validate spatial hazard or event localization.

Best prior real-curve result:

| model | curve RMSE mm |
|---|---:|
| `cox_discrete_time_curve_proxy_v1` | 2.1061 |
| `gbm_survival_curve_proxy_v1` | 2.3746 |
| `traditional_last_rate` | 2.5797 |
| `paris_prior_relu_bilstm_sa_v1` | 3.4945 |
| `hawkes_curve_intensity_proxy_v1` | 4.2995 |
| `traditional_paris_powerlaw` | 24.7203 |

This does not mean the hazard model is fully validated. It only says the curve proxy is competitive on the small NASA curve set. Spatial hazard still needs true event `(t, location, energy)` data or simulation-to-real transfer through DIC/mask/AE datasets.

## Pipeline Findings

1. The first 64L training attempt failed because the HDF5 reader was doing many random single-image reads from the external SSD after an accidental disconnect. The script was patched to read sequential batches before sampling pixels.
2. XGBoost CUDA training works and trained 30.5M rows in under one minute.
3. XGBoost prediction warns about CPU NumPy input being transferred to the CUDA booster. Evaluation is still fast enough for this baseline, but a future CuPy/DMatrix path would be cleaner.
4. The 128L checksum mismatch is treated as a data integrity problem, not ignored.

## Output Files

| artifact | path |
|---|---|
| XGBoost metrics | `<workspace>\crackle_runs\crackle_v1_7_bigdata_20260605\crackmnist_64L_xgb_cuda\crackmnist_xgb_metrics.csv` |
| DIC rule metrics | `<workspace>\crackle_runs\crackle_v1_7_bigdata_20260605\crackmnist_64L_dic_rules\crackmnist_rule_metrics.csv` |
| Combined comparison | `<workspace>\crackle_runs\crackle_v1_7_bigdata_20260605\crackmnist_64L_model_comparison.csv` |
| Three-seed aggregate | `<workspace>\crackle_runs\crackle_v1_7_bigdata_20260605\crackmnist_64L_xgb_seed_aggregate.csv` |
| v1.7 summary | `<workspace>\crackle_runs\crackle_v1_7_bigdata_20260605\crackle_v1_7_summary.csv` |

## Decision

Use CrackMNIST 64L immediately for real DIC/mask spatial validation. Use CrackMNIST 128L only after checksum passes. Keep NASA PHM curves for curve/Paris competitor reporting, but do not confuse curve RMSE with spatial hazard validation.

