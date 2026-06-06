# Crackle v1.8 DLR Thesis Correction

Date: 2026-06-05

## Correction

The v1.7 CrackMNIST / DIC result must not be claimed as a win over the real DLR baseline. It only proves that:

1. the real DIC/mask data path is usable and checksum-clean;
2. the learned XGBoost baseline is not a trivial DIC magnitude/gradient threshold;
3. XGBoost CUDA can train quickly on large pixel samples.

It does not prove superiority over the fair DLR baseline, which should be a CNN-style crack-tip detector plus the Williams-series / high-fidelity correction workflow used by the dataset authors. It also does not validate the original Crackle thesis, because current-frame DIC crack-tip detection is not next-event crack forecast.

## DLR Spatial Detection Audit

Dataset:

`<workspace>\external_datasets\real_fracture_20260605\dlr_dic_zenodo_5740216\S_160_4.7.zip`

Protocol:

- 835 real load stages, left/right labels = 1670 samples.
- first 30% stages for calibration/domain adaptation.
- final 70% stages for test.
- DLR nodemap `u/v` was rasterized to 64x64 DIC images.
- GroundTruth labels: `1 = crack path`, `2 = crack tip`.

Key result:

| model | target | path/tip IoU | pixel F1 | argmax tip err px256 | frontier tip err px256 | hit <=4px |
|---|---|---:|---:|---:|---:|---:|
| `xgb_crackmnist_cold_transfer` | tip | 0.0025 | 0.0050 | 88.26 | 141.79 | 0.005 |
| `xgb_dlr_front30_adapt_tip` | tip | 0.0036 | 0.0072 | 77.12 | 95.92 | 0.000 |
| `xgb_dlr_front30_adapt_path_tip` | path_tip | 0.2568 | 0.4087 | 116.40 | 134.69 | 0.000 |
| `paris_lefm_straight_tip_extrapolator` | path_tip | 0.2405 | 0.3877 | 5.95 | n/a | 0.405 |

Readout:

- Cold sim/CrackMNIST-to-DLR transfer fails.
- Front-30 DLR adaptation can slightly improve path-mask IoU over the straight-line geometry baseline, but it fails the crack-tip localization metric badly.
- Therefore this is not a spatial victory. At best it shows that a path mask and a crack-tip head must be separated.

Output:

`<workspace>\crackle_runs\crackle_v1_8_dlr_spatial_20260605\dlr_s160_47_xgb64_frontier_readout\dlr_spatial_metrics.csv`

## DLR Next-Stage Tip Forecast Audit

This test returns to the actual forecast framing: predict the next stage crack tip from current/past state.

Causal features:

- current crack-tip coordinates,
- previous tip delta,
- current force,
- current crack length,
- current local DIC `u/v` and gradient features.

Result on final 70% stages:

| model | mean next-tip err px256 | median | p90 | hit <=4px | verdict |
|---|---:|---:|---:|---:|---|
| `persistence_current_tip` | 0.1666 | 0.0000 | 1.0000 | 1.000 | baseline already saturated |
| `last_delta_extrapolation` | 0.3332 | 0.0000 | 1.0000 | 1.000 | also saturated |
| `front30_linear_tip_fit` | 5.9379 | 5.6100 | 11.6690 | 0.404 | weaker long extrapolator |
| `xgb_temporal_dic_next_tip_v1` | 0.1847 | 0.0261 | 0.6686 | 1.000 | does not beat persistence |

Readout:

- DLR `S_160_4.7` next-frame tip forecast is too easy under this protocol because the crack tip often moves 0-1 pixel per stage.
- A simple persistence baseline is already essentially perfect.
- This sequence does not unlock a meaningful forecast win.

Output:

`<workspace>\crackle_runs\crackle_v1_8_dlr_spatial_20260605\dlr_tip_forecast_xgb\dlr_tip_forecast_metrics.csv`

## GPU Reality

Current local Python stack:

- PyTorch is CPU-only: `2.9.1+cpu`, `torch.cuda.is_available() = False`.
- XGBoost CUDA works for training.
- XGBoost prediction still warns that input arrays are CPU-side, so inference does CPU feature construction and CPU-to-GPU transfer.
- These DLR/CrackMNIST tasks are small enough that they do not fill the RTX 4080 Laptop GPU. Bigger synthetic/hard 3D active-set workloads are needed to measure true GPU utilization.

GPU transition requirement:

1. install/use a CUDA tensor backend (`torch` CUDA or `cupy-cuda12x`) outside `C:`;
2. move feature construction to GPU;
3. feed XGBoost with GPU arrays or replace the pixel model with a CUDA CNN;
4. benchmark data transfer, feature time, predict time, and end-to-end wall clock separately.

## Decision

No claim is allowed from v1.7/v1.8 that Crackle beats traditional or DLR methods on real DIC.

Allowed claim:

```text
We now have checksum-clean real DIC/mask data and can run honest spatial audits.
Simple DIC thresholds are not competitive, but the fair CNN/Williams DLR baseline is still missing.
Cold transfer fails; front-30 adaptation gives only a modest path IoU signal and fails tip localization.
DLR one-step forecast is saturated by persistence and is not a meaningful win condition.
```

Next honest options:

1. implement or reproduce the DLR CNN/Williams detector as the fair detection baseline;
2. acquire AE/event tables or hard mixed-mode/heterogeneous crack sequences for true hazard validation;
3. keep Crackle novelty on hard event forecasting, not current-frame DIC detection.

