# Crackle Final Archive and Open-Source Readout

Date: 2026-06-05

## Final Verdict

Crackle should be archived as a rigorous research prototype with useful code and honest negative results.

The last technical push improved the engineering stack substantially:

- installed CUDA PyTorch on `G:` without filling `C:`;
- trained a real CUDA CNN baseline;
- reached strong CrackMNIST detection metrics;
- measured high GPU utilization;
- reran DLR real spatial audits.

But it still did not establish the original thesis on real data.

## Best Positive Result

CrackMNIST 64L current-frame crack-tip detection:

| model | split | pixel F1 | pixel IoU | top-k IoU | top-1 hit | tip err px | train time | eval speed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `tiny_unet_crackmnist_cuda_v1` | test | 0.8008 | 0.6678 | 0.6798 | 0.8922 | 1.3311 | 607.6 s | 5170 img/s |

GPU telemetry:

| avg util | p95 util | peak memory | avg power | avg temp |
|---:|---:|---:|---:|---:|
| 63.7% | 100.0% | 9796 MB | 67.9 W | 58.6 C |

This is a meaningful implementation result. It is not a thesis win, because CrackMNIST is a current-frame crack-tip detection task and the fair baseline should be the DLR CNN/Williams-series detector.

## DLR Real Spatial Audit

Protocol:

- DLR `S_160_4.7`
- 835 real load stages, left/right labels = 1670 samples
- first 30% stages for thresholding/fine-tuning
- final 70% stages for testing

| model | target | pixel F1 | IoU | top-k IoU | tip err px256 | frontier err px256 | hit <=4px |
|---|---|---:|---:|---:|---:|---:|---:|
| `tiny_unet_crackmnist_cold_transfer` | tip | 0.0065 | 0.0033 | 0.0020 | 101.25 | 120.11 | 0.000 |
| `tiny_unet_dlr_front30_finetune_tip` | tip | 0.0572 | 0.0294 | 0.0185 | 65.56 | 77.30 | 0.020 |
| `tiny_unet_dlr_front30_finetune_path_tip` | path_tip | 0.4098 | 0.2577 | 0.2732 | 91.93 | 130.72 | 0.000 |
| `paris_lefm_straight_tip_extrapolator` | path_tip | 0.3877 | 0.2405 | n/a | 5.95 | n/a | 0.405 |

Readout:

- CNN path IoU slightly beats the simple straight-line geometry baseline.
- CNN crack-tip localization is much worse than the geometry baseline.
- The result is not a real-data crack-tip victory.

## DLR Next-Stage Forecast Audit

True forecast framing:

| model | mean next-tip err px256 | p90 | hit <=4px | readout |
|---|---:|---:|---:|---|
| `persistence_current_tip` | 0.1666 | 1.0000 | 1.000 | saturated trivial baseline |
| `last_delta_extrapolation` | 0.3332 | 1.0000 | 1.000 | also saturated |
| `xgb_temporal_dic_next_tip_v1` | 0.1847 | 0.6686 | 1.000 | does not beat persistence |

Readout:

DLR `S_160_4.7` one-step forecast is too easy. The tip often moves 0-1 px per stage, so persistence already saturates the metric. This does not validate event-hazard forecasting.

## Why Archive

The project is valuable as code and negative evidence, but not as the originally hoped thesis:

- real curve data supports only 1D curve forecasting, not spatial hazard;
- real DIC data supports current-frame detection, not future event prediction;
- DLR one-step forecast is saturated by persistence;
- the missing real AE/event table remains the key blocker for hazard validation.

## Open-Source Boundary

Commit:

- code under `crackle/`,
- tests under `tests/`,
- reports under `reports/`,
- `requirements-crackle.txt`,
- `LICENSE`,
- README pointers.

Do not commit:

- external datasets,
- model weights,
- `.pt/.pth` checkpoints,
- local G-drive run artifacts,
- API keys or private notes.

## Suggested Repository Description

```text
Research archive for crack-event forecasting and DIC crack-tip localization experiments, including CUDA baselines, real-data audits, and honest negative results.
```

## Final Claim

```text
Crackle is an auditable fracture-AI research archive. It provides reproducible baselines and clear negative results, but it does not yet validate a real-data hazard-forecasting breakthrough.
```

