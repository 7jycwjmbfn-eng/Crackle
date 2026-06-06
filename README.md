# Crackle

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-research%20archive-lightgrey.svg)](#claim-boundary)

Crackle is a research archive for fracture-AI experiments: crack-event ranking, DIC crack-tip localization, fatigue-curve forecasting, and controlled rollout-vs-one-shot fracture studies.

The archive is intentionally conservative. It preserves code, reports, and negative results so the work can be audited, reproduced, reused, or closed cleanly. It does not claim a real-data breakthrough in crack forecasting.

## What Is Inside

| Area | What is included |
|---|---|
| Event forecasting | Causal risk-set construction, Hawkes-style baselines, deterministic rankers, Cox/logistic survival heads, GBM hazard ranking. |
| Crack-tip localization | CrackMNIST DIC rule baselines, XGBoost pixel baseline, CUDA Tiny-UNet baseline. |
| Real-data audits | DLR DIC spatial transfer, first-30% fine-tuning, next-stage tip forecasting, NASA PHM crack-length curve benchmarks. |
| Rollout studies | Controlled heterogeneous-toughness experiments comparing active rollout against one-shot DEM/INR-style fields. |
| Reporting | Versioned markdown audit reports with metrics, failure modes, and claim boundaries. |

## Repository Layout

```text
crackle/
  baselines/      deterministic, Hawkes, Cox/logistic, GBM baselines
  benchmarks/     synthetic notched-plate helpers and compatibility entry points
  data/           event catalogs, risk sets, causal feature builders
  eval/           CrackMNIST, DLR, NASA curve, wall-clock, and head-to-head evaluators
  experiments/    heterogeneous pinning / rollout-cut experiments
  metrics/        fracture and point-process metrics
  train/          training entry points

reports/          audit reports and final readouts
tests/            lightweight contract tests
```

## Main Findings

### 1. CrackMNIST detection works, but it is not forecasting

The strongest positive engineering result is a CUDA Tiny-UNet on CrackMNIST 64L current-frame crack-tip detection:

| model | split | pixel F1 | pixel IoU | top-1 hit | tip error |
|---|---|---:|---:|---:|---:|
| `tiny_unet_crackmnist_cuda_v1` | test | 0.8008 | 0.6678 | 0.8922 | 1.33 px |

This is useful, but it is a detection task. It should not be presented as forward crack-event forecasting.

### 2. DLR real-data transfer did not validate the original thesis

On DLR `S_160_4.7`:

- cold CrackMNIST-to-DLR transfer failed;
- first-30% DLR fine-tuning improved path masks but not precise tip localization;
- one-step next-tip forecasting was saturated by a trivial persistence baseline.

The real-data result is therefore modest and mostly negative.

### 3. Full correlation-length rollout-cut study favors selective rollout

The controlled heterogeneous-toughness study tested:

```text
correlation length: small, medium, large
contrast:           1.0, 1.5, 2.0, 3.0, 5.0
seeds:              3
cases:              45
```

Readout:

- medium correlation length showed a real pinning signal;
- large correlation length did not;
- active rollout remained more accurate than one-shot fields in every tested regime.

Final direction:

```text
Do not delete rollout globally.
Use one-shot fields as a prior, then apply selective local rollout in divergence regions.
```

See `reports/hetero_pinning_fullcl_final_readout_20260606.md`.

## Installation

Create an environment with Python 3.10+:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[ml]
```

CUDA experiments need PyTorch installed for your local CUDA version. For example:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

Core tests:

```powershell
python -m unittest tests.test_crackle_contract
```

## Data

External datasets are not committed. Put downloaded data in a local folder such as:

```text
data/external/
```

Useful public sources:

- CrackMNIST: https://zenodo.org/records/18454958
- DLR DIC: https://zenodo.org/records/5740216
- NASA PHM 2019: https://c3.ndc.nasa.gov/dashlink/resources/1014/

The repository stores code and reports only. It intentionally excludes raw datasets, derived caches, checkpoints, and model weights.

## Example Commands

CrackMNIST CUDA baseline:

```powershell
python -m crackle.eval.crackmnist_cnn_baseline `
  --data-root .\data\external\crackmnist_cache `
  --out .\runs\crackmnist_64L_tiny_unet_cuda `
  --pixels 64 --size L --device cuda --epochs 18 --batch-size 512
```

DLR DIC spatial validation:

```powershell
python -m crackle.eval.dlr_cnn_spatial_validation `
  --dlr-zip .\data\external\dlr_dic\S_160_4.7.zip `
  --cache-path .\runs\dlr_s160_47_64_cache.npz `
  --checkpoint .\runs\crackmnist_64L_tiny_unet_cuda\tiny_unet_crackmnist_cuda_v1.pt `
  --out .\runs\dlr_s160_47_tiny_unet_front30 `
  --model-pixels 64 --device cuda
```

Heterogeneous pinning study:

```powershell
python -m crackle.experiments.hetero_pinning `
  --out .\runs\hetero_pinning_fullcl_thr02 `
  --corr-lengths small,medium,large `
  --contrasts 1.0,1.5,2.0,3.0,5.0 `
  --num-seeds 3 `
  --damage-threshold 0.2 `
  --device cuda
```

## Key Reports

- `reports/Crackle_Final_Archive_and_OpenSource_20260605.md`
- `reports/Crackle_v1_7_BigData_Run_20260605.md`
- `reports/Crackle_v1_8_DLR_Thesis_Correction_20260605.md`
- `reports/hetero_pinning_fullcl_final_readout_20260606.md`
- `reports/hetero_pinning_xdem_asset_audit_20260606.md`

## Claim Boundary

Allowed:

```text
Crackle provides reproducible crack DIC and crack-event benchmark code, including GPU baselines, real-data audits, and honest negative results.
```

Not allowed:

```text
Crackle beats DLR crack-tip detection.
Crackle beats traditional fracture forecasting on real data.
Crackle validates real hazard/event prediction without true event-time data.
Crackle proves one-shot fracture solving can replace rollout.
```

## License

MIT License. See [LICENSE](LICENSE).
