# Crackle Research Archive

Crackle is a fracture forecasting / crack-tip localization research archive inside this repository.

This archive is intentionally conservative: several experiments failed to support the original thesis. The code and reports are kept so the work can be audited, reproduced, reused, or closed cleanly.

## What This Contains

- Synthetic point-process / survival baselines for crack event ranking.
- NASA PHM crack-length curve forecasting benchmarks.
- CrackMNIST DIC crack-tip mask baselines:
  - DIC hand-written rules,
  - XGBoost CUDA pixel baseline,
  - CUDA Tiny-UNet baseline.
- DLR `S_160_4.7` real DIC spatial audits:
  - cold transfer,
  - first-30%-stage fine-tuning,
  - next-stage crack-tip forecasting.

## Main Finding

The project did not establish a robust real-data win for the original Crackle thesis.

The strongest positive result is a CUDA CNN baseline on CrackMNIST 64L:

| model | split | pixel F1 | pixel IoU | top-1 hit | tip err |
|---|---|---:|---:|---:|---:|
| `tiny_unet_crackmnist_cuda_v1` | test | 0.8008 | 0.6678 | 0.8922 | 1.33 px |

But this is current-frame DIC crack-tip detection, not forward crack-event forecasting, and it should be compared against the DLR/CNN/Williams-series baseline rather than weak hand-written DIC thresholds.

On DLR `S_160_4.7`, real spatial transfer remained disappointing:

- cold CrackMNIST-to-DLR transfer failed;
- front-30% DLR fine-tuning gave modest path IoU but poor crack-tip localization;
- one-step next-tip forecasting was saturated by the trivial persistence baseline.

## Reproduction Pointers

External data is not committed. Download data to a local workspace such as:

```text
G:\GaussMoE_Workspace\external_datasets\real_fracture_20260605
```

Important sources:

- CrackMNIST: https://zenodo.org/records/18454958
- DLR DIC: https://zenodo.org/records/5740216
- NASA PHM 2019: https://c3.ndc.nasa.gov/dashlink/resources/1014/

CUDA PyTorch was installed outside the repo:

```powershell
$py='C:\Users\hp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$target='G:\GaussMoE_Workspace\python_pkgs\torch_cuda_128'
$tmp='G:\GaussMoE_Workspace\python_pkgs\pip_tmp_torch_cuda'
$cache='G:\GaussMoE_Workspace\python_pkgs\pip_cache'
$env:TEMP=$tmp; $env:TMP=$tmp
& $py -m pip install --target $target --cache-dir $cache --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

Example runs:

```powershell
$env:PYTHONPATH='G:\GaussMoE_Workspace\python_pkgs\torch_cuda_128;G:\GaussMoE_Workspace\python_pkgs\crackmnist_2_0_1;G:\GaussMoE_Workspace\python_pkgs\crackle_v1_3;.'

python -m crackle.eval.crackmnist_cnn_baseline `
  --data-root 'G:\GaussMoE_Workspace\external_datasets\real_fracture_20260605\crackmnist_cache' `
  --out 'G:\GaussMoE_Workspace\crackle_runs\crackle_v1_9_final_20260605\crackmnist_64L_tiny_unet_cuda' `
  --pixels 64 --size L --device cuda --epochs 18 --batch-size 512

python -m crackle.eval.dlr_cnn_spatial_validation `
  --dlr-zip 'G:\GaussMoE_Workspace\external_datasets\real_fracture_20260605\dlr_dic_zenodo_5740216\S_160_4.7.zip' `
  --cache-path 'G:\GaussMoE_Workspace\crackle_runs\crackle_v1_8_dlr_spatial_20260605\dlr_s160_47_64_cache.npz' `
  --checkpoint 'G:\GaussMoE_Workspace\crackle_runs\crackle_v1_9_final_20260605\crackmnist_64L_tiny_unet_cuda\tiny_unet_crackmnist_cuda_v1.pt' `
  --out 'G:\GaussMoE_Workspace\crackle_runs\crackle_v1_9_final_20260605\dlr_s160_47_tiny_unet_front30' `
  --model-pixels 64 --device cuda
```

## Reports

- `reports/Crackle_v1_7_BigData_Run_20260605.md`
- `reports/Crackle_v1_8_DLR_Thesis_Correction_20260605.md`
- `reports/Crackle_Final_Archive_and_OpenSource_20260605.md`

## Claim Boundary

Allowed:

```text
This archive provides reproducible crack DIC and crack-event benchmark code, including GPU baselines and honest negative results.
```

Not allowed:

```text
Crackle beats DLR crack-tip detection.
Crackle beats traditional fracture forecasting on real data.
Crackle validates real hazard/event prediction without true event-time data.
```
