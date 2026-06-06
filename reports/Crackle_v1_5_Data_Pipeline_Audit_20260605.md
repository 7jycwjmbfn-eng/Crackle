# Crackle v1.5 Data and Pipeline Audit

Date: 2026-06-05

## 0. Bottom Line

This run found a real improvement, but also clarified the boundary.

- Best balanced crack event model after this pass: `crackle_cox_fast_ranker_v1`, full-train BCE-only, `active_topk=1024`.
- It beats `cox_discrete_time_v1` on event recall and location error while keeping wall-clock speedup over dense PD.
- Full `crackle_cox_mechanics_ranker_v1` is still the quality ceiling, but it is slower than dense PD in the current CPU/Numpy feature path.
- Downloaded public crack image datasets are useful for spatial crack mask/path validation and pretraining, not for hazard/event-process training.
- No currently downloaded public dataset contains the valuable `(event_time, location, energy)` crack event table needed to validate hazard directly.

## 1. External Data Kept and Deleted

Root:

```text
G:\GaussMoE_Workspace\external_datasets
```

Total retained local external files:

```text
60 files, about 5.92 GB
```

### Kept: segmentation / mask data

These are useful for sim-to-real spatial validation, crack path mask metrics, segmentation pretraining, and visual QA.

| dataset | rows | role |
|---|---:|---|
| `rimvydasrub/crackseg9k` | 9,159 | primary segmentation |
| `varcoder/crack-segmentation-dataset` | 11,298 | primary segmentation |
| `fcakyon/crack-instance-segmentation` | 433 | instance segmentation auxiliary |
| `kniemiec/crack-segmentation` | 32 | segmentation mask |
| `rishitunu/ECC_crackdataset_withsplit` | 1,289 | segmentation mask |
| `rishitunu/ecc_crackdetector_dataset_exhaustive` | 1,289 | segmentation mask |

Subtotal:

```text
23,500 mask/segmentation rows
```

### Kept: classification / image-only data

These are useful for crack/non-crack pretraining, domain hard negatives, and image QA. They are not sufficient for event hazard.

| dataset | rows | role |
|---|---:|---|
| `mohammadnajeeb/concrete_crack_images` | 40,000 | classification auxiliary |
| `Taki3d/CrackDetection` | 11,025 | classification auxiliary |
| `aliasghar-j/concrete-crack-dataset` | 273 | small classification auxiliary |
| `Manshika13/concrete-crack-dataset-v2` | 1,600 | classification auxiliary |
| `ShawnYang2001/UAV-Crack` | 390 | classification auxiliary |
| `ZhiyaYang/sewer-defect-crack-dataset` | 585 | classification auxiliary |
| `xcll/CRACK500_testdata` | 400 | classification auxiliary |
| `xcll/crack500_and_deepcrack` | 724 | classification auxiliary |

Subtotal:

```text
54,997 classification/image-only rows
```

### Deleted as useless residue

Verified each path was under `G:\GaussMoE_Workspace\external_datasets` before deletion.

```text
G:\GaussMoE_Workspace\external_datasets\crack_public_hf_20260605\.python_pkgs
G:\GaussMoE_Workspace\external_datasets\crack_public_hf_20260605\raw_parquet\crackedcity__nvidia-hackathon-dataset
G:\GaussMoE_Workspace\external_datasets\crack_public_hf_more_20260605\raw_parquet\rievil__crackenpy_dataset
```

Reasons:

- `.python_pkgs`: local dependency residue, not dataset.
- `crackedcity/nvidia-hackathon-dataset`: manifest/index-only row, no usable raw data.
- `rievil/crackenpy_dataset`: only a `401 Unauthorized` parquet manifest, no downloaded parquet data.

## 2. Data Use Verdict

Current external public downloads:

```text
Good for:
  spatial mask/path validation
  segmentation pretraining
  crack/non-crack image pretraining
  domain shift checks

Not good for:
  event hazard training
  Hawkes/point-process validation
  crack propagation timing
  acoustic energy event modeling
```

The valuable next data types are:

1. AE event data with event time, sensor/location, energy/amplitude. This can validate or fine-tune hazard.
2. DIC full-field strain/displacement data. This can validate mechanics covariates and spatial field alignment.
3. Fatigue crack growth curves with cycle count and crack length. This is suitable for the Paris-law competitor head-to-head, but not spatial hazard.

Useful public leads checked:

- Zenodo DIC fatigue crack growth full-field displacement/strain dataset: https://zenodo.org/record/5740216
- Zenodo bonded composite fatigue crack growth data with DIC pictures and processed crack length/growth sheets: https://zenodo.org/records/10895431
- NASA fatigue crack growth in aluminum lap joint data: https://data.nasa.gov/dataset/fatigue-crack-growth-in-aluminum-lap-joint/resource/912c3d05-f207-41dc-8d8d-347c8e04bc98
- NIST acoustic emission slow crack growth paper context: https://www.nist.gov/publications/detectability-slow-crack-growth-bridge-seels-acoustic-emission
- CrackSeg9k paper: https://arxiv.org/abs/2208.13054
- CrackSeg9k Hugging Face page: https://huggingface.co/datasets/rimvydasrub/crackseg9k

## 3. Code Changes

Patched:

```text
crackle/eval/eval_crackle.py
crackle/eval/active_sweep_v2.py
crackle/eval/wallclock_benchmark.py
crackle/train/train_crackle.py
```

Implemented:

- Added `crackle_cox_fast_ranker_v1`.
- It uses the 17-dimensional `HYBRID_FEATURE_NAMES` path instead of the slower 35-dimensional mechanics ranker feature path.
- Added LogisticHazard object caching in evaluation.
- Added batched Logistic active-sweep inference, one sample's step features predicted together.
- Repaired active sweep total wall-clock accounting to include sample-level preprocessing/history overhead.
- Replaced precomputed full `hawkes_history_by_step` lists in active sweep/wallclock benchmark with online causal history update.

The online update is causal:

```text
score step t with history from <= t
then update history using the observed event at t
use updated history at t+1
```

## 4. Tests

Commands run:

```bash
python -m py_compile crackle/eval/eval_crackle.py crackle/eval/active_sweep_v2.py crackle/eval/wallclock_benchmark.py crackle/train/train_crackle.py
python -m unittest tests.test_crackle_contract -v
```

Result:

```text
6 tests passed
```

Important contract tests still passing:

- no future-label access for deterministic ranker
- shuffled future labels do not change causal features
- ranker features are causal and include regime
- riskset contains positive and censored bonds

## 5. Training Runs

Data:

```text
G:\GaussMoE_Workspace\crackle_runs\crackle_v1_2_hard64_20260605\data
dataset_hash = 900775233e6adaff
64 total cases
52 train cases
6 test cases
```

Full fast BCE-only training:

```text
model = crackle_cox_fast_ranker_v1
seeds = 20260605, 20260606, 20260607
train rows per seed = 1,797,887
positive rows per seed = 86,299
negative rows per seed = 1,711,588
rank_weight = 0.0
pairwise_rows = 0
```

Main output:

```text
G:\GaussMoE_Workspace\crackle_runs\crackle_v1_5_fast_ranker_20260605\sweep_full_3seed_online_history\crackle_active_sweep_v2.csv
```

## 6. Final 3-Seed Active Sweep

Primary comparison uses `prob_threshold = none`, meaning fixed `top_k=64` event selection inside the active set.

### active_topk = 512

| model | recall mean | loc err mean | total ms | speedup vs dense |
|---|---:|---:|---:|---:|
| `cox_discrete_time_v1` | 0.9070 ± 0.0007 | 0.6054 ± 0.0040 | 3.9997 ± 0.0408 | 1.3159 ± 0.0134 |
| `crackle_cox_fast_ranker_v1` | 0.9461 ± 0.0223 | 0.2553 ± 0.0738 | 4.0094 ± 0.0528 | 1.3128 ± 0.0173 |
| `crackle_cox_mechanics_ranker_v1` | 0.9681 ± 0.0001 | 0.1896 ± 0.0010 | 6.4908 ± 0.5277 | 0.8159 ± 0.0628 |

### active_topk = 1024

| model | recall mean | loc err mean | total ms | speedup vs dense |
|---|---:|---:|---:|---:|
| `cox_discrete_time_v1` | 0.9070 ± 0.0007 | 0.6054 ± 0.0040 | 4.1495 ± 0.0304 | 1.2683 ± 0.0093 |
| `crackle_cox_fast_ranker_v1` | 0.9624 ± 0.0031 | 0.2007 ± 0.0121 | 4.3186 ± 0.1099 | 1.2194 ± 0.0305 |
| `crackle_cox_mechanics_ranker_v1` | 0.9681 ± 0.0001 | 0.1898 ± 0.0010 | 6.7069 ± 0.1163 | 0.7849 ± 0.0135 |

## 7. Current Model Choice

Recommended main spine now:

```text
crackle_cox_fast_ranker_v1
active_topk = 1024
rank_weight = 0.0
```

Reason:

- Beats Cox recall by about 5.5 points at active_topk 1024.
- Cuts next-event location error from about 0.605 to about 0.201.
- Keeps wall-clock speedup above dense PD in the current corrected active-sweep harness.
- Gets close to the full mechanics ranker while avoiding most of its feature overhead.

Quality oracle / ablation:

```text
crackle_cox_mechanics_ranker_v1
```

Reason:

- Highest recall and best location error.
- Still too slow for the main speed claim.

Do not choose as main:

```text
cox_discrete_time_v1
```

Reason:

- Fast, but much worse location error and recall.

## 8. Traditional / Competitor Position

Do not claim this replaces Paris/NASGRO/FEM/PD at equal physical fidelity.

Allowed claim after current runs:

```text
On the local hard synthetic PD benchmark, the fast causal event ranker improves event recall and next-event localization over Cox while retaining wall-clock speedup against the recorded dense-PD step time.
```

Against Paris-law style competitors, our advantage is not 1D curve fitting on clean da/dN. Paris-style methods are strong there. Our advantage should be claimed only where we report evidence:

- spatial next-event localization
- branching/hole/heterogeneous/arrest regimes
- active-set wall-clock reduction
- full-domain event recall

Therefore, the next real head-to-head should use:

```text
front 30 percent known -> forecast remaining 70 percent
curve_forecast_rmse for shared 1D crack-length comparison
plus our extra spatial metrics: next_event_loc_err, branch_detect_f1, arrest_rate_error
```

## 9. Next Step

Immediate next work:

1. Add regime-wise aggregation to this final `sweep_full_3seed_online_history` table.
2. Download one DIC fatigue crack growth dataset and convert it to spatial validation format.
3. Download one fatigue crack growth curve dataset for Paris competitor head-to-head.
4. Keep searching specifically for AE event tables. Do not use AE spectrogram-only data as point-process ground truth unless timestamps/events are recoverable.
5. If speed matters more, optimize feature construction next. Current 1024 fast path still spends about:

```text
active_selection_ms          = 0.9247
mechanics_feature_update_ms  = 0.6241
intensity_eval_ms            = 2.7441
total_step_ms                = 4.3186
```

The next performance target is `intensity_eval_ms`, not model training.
