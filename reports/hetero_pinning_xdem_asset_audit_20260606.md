# Heterogeneous Pinning Rollout-Cut Audit

Date: 2026-06-06

Scope: first controlled measurement for the "cut autoregressive rollout" hypothesis in quasi-static brittle fracture. This is not a final fracture solver claim.

## 1. What Was Actually Reused From Old X-DEM-INR Assets

The user pointed to the prior X-DEM-INR/DEM-INR asset family under `<prior-local-asset-root>`. I inspected the relevant documents, code, and run reports before trusting the new experiment.

Most relevant old assets:

- `<prior-local-asset-root>\<prior-project>-recovered-local\workflow\research\dem_inr_training_plan.md`
- `<prior-local-asset-root>\cloud_upload_20260524_2235\<prior-project-workspace>\<prior-project>\workflow\research\x_dem_inr_implementation_plan.md`
- `<prior-local-asset-root>\cloud_upload_20260524_2235\<prior-project-workspace>\<prior-project>\workflow\research\ast_transdem_implementation_plan_2026-05-24.md`
- `<prior-local-asset-root>\<prior-project>-recovered-local\workflow\ml\x_dem_inr_2d_smoke.py`
- `<prior-local-asset-root>\cloud_upload_20260524_2235\<prior-project-workspace>\<prior-project>\workflow\ml\x_dem_inr_p2_hybrid_trainer.py`
- `<prior-local-asset-root>\cloud_upload_20260524_2235\<prior-project-workspace>\<prior-project>\workflow\ml\x_dem_inr_evaluator.py`
- `<prior-local-asset-root>\cloud_results\20260525_final_xdem_pno_42773\xdem_true_dem_p3_rar_cont_report.json`

Key reused implementation ideas:

- Exact fixed-boundary residual ansatz: `u = u_analytic + D_fixed * residual`, adapted from old `XDEMINR.forward()`.
- Multiresolution hash-grid features plus sine/SIREN-style worker, instead of a plain MLP.
- Autograd-based strain and damage gradients through coordinates.
- Energy integration as explicit `sum(dV_i * energy_density_i)`.
- Coordinate-wise heterogeneous field conditioning, not a global latent average.

What was not reused as a claim:

- The old structural L-bracket/X-DEM-INR results are not crack propagation validation.
- Old `external_work_weight != 1` reports were treated as numerical exploration, not physical proof.
- The new experiment does not claim to beat FEM, PD, phase-field, or DLR/CNN baselines.

## 2. New Code Added

Main script:

`crackle/experiments/hetero_pinning.py`

It implements:

- Controlled single-edge-notch heterogeneous toughness sweep.
- Dense marched reference proxy using incremental bond-break mechanics.
- One-shot DEM/INR variants:
  - `G1_pure_global`
  - `G2_global_irreversible`
- Active rollout proxy baseline:
  - `A_active_rollout`
- Metrics:
  - `crack_path_iou`
  - `crack_path_hausdorff_mm`
  - `crack_path_rms_err_mm`
  - `nucleation_location_err_mm`
  - `energy_gap_vs_reference`
  - drift curve for rollout
- Outputs:
  - case CSV
  - aggregate CSV
  - gate JSON
  - markdown report
  - SVG plots

Implemented guardrails from the work order:

- H1: explicit volume-weighted energy integral.
- H2: no detached coordinate path for serious energy gradients.
- H3: `G_c(x)` enters coordinate-wise.
- H4: smooth notch seed/opening, no hard min/max in the differentiable path.
- H5: no static FEM/PD anchor mix in this DEM optimization.
- H6: bbox/origin/unit/range are written into manifest.

## 3. Important Claim Boundary

The marched reference in this experiment is a dense incremental bond-break proxy, not a high-fidelity fracture truth.

The one-shot solver is a DEM-style phase-field coordinate network used as a measurement instrument. It is not a validated new fracture solver.

The downloaded real fracture datasets are not used in this controlled heterogeneity sweep. They are useful for later sim-to-real validation, DIC/crack-tip/path checks, AE hazard validation if event tables are available, and Paris-style curve baselines. They cannot replace the controlled `G_c(x)` contrast sweep because the hypothesis needs known heterogeneity strength.

## 4. Runs And Results

### 4.1 Abandoned Probe

Path:

`<workspace>\crackle_runs\hetero_pinning_v1_topk_full_20260606`

Status:

- Stopped after the user pointed to old X-DEM-INR assets.
- It had partial rows only.
- Do not use this run for conclusions.

### 4.2 Strict Crack-Core Run

Path:

`<workspace>\crackle_runs\hetero_pinning_xdemcore_smallmatrix_20260606`

Settings:

- Contrast: `1.0, 1.5, 2.0, 3.0, 5.0`
- Correlation length: `small`
- Seeds: `3`
- Path threshold: `damage_threshold = 0.5`

Verdict:

- `G0_sanity = false`
- This run is too strict for the phase-field output and cannot support a scientific conclusion.

Selected aggregate RMS error:

| model | contrast 1.0 | contrast 5.0 |
|---|---:|---:|
| `A_active_rollout` | 0.522 mm | 3.205 mm |
| `G1_pure_global` | 20.398 mm | 18.413 mm |
| `G2_global_irreversible` | 11.700 mm | 11.915 mm |

Interpretation:

The strict core metric shows the global field is not matching the discrete crack core. Since G0 fails, this is an implementation/measurement failure mode, not evidence for or against the physics hypothesis.

### 4.3 Path-Band Run

Path:

`<workspace>\crackle_runs\hetero_pinning_xdemcore_smallmatrix_thr02_20260606`

Settings:

- Contrast: `1.0, 1.5, 2.0, 3.0, 5.0`
- Correlation length: `small`
- Seeds: `3`
- Path threshold: `damage_threshold = 0.2`

Gate report:

```json
{
  "G0_sanity": true,
  "G1_gap_vs_contrast": false,
  "G1_pure_global_slope": 0.03358089992354808,
  "G2_global_irreversible_slope": 0.0296378343609078
}
```

Aggregate readout:

| model | contrast | path IoU | RMS err mm |
|---|---:|---:|---:|
| `A_active_rollout` | 1.0 | 0.9176 | 0.6459 |
| `A_active_rollout` | 5.0 | 0.8099 | 0.8061 |
| `G1_pure_global` | 1.0 | 0.7789 | 1.2207 |
| `G1_pure_global` | 5.0 | 0.6403 | 1.3419 |
| `G2_global_irreversible` | 1.0 | 0.7759 | 1.2187 |
| `G2_global_irreversible` | 5.0 | 0.6350 | 1.3333 |

Interpretation:

G0 passes under path-band scoring, so this run can be read as a controlled measurement.

The key hypothesis is not supported in this setting. If heterogeneity pinned the path strongly enough to make one-shot global DEM more valid, path error should decrease as contrast rises. Instead:

- `G1_pure_global` RMS slope is positive.
- `G2_global_irreversible` RMS slope is positive.
- IoU declines from contrast `1.0` to `5.0`.

So, for this proxy and small-correlation-length sweep, stronger heterogeneity did not rescue one-shot global solving. It made the one-shot solution less aligned with the marched reference.

## 5. Current Scientific Verdict

Do not claim:

- one-shot global DEM replaces rollout;
- heterogeneity pinning is validated;
- this beats traditional fracture methods;
- this beats high-fidelity PD/FEM/phase-field;
- this is ready for publication as a solver.

Allowed claim:

The controlled first cut did not support the heterogeneity-pinning rollout-removal hypothesis. Under a path-band metric where the one-shot field passes sanity, the global solution's path error does not decrease with toughness contrast. This suggests path dependence survives strong heterogeneity in the tested proxy.

## 6. Hardware / Pipeline Readout

The run used CUDA for the DEM/INR optimization path, but each case is still small and sequential. GPU memory was modest and utilization was not the limiting proof point.

Pipeline lesson:

- For real throughput, batch multiple cases and larger grids together.
- For this hypothesis, it was more valuable to get the gate readout than to burn hours on a larger version of a negative trend.
- If continuing, scale only after adding medium/large correlation lengths and a more faithful PD/phase-field reference.

## 7. Recommended Next Decision

Best next path if the project continues:

1. Do not open Phase 2-A amortized no-rollout field yet. Phase 1 did not pass.
2. If continuing, run Phase 1 again with:
   - medium and large correlation lengths;
   - stronger/more faithful dense PD or phase-field reference;
   - batched GPU cases;
   - fixed compute budget and `>=3` seeds.
3. If the negative trend survives, pivot to Phase 2-B only:
   - global field as initialization;
   - selective rollout only in divergence-map regions.

Most honest current project conclusion:

The old X-DEM-INR assets are useful as a neural variational field scaffold, but the new crack experiment does not yet justify deleting rollout. The cleanest direction is selective rollout, not full rollout removal.

