# Heterogeneous Pinning Full-CL Final Readout

Date: 2026-06-06

Run:

`<workspace>\crackle_runs\hetero_pinning_xdemcore_fullcl_thr02_20260606`

## 1. Scope

This is the completed bounded experiment for the heterogeneity-pinning rollout-cut hypothesis.

Matrix:

- correlation lengths: `small, medium, large`
- contrasts: `1.0, 1.5, 2.0, 3.0, 5.0`
- seeds: `3`
- cases: `45`
- metric rows: `135`
- path metric: path-band, `damage_threshold=0.2`
- device: CUDA
- duration: `1102.654 s`

Claim boundary:

- Reference is still a dense incremental bond-break proxy, not high-fidelity fracture truth.
- `G1/G2` are X-DEM-INR-style one-shot measurement prototypes, not validated solvers.
- This run answers the controlled hypothesis direction; it does not claim to beat traditional fracture methods.

## 2. Top-Level Gate

The script-level mixed-CL gate says:

```text
G0_sanity = pass
G1_gap_vs_contrast = pass
```

Mixed over all correlation lengths:

| model | RMS slope | RMS c=1 | RMS c=5 | readout |
|---|---:|---:|---:|---|
| `G1_pure_global` | -0.0193 | 0.8368 | 0.7677 | improves when CLs are pooled |
| `G2_global_irreversible` | -0.0190 | 0.8406 | 0.7748 | improves when CLs are pooled |

This mixed pass is real, but it is not enough. The decisive check is per correlation length.

## 3. Per-Correlation-Length Verdict

| corr length | model | RMS slope | RMS c=1 | RMS c=5 | IoU c=1 | IoU c=5 | improves? |
|---|---|---:|---:|---:|---:|---:|---|
| small | `G1_pure_global` | -0.0190 | 0.8412 | 0.7831 | 0.8349 | 0.8379 | yes, slight |
| small | `G2_global_irreversible` | -0.0219 | 0.8406 | 0.7756 | 0.8326 | 0.8370 | yes, slight |
| medium | `G1_pure_global` | -0.0444 | 0.8346 | 0.6459 | 0.8358 | 0.8593 | yes |
| medium | `G2_global_irreversible` | -0.0506 | 0.8406 | 0.6397 | 0.8326 | 0.8593 | yes |
| large | `G1_pure_global` | +0.0055 | 0.8346 | 0.8742 | 0.8358 | 0.8344 | no |
| large | `G2_global_irreversible` | +0.0156 | 0.8406 | 0.9091 | 0.8326 | 0.8232 | no |

Readout:

- `medium` correlation length supports the pinning intuition most clearly.
- `small` improves slightly in this rerun, but the gain is modest.
- `large` correlation length fails the hypothesis: RMS worsens and IoU does not improve.

The originally expected "large CL should be strongest" did not appear.

## 4. Rollout Still Wins In Absolute Accuracy

At contrast `5.0`:

| corr length | `A_active_rollout` RMS | best one-shot RMS | winner |
|---|---:|---:|---|
| small | 0.4093 | 0.7756 | rollout |
| medium | 0.3240 | 0.6397 | rollout |
| large | 0.5673 | 0.8742 | rollout |

At contrast `1.0`:

| corr length | `A_active_rollout` RMS | best one-shot RMS | winner |
|---|---:|---:|---|
| small | 0.2718 | 0.8406 | rollout |
| medium | 0.2718 | 0.8346 | rollout |
| large | 0.2718 | 0.8346 | rollout |

This matters more than the slope. Even where one-shot improves with contrast, it does not overtake the marched/active rollout proxy.

## 5. Final Decision

The full bounded experiment does not justify deleting rollout.

More precise verdict:

```text
Heterogeneous pinning exists partially in this proxy, strongest at medium correlation length.
It is not strong enough, nor consistent enough across large correlation length, to replace rollout.
Rollout remains the accuracy winner in every tested correlation-length regime.
```

By the user's rule:

```text
large CL also fails -> hypothesis dies as a full no-rollout thesis -> choose P2-B
```

So the next architecture direction is:

```text
P2-B selective rollout:
  global one-shot field as initialization / prior
  local incremental rollout only in divergence-map regions
```

Do not continue with:

```text
P2-A fully amortized no-rollout field
```

unless a future, higher-fidelity reference changes this result.

## 6. Files Produced

Main run files:

- `hetero_pinning_case_metrics.csv`
- `hetero_pinning_aggregate.csv`
- `hetero_pinning_by_corr_aggregate.csv`
- `hetero_pinning_by_corr_slopes.csv`
- `hetero_pinning_gate_report.json`
- `hetero_pinning_manifest.json`
- `hetero_pinning_report.md`
- `figures/gap_vs_contrast.svg`
- `figures/drift_curve.svg`

Auxiliary report:

- `reports/hetero_pinning_xdem_asset_audit_20260606.md`


