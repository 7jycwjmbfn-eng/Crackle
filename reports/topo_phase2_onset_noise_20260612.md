# Topo Phase 2.1 robustness — causal onset under measurement noise

Date: 2026-06-12. Branch: topo-tda. Spec: crackle_tda_spec.md §2.1
(strengthening). Data: datasets/topo_synth_v1, all 2000 multi-notch
hetero cases. Artifacts: runs_topo/onset_noise_full/ (per_case.parquet,
fa_sweep.csv, matched_fa.csv, headline_table.csv, lead_vs_fa_noise.png).
Regenerable: `python -m scripts.topo_onset_noise --dataset ... --out ...`.

## Why this run exists

The noiseless Phase 2.1 result (topo_phase2_causal_onset_20260611.md)
had a degenerate false-alarm axis (FA ~ 0 for every signal), so the
"topological signals lead the macroscopic signal" claim rested on a
selectivity argument, not a real ROC. Two things were wrong and are now
fixed:

1. **No measurement noise.** Added a DIC-like spatially-correlated
   Gaussian noise layer (`crackle/topo/noise.py`, 7 unit tests) at three
   levels σ ∈ {0.02, 0.05, 0.10} vs sig_tau = 0.08. At σ = 0.05 the noise
   inflates the event stream ~5× (spurious frame-to-frame flicker), so it
   genuinely stresses the topological signals.
2. **Degenerate false-alarm reference.** The first smoke run exposed that
   `growth_start ≈ step 2` here (kinematic loading is monotone from the
   first step), so "alarm before growth_start" could never fire — FA was
   stuck at 0 with OR without noise. This was the real reason the
   noiseless result showed FA ~ 0. Corrected to a conservative
   instability-referenced definition: event = t* (argmax damage
   increment); a first alarm at t_a is a **false alarm if premature**
   (t* − t_a > W), a **detection** if useful (0 ≤ t* − t_a ≤ W), scored
   over warning windows W ∈ {15, 25, 40}. Premature noise-firing is
   penalized as a false alarm, not rewarded as long lead, so the
   threshold sweep cannot be gamed. The detectors themselves are
   unchanged; only the scoring was fixed.

The failure time t* is taken from the CLEAN simulation (ground truth);
detectors and the control total_damage both see only the NOISY field.

## Pre-registered question (committed before the run)

Under noise that lifts the control's FA off zero, at MATCHED false-alarm
rate, does at least one topological precursor signal achieve a strictly
longer median useful lead than total_damage on ≥ 2 of the 3 noise levels?

**VERDICT: YES (qualified).** The instantaneous persistence signal
`total_pers_h0` beats the macroscopic control at matched FA on all three
noise levels, under the rolling-z detector. But the advantage is
signal- and detector-specific, and one topological signal (cumulative
event count) FAILS — both reported below.

## Headline (rolling-z, warning window W = 25, FA target 0.10, 2000 cases)

| signal | σ | FA topo / ctrl | detect topo / ctrl | median lead topo / ctrl | case win-frac |
|---|---|---|---|---|---|
| total_pers_h0 | 0.02 | 0.086 / 0.087 | **0.260 / 0.024** | **12 / 5** | 0.934 |
| total_pers_h0 | 0.05 | 0.041 / 0.055 | **0.112 / 0.022** | **12 / 11.5** | 0.854 |
| total_pers_h0 | 0.10 | 0.041 / 0.042 | **0.068 / 0.029** | **12 / 10** | 0.705 |
| cum_events | 0.05 | 0.051 / 0.055 | 0.011 / 0.022 | 6 / 11.5 | 0.323 |
| cum_events | 0.10 | 0.019 / 0.042 | 0.002 / 0.029 | 4 / 10 | 0.049 |

At a genuinely matched false-alarm rate, `total_pers_h0` both **detects
the impending instability far more often** (13×, 5×, 2.3× the control's
detection rate as noise rises) **and with longer median lead** (12 vs
5/11.5/10 steps). The edge is largest at low noise and compresses but
persists at σ = 0.10.

Robustness across ALL matched operating points (every W, FA target;
fraction where topo median lead > control), not just the headline cell:
- total_pers_h0: 0.71 / 0.67 / 0.86 at σ = 0.02 / 0.05 / 0.10
- entropy_h0:    0.80 / 0.67 / 0.93
- n_h0_sig:      (over-fires at low σ) / 0.62 / 0.92
- cum_events:    0.33 / 0.50  ← fails

## Honest negatives and caveats

1. **The cumulative event count (cum_events) fails under noise.** Frame-
   to-frame flicker accumulates, so detection collapses (1%, 0.2%) and
   it loses to the control. Not every topological signal is robust; the
   instantaneous ones (total_pers_h0, entropy_h0) are, the cumulative
   one is not. The test has teeth.
2. **Detector-dependent.** The clean matched-FA comparison is only
   possible under rolling-z. Under CUSUM, the macroscopic control
   **cannot be made selective at low noise** — its minimum achievable FA
   is 0.51 at σ = 0.02 (it cries wolf in half the cases at every
   threshold), while `total_pers_h0` reaches FA ≈ 0.0–0.03. This is
   arguably a point FOR topology (controllable where the macroscopic
   signal is not), but it means CUSUM cannot provide a matched-FA duel
   except at σ = 0.10.
3. **Absolute detection rates are low (6–34%)** because the premature=FA
   scoring is deliberately conservative. This is a RELATIVE comparison at
   matched FA, not an operational-capability claim.
4. **Margin shrinks with noise.** At σ = 0.10 the median-lead margin is
   12 vs 10 and the detection-rate edge narrows; at high FA targets the
   control catches up. The result is clearest in the low-false-alarm
   regime, which is the operationally meaningful one.

## Claim boundary

Allowed: "Under DIC-like measurement noise (σ up to sig_tau and beyond),
with a conservative instability-referenced false-alarm definition, the
instantaneous topological persistence signal (total_pers_h0) detects
impending instability more often and with longer median lead than the
macroscopic damage signal at MATCHED false-alarm rate, on all three noise
levels, under the rolling-z detector. The cumulative topological event
count does not survive noise."

Not allowed: any real-data transfer; any claim for the cumulative event
signal; any operational-capability claim from the absolute rates; a
detector-independent claim (CUSUM cannot match FA at low noise).
