# Topo Phase 2.1 — causal onset detection (2000-case multi-notch world)

Date: 2026-06-11. Branch: topo-tda. Spec: crackle_tda_spec.md §2.1.
Data: datasets/topo_synth_v1 (2000 cases, 48×29×81, uint8 shards).
Detectors: rolling z and one-sided CUSUM (k=0.5 and k=2.0) on causally
standardized values — trailing 10-step window, no max-normalization, no
future access. Signals: n_h0_sig, entropy_h0, total_pers_h0, cum_events;
control: total_damage. Artifacts: runs_topo/phase2_onset_full/
(sweep.csv, duel_per_threshold.csv, per_case.parquet, lead_vs_fa.png;
regenerable via `python -m scripts.topo_causal_onset`).

## Evaluation conventions

- t* = argmax damage increment; growth_start = first run of 3 consecutive
  positive damage increments; alarm before growth_start = false alarm;
  lead = t* − first alarm (non-false alarms only).
- In this NOISELESS kinematic world false alarms are structurally ~0 for
  every signal and setting, so the lead-vs-FA curve degenerates and the
  detector threshold (selectivity) becomes the operating axis. Both z and
  standardized CUSUM are scale-free, so topo-vs-control comparisons are
  made at the IDENTICAL (detector, threshold) setting ("duel"). Real
  noisy data (Phase 3) will repopulate the FA axis.

## Result: a selectivity crossover, not a uniform winner

Permissive regime (z ≤ 4; CUSUM k=0.5 any h; CUSUM k=2 h ≤ 5):
the control wins every duel (win_frac ≈ 0). But its alarm is degenerate:
at z=1.5, 97.6% of cases alarm within 3 steps of growth_start — the
detector recognizes that LOADING has begun, in every case, and its
median "lead" of 47 steps is just t* − growth_start, the trivial upper
bound available to any detector that fires immediately. A monotone ramp
holds a standardized z of ~1.9 against its own trailing window, so any
threshold below that fires on the ramp itself; CUSUM with drift k=0.5
accumulates ~1.4/step on a ramp and crosses every swept h the same way.

Acceleration-selective regime (z ≥ 5; CUSUM k=2 h ≥ 12) — the regime
that actually asks "is the process accelerating beyond its own recent
behavior?":

| setting | topo signal | alarm rate topo/ctrl | median lead topo/ctrl | win frac |
|---|---|---|---|---|
| z=5 | n_h0_sig | 0.95 / 0.20 | 23 / 44 | 0.82 |
| z=6 | n_h0_sig | 0.94 / 0.05 | 23 / 25 | 0.97 |
| z=6 | cum_events | 1.00 / 0.05 | 32 / 25 | 0.97 |
| z=8 | n_h0_sig | 0.93 / 0.01 | 24 / 16 | 1.00 |
| z=8 | cum_events | 0.99 / 0.01 | 32 / 16 | 1.00 |
| cusum k=2, h=12 | cum_events | 0.99 / 0.34 | 32 / 8 | 0.96 |
| cusum k=2, h=16 | n_h0_sig | 0.93 / 0.16 | 24 / 3 | 0.99 |

The macroscopic curve is too smooth to produce acceleration alarms: its
detection rate collapses (0.2–20%) and where it does fire, leads shrink
to 0–25 steps. The topological signals — which quantize damage-field
reorganization into discrete jumps — keep firing in 47–100% of cases
with median leads of 23–37 steps. Win fractions 0.81–1.00 clear the
pre-stated 2/3 bar at every acceleration-selective setting, for all four
topo signals, under two detector families.

Among topo signals: cum_events is the most reliable (≥99% detection at
every setting, stable 32-step median lead); n_h0_sig detects ~93–96%
with ~24-step leads; entropy_h0 and total_pers_h0 trade detection rate
for slightly longer leads.

## Claim boundary

Allowed: "Under causal, acceleration-selective change detection at
matched detector settings (equal false-alarm rate, both ~0 in this
noiseless world), topological event/summary signals alarm in ≥93% of
heterogeneous multi-notch cases with median leads of 23–37 steps before
macroscopic instability, while the total-damage control alarms in ≤20%
of cases — satisfying the pre-stated ≥2/3 dominance criterion."

Also allowed (and required for honesty): "At permissive settings the
macroscopic signal alarms earlier in every case; that alarm is
ramp-following (97.6% within 3 steps of growth onset) and carries no
case-specific timing information."

Not allowed: any transfer to noisy/real data (FA axis untested there);
any claim about event-level prediction (that is Phase 2.2's hazard
question); leads here are detector leads on global curves, not forecasts
of where/when individual events occur.
