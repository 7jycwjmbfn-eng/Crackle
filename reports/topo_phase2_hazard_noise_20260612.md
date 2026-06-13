# Topo Phase 2.2 robustness — does the topology beat traditional methods UNDER noise?

Date: 2026-06-12. Branch: topo-tda. Spec: crackle_tda_spec.md §2.2
(strengthening). This re-runs the hazard feature ablation on risk sets
whose FEATURES are computed from a DIC-noised field (σ = 0.05, comparable
to sig_tau = 0.08) while the LABELS remain the clean ground-truth events
(forecast true failure from a noisy observation). Catalog:
datasets/topo_synth_v1/catalog_noise05 (2000 cases). Artifacts:
runs_topo/phase2_hazard_noise/.

The clean ablation (topo_phase2_hazard_20260611.md) already passed; the
open question is whether the topological advantage over TRADITIONAL
methods survives measurement noise, which contaminates the topological
features (the event stream inflates ~5× under this noise). If the
topology only wins on clean simulation data, it is not a real method.

## Pre-registered criterion (committed BEFORE running the noisy ablation)

Under σ = 0.05 noise, topological features count as beating the
traditional baseline iff ablation (c) local+topo+history improves test
binary NLL over BOTH (a) local-damage-only AND every classical referee
(base rate, carry-forward two-bucket, hawkes-logistic) by a margin
exceeding the GBM 3-seed std, on ≥ 2 of the 3 horizons {3, 5, 10}.

Traditional method = local damage features + GBM/logistic, plus the
classical referees. New method = (c) with global topological curves and
topological event-history features. Same model classes, same protocol,
same seeds as the clean run; only the feature field is noised.

<!-- RESULTS FILLED IN AFTER THE RUN (numbers are script-derived) -->
