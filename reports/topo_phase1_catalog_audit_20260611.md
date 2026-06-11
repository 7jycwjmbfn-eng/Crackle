# Topo Phase 1.3 catalog audit

dataset: datasets\topo_synth_v1  cases: 2000  wall: 95s
config: sig_tau=0.08 roi_margin_k=1.5 matcher=wasserstein tile=6 horizons=[3, 5, 10] decay=0.85
git: 114cd7dd09a46ab535f70cc1ec40e9e5d8a7fcbc

## Event stream

events/step per case: mean 0.407, p10 0.163, p90 0.713
KILL RULE 1 (>= 0.3 events/step mean): PASS

| kind | events |
|---|---|
| h0_born | 24741 |
| h0_died | 20625 |
| h1_born | 10884 |
| h1_died | 8888 |

## Splits (by case)

| split | cases | riskset rows |
|---|---|---|
| ood | 495 | 962280 |
| test | 222 | 431568 |
| train | 1052 | 2045088 |
| val | 231 | 449064 |

## Class balance (positives / non-censored, all cases)

| label | pos | neg | censored | pos rate |
|---|---|---|---|---|
| label_any_H10 | 387421 | 3020579 | 480000 | 0.1137 |
| label_any_H3 | 155089 | 3588911 | 144000 | 0.0414 |
| label_any_H5 | 234425 | 3413575 | 240000 | 0.0643 |
| label_h0_born_H10 | 203367 | 3204633 | 480000 | 0.0597 |
| label_h0_born_H3 | 69445 | 3674555 | 144000 | 0.0185 |
| label_h0_born_H5 | 111661 | 3536339 | 240000 | 0.0306 |
| label_h0_died_H10 | 142441 | 3265559 | 480000 | 0.0418 |
| label_h0_died_H3 | 48602 | 3695398 | 144000 | 0.0130 |
| label_h0_died_H5 | 78158 | 3569842 | 240000 | 0.0214 |
| label_h1_born_H10 | 88617 | 3319383 | 480000 | 0.0260 |
| label_h1_born_H3 | 31448 | 3712552 | 144000 | 0.0084 |
| label_h1_born_H5 | 49985 | 3598015 | 240000 | 0.0137 |
| label_h1_died_H10 | 63622 | 3344378 | 480000 | 0.0187 |
| label_h1_died_H3 | 24080 | 3719920 | 144000 | 0.0064 |
| label_h1_died_H5 | 37317 | 3610683 | 240000 | 0.0102 |

Claim boundary: this is an event/feature CATALOG audit; it licenses
Phase 2 model training, no forecasting claim. Features use frames <= t
only (causality unit-tested); labels use (t, t+H].