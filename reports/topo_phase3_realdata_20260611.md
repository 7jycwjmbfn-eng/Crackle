# Topo Phase 3 — real-data status readout (datasets retrieved + inspected)

Date: 2026-06-11. Branch: topo-tda. Spec: crackle_tda_spec.md §3 +
addendum v1.1 A item 3 ("INSPECT each dataset's actual content before
writing loaders; registry descriptions are leads, not guarantees").

## 3.1 DLR robustness check — BLOCKED on this machine

Needs fine-tuned UNet path masks from
`crackle.eval.dlr_cnn_spatial_validation`. No `runs/` artifacts in this
clone (run outputs are .gitignored) and the DLR raw data is absent.
Deferred until those artifacts are restored or the pipeline is re-run.
No claim made.

## 3.2 Multi-crack real data — four sources retrieved and inspected

All downloads in `external_datasets/` (not committed; .gitignored). The
addendum's warning is confirmed empirically: the registry descriptions
were leads, and none of the retrieved sources provides a per-frame 2D
crack/damage FIELD, which is what the (T, ny, nx) movie contract needs.

| dataset | retrieved content | usable as movie? |
|---|---|---|
| **Rimkus & Gribniak RC ties** (DiB 2017, priority 1) | obtained via EuropePMC supplementary package: 22-specimen load–strain curves (P[kN], mean concrete strain εc, mean steel strain εs; 4792 rows in mmc2.xlsx "Appendix A"), specimen-geometry + rig figures (gr1/gr2), supplement PDF | **NO** — 1D load–strain curves, not 2D fields. The DIC crack-development schemes are raster figures embedded in the article's Table 3, not extractable per-frame field data |
| kth_dic_concrete (Mendeley) | paper PDF + lab-test xlsx only | NO — DIC image series not in the record |
| craquelure_paintings (Zenodo) | FEM stress-vs-RH figure data (.dat) | NO — simulation curves, not crack imagery |
| desiccation_slope (Zenodo) | DPM simulation .avi + COMSOL .mph | NO — simulation movies |

### What the Rimkus retrieval DOES establish

Progress beyond the prior "BLOCKED, not retrieved" state: the priority-1
dataset was located (EuropePMC OA supplementary endpoint, not the
publisher paywall) and its true content is now documented. The public
Rimkus release supports **scalar load–response analysis** (and could feed
the macroscopic-instability/onset side of the pipeline as a real
load–strain control signal), but NOT the topological event pipeline,
which is field-based. Extracting topology from this study would require
the original DIC displacement/strain FIELD exports, which are not in the
public supplement — author contact or the LaVision project files would
be needed.

## Honest verdict

The synthetic-world results (Phases 1–2, Tracks A/C positive, B negative)
stand on their own. Real multi-crack FIELD data is not obtainable from
the public API/OA sources inspected here; the registry's most promising
lead (Rimkus) publishes load–strain curves, not crack fields. Real-data
topological validation remains open and now has a precise blocker:
per-frame DIC field exports, not summary curves.

## Path forward (unchanged priority, sharpened)

1. Rimkus: request the original DIC field exports from the authors, OR
   use the retrieved load–strain curves as a real control signal for a
   scalar-only onset check (no topology).
2. KTH: locate the companion image-series record referenced by the PDF.
3. Restore DLR UNet mask artifacts for §3.1.

## Claim boundary

Allowed: "Public API/OA sources for the registry's multi-crack datasets,
including the priority-1 Rimkus RC ties, do not contain per-frame crack-
FIELD data; the Rimkus public release is load–strain curves. Real-data
topological validation requires DIC field exports not in those releases."

Not allowed: any topological claim about real data (no field data was
obtained); any transfer of the synthetic findings to real specimens.
