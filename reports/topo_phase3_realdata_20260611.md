# Topo Phase 3 — real-data status readout (dataset inspection)

Date: 2026-06-11. Branch: topo-tda. Spec: crackle_tda_spec.md §3 +
addendum v1.1 A item 3 ("INSPECT each dataset's actual content before
writing loaders; registry descriptions are leads, not guarantees").

## 3.1 DLR robustness check — BLOCKED on this machine

The check needs fine-tuned UNet path masks from the existing DLR pipeline
(`crackle.eval.dlr_cnn_spatial_validation` outputs). No `runs/` artifacts
exist in this clone (run outputs are .gitignored by policy) and the DLR
raw data is not present. Deferred until the original run artifacts are
restored or the pipeline is re-run. No claim is made.

## 3.2 Multi-crack real data — registry inspection results

All three API-fetchable registry entries were downloaded and inspected
(`external_datasets/`, not committed). All three FAIL content inspection
for the movie-loader purpose:

| dataset | content found | verdict |
|---|---|---|
| kth_dic_concrete (Mendeley dns97tfdjn) | paper PDF (2 MB) + lab-test xlsx; latest version IS v1 | DIC image series NOT in the record; need companion record or author contact |
| craquelure_paintings (Zenodo 17862067) | FEM figure data (.dat: stress along diagonals vs RH drops) | simulation curves, not crack-network imagery; unusable as a pattern-topology testbed |
| desiccation_slope (Zenodo 10199729) | 2 DPM simulation .avi + COMSOL .mph | simulation movies, not desiccation crack time-lapse |

The addendum's warning is confirmed empirically: none of the three
registry leads contains what its description suggested.

## Path forward (unchanged priority order)

1. Rimkus & Gribniak RC ties (Data in Brief 2017,
   doi:10.1016/j.dib.2017.05.038) — manual download of the article
   supplement from the DOI landing page; remains priority 1 and is the
   intended first loader target (expect h0_born-dominated sequential
   transverse cracking).
2. KTH: locate the companion image-series record (the PDF references the
   full test series) or contact authors.
3. arXiv:2411.04620 RC slab — availability still unverified.

## Claim boundary

Allowed: "the public API-fetchable leads in the current registry do not
contain usable multi-crack movie data; real-data validation requires
manual supplement retrieval." Nothing else — no topological claim about
real data is licensed by this phase in its current state.
