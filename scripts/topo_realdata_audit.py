"""Phase 3.2 - topological audit of a REAL crack-mask sequence.

Loads an ordered set of binary crack-segmentation masks (one physical
specimen, ordered by load epoch), block-averages them to a tractable grid
(a crack-density field; topology is scale-robust, see the res96 probe),
and runs the existing Phase 0 topological machinery: per-frame summaries
(H0 crack components, H1 enclosed loops) and the frame-to-frame event
stream (h0/h1 born/died). The deliverable is the event-kind distribution
and topological evolution, compared against the synthetic multi-notch
world (which predicted h0_born-dominated sequential nucleation followed by
h1 loop formation as the network closes).

Crack = white = high value, so superlevel persistence captures cracks
directly (no inversion). Real masks are clean binary (no measurement-noise
floor), so sig_tau only filters tiny block-averaged ridges.

Example (PowerShell):
  python -m scripts.topo_realdata_audit `
    --masks-dir .\\external_datasets\\harb_rc_slab\\targets `
    --downscale 16 --sig-tau 0.05 --out .\\runs_topo\\phase3_harb
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crackle.experiments.hetero_pinning import _git_commit
from crackle.topo import (
    event_count_curves,
    extract_events,
    load_mask_sequence,
    natural_key,
    sequence_summaries,
)

EVENT_COLORS = {"h0_born": "#1f77b4", "h0_died": "#d62728",
                "h1_born": "#2ca02c", "h1_died": "#9467bd"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="*.tif")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--downscale", type=int, default=16)
    parser.add_argument("--sig-tau", type=float, default=0.05)
    parser.add_argument("--match-dist", type=float, default=6.0)
    parser.add_argument("--matcher", type=str, default="wasserstein",
                        choices=["greedy", "wasserstein"])
    parser.add_argument("--label", type=str, default="real")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # the masks are ~92 MP; they are trusted

    paths = sorted(Path(args.masks_dir).glob(args.glob), key=natural_key)
    if not paths:
        parser.error(f"no files matched {args.masks_dir}/{args.glob}")
    print(f"{len(paths)} masks; first={paths[0].name} last={paths[-1].name}",
          flush=True)

    t0 = time.perf_counter()
    movie = load_mask_sequence(paths, threshold=None, downscale=args.downscale)
    print(f"movie {movie.shape} loaded in {time.perf_counter()-t0:.1f}s",
          flush=True)

    rows, diagrams = sequence_summaries(movie, sig_tau=args.sig_tau)
    events = extract_events(diagrams, sig_tau=args.sig_tau,
                            max_dist=args.match_dist, method=args.matcher)
    n_steps = movie.shape[0]
    counts = event_count_curves(events, n_steps)
    curves = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "step"}

    # white (crack) fraction per frame = a real macroscopic damage proxy
    crack_frac = (movie > 0).reshape(n_steps, -1).mean(axis=1)

    kind_tot = {k: int(counts[k].sum()) for k in
                ("h0_born", "h0_died", "h1_born", "h1_died")}
    summary = {
        "dataset": str(args.masks_dir), "n_frames": n_steps,
        "grid": list(movie.shape[1:]), "downscale": args.downscale,
        "sig_tau": args.sig_tau, "matcher": args.matcher,
        "events_total": int(counts["all"].sum()),
        "events_per_step": float(counts["all"].sum() / max(n_steps - 1, 1)),
        "kind_counts": kind_tot,
        "h1_events": kind_tot["h1_born"] + kind_tot["h1_died"],
        "n_h0_sig_first": float(curves["n_h0_sig"][0]),
        "n_h0_sig_last": float(curves["n_h0_sig"][-1]),
        "n_h1_sig_first": float(curves["n_h1_sig"][0]),
        "n_h1_sig_last": float(curves["n_h1_sig"][-1]),
        "crack_frac_first": float(crack_frac[0]),
        "crack_frac_last": float(crack_frac[-1]),
        "wall_s": round(time.perf_counter() - t0, 1),
        "git_commit": _git_commit(Path.cwd()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2),
                                           encoding="utf-8")
    with (args.out / "topo_events.csv").open("w", newline="",
                                             encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["step", "kind", "y", "x",
                                          "persistence", "birth_value"])
        w.writeheader()
        w.writerows([e.as_row() for e in events])

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    a = ax[0, 0]
    a.plot(curves["n_h0_sig"], "o-", label="# sig H0 (crack components)")
    a.plot(curves["n_h1_sig"], "s-", label="# sig H1 (enclosed loops)")
    a.set_xlabel("load epoch index"); a.set_title(
        f"{args.label}: topological evolution"); a.legend(); a.grid(alpha=0.3)
    a = ax[0, 1]
    a.plot(crack_frac, "k.-"); a.set_title("crack (white) area fraction")
    a.set_xlabel("load epoch index"); a.grid(alpha=0.3)
    a = ax[1, 0]
    kinds = ["h0_born", "h0_died", "h1_born", "h1_died"]
    for ki, k in enumerate(kinds):
        steps = [e.step for e in events if e.kind == k]
        a.scatter(steps, [ki] * len(steps), s=18, color=EVENT_COLORS[k])
    a.set_yticks(range(4)); a.set_yticklabels(kinds)
    a.set_xlim(0, n_steps); a.set_title("topological event raster")
    a.set_xlabel("load epoch index")
    a = ax[1, 1]
    a.imshow(movie[-1], origin="lower", cmap="inferno", vmin=0, vmax=1)
    a.set_title(f"final crack-density field ({movie.shape[2]}x{movie.shape[1]})")
    fig.tight_layout(); fig.savefig(args.out / f"audit_{args.label}.png", dpi=130)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
