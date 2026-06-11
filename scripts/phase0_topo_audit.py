"""Phase 0 topological audit for Crackle damage-field movies.

Answers ONE question with numbers and pictures: do the damage fields of the
hetero_pinning world carry enough topological events, and do topological
curves move before macroscopic instability?

Two input modes:
  --synthetic              regenerate reference cases by importing the
                           existing hetero_pinning simulator (numpy-only,
                           no torch needed)
  --npz path [path ...]    audit saved case_arrays.npz files from real runs
                           (key selected by --field-key)

Outputs per case under --out/<case_id>/:
  topo_features.csv   per-step topological summaries
  topo_events.csv     discrete topological event stream
  lead_time.csv       precursor onset vs macroscopic instability
  audit_<case>.png    4-panel figure
Plus a global summary: phase0_event_density.csv, phase0_report.md,
phase0_summary.png.

TDA here is CPU-only and fast (cubical persistence on ~48x29 grids runs in
milliseconds); CUDA is not needed for this phase.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crackle.topo import (
    event_count_curves,
    extract_events,
    horizon_margin_mask,
    instability_step,
    lead_time_table,
    load_case_npz,
    sequence_summaries,
)

EVENT_COLORS = {
    "h0_born": "#1f77b4",
    "h0_died": "#d62728",
    "h1_born": "#2ca02c",
    "h1_died": "#9467bd",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_cases(args: argparse.Namespace) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    from crackle.experiments.hetero_pinning import (
        CORR_LENGTHS_MM,
        correlated_toughness_field,
        make_grid,
        simulate_reference,
    )
    from crackle.topo.io import flat_to_fields

    cases: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    points, _, _, _ = make_grid(args.nx, args.ny, args.length, args.height)
    for contrast in args.contrasts:
        for corr_label in args.corr_lengths:
            for seed in range(args.num_seeds):
                gc_field = correlated_toughness_field(
                    nx=args.nx, ny=args.ny, length=args.length, height=args.height,
                    contrast=contrast, corr_length=CORR_LENGTHS_MM[corr_label],
                    seed=args.seed + seed,
                )
                ref = simulate_reference(
                    points=points,
                    gc_nodes=gc_field.reshape(-1),
                    steps=args.pd_steps,
                    length=args.length, height=args.height, horizon=args.horizon,
                    notch_length=args.notch_length,
                    tension_strain=args.tension_strain,
                    critical_stretch=args.critical_stretch,
                    notch_opening_factor=args.notch_opening_factor,
                    poisson=args.poisson,
                )
                fields = flat_to_fields(ref["damage"], points)
                case_id = f"c{contrast:g}_{corr_label}_s{seed}"
                meta = {
                    "contrast": float(contrast), "corr_length": corr_label,
                    "seed": int(seed), "source": "synthetic_reference",
                }
                cases.append((case_id, fields, meta))
    return cases


def npz_cases(paths: list[Path], field_key: str) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    cases = []
    for path in paths:
        fields, _, _ = load_case_npz(path, key=field_key)
        case_id = path.parent.name or path.stem
        cases.append((case_id, fields, {"source": str(path), "field_key": field_key}))
    return cases


def audit_case(
    case_id: str,
    fields: np.ndarray,
    meta: dict[str, Any],
    out_dir: Path,
    *,
    sig_tau: float,
    match_dist: float,
    connectivity: str,
    roi_k: float = 0.0,
    height: float = 40.0,
    horizon: float = 5.2,
    matcher: str = "wasserstein",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ny, nx = fields.shape[1], fields.shape[2]
    # boundary attribution uses the same margin whether or not ROI filters,
    # so ROI-off and ROI-on reports stay comparable
    attr_k = roi_k if roi_k > 0 else 1.5
    margin_rows = int(np.ceil(attr_k * horizon / (height / max(ny - 1, 1))))
    roi = (
        horizon_margin_mask(ny, nx, height=height, horizon=horizon, k=roi_k)
        if roi_k > 0
        else None
    )
    rows, diagrams = sequence_summaries(
        fields, sig_tau=sig_tau, connectivity=connectivity, roi=roi
    )
    events = extract_events(
        diagrams, sig_tau=sig_tau, max_dist=match_dist, roi=roi, method=matcher
    )
    n_steps = fields.shape[0]
    curves_raw = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "step"}
    counts = event_count_curves(events, n_steps)
    total_damage = fields.reshape(n_steps, -1).sum(axis=1)
    t_star = instability_step(total_damage)

    precursor_curves = {
        "n_h0_sig": curves_raw["n_h0_sig"],
        "n_h1_sig": curves_raw["n_h1_sig"],
        "total_pers_h0": curves_raw["total_pers_h0"],
        "total_pers_h1": curves_raw["total_pers_h1"],
        "entropy_h0": curves_raw["entropy_h0"],
        "entropy_h1": curves_raw["entropy_h1"],
        "cum_topo_events": np.cumsum(counts["all"]).astype(float),
        "total_damage(ref)": total_damage,
    }
    leads = lead_time_table(precursor_curves, total_damage)

    case_dir = out_dir / case_id
    _write_csv(case_dir / "topo_features.csv", rows)
    _write_csv(case_dir / "topo_events.csv", [e.as_row() for e in events])
    _write_csv(case_dir / "lead_time.csv", leads)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(total_damage, color="k", label="total damage")
    ax2 = ax.twinx()
    ax2.plot(np.diff(total_damage, prepend=total_damage[0]), color="orange",
             alpha=0.7, label="increment")
    ax.axvline(t_star, color="red", ls="--", lw=1, label=f"instability t*={t_star}")
    ax.set_title(f"{case_id}: macroscopic damage"); ax.legend(loc="upper left", fontsize=8)

    ax = axes[0, 1]
    ax.plot(curves_raw["n_h0_sig"], label="# sig H0 (hotspots)")
    ax.plot(curves_raw["n_h1_sig"], label="# sig H1 (loops)")
    ax.plot(curves_raw["entropy_h0"], label="H0 entropy", ls=":")
    ax.axvline(t_star, color="red", ls="--", lw=1)
    ax.set_title("topological precursor curves"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    kinds = ["h0_born", "h0_died", "h1_born", "h1_died"]
    for k_i, kind in enumerate(kinds):
        steps = [e.step for e in events if e.kind == kind]
        ax.scatter(steps, [k_i] * len(steps), s=14, color=EVENT_COLORS[kind])
    ax.set_yticks(range(len(kinds))); ax.set_yticklabels(kinds, fontsize=8)
    ax.axvline(t_star, color="red", ls="--", lw=1)
    ax.set_xlim(0, n_steps); ax.set_title("topological event raster"); ax.set_xlabel("load step")

    ax = axes[1, 1]
    im = ax.imshow(fields[-1], origin="lower", cmap="inferno", vmin=0, vmax=1)
    for e in events:
        if e.kind == "h1_born":
            ax.scatter(e.x, e.y, marker="o", s=60, facecolors="none", edgecolors="lime", lw=1.5)
        elif e.kind == "h0_died":
            ax.scatter(e.x, e.y, marker="x", s=40, color="cyan", lw=1.5)
    ax.set_title("final damage + events (o=loop born, x=merge)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(case_dir / f"audit_{case_id}.png", dpi=130)
    plt.close(fig)

    elapsed = time.perf_counter() - t0
    n_boundary = sum(
        1 for e in events if e.y < margin_rows or e.y >= ny - margin_rows
    )
    summary: dict[str, Any] = {
        "case_id": case_id, **meta,
        "n_steps": int(n_steps),
        "roi_k": float(roi_k),
        "margin_rows": int(margin_rows),
        "n_events_boundary": int(n_boundary),
        "n_events_interior": int(len(events) - n_boundary),
        "n_h0_born": int(counts["h0_born"].sum()),
        "n_h0_died": int(counts["h0_died"].sum()),
        "n_h1_born": int(counts["h1_born"].sum()),
        "n_h1_died": int(counts["h1_died"].sum()),
        "n_events_total": int(counts["all"].sum()),
        "events_per_step": float(counts["all"].sum() / max(n_steps - 1, 1)),
        "instability_step": int(t_star),
        "best_lead_steps": max(
            (r["lead_steps"] for r in leads
             if r["lead_steps"] is not None and r["signal"] != "total_damage(ref)"),
            default=None,
        ),
        "tda_wall_s": round(elapsed, 3),
    }
    return summary


def write_report(out_dir: Path, summaries: list[dict[str, Any]], args_ns: argparse.Namespace) -> None:
    lines = [
        "# Phase 0 Topological Audit",
        "",
        f"sig_tau={args_ns.sig_tau}, connectivity={args_ns.connectivity}, "
        f"match_dist={args_ns.match_dist}, roi_margin_k={args_ns.roi_margin_k}, "
        f"matcher={args_ns.matcher}",
        "",
        "Question: is there enough topological signal in these damage movies "
        "to define an event stream worth forecasting?",
        "",
        "| case | events total | ev/step | boundary/interior | H0 born/died "
        "| H1 born/died | t* | best lead |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['case_id']} | {s['n_events_total']} | {s['events_per_step']:.2f} "
            f"| {s['n_events_boundary']}/{s['n_events_interior']} "
            f"| {s['n_h0_born']}/{s['n_h0_died']} | {s['n_h1_born']}/{s['n_h1_died']} "
            f"| {s['instability_step']} | {s['best_lead_steps']} |"
        )
    ev = np.array([s["n_events_total"] for s in summaries], dtype=float)
    lines += [
        "",
        f"Mean events/case: {ev.mean():.1f} (min {ev.min():.0f}, max {ev.max():.0f}).",
        "",
        "Claim boundary: greedy matching is a heuristic; leads use a "
        "retrospective onset detector (max-normalized). Positive event density "
        "licenses Phase 1 (event catalog -> risk sets), not any forecasting claim.",
    ]
    (out_dir / "phase0_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs_topo/phase0"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--npz", type=Path, nargs="*", default=None)
    parser.add_argument("--field-key", type=str, default="reference_damage")
    parser.add_argument("--sig-tau", type=float, default=0.08,
                        help="persistence threshold for significant features")
    parser.add_argument("--match-dist", type=float, default=6.0,
                        help="max grid-cell distance for frame-to-frame matching")
    parser.add_argument("--connectivity", type=str, default="8", choices=["4", "8"])
    parser.add_argument("--roi-margin-k", type=float, default=0.0,
                        help="exclude k*horizon from top/bottom edges at the "
                             "diagram level (0 = off); spec 1.1 default 1.5")
    parser.add_argument("--matcher", type=str, default="wasserstein",
                        choices=["greedy", "wasserstein"])
    # synthetic-mode physics arguments mirror hetero_pinning defaults
    parser.add_argument("--contrasts", type=float, nargs="*", default=[1.0, 3.0, 5.0])
    parser.add_argument("--corr-lengths", type=str, nargs="*", default=["small", "medium"])
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=29)
    parser.add_argument("--length", type=float, default=100.0)
    parser.add_argument("--height", type=float, default=40.0)
    parser.add_argument("--horizon", type=float, default=5.2)
    parser.add_argument("--notch-length", type=float, default=22.0)
    parser.add_argument("--pd-steps", type=int, default=80)
    parser.add_argument("--tension-strain", type=float, default=0.075)
    parser.add_argument("--critical-stretch", type=float, default=0.045)
    parser.add_argument("--notch-opening-factor", type=float, default=0.54)
    parser.add_argument("--poisson", type=float, default=0.28)
    args = parser.parse_args(argv)

    if not args.synthetic and not args.npz:
        parser.error("choose --synthetic and/or --npz <files>")
    args.out.mkdir(parents=True, exist_ok=True)

    cases: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    if args.synthetic:
        print("generating synthetic reference cases via hetero_pinning ...", flush=True)
        cases += synthetic_cases(args)
    if args.npz:
        cases += npz_cases(list(args.npz), args.field_key)

    summaries = []
    for index, (case_id, fields, meta) in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case_id} fields={fields.shape}", flush=True)
        summaries.append(
            audit_case(case_id, fields, meta, args.out,
                       sig_tau=args.sig_tau, match_dist=args.match_dist,
                       connectivity=args.connectivity,
                       roi_k=args.roi_margin_k,
                       height=args.height, horizon=args.horizon,
                       matcher=args.matcher)
        )
    _write_csv(args.out / "phase0_event_density.csv", summaries)
    write_report(args.out, summaries, args)

    if args.synthetic and len({s.get("contrast") for s in summaries} - {None}) > 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        for corr in sorted({s["corr_length"] for s in summaries if "corr_length" in s}):
            pts = [(s["contrast"], s["n_events_total"]) for s in summaries
                   if s.get("corr_length") == corr]
            xs = sorted({p[0] for p in pts})
            ys = [np.mean([p[1] for p in pts if p[0] == x]) for x in xs]
            ax.plot(xs, ys, "o-", label=f"corr={corr}")
        ax.set_xlabel("toughness contrast"); ax.set_ylabel("topological events / case")
        ax.set_title("Phase 0: event density vs heterogeneity"); ax.legend()
        fig.tight_layout(); fig.savefig(args.out / "phase0_summary.png", dpi=130)
        plt.close(fig)

    print(json.dumps({"cases": len(summaries),
                      "report": str(args.out / "phase0_report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
