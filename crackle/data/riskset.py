from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from crackle.data.common import dataset_hash, load_json, load_labels, ok_samples, write_json
from crackle.data.features import bond_geometry, frontier_bond_mask, local_load_magnitude, material_toughness


def sample_riskset_summary(sample: dict[str, Any], labels: dict[str, np.ndarray], out_dir: Path, *, partial_observation_rate: float = 1.0) -> dict[str, Any]:
    bonds = labels["bonds"].astype(np.int64)
    alive = labels["bond_alive"].astype(bool)
    geom = bond_geometry(labels["reference_x"], bonds)
    toughness = material_toughness(labels, bonds)
    final_alive = alive[-1]
    censored = final_alive.copy()
    frontier_any = np.zeros((bonds.shape[0],), dtype=bool)
    for step in range(alive.shape[0]):
        frontier_any |= frontier_bond_mask(labels["crack_tip_mask"][step], bonds)
    rng = np.random.default_rng(abs(hash(sample.get("id"))) % (2**32))
    partial_mask = rng.random((alive.shape[0], bonds.shape[0])) < float(partial_observation_rate)
    sample_dir = out_dir / "samples" / str(sample.get("id"))
    sample_dir.mkdir(parents=True, exist_ok=True)
    static_path = sample_dir / "riskset_static.npz"
    np.savez_compressed(
        static_path,
        material_toughness_bond=toughness,
        boundary_distance_bond=geom.boundary_distance,
        bond_center=geom.centers,
        bond_orientation=geom.orientation,
        frontier_ever_mask=frontier_any,
        censored_bond_mask=censored,
        partial_observation_mask=partial_mask,
        load_state_bond=local_load_magnitude(labels, bonds),
    )
    positive_count = int(np.count_nonzero(alive[:-1] & ~alive[1:]))
    censored_count = int(np.count_nonzero(censored))
    if positive_count <= 0:
        raise ValueError(f"risk set for {sample.get('id')} has no positive events")
    if censored_count <= 0:
        raise ValueError(f"risk set for {sample.get('id')} has no censored alive bonds")
    return {
        "sample_id": sample.get("id"),
        "benchmark_case_id": sample.get("benchmark_case_id"),
        "split_id": sample.get("split_id") or sample.get("split"),
        "source_crack_npz": sample.get("crack_npz"),
        "riskset_static_npz": str(static_path),
        "num_steps": int(alive.shape[0]),
        "num_bonds": int(bonds.shape[0]),
        "positive_event_count": positive_count,
        "censored_bond_count": censored_count,
        "partial_observation_rate": float(partial_observation_rate),
    }


def build_riskset(manifest_path: Path, out_dir: Path, *, split: str = "all", partial_observation_rate: float = 1.0) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    samples = ok_samples(manifest, split=split)
    risk_root = out_dir / "crackle_riskset.zarr"
    rows = []
    for sample in samples:
        labels = load_labels(sample)
        rows.append(sample_riskset_summary(sample, labels, risk_root, partial_observation_rate=partial_observation_rate))
    metadata = {
        "schema": "crackle_reference_backed_riskset_v1",
        "storage_note": "Directory is named .zarr for contract compatibility, but large causal tensors are reference-backed by source crack_npz files; derived static arrays are stored as compressed NPZ.",
        "source_manifest": str(manifest_path),
        "dataset_hash": manifest.get("dataset_hash") or dataset_hash(manifest_path),
        "split": split,
        "num_samples": len(rows),
        "riskset_contains_positive_and_censored_bonds": all(row["positive_event_count"] > 0 and row["censored_bond_count"] > 0 for row in rows),
        "samples": rows,
    }
    write_json(risk_root / "riskset_manifest.json", metadata)
    write_json(out_dir / "riskset_metadata.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Crackle risk set metadata and derived causal arrays.")
    parser.add_argument("--manifest", "--input", dest="manifest", type=Path, required=True)
    parser.add_argument("--output", "--out", dest="output", type=Path, required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--partial-observation-rate", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_riskset(args.manifest, args.output, split=args.split, partial_observation_rate=args.partial_observation_rate)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
