from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from crackle.data.common import dataset_hash, load_json, load_labels, ok_samples, write_csv, write_json
from crackle.data.features import bond_geometry, material_toughness, node_to_bond_mean


EVENT_COLUMNS = [
    "event_id",
    "sample_id",
    "benchmark_case_id",
    "split_id",
    "t_event",
    "step_event",
    "bond_id",
    "bond_i",
    "bond_j",
    "bond_center_x",
    "bond_center_y",
    "bond_center_z_optional",
    "bond_orientation_x",
    "bond_orientation_y",
    "bond_orientation_z_optional",
    "event_energy",
    "event_mark",
    "failure_mode_optional",
]


def bond_break_events(sample: dict[str, Any], labels: dict[str, np.ndarray], *, event_id_start: int = 0) -> list[dict[str, Any]]:
    bonds = labels["bonds"].astype(np.int64)
    alive = labels["bond_alive"].astype(bool)
    stretch = labels["bond_stretch"]
    energy_node = labels["strain_energy"]
    toughness = material_toughness(labels, bonds)
    geom = bond_geometry(labels["reference_x"], bonds)
    dt = float(sample.get("time_step_dt") or 1.0 / max(alive.shape[0] - 1, 1))
    events: list[dict[str, Any]] = []
    event_id = int(event_id_start)
    for step in range(alive.shape[0] - 1):
        broke = np.asarray(alive[step] & ~alive[step + 1], dtype=bool)
        ids = np.flatnonzero(broke)
        if ids.size == 0:
            continue
        energy_bond = node_to_bond_mean(energy_node[step + 1], bonds, ids)
        for local_index, bond_id in enumerate(ids):
            i, j = int(bonds[bond_id, 0]), int(bonds[bond_id, 1])
            events.append(
                {
                    "event_id": event_id,
                    "sample_id": sample.get("id"),
                    "benchmark_case_id": sample.get("benchmark_case_id"),
                    "split_id": sample.get("split_id") or sample.get("split"),
                    "t_event": float((step + 1) * dt),
                    "step_event": int(step + 1),
                    "bond_id": int(bond_id),
                    "bond_i": i,
                    "bond_j": j,
                    "bond_center_x": float(geom.centers[bond_id, 0]),
                    "bond_center_y": float(geom.centers[bond_id, 1]),
                    "bond_center_z_optional": None,
                    "bond_orientation_x": float(geom.orientation[bond_id, 0]),
                    "bond_orientation_y": float(geom.orientation[bond_id, 1]),
                    "bond_orientation_z_optional": None,
                    "event_energy": float(energy_bond[local_index]),
                    "event_mark": float(stretch[step + 1, bond_id] / max(toughness[bond_id], 1e-12)),
                    "failure_mode_optional": "stretch_gt_local_critical",
                }
            )
            event_id += 1
    expected = int(np.count_nonzero(alive[:-1] & ~alive[1:]))
    if expected != len(events):
        raise ValueError(f"event count mismatch for {sample.get('id')}: expected {expected}, built {len(events)}")
    return events


def write_events(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(out_dir / "crackle_events.csv", rows, EVENT_COLUMNS)
    parquet_path = out_dir / "crackle_events.parquet"
    parquet_written = False
    try:
        import pandas as pd

        pd.DataFrame(rows, columns=EVENT_COLUMNS).to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception as exc:
        raise RuntimeError(
            "Writing crackle_events.parquet requires pandas with a parquet engine such as pyarrow. "
            "Install pyarrow or set PYTHONPATH to the local pyarrow package."
        ) from exc
    return {"csv": str(csv_path), "parquet": str(parquet_path), "parquet_written": parquet_written}


def build_event_catalog(manifest_path: Path, out_dir: Path, *, split: str = "all") -> dict[str, Any]:
    manifest = load_json(manifest_path)
    rows: list[dict[str, Any]] = []
    samples = ok_samples(manifest, split=split)
    for sample in samples:
        labels = load_labels(sample)
        rows.extend(bond_break_events(sample, labels, event_id_start=len(rows)))
    paths = write_events(out_dir, rows)
    metadata = {
        "schema": "crackle_event_catalog_v1",
        "source_manifest": str(manifest_path),
        "dataset_hash": manifest.get("dataset_hash") or dataset_hash(manifest_path),
        "split": split,
        "num_samples": len(samples),
        "num_events": len(rows),
        "event_count_matches_alive_drop": True,
        "paths": paths,
    }
    write_json(out_dir / "metadata.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert dense-PD crack rollout labels into Crackle event catalog.")
    parser.add_argument("--manifest", "--input", dest="manifest", type=Path, required=True)
    parser.add_argument("--output", "--out", dest="output", type=Path, required=True)
    parser.add_argument("--split", default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_event_catalog(args.manifest, args.output, split=args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
