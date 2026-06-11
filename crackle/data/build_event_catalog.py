from __future__ import annotations

import argparse
import json
from pathlib import Path

from crackle.data.event_catalog import build_event_catalog
from crackle.data.riskset import build_riskset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Crackle event catalog and riskset from a dense-PD crack manifest.")
    parser.add_argument("--input", "--manifest", dest="manifest", type=Path, required=True)
    parser.add_argument("--output", "--out", dest="output", type=Path, required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--partial-observation-rate", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    events = build_event_catalog(args.manifest, args.output, split=args.split)
    riskset = build_riskset(
        args.manifest,
        args.output,
        split=args.split,
        partial_observation_rate=args.partial_observation_rate,
    )
    result = {"events": events, "riskset": {"path": str(args.output / "crackle_riskset.zarr"), "num_samples": riskset["num_samples"]}}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
