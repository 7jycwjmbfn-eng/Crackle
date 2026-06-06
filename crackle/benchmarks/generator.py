from __future__ import annotations

import argparse
import json
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Crackle hard-benchmark dense-PD rollouts.")
    parser.add_argument("--out", "--output", dest="out", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--num-particles", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--horizon", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        from gaussmoe_physics.data.make_crack_notched_plate import main as make_crack_main
    except ImportError as exc:
        raise SystemExit(
            "crackle.benchmarks.generator is a compatibility wrapper for the older "
            "GaussMoE dense-PD dataset generator. Install or place gaussmoe_physics "
            "on PYTHONPATH to use this entry point. Core Crackle models, metrics, "
            "real-data audits, and hetero_pinning do not require this import."
        ) from exc

    args = build_parser().parse_args(argv)
    cmd = [
        "--out",
        str(args.out),
        "--num-samples",
        str(args.num_samples),
        "--num-particles",
        str(args.num_particles),
        "--steps",
        str(args.steps),
        "--horizon",
        str(args.horizon),
        "--seed",
        str(args.seed),
        "--hard-bench-v1",
    ]
    if args.quiet:
        cmd.append("--quiet")
    rc = make_crack_main(cmd)
    manifest = args.out / "dataset_manifest.json"
    result = {"manifest": str(manifest), "return_code": rc, "note": "partial_observation_variant is applied during Crackle riskset construction via partial_observation_mask."}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
