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
    args = build_parser().parse_args(argv)
    result = {
        "requested_out": str(args.out),
        "return_code": 2,
        "note": (
            "The historical dense-PD dataset generator is not bundled in this public archive. "
            "Use crackle.experiments.hetero_pinning for the self-contained synthetic fracture "
            "study, or provide your own event catalog and build risk sets with crackle.data."
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
