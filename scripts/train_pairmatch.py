#!/usr/bin/env python3
"""Independent ESIM-style training; never opens the test split."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must precede importing torch; preserve an explicitly configured valid value.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pairmatch.data import DataError  # noqa: E402
from pairmatch.runner import CONDITIONS, train  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--subset-dir", type=Path, required=True)
    p.add_argument("--verification-report", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True, help="New local directory; never overwritten")
    p.add_argument("--confirm-data-rights", action="store_true")
    p.add_argument("--confirm-reviewed-data", action="store_true", help="Human review done and remaining limitations documented")
    p.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    p.add_argument("--term-lexicon", type=Path, help="UTF-8, one real term per line; required for U losses")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--embedding-dim", type=int, default=300)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float,
                   help="CE/A0 defaults to 0.0005; other conditions require a value or tuning lock")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--tuning-lock", type=Path, help="Frozen validation-selected configuration")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--u-batch-size", type=int, default=16)
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--stage1-epochs", type=int, help="Default: 32 for one stage, 16 for two stages")
    p.add_argument("--stage2-epochs", type=int, default=16)
    p.add_argument("--stage1-steps", type=int, help="Use step-based mode instead of epoch-based mode")
    p.add_argument("--stage2-steps", type=int, help="Second-stage limit in step-based mode")
    p.add_argument("--eval-every", type=int, default=200, help="Validation interval in step-based mode")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min-delta", type=float, default=0.0001)
    p.add_argument("--fixed-steps", action="store_true", help="Disable early stopping for predeclared matched-step controls")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--scl-weight", type=float, default=0.5)
    p.add_argument("--ul-weight", type=float, default=0.5)
    p.add_argument("--confidence", type=float, default=0.95, help="Pseudo-label confidence for self-training/consistency")
    p.add_argument("--teacher-ema", type=float, default=0.999, help="Mean-teacher EMA after each optimizer update")
    p.add_argument("--mask-rate", type=float, default=0.20)
    p.add_argument("--u-fraction", type=float, choices=(0.25, 0.5, 1.0), default=1.0)
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        result = train(args)
        print(f"Training complete: {result['total_executed_steps_including_rolled_back']} optimizer updates; outputs are local.")
        return 0
    except (DataError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Training failed ({type(exc).__name__}): {exc if isinstance(exc, DataError) else 'check local inputs and run record'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
