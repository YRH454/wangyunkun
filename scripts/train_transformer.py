#!/usr/bin/env python3
"""Fine-tune a licensed locally available pretrained Transformer without accessing U or test data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pairmatch.data import DataError  # noqa: E402
from pairmatch.transformer_runner import MODEL_IDS, train  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--subset-dir", type=Path, required=True)
    p.add_argument("--verification-report", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="Local model and tokenizer; no network access")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", choices=MODEL_IDS, required=True)
    p.add_argument("--confirm-data-rights", action="store_true")
    p.add_argument("--confirm-reviewed-data", action="store_true")
    p.add_argument("--confirm-model-rights-and-revision", action="store_true")
    p.add_argument("--allow-unsafe-pickle-weights", action="store_true", help="Only for trusted checkpoint files")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float,
                   help="Defaults to 2e-5 for BERT-base and 5e-5 for the four-layer model")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--tuning-lock", type=Path, help="Frozen validation-selected configuration")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--steps", type=int, help="Use step-based mode instead of epoch-based mode")
    p.add_argument("--eval-every", type=int, default=200, help="Validation interval in step-based mode")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min-delta", type=float, default=0.0001)
    p.add_argument("--fixed-steps", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        result = train(args)
        print(f"Transformer training complete: {result['total_executed_steps_including_rolled_back']} optimizer updates; outputs are local.")
        return 0
    except (DataError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Training failed ({type(exc).__name__}): {exc if isinstance(exc, DataError) else 'check local inputs and run record'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
