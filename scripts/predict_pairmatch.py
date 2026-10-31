#!/usr/bin/env python3
"""Evaluate a frozen checkpoint on the matching validation or explicitly locked test split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from pairmatch.data import DataError, checked_rows, digest, read_json, truncation  # noqa: E402
from pairmatch.runner import dump, make_model, validation_result  # noqa: E402


def predict(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise DataError("output directory exists; refusing to overwrite")
    if any(path.resolve().is_relative_to(args.output_dir.resolve()) for path in (args.checkpoint, args.split_dir)):
        raise DataError("an input is inside the output directory")
    if args.split == "test" and not args.confirm_test_locked:
        raise DataError("test evaluation requires explicit confirmation of frozen choices")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise DataError("CUDA was requested but is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise DataError("unsupported checkpoint format")
    if args.split == "test":
        summary = read_json(args.checkpoint.parent / "summary.json")
        if summary.get("status") != "COMPLETE" or summary.get("final_checkpoint") != args.checkpoint.name:
            raise DataError("test requires the final selected checkpoint of a completed run")
    manifest_path = args.split_dir / "manifest.json"
    if digest(manifest_path) != checkpoint["provenance"]["split_manifest_sha256"]:
        raise DataError("checkpoint and split manifest do not match")
    manifest = read_json(manifest_path)
    filename = f"{args.split}.jsonl"
    expected = manifest["output_sha256"][filename]
    checkpoint_expected = checkpoint["provenance"]["test_sha256_expected_not_read" if args.split == "test" else "validation_sha256"]
    if expected != checkpoint_expected:
        raise DataError("checkpoint and evaluation file hash do not match")
    data = checked_rows(args.split_dir / filename, expected, True)
    if not data:
        raise DataError("evaluation split is empty")
    config, vocab = checkpoint["config"], checkpoint["vocab"]
    if vocab.get("<pad>") != 0 or vocab.get("<unk>") != 1 or vocab.get("<mask>") != 2:
        raise DataError("invalid checkpoint vocabulary")
    if args.batch_size < 1 or args.cpu_threads < 1:
        raise DataError("batch size and CPU threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    model = make_model(config, len(vocab)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics, probabilities = validation_result(model, data, vocab, config["max_length"], args.batch_size, device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"status": "COMPLETE", "split": args.split, "split_sha256": expected,
              "checkpoint_sha256": digest(args.checkpoint), "training_config": config,
              "training_provenance": checkpoint["provenance"], "checkpoint_stage": checkpoint["stage"],
              "checkpoint_step": checkpoint["stage_step"], "count": len(data), "metrics": metrics,
              "truncation": truncation(data, config["max_length"]),
              "test_lock_is_user_attestation_not_automatically_proven": args.split == "test"}
    dump(args.output_dir / "metrics.json", result)
    with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row, probability in zip(data, probabilities):
            handle.write(json.dumps({"sample_id": row["sample_id"], "gold_label": row["gold_label"],
                                     "probability_positive": probability, "prediction": int(probability >= 0.5)},
                                    ensure_ascii=False, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-test-locked", action="store_true", help="Choices and analysis plan frozen before test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    try:
        result = predict(args)
        print(f"Evaluation complete: {result['count']} {args.split} pairs; results are local.")
        return 0
    except (DataError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Evaluation failed ({type(exc).__name__}): {exc if isinstance(exc, DataError) else 'check inputs and checkpoint'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
