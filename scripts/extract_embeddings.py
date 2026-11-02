#!/usr/bin/env python3
"""Extract a fixed local evaluation sample's pair embeddings for descriptive t-SNE."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pairmatch.data import DataError, checked_rows, digest, read_json  # noqa: E402
from pairmatch.model import batch_encode  # noqa: E402
from pairmatch.runner import dump, make_model, pairs_for  # noqa: E402


def extract(args: argparse.Namespace) -> dict:
    if not args.confirm_prespecified_analysis:
        raise DataError("confirm feature sampling and descriptive analysis were prespecified")
    if args.output_dir.exists():
        raise DataError("output directory exists; refusing to overwrite")
    if any(path.resolve().is_relative_to(args.output_dir.resolve()) for path in (args.checkpoint, args.split_dir)):
        raise DataError("an input is inside output directory")
    if args.max_samples < 2 or args.batch_size < 1 or args.cpu_threads < 1:
        raise DataError("invalid sample count, batch size or thread count")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise DataError("CUDA is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") not in (1, 2):
        raise DataError("unsupported checkpoint")
    summary = read_json(args.checkpoint.parent / "summary.json")
    if summary.get("status") != "COMPLETE" or summary.get("final_checkpoint") != args.checkpoint.name:
        raise DataError("feature extraction requires the completed final selected checkpoint")
    manifest_path = args.split_dir / "manifest.json"
    if digest(manifest_path) != checkpoint["provenance"]["split_manifest_sha256"]:
        raise DataError("checkpoint and split do not match")
    manifest = read_json(manifest_path)
    filename = args.split + ".jsonl"
    expected = manifest["output_sha256"][filename]
    checkpoint_hash = checkpoint["provenance"]["test_sha256_expected_not_read" if args.split == "test" else "validation_sha256"]
    if expected != checkpoint_hash:
        raise DataError("evaluation file hash mismatch")
    data = checked_rows(args.split_dir / filename, expected, True)
    data = sorted(data, key=lambda row: row["sample_id"])
    random.Random(args.seed).shuffle(data)
    data = data[:args.max_samples]
    if len(data) < 2:
        raise DataError("at least two evaluation samples are required")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    config = checkpoint["config"]
    if checkpoint["format_version"] == 1:
        vocab = checkpoint["vocab"]
        model = make_model(config, len(vocab)).to(device)
        model.load_state_dict(checkpoint["model_state"])
        def forward(batch):
            return model(*batch_encode(pairs_for(batch, vocab, config["max_length"]), device))[1]
    else:
        from pairmatch.transformer_runner import restore, tokenize
        model, tokenizer = restore(checkpoint, args.checkpoint.parent / "tokenizer", device)
        def forward(batch):
            return model(tokenize(tokenizer, batch, config["max_length"], device))[1]
    model.eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(data), args.batch_size):
            vectors.append(forward(data[start:start + args.batch_size]).cpu().numpy())
    vectors_array = np.concatenate(vectors, axis=0)
    if not np.isfinite(vectors_array).all():
        raise DataError("non-finite embeddings")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "embeddings.npz", embeddings=vectors_array,
                        sample_ids=np.asarray([row["sample_id"] for row in data]),
                        labels=np.asarray([row["gold_label"] for row in data], dtype=np.int64))
    report = {"status": "COMPLETE", "split": args.split, "split_sha256": expected,
              "split_manifest_sha256": checkpoint["provenance"]["split_manifest_sha256"],
              "checkpoint_sha256": digest(args.checkpoint), "condition": config["condition"],
              "sample_count": len(data), "sample_seed": args.seed, "sample_max": args.max_samples,
              "selection": "sorted IDs, seeded shuffle, take prefix; identical on a shared split",
              "embedding": "pair representation immediately before the classifier", "embedding_dimension": vectors_array.shape[1],
              "embeddings_sha256": digest(args.output_dir / "embeddings.npz"),
              "prespecified_analysis_user_attestation": True,
              "warning": "Descriptive representation only; labels and IDs remain private, no test-based model selection."}
    dump(args.output_dir / "manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-prespecified-analysis", action="store_true")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        result = extract(args)
        print(f"Extracted {result['sample_count']} private pair embeddings for descriptive analysis.")
        return 0
    except (DataError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Extraction failed ({type(exc).__name__}): {exc if isinstance(exc, DataError) else 'check local inputs'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
