#!/usr/bin/env python3
"""Measure CPU batch-1 latency on a locked validation input set, never the test split."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from pairmatch.data import DataError, checked_rows, digest, read_json, truncation  # noqa: E402
from pairmatch.model import batch_encode  # noqa: E402
from pairmatch.runner import dump, make_model, pairs_for  # noqa: E402


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * 0.95
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return {"median_ms": statistics.median(ordered), "p95_ms": ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower),
            "min_ms": ordered[0], "max_ms": ordered[-1]}


def benchmark(args: argparse.Namespace) -> dict:
    if not args.confirm_data_rights:
        raise DataError("confirm the right to process the data")
    if args.output.exists():
        raise DataError("output file exists; refusing to overwrite")
    if min(args.warmup, args.repeats, args.sample_count, args.threads) < 1:
        raise DataError("warmup, repeats, sample count and threads must be positive")
    summary = read_json(args.checkpoint.parent / "summary.json")
    if summary.get("status") != "COMPLETE" or summary.get("final_checkpoint") != args.checkpoint.name:
        raise DataError("benchmark requires the final selected checkpoint of a completed run")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") not in (1, 2):
        raise DataError("unsupported checkpoint format")
    manifest_path = args.split_dir / "manifest.json"
    if digest(manifest_path) != checkpoint["provenance"]["split_manifest_sha256"]:
        raise DataError("checkpoint and split manifest do not match")
    manifest = read_json(manifest_path)
    expected = manifest["output_sha256"]["validation.jsonl"]
    if expected != checkpoint["provenance"]["validation_sha256"]:
        raise DataError("validation hash mismatch")
    data = checked_rows(args.split_dir / "validation.jsonl", expected, True)[:args.sample_count]
    if not data:
        raise DataError("validation input is empty")
    config = checkpoint["config"]
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    if checkpoint["format_version"] == 1:
        vocab = checkpoint["vocab"]
        model = make_model(config, len(vocab)).to(device).eval()
        model.load_state_dict(checkpoint["model_state"])
        def encode_one(row):
            return batch_encode(pairs_for([row], vocab, config["max_length"]), device)
        def forward(inputs):
            return model(*inputs)
        truncation_info = truncation(data, config["max_length"])
    else:
        from pairmatch.transformer_runner import restore, tokenize, token_statistics
        model, tokenizer = restore(checkpoint, args.checkpoint.parent / "tokenizer", device)
        model.eval()
        def encode_one(row):
            return tokenize(tokenizer, [row], config["max_length"], device)
        def forward(inputs):
            return model(inputs)
        truncation_info = token_statistics(tokenizer, data, config["max_length"])
    cached = [encode_one(row) for row in data]
    times = {}
    with torch.inference_mode():
        for kind in ("model_only", "end_to_end_without_disk"):
            durations = []
            for iteration in range(args.warmup + args.repeats):
                index = iteration % len(data)
                start = time.perf_counter_ns()
                inputs = cached[index] if kind == "model_only" else encode_one(data[index])
                forward(inputs)
                elapsed = (time.perf_counter_ns() - start) / 1_000_000
                if iteration >= args.warmup:
                    durations.append(elapsed)
            times[kind] = distribution(durations)
    try:
        import psutil
        memory = psutil.Process().memory_info()
        process_memory = {"current_rss_bytes": memory.rss,
                          "windows_peak_working_set_bytes": getattr(memory, "peak_wset", None),
                          "psutil_version": psutil.__version__,
                          "scope": "whole process since process start, including Python, dependencies and checkpoint loading"}
    except ImportError:
        process_memory = {"status": "unavailable_install_psutil"}
    result = {"status": "COMPLETE", "device": "cpu", "platform": platform.platform(),
              "processor": platform.processor(), "torch_version": str(torch.__version__), "threads": args.threads,
              "interop_threads": torch.get_num_interop_threads(),
              "batch_size": 1, "warmup_per_measurement": args.warmup, "timed_repetitions_per_measurement": args.repeats,
              "input_selection": "first N validation pairs, cycled in fixed order; no disk reading in timing",
              "selected_pairs": len(data), "validation_sha256": expected,
              "truncation": truncation_info, "checkpoint_sha256": digest(args.checkpoint),
              "checkpoint_file_size_bytes_including_metadata": args.checkpoint.stat().st_size,
              "benchmark_script_sha256": digest(Path(__file__)),
              "parameters_total_including_embeddings": sum(p.numel() for p in model.parameters()),
              "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "timings": times, "cpu_process_memory": process_memory, "flops": "not measured",
              "note": "Both timings end at model logits; no softmax/postprocessing or disk I/O. Model-only excludes encoding; end-to-end includes it."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local JSON file")
    parser.add_argument("--confirm-data-rights", action="store_true")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    try:
        torch.set_num_interop_threads(1)
        benchmark(args)
        print("CPU benchmark complete; measurement is local to the current host.")
        return 0
    except (DataError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Benchmark failed ({type(exc).__name__}): {exc if isinstance(exc, DataError) else 'check inputs and checkpoint'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
