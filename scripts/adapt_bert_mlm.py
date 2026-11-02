#!/usr/bin/env python3
"""Continue a local BERT masked-language model on training U only, with a fixed budget."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402
from pairmatch.data import BatchStream, DataError, digest, load_training_data  # noqa: E402
from pairmatch.runner import append, dump, random_seed, version_info  # noqa: E402
from pairmatch.transformer_runner import artifact_hashes  # noqa: E402


def masked_batch(tokenizer, batch: list[dict], max_length: int, probability: float,
                 device: torch.device) -> tuple[dict, int, bool]:
    encoded = tokenizer([row["sentence1"] for row in batch], [row["sentence2"] for row in batch],
                        padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt", return_special_tokens_mask=True)
    special = encoded.pop("special_tokens_mask").bool()
    input_ids = encoded["input_ids"]
    eligible = ~special & encoded["attention_mask"].bool()
    if not eligible.any():
        raise DataError("no maskable tokens in an unlabeled microbatch")
    selected = (torch.rand(input_ids.shape) < probability) & eligible
    forced = not bool(selected.any())
    if forced:
        positions = eligible.nonzero(as_tuple=False)
        index = int(torch.randint(len(positions), (1,)).item())
        selected[positions[index, 0], positions[index, 1]] = True
    labels = input_ids.clone()
    labels[~selected] = -100
    actions = torch.rand(input_ids.shape)
    input_ids[selected & (actions < 0.8)] = tokenizer.mask_token_id
    replace_random = selected & (actions >= 0.8) & (actions < 0.9)
    random_ids = torch.randint(len(tokenizer), input_ids.shape, dtype=torch.long)
    input_ids[replace_random] = random_ids[replace_random]
    encoded["labels"] = labels
    return {key: value.to(device) for key, value in encoded.items()}, int(selected.sum().item()), forced


def adapt(args: argparse.Namespace) -> dict:
    if not (args.confirm_data_rights and args.confirm_reviewed_data and args.confirm_model_rights_and_revision):
        raise DataError("confirm data rights, review and model rights/revision")
    if args.output_dir.exists():
        raise DataError("output directory exists; refusing to overwrite")
    if any(path.resolve().is_relative_to(args.output_dir.resolve()) for path in
           (args.split_dir, args.subset_dir, args.verification_report, args.checkpoint_dir)):
        raise DataError("an input is inside output directory")
    if (min(args.seed, args.steps, args.batch_size, args.accumulation, args.max_length, args.cpu_threads) < 1
            or not all(math.isfinite(x) for x in (args.learning_rate, args.weight_decay, args.mask_probability))
            or args.learning_rate <= 0 or args.weight_decay < 0 or not 0 < args.mask_probability < 1):
        raise DataError("invalid adaptation settings")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise DataError("CUDA is unavailable")
    _, _, unlabeled, provenance = load_training_data(args.split_dir, args.subset_dir, args.verification_report, True)
    if provenance["sampling_seed"] != args.seed:
        raise DataError("adaptation seed must match the U sampling seed")
    source_hashes = artifact_hashes(args.checkpoint_dir)
    safe = any(name.endswith(".safetensors") for name in source_hashes)
    if not safe and not args.allow_unsafe_pickle_weights:
        raise DataError("pickle weights require explicit trusted-source confirmation")
    torch.set_num_threads(args.cpu_threads)
    random_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True, trust_remote_code=False)
    if tokenizer.mask_token_id is None:
        raise DataError("tokenizer has no mask token")
    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint_dir, local_files_only=True, trust_remote_code=False,
                                                use_safetensors=safe).to(device)
    if model.config.model_type != "bert" or artifact_hashes(args.checkpoint_dir) != source_hashes:
        raise DataError("BERT checkpoint invalid or changed during loading")
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        raise DataError("tokenizer vocabulary size does not match checkpoint embeddings")
    recipe = {"steps": args.steps, "batch_size": args.batch_size, "accumulation": args.accumulation,
              "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
              "max_length": args.max_length, "mask_probability": args.mask_probability,
              "mask_policy": "BERT 80% mask / 10% random / 10% unchanged; special tokens protected; force one per empty microbatch",
              "selection": "fixed_final_step_without_validation_or_test", "precision": "fp32"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    total_steps = 0
    try:
        dump(args.output_dir / "recipe.json", recipe)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        warmup = max(1, math.ceil(args.steps * 0.1))
        def scale(step):
            return (step + 1) / warmup if step < warmup else max(0.0, (args.steps - step) / max(1, args.steps - warmup))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
        stream = BatchStream(unlabeled, args.batch_size, args.seed + 3000)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        exposures = masked_count = forced_batches = 0
        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.0
            for _ in range(args.accumulation):
                batch = stream.next()
                inputs, masked, forced = masked_batch(tokenizer, batch, args.max_length, args.mask_probability, device)
                loss = model(**inputs, return_dict=True).loss
                if not torch.isfinite(loss):
                    raise DataError("non-finite MLM loss")
                (loss / args.accumulation).backward()
                mean_loss += float(loss.detach().item()) / args.accumulation
                exposures += len(batch)
                masked_count += masked
                forced_batches += int(forced)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            total_steps += 1
            append(args.output_dir / "mlm_log.jsonl", {"step": step, "learning_rate": learning_rate,
                                                       "mlm_loss": mean_loss, "u_pair_exposures_cumulative": exposures})
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        adapted = args.output_dir / "adapted_model"
        # Save only the encoder used for pair classification, not the discarded MLM head.
        model.base_model.save_pretrained(adapted, safe_serialization=True)
        tokenizer.save_pretrained(adapted)
        adapted_hashes = artifact_hashes(adapted)
        root = Path(__file__).resolve().parent.parent
        implementation = {str(path.relative_to(root)).replace("\\", "/"): digest(path) for path in
                          [Path(__file__).resolve(), root / "pairmatch" / "data.py", root / "pairmatch" / "transformer_runner.py"]}
        report = {"status": "COMPLETE", "split_manifest_sha256": provenance["split_manifest_sha256"],
                  "ldp_manifest_sha256": provenance["ldp_manifest_sha256"],
                  "unlabeled_sha256": provenance["unlabeled_sha256_expected"], "unlabeled_count": len(unlabeled),
                  "unlabeled_mode": provenance["unlabeled_mode"], "seed": args.seed,
                  "base_model": "google-bert/bert-base-chinese", "base_revision_user_attested": True,
                  "base_checkpoint_files_sha256": source_hashes, "adapted_files_sha256": adapted_hashes,
                  "recipe": recipe, "implementation_sha256": implementation,
                  "training_u_pair_exposures": exposures, "masked_tokens": masked_count,
                  "forced_nonempty_microbatches": forced_batches, "elapsed_seconds": elapsed,
                  "environment": version_info(device),
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                  "validation_or_test_text_used_for_mlm": False,
                  "test_file_read_during_adaptation": False,
                  "validation_read_only_for_split_and_loader_checks": True,
                  "warning": "Rights and pinned revision are user attestations; pretraining exposure remains unknown."}
        dump(adapted / "adaptation_manifest.json", report)
        dump(args.output_dir / "summary.json", report)
        return report
    except Exception as exc:
        dump(args.output_dir / "failure.json", {"status": "FAILED", "error_type": type(exc).__name__,
                                                 "executed_steps": total_steps})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--verification-report", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-data-rights", action="store_true")
    parser.add_argument("--confirm-reviewed-data", action="store_true")
    parser.add_argument("--confirm-model-rights-and-revision", action="store_true")
    parser.add_argument("--allow-unsafe-pickle-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    args = parser.parse_args()
    try:
        result = adapt(args)
        print(f"Offline U-only BERT adaptation complete: {result['recipe']['steps']} updates; adapted model is local.")
        return 0
    except (DataError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Adaptation failed ({type(exc).__name__}): {exc if isinstance(exc, DataError) else 'check inputs and local run'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
