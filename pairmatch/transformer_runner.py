"""Offline, independently implemented pretrained Transformer pair baseline."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers

from .data import BatchStream, DataError, FullUpdateEpochBatchStream, digest, load_training_data
from .metrics import calculate
from .runner import append, dump, random_seed, version_info


ROOT = Path(__file__).resolve().parent.parent
MODEL_IDS = ("bert_base_zh", "roberta_wwm_zh", "small_zh", "bert_domain_adaptation")


def model_metadata(model_id: str) -> dict:
    with (ROOT / "configs" / "models.json").open(encoding="utf-8") as handle:
        entries = json.load(handle)["models"]
    return next(item for item in entries if item["id"] == model_id)


def artifact_hashes(directory: Path, require_model_config: bool = True) -> dict[str, str]:
    if not directory.is_dir():
        raise DataError("pretrained checkpoint directory is unavailable")
    allowed = {".json", ".txt", ".safetensors", ".bin", ".model", ".vocab", ".merges"}
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in allowed
                   and ".git" not in path.relative_to(directory).parts)
    if not files or (require_model_config and not (directory / "config.json").is_file()):
        raise DataError("local checkpoint must contain a model config and tokenizer/weight files")
    return {str(path.relative_to(directory)).replace("\\", "/"): digest(path) for path in files}


class TransformerPair(nn.Module):
    """Use CLS hidden state with a freshly initialized binary classification head."""
    def __init__(self, encoder: nn.Module, hidden_size: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.encoder(**inputs, return_dict=True)
        representation = result.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(representation)), representation


def tokenize(tokenizer: Any, rows: list[dict], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    batch = tokenizer([row["sentence1"] for row in rows], [row["sentence2"] for row in rows],
                      padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


def token_statistics(tokenizer: Any, rows: list[dict], max_length: int) -> dict:
    total_full = total_kept = truncated = 0
    for start in range(0, len(rows), 128):
        batch = rows[start:start + 128]
        full = tokenizer([r["sentence1"] for r in batch], [r["sentence2"] for r in batch],
                         padding=False, truncation=False, return_attention_mask=False)["input_ids"]
        kept = tokenizer([r["sentence1"] for r in batch], [r["sentence2"] for r in batch],
                         padding=False, truncation=True, max_length=max_length, return_attention_mask=False)["input_ids"]
        total_full += sum(map(len, full))
        total_kept += sum(map(len, kept))
        truncated += sum(len(a) > len(b) for a, b in zip(full, kept))
    return {"pair_count": len(rows), "truncated_pairs": truncated,
            "truncated_fraction": truncated / len(rows) if rows else None,
            "encoded_token_coverage_including_special_tokens": total_kept / total_full if total_full else None,
            "note": "Counts model-specific tokens including special tokens, not Unicode characters."}


def evaluate(model: TransformerPair, tokenizer: Any, data: list[dict], max_length: int,
             batch_size: int, device: torch.device) -> tuple[dict, list[float]]:
    model.eval()
    probabilities: list[float] = []
    loss_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            logits, _ = model(tokenize(tokenizer, batch, max_length, device))
            labels = torch.tensor([r["gold_label"] for r in batch], dtype=torch.long, device=device)
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
            probabilities.extend(logits.softmax(-1)[:, 1].cpu().tolist())
    metrics = calculate([r["gold_label"] for r in data], probabilities)
    metrics["cross_entropy"] = loss_sum / len(data)
    return metrics, probabilities


def restore(checkpoint: dict, tokenizer_dir: Path, device: torch.device) -> tuple[TransformerPair, Any]:
    if artifact_hashes(tokenizer_dir, False) != checkpoint["tokenizer_sha256"]:
        raise DataError("saved tokenizer files changed after training")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    model_config = dict(checkpoint["encoder_config"])
    model_type = model_config.pop("model_type")
    encoder_config = AutoConfig.for_model(model_type, **model_config)
    encoder = AutoModel.from_config(encoder_config, trust_remote_code=False)
    model = TransformerPair(encoder, encoder_config.hidden_size, checkpoint["config"]["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, tokenizer


def train(args: Any) -> dict:
    if not (args.confirm_data_rights and args.confirm_reviewed_data and args.confirm_model_rights_and_revision):
        raise DataError("confirm data rights, review, and pretrained model rights/revision")
    if args.model not in MODEL_IDS:
        raise DataError("unknown pretrained model")
    epoch_mode = args.steps is None
    from .tuning import apply_lock
    tuning_lock = apply_lock(args, args.model)
    if args.learning_rate is None:
        args.learning_rate = {"bert_base_zh": 2e-5, "small_zh": 5e-5}.get(args.model)
        if args.learning_rate is None:
            raise DataError("set --learning-rate or use a tuning lock for this model")
    if args.run_dir.exists():
        raise DataError("run directory exists; refusing to overwrite")
    if any(path.resolve().is_relative_to(args.run_dir.resolve()) for path in
           (args.split_dir, args.subset_dir, args.verification_report, args.checkpoint_dir)):
        raise DataError("an input is inside the output directory")
    if (min(args.seed, args.max_length, args.batch_size, args.accumulation, args.epochs, args.eval_every,
            args.patience, args.cpu_threads) < 1 or not all(math.isfinite(x) for x in
            (args.dropout, args.learning_rate, args.weight_decay, args.min_delta))
            or (not epoch_mode and args.steps < 1)
            or not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.weight_decay < 0 or args.min_delta < 0):
        raise DataError("invalid training settings")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise DataError("CUDA was requested but is unavailable")
    if args.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise DataError("set CUBLAS_WORKSPACE_CONFIG before importing torch")
    meta = model_metadata(args.model)
    source_hashes = artifact_hashes(args.checkpoint_dir)
    has_safetensors = any(name.endswith(".safetensors") for name in source_hashes)
    if not has_safetensors and not args.allow_unsafe_pickle_weights:
        raise DataError("no safetensors found; pickle weights require explicit trusted-source confirmation")
    labeled, validation, _, provenance = load_training_data(args.split_dir, args.subset_dir, args.verification_report, False)
    updates_per_epoch = len(labeled) // (args.batch_size * args.accumulation)
    if epoch_mode:
        if updates_per_epoch < 1:
            raise DataError("labeled data must contain at least one complete optimizer batch")
        args.steps = args.epochs * updates_per_epoch
        args.eval_every = updates_per_epoch
    adaptation = None
    if args.model == "bert_domain_adaptation":
        from .data import read_json
        adaptation = read_json(args.checkpoint_dir / "adaptation_manifest.json")
        if (adaptation.get("status") != "COMPLETE"
                or adaptation.get("split_manifest_sha256") != provenance["split_manifest_sha256"]
                or adaptation.get("ldp_manifest_sha256") != provenance["ldp_manifest_sha256"]
                or adaptation.get("seed") != args.seed
                or adaptation.get("unlabeled_sha256") != provenance["unlabeled_sha256_expected"]
                or adaptation.get("adapted_files_sha256") != {key: value for key, value in source_hashes.items() if key != "adaptation_manifest.json"}):
            raise DataError("adapted checkpoint does not match this seed, U pool or split")
    torch.set_num_threads(args.cpu_threads)
    random_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(args.checkpoint_dir, local_files_only=True, trust_remote_code=False,
                                        use_safetensors=has_safetensors)
    if artifact_hashes(args.checkpoint_dir) != source_hashes:
        raise DataError("pretrained checkpoint changed while loading")
    if not hasattr(encoder.config, "hidden_size"):
        raise DataError("checkpoint does not expose a compatible hidden size")
    if len(tokenizer) != encoder.get_input_embeddings().num_embeddings:
        raise DataError("tokenizer vocabulary size does not match checkpoint embeddings")
    model = TransformerPair(encoder, encoder.config.hidden_size, args.dropout).to(device)
    config = {"condition": args.model, "model_family": "transformer_cross_encoder_fresh_head",
              "pretrained_checkpoint": meta["pretrained_checkpoint"], "revision_user_attested": meta["revision"],
              "seed": args.seed, "device": args.device, "dropout": args.dropout, "max_length": args.max_length,
              "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
              "accumulation": args.accumulation, "steps": args.steps, "eval_every": args.eval_every,
              "training_mode": "epoch" if epoch_mode else "step",
              "epochs": args.epochs if epoch_mode else None,
              "updates_per_epoch": updates_per_epoch if epoch_mode else None,
              "labeled_pairs_per_epoch": (updates_per_epoch * args.batch_size * args.accumulation
                                          if epoch_mode else None),
              "labeled_tail_skipped_per_epoch": (len(labeled) % (args.batch_size * args.accumulation)
                                                 if epoch_mode else None),
              "patience": args.patience, "min_delta": args.min_delta, "fixed_steps": args.fixed_steps,
              "cpu_threads": args.cpu_threads, "threshold": 0.5, "training_U_accessed": adaptation is not None,
              "source_weights_safetensors": has_safetensors,
              "local_checkpoint_files_sha256": adaptation["base_checkpoint_files_sha256"] if adaptation else source_hashes,
              "adaptation_recipe": adaptation["recipe"] if adaptation else None,
              "adaptation_implementation_sha256": adaptation["implementation_sha256"] if adaptation else None,
              "tuning_lock": tuning_lock}
    args.run_dir.mkdir(parents=True, exist_ok=False)
    try:
        tokenizer_dir = args.run_dir / "tokenizer"
        tokenizer.save_pretrained(tokenizer_dir)
        tokenizer_hashes = artifact_hashes(tokenizer_dir, False)
        code_files = sorted((ROOT / "pairmatch").glob("*.py")) + [ROOT / "scripts" / "train_transformer.py"]
        provenance = {**provenance, "environment": {**version_info(device), "transformers": transformers.__version__},
                      "implementation_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in code_files},
                      "labeled_count": len(labeled), "u_count_used": adaptation["unlabeled_count"] if adaptation else 0,
                      "adapted_model_files_sha256": source_hashes if adaptation else None,
                      "adaptation_manifest_sha256": digest(args.checkpoint_dir / "adaptation_manifest.json") if adaptation else None,
                      "validation_count": len(validation),
                      "tokenizer_file_sha256": tokenizer_hashes,
                      "training_rights_and_review_are_user_attestations_not_automatically_proven": True,
                      "pretrained_revision_and_rights_user_attested_not_automatically_proven": True,
                      "truncation": {"labeled": token_statistics(tokenizer, labeled, args.max_length),
                                     "validation": token_statistics(tokenizer, validation, args.max_length)},
                      "test_was_not_read_during_training": True}
        dump(args.run_dir / "config.json", config)
        dump(args.run_dir / "provenance.json", provenance)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        warmup = max(1, math.ceil(args.steps * 0.1))
        def scale(step: int) -> float:
            return (step + 1) / warmup if step < warmup else max(0.0, (args.steps - step) / max(1, args.steps - warmup))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
        stream = (FullUpdateEpochBatchStream(labeled, args.batch_size, args.accumulation, args.seed + 1000)
                  if epoch_mode else BatchStream(labeled, args.batch_size, args.seed + 1000))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        best, patience_best, best_step, bad, exposures = -1.0, -1.0, 0, 0, 0
        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_mean = 0.0
            micro_steps = args.accumulation
            for _ in range(micro_steps):
                batch = stream.next()
                logits, _ = model(tokenize(tokenizer, batch, args.max_length, device))
                target = torch.tensor([row["gold_label"] for row in batch], dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, target)
                if not torch.isfinite(loss):
                    raise DataError("non-finite loss; run stopped")
                (loss / micro_steps).backward()
                loss_mean += float(loss.detach().item()) / micro_steps
                exposures += len(batch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            append(args.run_dir / "train_log.jsonl", {"stage": 1, "stage_step": step,
                   "global_executed_step": step, "learning_rate": learning_rate, "ce": loss_mean, "total": loss_mean})
            if step % args.eval_every == 0 or step == args.steps:
                metrics, _ = evaluate(model, tokenizer, validation, args.max_length, args.batch_size, device)
                new_best = metrics["macro_f1"] > best
                significant = metrics["macro_f1"] > patience_best + args.min_delta
                if significant:
                    patience_best, bad = metrics["macro_f1"], 0
                else:
                    bad += 1
                if new_best:
                    best, best_step = metrics["macro_f1"], step
                    torch.save({"format_version": 2, "model_state": model.state_dict(),
                                "encoder_config": encoder.config.to_dict(), "tokenizer_sha256": tokenizer_hashes,
                                "config": config, "provenance": provenance, "stage": 1, "stage_step": step,
                                "global_executed_step": step, "validation_macro_f1": best}, args.run_dir / "stage_1_best.pt")
                append(args.run_dir / "validation_log.jsonl", {"stage": 1, "stage_step": step,
                       "global_executed_step": step, "new_best": new_best,
                       "improved_by_min_delta_for_patience": significant, **metrics})
                if not args.fixed_steps and bad >= args.patience:
                    break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        summary = {"status": "COMPLETE", "total_executed_steps_including_rolled_back": step,
                   "stages": [{"stage": 1, "executed_steps": step, "best_step": best_step,
                               "best_validation_macro_f1": best, "labeled_exposures": exposures, "unlabeled_exposures": 0}],
                   "parameters": {"total": sum(p.numel() for p in model.parameters()),
                                  "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
                   "training_seconds_including_validation": time.perf_counter() - started,
                   "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                   "final_checkpoint": "stage_1_best.pt",
                   "warning": "Pretraining information differs from randomly initialized BiLSTM baselines."}
        dump(args.run_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        dump(args.run_dir / "failure.json", {"status": "FAILED", "error_type": type(exc).__name__})
        raise
