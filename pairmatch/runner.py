"""Train independent baselines/ablations on a pre-audited, locked local split."""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .data import BatchStream, DataError, digest, encode, load_training_data, mask_view, truncation, vocabulary
from .metrics import calculate
from .model import PairESIM, batch_encode
from .objectives import supervised_contrastive, unsupervised_contrastive


# Each tuple is (SCL enabled, U loss enabled). Every stage always includes CE.
CONDITIONS = {
    "ce": [(False, False)], "ce_scl": [(True, False)],
    "self_train": [(False, False), (False, True)], "consistency": [(False, False), (False, True)],
    "mean_teacher": [(False, False), (False, True)],
    "dropout_pretrain": [(False, True), (False, False)],
    "A0": [(False, False), (False, False)], "A1": [(True, False), (False, False)],
    "A2": [(False, False), (False, True)], "A3": [(True, False), (False, True)],
    "A4": [(True, True)], "A5": [(True, False), (False, True)],
    "A6": [(True, False), (False, True)], "A7": [(True, False), (False, True)],
    "A8": [(False, True), (True, False)],
}


def dump(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def version_info(device: torch.device) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        driver = "unavailable"
    git_status = git("status", "--porcelain")
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": str(torch.__version__),
            "cuda_build": torch.version.cuda, "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "nvidia_smi": driver, "git_commit": git("rev-parse", "HEAD"),
            "git_worktree_dirty_including_untracked": None if git_status == "unavailable" else bool(git_status),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "precision": "fp32"}


def random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def pairs_for(rows: list[dict], vocab: dict[str, int], max_length: int) -> list[tuple[list[int], list[int]]]:
    return [(encode(row["sentence1"], vocab, max_length), encode(row["sentence2"], vocab, max_length)) for row in rows]


def validation_result(model: PairESIM, data: list[dict], vocab: dict, max_length: int,
                      batch_size: int, device: torch.device) -> tuple[dict, list[float]]:
    model.eval()
    probabilities: list[float] = []
    loss_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            logits, _ = model(*batch_encode(pairs_for(batch, vocab, max_length), device))
            labels = torch.tensor([r["gold_label"] for r in batch], dtype=torch.long, device=device)
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
            probabilities.extend(logits.softmax(-1)[:, 1].cpu().tolist())
    metrics = calculate([r["gold_label"] for r in data], probabilities)
    metrics["cross_entropy"] = loss_sum / len(data)
    return metrics, probabilities


def loss_u(model: PairESIM, batch: list[dict], vocab: dict, args: Any, terms: list[str],
           rng: random.Random, device: torch.device) -> tuple[torch.Tensor | None, dict]:
    originals, augmented, ids = [], [], []
    counts = {"presented": len(batch), "unchanged": 0, "positions": 0, "total_positions": 0,
              "masked_types": 0, "eligible_types": 0}
    for row in batch:
        a, b, result = mask_view(row, terms, args.mask_rate, rng, args.max_length, args.condition == "A6")
        for key in ("positions", "total_positions", "masked_types", "eligible_types"):
            counts[key] += result[key]
        if not result["positions"]:
            counts["unchanged"] += 1
            continue
        originals.append((encode(row["sentence1"], vocab, args.max_length), encode(row["sentence2"], vocab, args.max_length)))
        augmented.append(([vocab.get(x, 1) for x in a], [vocab.get(x, 1) for x in b]))
        ids.append(row["sample_id"])
    counts["valid_views"] = len(ids)
    if len(ids) < 2 or len(set(ids)) < 2:
        counts["skipped_batch"] = 1
        return None, counts
    _, original_z = model(*batch_encode(originals, device))
    _, augmented_z = model(*batch_encode(augmented, device))
    loss, valid = unsupervised_contrastive(original_z, augmented_z, ids, args.temperature)
    counts["skipped_batch"] = int(not valid)
    return loss if valid else None, counts


def loss_consistency(model: PairESIM, batch: list[dict], vocab: dict, args: Any, terms: list[str],
                     rng: random.Random, device: torch.device) -> tuple[torch.Tensor | None, dict]:
    original, augmented = [], []
    counts = {"presented": len(batch), "unchanged": 0, "positions": 0, "total_positions": 0,
              "masked_types": 0, "eligible_types": 0, "accepted": 0}
    for row in batch:
        a, b, info = mask_view(row, terms, args.mask_rate, rng, args.max_length)
        for key in ("positions", "total_positions", "masked_types", "eligible_types"):
            counts[key] += info[key]
        if not info["positions"]:
            counts["unchanged"] += 1
            continue
        original.append((encode(row["sentence1"], vocab, args.max_length), encode(row["sentence2"], vocab, args.max_length)))
        augmented.append(([vocab.get(x, 1) for x in a], [vocab.get(x, 1) for x in b]))
    counts["valid_views"] = len(original)
    if not original:
        counts["skipped_batch"] = 1
        return None, counts
    # Weak view is stop-gradient; the same training-time dropout policy is used.
    with torch.no_grad():
        weak_logits, _ = model(*batch_encode(original, device))
        confidence, pseudo = weak_logits.softmax(-1).max(dim=-1)
        selected = confidence >= args.confidence
    counts["accepted"] = int(selected.sum().item())
    if not counts["accepted"]:
        counts["skipped_batch"] = 1
        return None, counts
    chosen = [pair for pair, keep in zip(augmented, selected.cpu().tolist()) if keep]
    strong_logits, _ = model(*batch_encode(chosen, device))
    counts["skipped_batch"] = 0
    return F.cross_entropy(strong_logits, pseudo[selected]), counts


def loss_mean_teacher(model: PairESIM, teacher: PairESIM, batch: list[dict], vocab: dict, args: Any,
                      terms: list[str], rng: random.Random, device: torch.device) -> tuple[torch.Tensor | None, dict]:
    original, augmented = [], []
    counts = {"presented": len(batch), "unchanged": 0, "positions": 0, "total_positions": 0,
              "masked_types": 0, "eligible_types": 0}
    for row in batch:
        a, b, info = mask_view(row, terms, args.mask_rate, rng, args.max_length)
        for key in ("positions", "total_positions", "masked_types", "eligible_types"):
            counts[key] += info[key]
        if not info["positions"]:
            counts["unchanged"] += 1
            continue
        original.append((encode(row["sentence1"], vocab, args.max_length), encode(row["sentence2"], vocab, args.max_length)))
        augmented.append(([vocab.get(x, 1) for x in a], [vocab.get(x, 1) for x in b]))
    counts["valid_views"] = len(original)
    counts["skipped_batch"] = int(not original)
    if not original:
        return None, counts
    teacher.eval()
    with torch.no_grad():
        weak, _ = teacher(*batch_encode(original, device))
        target = weak.softmax(-1)
    strong, _ = model(*batch_encode(augmented, device))
    return F.mse_loss(strong.softmax(-1), target), counts


def update_teacher(teacher: PairESIM, student: PairESIM, decay: float) -> None:
    with torch.no_grad():
        for old, new in zip(teacher.parameters(), student.parameters()):
            old.mul_(decay).add_(new, alpha=1 - decay)
        for old, new in zip(teacher.buffers(), student.buffers()):
            old.copy_(new)


def pseudo_labels(model: PairESIM, data: list[dict], vocab: dict, args: Any,
                  device: torch.device) -> tuple[list[dict], dict]:
    """One frozen-model pass; no gold U labels are present or accessed."""
    model.eval()
    accepted: list[dict] = []
    class_counts = {"0": 0, "1": 0}
    with torch.inference_mode():
        for start in range(0, len(data), args.u_batch_size):
            batch = data[start:start + args.u_batch_size]
            logits, _ = model(*batch_encode(pairs_for(batch, vocab, args.max_length), device))
            confidence, labels = logits.softmax(-1).max(-1)
            for row, confidence_value, label in zip(batch, confidence.cpu().tolist(), labels.cpu().tolist()):
                if confidence_value >= args.confidence:
                    accepted.append({**row, "pseudo_label": label})
                    class_counts[str(label)] += 1
    return accepted, {"candidate_count": len(data), "accepted_count": len(accepted),
                      "accepted_fraction": len(accepted) / len(data), "pseudo_label_counts": class_counts,
                      "threshold": args.confidence}


def read_terms(path: Path | None) -> list[str]:
    if path is None:
        raise DataError("a real term lexicon is required for U-loss conditions")
    with path.open(encoding="utf-8-sig") as handle:
        terms = sorted({line.strip() for line in handle if line.strip()})
    if not terms:
        raise DataError("term lexicon is empty")
    return terms


def make_model(config: dict, vocab_size: int) -> PairESIM:
    return PairESIM(vocab_size, config["embedding_dim"], config["hidden"], config["dropout"])


def train(args: Any) -> dict:
    if not args.confirm_data_rights or not args.confirm_reviewed_data:
        raise DataError("confirm data rights and completed/documented review before training")
    if args.condition not in CONDITIONS:
        raise DataError("unknown condition")
    from .tuning import apply_lock
    tuning_lock = apply_lock(args, args.condition)
    if args.run_dir.exists():
        raise DataError("run directory exists; refusing to overwrite")
    resolved_out = args.run_dir.resolve()
    if any(path.resolve().is_relative_to(resolved_out) for path in (args.split_dir, args.subset_dir, args.verification_report)):
        raise DataError("an input is inside the output directory")
    if args.term_lexicon and args.term_lexicon.resolve().is_relative_to(resolved_out):
        raise DataError("term lexicon is inside the output directory")
    finite = (args.learning_rate, args.weight_decay, args.dropout, args.temperature, args.mask_rate,
              args.scl_weight, args.ul_weight, args.min_delta, args.u_fraction, args.confidence, args.teacher_ema)
    if not all(math.isfinite(x) for x in finite):
        raise DataError("numeric settings must be finite")
    if (args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1 or args.temperature <= 0
            or not 0 <= args.mask_rate <= 1 or not 0.5 <= args.confidence <= 1 or not 0 <= args.teacher_ema < 1
            or min(args.scl_weight, args.ul_weight, args.min_delta) < 0
            or min(args.max_length, args.embedding_dim, args.hidden, args.batch_size, args.u_batch_size,
                   args.accumulation, args.eval_every, args.patience, args.stage1_steps, args.stage2_steps,
                   args.cpu_threads) < 1):
        raise DataError("invalid training setting")
    if args.u_fraction != 1.0 and args.condition != "A7":
        raise DataError("U fraction other than 1 is reserved for A7")
    if args.u_fraction not in (0.25, 0.5, 1.0):
        raise DataError("U fraction must be 0.25, 0.5 or 1")
    uses_u = any(u for _, u in CONDITIONS[args.condition])
    if uses_u and args.condition not in ("self_train", "consistency", "mean_teacher") and args.u_batch_size < 2:
        raise DataError("U microbatch must contain at least two pairs")
    needs_terms = uses_u and args.condition not in ("self_train", "dropout_pretrain")
    if args.condition == "dropout_pretrain" and args.dropout <= 0:
        raise DataError("dropout pretraining requires nonzero dropout")
    if needs_terms and args.term_lexicon is None:
        raise DataError("a real term lexicon is required for U-loss conditions")
    term_hash = digest(args.term_lexicon) if needs_terms else None
    terms = read_terms(args.term_lexicon) if needs_terms else []
    if needs_terms and digest(args.term_lexicon) != term_hash:
        raise DataError("term lexicon changed while reading")
    labeled, validation, unlabeled, provenance = load_training_data(
        args.split_dir, args.subset_dir, args.verification_report, uses_u and args.condition != "A5")
    if provenance["sampling_seed"] != args.seed:
        raise DataError("training seed must match the LDP sampling seed")
    if uses_u and args.condition == "A5":
        unlabeled = [{key: value for key, value in row.items() if key != "gold_label"} for row in labeled]
    if uses_u and args.condition == "A7":
        unlabeled = sorted(unlabeled, key=lambda r: r["sample_id"])
        random.Random(f"u-size:{provenance['sampling_seed']}").shuffle(unlabeled)
        unlabeled = unlabeled[:max(2, math.floor(len(unlabeled) * args.u_fraction))]
    if uses_u and len(unlabeled) < 2:
        raise DataError("at least two U pairs are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise DataError("CUDA was requested but is not available")
    if args.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise DataError("set CUBLAS_WORKSPACE_CONFIG before importing torch")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    random_seed(args.seed)
    vocab = vocabulary(labeled, args.max_length)
    config = {"condition": args.condition, "seed": args.seed, "device": args.device,
              "embedding_dim": args.embedding_dim, "hidden": args.hidden, "dropout": args.dropout,
              "max_length": args.max_length, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
              "batch_size": args.batch_size, "u_batch_size": args.u_batch_size, "accumulation": args.accumulation,
              "stage1_steps": args.stage1_steps, "stage2_steps": args.stage2_steps, "eval_every": args.eval_every,
              "patience": args.patience, "min_delta": args.min_delta, "fixed_steps": args.fixed_steps,
              "temperature": args.temperature, "scl_weight": args.scl_weight, "ul_weight": args.ul_weight,
              "mask_rate": args.mask_rate, "u_fraction": args.u_fraction, "cpu_threads": args.cpu_threads,
              "term_lexicon_sha256": term_hash,
              "pseudo_label_confidence": args.confidence,
              "teacher_ema": args.teacher_ema,
              "vocabulary_fitted_on": "selected_labeled_training_text_only",
              "stages_scl_ul": CONDITIONS[args.condition], "threshold": 0.5,
              "tuning_lock": tuning_lock}
    args.run_dir.mkdir(parents=True, exist_ok=False)
    dump(args.run_dir / "config.json", config)
    root = Path(__file__).resolve().parent.parent
    code_files = sorted((root / "pairmatch").glob("*.py")) + [root / "scripts" / "train_pairmatch.py"]
    provenance = {**provenance, "environment": version_info(device),
                  "implementation_sha256": {str(path.relative_to(root)).replace("\\", "/"): digest(path) for path in code_files},
                  "labeled_count": len(labeled), "u_count_used": len(unlabeled), "validation_count": len(validation),
                  "training_rights_and_review_are_user_attestations_not_automatically_proven": True,
                  "truncation": {"labeled": truncation(labeled, args.max_length), "validation": truncation(validation, args.max_length),
                                 "unlabeled_used": truncation(unlabeled, args.max_length)},
                  "test_was_not_read_during_training": True}
    dump(args.run_dir / "provenance.json", provenance)
    try:
        model = make_model(config, len(vocab)).to(device)
        parameters = {"total": sum(p.numel() for p in model.parameters()),
                      "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
    except Exception as exc:
        dump(args.run_dir / "failure.json", {"status": "FAILED", "error_type": type(exc).__name__,
                                              "phase": "model_initialization", "executed_steps": 0})
        raise
    started = time.perf_counter()
    total_steps = 0
    stage_summaries = []
    pseudo_pool: list[dict] = []
    pseudo_summary = None
    try:
        for stage_index, (use_scl, use_ul) in enumerate(CONDITIONS[args.condition], 1):
            limit = args.stage1_steps if stage_index == 1 else args.stage2_steps
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            teacher = copy.deepcopy(model).eval() if args.condition == "mean_teacher" and stage_index == 2 else None
            if teacher is not None:
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)
            warmup = max(1, math.ceil(limit * 0.1))
            def scale(step: int) -> float:
                if step < warmup:
                    return (step + 1) / warmup
                return max(0.0, (limit - step) / max(1, limit - warmup))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
            stream_l = BatchStream(labeled, args.batch_size, args.seed + 1000 * stage_index)
            u_pool = pseudo_pool if args.condition == "self_train" else unlabeled
            stream_u = BatchStream(u_pool, args.u_batch_size, args.seed + 1000 * stage_index + 1) if use_ul and u_pool else None
            mask_rng = random.Random(args.seed + 1000 * stage_index + 2)
            best, patience_best, best_step, bad = -1.0, -1.0, 0, 0
            counts = {"labeled_exposures": 0, "unlabeled_exposures": 0, "u_valid_views": 0,
                      "u_unchanged": 0, "u_skipped_batches": 0, "masked_positions": 0,
                      "mask_total_positions": 0, "masked_types": 0, "eligible_types": 0,
                      "scl_valid_anchors": 0, "scl_zero_anchor_batches": 0, "pseudo_accepted_exposures": 0}
            best_path = args.run_dir / f"stage_{stage_index}_best.pt"
            if args.condition == "dropout_pretrain" and stage_index == 1:
                # The classifier is not supervised in stage 1. Use the predeclared
                # final pretraining update, not validation labels, to choose it.
                assert stream_u is not None
                for step in range(1, limit + 1):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    mean_ul = 0.0
                    for _ in range(args.accumulation):
                        batch = stream_u.next()
                        inputs = batch_encode(pairs_for(batch, vocab, args.max_length), device)
                        _, first = model(*inputs)
                        _, second = model(*inputs)
                        loss, valid = unsupervised_contrastive(first, second, [row["sample_id"] for row in batch], args.temperature)
                        if not valid or not torch.isfinite(loss):
                            raise DataError("invalid dropout contrastive microbatch")
                        (loss / args.accumulation).backward()
                        mean_ul += float(loss.detach().item()) / args.accumulation
                        counts["unlabeled_exposures"] += len(batch)
                        counts["u_valid_views"] += len(batch)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    used_learning_rate = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    total_steps += 1
                    append(args.run_dir / "train_log.jsonl", {"stage": stage_index, "stage_step": step,
                           "global_executed_step": total_steps, "learning_rate": used_learning_rate,
                           "ce": 0.0, "scl": 0.0, "ul": mean_ul, "total": mean_ul})
                torch.save({"format_version": 1, "model_state": model.state_dict(), "vocab": vocab,
                            "config": config, "provenance": provenance, "stage": stage_index,
                            "stage_step": limit, "global_executed_step": total_steps,
                            "validation_macro_f1": None}, best_path)
                stage_summaries.append({"stage": stage_index, "executed_steps": limit, "best_step": limit,
                                        "best_validation_macro_f1": None,
                                        "selection": "fixed_final_pretraining_step_without_validation_selection", **counts})
                continue
            for step in range(1, limit + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                sums = {"ce": 0.0, "scl": 0.0, "ul": 0.0, "total": 0.0}
                for _ in range(args.accumulation):
                    batch = stream_l.next()
                    logits, z = model(*batch_encode(pairs_for(batch, vocab, args.max_length), device))
                    labels = torch.tensor([r["gold_label"] for r in batch], dtype=torch.long, device=device)
                    ce = F.cross_entropy(logits, labels)
                    total = ce
                    scl_value = ul_value = 0.0
                    counts["labeled_exposures"] += len(batch)
                    if use_scl:
                        scl, valid = supervised_contrastive(z, labels, args.temperature)
                        total = total + args.scl_weight * scl
                        scl_value = float(scl.detach().item())
                        counts["scl_valid_anchors"] += valid
                        counts["scl_zero_anchor_batches"] += int(valid == 0)
                    if stream_u:
                        u_batch = stream_u.next()
                        if args.condition == "self_train":
                            pseudo_logits, _ = model(*batch_encode(pairs_for(u_batch, vocab, args.max_length), device))
                            pseudo_y = torch.tensor([row["pseudo_label"] for row in u_batch], dtype=torch.long, device=device)
                            ul = F.cross_entropy(pseudo_logits, pseudo_y)
                            info = {"presented": len(u_batch), "valid_views": len(u_batch), "unchanged": 0,
                                    "skipped_batch": 0, "positions": 0, "total_positions": 0,
                                    "masked_types": 0, "eligible_types": 0, "accepted": len(u_batch)}
                        elif args.condition == "consistency":
                            ul, info = loss_consistency(model, u_batch, vocab, args, terms, mask_rng, device)
                        elif teacher is not None:
                            ul, info = loss_mean_teacher(model, teacher, u_batch, vocab, args, terms, mask_rng, device)
                        else:
                            ul, info = loss_u(model, u_batch, vocab, args, terms, mask_rng, device)
                        counts["unlabeled_exposures"] += info["presented"]
                        counts["pseudo_accepted_exposures"] += info.get("accepted", 0)
                        counts["u_valid_views"] += info["valid_views"]
                        counts["u_unchanged"] += info["unchanged"]
                        counts["u_skipped_batches"] += info["skipped_batch"]
                        counts["masked_positions"] += info["positions"]
                        counts["mask_total_positions"] += info["total_positions"]
                        counts["masked_types"] += info["masked_types"]
                        counts["eligible_types"] += info["eligible_types"]
                        if ul is not None:
                            total = total + args.ul_weight * ul
                            ul_value = float(ul.detach().item())
                    if not torch.isfinite(total):
                        raise DataError("non-finite loss; run stopped")
                    (total / args.accumulation).backward()
                    for key, value in (("ce", float(ce.detach().item())), ("scl", scl_value),
                                       ("ul", ul_value), ("total", float(total.detach().item()))):
                        sums[key] += value / args.accumulation
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                used_learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                if teacher is not None:
                    update_teacher(teacher, model, args.teacher_ema)
                scheduler.step()
                total_steps += 1
                append(args.run_dir / "train_log.jsonl", {"stage": stage_index, "stage_step": step,
                       "global_executed_step": total_steps, "learning_rate": used_learning_rate, **sums})
                if step % args.eval_every == 0 or step == limit:
                    metrics, _ = validation_result(model, validation, vocab, args.max_length, args.batch_size, device)
                    new_best = metrics["macro_f1"] > best
                    significant = metrics["macro_f1"] > patience_best + args.min_delta
                    if significant:
                        patience_best, bad = metrics["macro_f1"], 0
                    else:
                        bad += 1
                    if new_best:
                        best, best_step = metrics["macro_f1"], step
                        # This file is the current best within this NEW run, not a previous run.
                        torch.save({"format_version": 1, "model_state": model.state_dict(), "vocab": vocab,
                                    "config": config, "provenance": provenance, "stage": stage_index,
                                    "stage_step": step, "global_executed_step": total_steps,
                                    "validation_macro_f1": best}, best_path)
                    append(args.run_dir / "validation_log.jsonl", {"stage": stage_index, "stage_step": step,
                           "global_executed_step": total_steps, "new_best": new_best,
                           "improved_by_min_delta_for_patience": significant, **metrics})
                    if not args.fixed_steps and bad >= args.patience:
                        break
            # Only the best first-stage weights proceed; the optimizer/scheduler are recreated above.
            checkpoint = torch.load(best_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state"])
            stage_summaries.append({"stage": stage_index, "executed_steps": step, "best_step": best_step,
                                    "best_validation_macro_f1": best, **counts})
            if args.condition == "self_train" and stage_index == 1:
                pseudo_pool, pseudo_summary = pseudo_labels(model, unlabeled, vocab, args, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        summary = {"status": "COMPLETE", "total_executed_steps_including_rolled_back": total_steps,
                   "stages": stage_summaries, "pseudo_label_selection": pseudo_summary, "parameters": parameters,
                   "training_seconds_including_validation": time.perf_counter() - started,
                   "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                   "final_checkpoint": f"stage_{len(stage_summaries)}_best.pt",
                   "warning": "Independent implementation; no claim to reproduce original authors' results."}
        dump(args.run_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        dump(args.run_dir / "failure.json", {"status": "FAILED", "error_type": type(exc).__name__,
                                              "executed_steps": total_steps})
        raise
