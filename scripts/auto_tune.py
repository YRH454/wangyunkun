#!/usr/bin/env python3
"""Run a predeclared validation-only grid and freeze the selected settings."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pairmatch.data import DataError, digest, read_json  # noqa: E402
from pairmatch.runner import CONDITIONS, dump  # noqa: E402
from pairmatch.transformer_runner import MODEL_IDS  # noqa: E402


PAIR_SETTINGS = ("device", "cpu_threads", "max_length", "embedding_dim", "hidden", "dropout", "batch_size",
                 "u_batch_size", "accumulation", "stage1_steps", "stage2_steps", "eval_every", "patience",
                 "min_delta", "fixed_steps", "temperature", "scl_weight", "ul_weight", "mask_rate", "u_fraction")
TRANSFORMER_SETTINGS = ("device", "cpu_threads", "max_length", "dropout", "batch_size", "accumulation",
                        "steps", "eval_every", "patience", "min_delta", "fixed_steps")

# Candidate spaces explicitly stated for these three comparisons. Other methods
# need a separately supplied grid; these entries are not training results.
PAPER_GRIDS = {
    "ce": ([1e-3, 5e-4, 1e-4], [16, 32, 64]),
    "bert_base_zh": ([1e-5, 2e-5, 3e-5], [16, 32]),
    "small_zh": ([3e-5, 5e-5, 1e-4], [16, 32]),
}


def resolve_grid(args: argparse.Namespace) -> dict:
    is_transformer = args.method in MODEL_IDS
    specified = PAPER_GRIDS.get(args.method)
    rates = args.learning_rates if args.learning_rates is not None else (specified[0] if specified else None)
    micro_batch = args.batch_size if args.batch_size is not None else (8 if is_transformer else 16)
    effective = getattr(args, "effective_batch_sizes", None)
    if effective is not None and args.accumulation is not None and effective != [micro_batch * args.accumulation]:
        raise DataError("accumulation conflicts with the effective-batch candidates")
    # Preserve the former explicit --batch-size/--accumulation interface.
    if effective is None and args.accumulation is not None:
        effective = [micro_batch * args.accumulation]
    if effective is None:
        effective = specified[1] if specified else None
    legacy_decays = getattr(args, "weight_decays", None)
    if legacy_decays is not None and len(legacy_decays) != 1:
        raise DataError("weight decay is fixed; supply exactly one value")
    decay = legacy_decays[0] if legacy_decays is not None else getattr(args, "weight_decay", 0.01)
    if (not rates or not effective or micro_batch < 1 or len(set(rates)) != len(rates)
            or len(set(effective)) != len(effective)
            or any(not math.isfinite(x) or x <= 0 for x in rates)
            or any(x < 1 or x % micro_batch != 0 for x in effective)
            or not math.isfinite(decay) or decay < 0):
        raise DataError("invalid or missing learning-rate/effective-batch grid")
    standard = bool(specified and rates == specified[0] and effective == specified[1])
    if not standard and not args.allow_nonstandard_grid:
        raise DataError("nonstandard grid requires --allow-nonstandard-grid")
    return {"learning_rates": rates, "effective_batches": effective, "micro_batch": micro_batch,
            "weight_decay": decay, "standard_grid": standard}


def tune(args: argparse.Namespace) -> dict:
    if not args.confirm_data_rights or not args.confirm_reviewed_data:
        raise DataError("confirm data rights and reviewed data before tuning")
    is_transformer = args.method in MODEL_IDS
    if args.method not in CONDITIONS and not is_transformer:
        raise DataError("unknown tuning method")
    if is_transformer and (not args.checkpoint_dir or not args.confirm_model_rights_and_revision):
        raise DataError("a local checkpoint and model rights/revision confirmation are required")
    if args.output_dir.exists():
        raise DataError("tuning output exists; refusing to overwrite")
    inputs = [args.split_dir, args.subset_dir, args.verification_report]
    inputs.extend(path for path in (args.checkpoint_dir, args.term_lexicon) if path)
    if any(path.resolve().is_relative_to(args.output_dir.resolve()) for path in inputs):
        raise DataError("an input is inside the output directory")
    parent_hash = digest(args.split_dir / "manifest.json")
    child = read_json(args.subset_dir / "manifest.json")
    verification = read_json(args.verification_report)
    if (child.get("status") != "COMPLETE" or child.get("split_manifest_sha256") != parent_hash
            or verification.get("status") != "EXACT_AND_CHOSEN_LEXICAL_CHECKS_PASS"
            or verification.get("split_manifest_sha256") != parent_hash):
        raise DataError("split, subset and verification do not match")
    sampling_seed = child.get("sampling_seed")
    labeled_percent = child.get("percent_of_locked_labeled_training_pool")
    if not isinstance(sampling_seed, int) or not isinstance(labeled_percent, (int, float)) or labeled_percent <= 0:
        raise DataError("the subset manifest must record its sampling seed and labeled percentage")
    grid = resolve_grid(args)
    rates, effective_batches = grid["learning_rates"], grid["effective_batches"]
    batch, decay, standard_grid = grid["micro_batch"], grid["weight_decay"], grid["standard_grid"]
    if min(args.max_length, args.cpu_threads, args.eval_every) < 1:
        raise DataError("invalid fixed training settings")
    script = ROOT / "scripts" / ("train_transformer.py" if is_transformer else "train_pairmatch.py")
    one_stage = is_transformer or len(CONDITIONS[args.method]) == 1
    stage1_epochs = getattr(args, "stage1_epochs", None)
    if stage1_epochs is None:
        stage1_epochs = 32 if one_stage else 16
    stage2_epochs = getattr(args, "stage2_epochs", 16)
    epochs = getattr(args, "epochs", 32)
    step_mode = args.stage1_steps is not None or args.stage2_steps is not None
    default_steps = 4000 if one_stage else 2000
    steps = args.stage1_steps if args.stage1_steps is not None else default_steps
    stage2_steps = args.stage2_steps if args.stage2_steps is not None else 2000
    if min(steps, stage2_steps, stage1_epochs, stage2_epochs, epochs, args.embedding_dim, args.hidden) < 1:
        raise DataError("invalid model or training-budget setting")
    common = [sys.executable, "-B", str(script), "--split-dir", str(args.split_dir), "--subset-dir", str(args.subset_dir),
              "--verification-report", str(args.verification_report), "--confirm-data-rights", "--confirm-reviewed-data",
              "--seed", "101", "--device", args.device, "--cpu-threads", str(args.cpu_threads),
              "--max-length", str(args.max_length), "--batch-size", str(batch),
              "--eval-every", str(args.eval_every)]
    if is_transformer:
        common += ["--model", args.method, "--checkpoint-dir", str(args.checkpoint_dir),
                   "--confirm-model-rights-and-revision"]
        common += ["--steps", str(steps)] if step_mode else ["--epochs", str(epochs)]
        if args.allow_unsafe_pickle_weights:
            common.append("--allow-unsafe-pickle-weights")
    else:
        common += ["--condition", args.method, "--embedding-dim", str(args.embedding_dim), "--hidden", str(args.hidden)]
        common += (["--stage1-steps", str(steps), "--stage2-steps", str(stage2_steps)] if step_mode
                   else ["--stage1-epochs", str(stage1_epochs), "--stage2-epochs", str(stage2_epochs)])
        if any(use_ul for _, use_ul in CONDITIONS[args.method]):
            common += ["--u-batch-size", str(batch)]
        if args.term_lexicon:
            common += ["--term-lexicon", str(args.term_lexicon)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = {"status": "VALIDATION_GRID_STARTED", "method": args.method, "split_manifest_sha256": parent_hash,
            "subset_manifest_sha256": digest(args.subset_dir / "manifest.json"), "sampling_seed": sampling_seed,
            "tuning_seed": 101, "labeled_percent": labeled_percent, "learning_rates": rates,
            "effective_batch_sizes": effective_batches, "micro_batch_size": batch, "fixed_weight_decay": decay,
            "standard_grid": standard_grid, "training_mode": "step" if step_mode else "epoch",
            "selection": "highest validation macro-F1; earliest listed candidate on tie",
            "test_file_accessed": False}
    dump(args.output_dir / "tuning_plan.json", plan)
    trials = []
    for index, (rate, effective_batch) in enumerate(itertools.product(rates, effective_batches), 1):
        trial_dir = args.output_dir / f"trial_{index:02d}"
        accumulation = effective_batch // batch
        command = common + ["--run-dir", str(trial_dir), "--learning-rate", str(rate), "--weight-decay", str(decay),
                            "--accumulation", str(accumulation)]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        with (args.output_dir / f"trial_{index:02d}_process.txt").open("x", encoding="utf-8") as handle:
            handle.write(process.stdout + "\n" + process.stderr)
        if process.returncode != 0:
            trials.append({"trial": index, "learning_rate": rate, "weight_decay": decay,
                           "effective_batch_size": effective_batch, "status": "FAILED",
                           "returncode": process.returncode})
            dump(args.output_dir / "tuning_summary.json", {"status": "FAILED_NO_SELECTION", "trials": trials,
                 "note": "A failed candidate was not silently omitted; use a new directory to rerun."})
            raise DataError("a tuning trial failed; no winner was selected")
        summary = read_json(trial_dir / "summary.json")
        if summary.get("status") != "COMPLETE":
            raise DataError("tuning trial did not finish")
        trials.append({"trial": index, "learning_rate": rate, "weight_decay": decay,
                       "effective_batch_size": effective_batch, "status": "COMPLETE",
                       "validation_macro_f1": summary["stages"][-1]["best_validation_macro_f1"],
                       "executed_optimizer_steps": summary["total_executed_steps_including_rolled_back"]})
    selected = max(trials, key=lambda item: (item["validation_macro_f1"], -item["trial"]))
    selected_dir = args.output_dir / f"trial_{selected['trial']:02d}"
    config, provenance = read_json(selected_dir / "config.json"), read_json(selected_dir / "provenance.json")
    names = TRANSFORMER_SETTINGS if is_transformer else PAIR_SETTINGS
    fixed = {name: config[name] for name in names}
    if config["training_mode"] == "epoch":
        # The training runners materialize step counts from dataset size after
        # applying a lock. Lock the caller's epoch mode, not those derived counts.
        fixed.pop("eval_every")
        if is_transformer:
            fixed["steps"] = None
            fixed["epochs"] = config["epochs"]
        else:
            fixed["stage1_steps"] = fixed["stage2_steps"] = None
            fixed["stage1_epochs"] = config["stage1_epochs"]
            fixed["stage2_epochs"] = config["stage2_epochs"]
    if not is_transformer:
        fixed["confidence"] = config["pseudo_label_confidence"]
        fixed["teacher_ema"] = config["teacher_ema"]
    reference_budget = (not step_mode and batch == (8 if is_transformer else 16)
                        and (epochs == 32 if is_transformer else
                             (stage1_epochs == (32 if one_stage else 16) and stage2_epochs == 16)))
    lock = {"status": "VALIDATION_TUNING_COMPLETE", "method": args.method, "split_manifest_sha256": parent_hash,
            "sampling_seed": sampling_seed, "tuning_seed": 101, "labeled_percent": labeled_percent,
            "unlabeled_mode": child["unlabeled_mode"],
            "unlabeled_cap": child["unlabeled_cap"], "protocol_compliant": standard_grid and reference_budget,
            "standard_grid": standard_grid, "reference_budget": reference_budget,
            "selected": selected, "fixed_arguments": fixed,
            "term_lexicon_sha256": None if is_transformer else config["term_lexicon_sha256"],
            "pretrained_files_sha256": config.get("local_checkpoint_files_sha256") if is_transformer else None,
            "adaptation_recipe": config.get("adaptation_recipe") if is_transformer else None,
            "adaptation_implementation_sha256": config.get("adaptation_implementation_sha256") if is_transformer else None,
            "implementation_sha256": provenance["implementation_sha256"],
            "warning": "A matching grid is not proof of data rights, independent sources, fair tuning or real test performance."}
    dump(args.output_dir / "selected_lock.json", lock)
    report = {"status": "COMPLETE", "method": args.method, "trials": trials, "selected": selected,
              "lock_sha256": digest(args.output_dir / "selected_lock.json"), "test_file_accessed": False}
    dump(args.output_dir / "tuning_summary.json", report)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=tuple(CONDITIONS) + MODEL_IDS, required=True)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--subset-dir", type=Path, required=True, help="Training subset; its sampling seed is recorded separately")
    p.add_argument("--verification-report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--confirm-data-rights", action="store_true")
    p.add_argument("--confirm-reviewed-data", action="store_true")
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--confirm-model-rights-and-revision", action="store_true")
    p.add_argument("--allow-unsafe-pickle-weights", action="store_true")
    p.add_argument("--term-lexicon", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, help="Micro-batch size, default 16 for ESIM and 8 for Transformer")
    p.add_argument("--accumulation", type=int, help="Legacy shortcut for a single effective-batch candidate")
    p.add_argument("--effective-batch-sizes", nargs="+", type=int, help="Effective labeled batch candidates")
    p.add_argument("--epochs", type=int, default=32, help="Epoch budget for Transformer trials")
    p.add_argument("--stage1-epochs", type=int, help="Default 32 for one-stage, 16 for two-stage trials")
    p.add_argument("--stage2-epochs", type=int, default=16)
    p.add_argument("--stage1-steps", type=int, help="Use step-based mode instead of epoch-based mode")
    p.add_argument("--stage2-steps", type=int, help="Second-stage budget in step-based mode")
    p.add_argument("--eval-every", type=int, default=200, help="Validation interval only in step-based mode")
    p.add_argument("--embedding-dim", type=int, default=300)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--learning-rates", nargs="+", type=float)
    p.add_argument("--weight-decay", type=float, default=0.01, help="Fixed AdamW weight decay, not a search axis")
    p.add_argument("--weight-decays", nargs="+", type=float, help="Legacy alias; exactly one value")
    p.add_argument("--allow-nonstandard-grid", action="store_true")
    args = p.parse_args()
    try:
        result = tune(args)
        print(f"Validation-only tuning complete: {len(result['trials'])} candidates; lock saved locally.")
        return 0
    except (DataError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Tuning failed ({type(exc).__name__}): {exc if isinstance(exc, DataError) else 'check inputs and trial logs'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
