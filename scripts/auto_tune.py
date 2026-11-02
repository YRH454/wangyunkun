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
    if child.get("sampling_seed") != 101 or child.get("percent_of_locked_labeled_training_pool") != 30:
        raise DataError("tuning requires the predeclared 30-percent subset with sampling seed 101")
    default_rates = [1e-5, 2e-5, 3e-5] if is_transformer else [1e-4, 3e-4, 1e-3]
    rates = args.learning_rates if args.learning_rates is not None else default_rates
    decays = args.weight_decays if args.weight_decays is not None else [0.0, 0.01]
    if (not rates or not decays or len(set(rates)) != len(rates) or len(set(decays)) != len(decays)
            or any(not math.isfinite(x) or x <= 0 for x in rates)
            or any(not math.isfinite(x) or x < 0 for x in decays)):
        raise DataError("invalid or duplicated search values")
    standard_grid = rates == default_rates and decays == [0.0, 0.01]
    if not standard_grid and not args.allow_nonstandard_grid:
        raise DataError("nonstandard grid requires --allow-nonstandard-grid and must be reported")
    if min(args.max_length, args.cpu_threads, args.eval_every, args.stage2_steps) < 1:
        raise DataError("invalid fixed training settings")
    script = ROOT / "scripts" / ("train_transformer.py" if is_transformer else "train_pairmatch.py")
    batch = args.batch_size if args.batch_size is not None else (8 if is_transformer else 16)
    accumulation = args.accumulation if args.accumulation is not None else (4 if is_transformer else 2)
    default_steps = 2000 if args.method == "bert_domain_adaptation" else (4000 if is_transformer or len(CONDITIONS[args.method]) == 1 else 2000)
    steps = args.stage1_steps if args.stage1_steps is not None else default_steps
    if min(batch, accumulation, steps, args.embedding_dim, args.hidden) < 1:
        raise DataError("invalid batch, model or step setting")
    common = [sys.executable, "-B", str(script), "--split-dir", str(args.split_dir), "--subset-dir", str(args.subset_dir),
              "--verification-report", str(args.verification_report), "--confirm-data-rights", "--confirm-reviewed-data",
              "--seed", "101", "--device", args.device, "--cpu-threads", str(args.cpu_threads),
              "--max-length", str(args.max_length), "--batch-size", str(batch), "--accumulation", str(accumulation),
              "--eval-every", str(args.eval_every)]
    if is_transformer:
        common += ["--model", args.method, "--checkpoint-dir", str(args.checkpoint_dir),
                   "--confirm-model-rights-and-revision", "--steps", str(steps)]
        if args.allow_unsafe_pickle_weights:
            common.append("--allow-unsafe-pickle-weights")
    else:
        common += ["--condition", args.method, "--stage1-steps", str(steps), "--stage2-steps", str(args.stage2_steps),
                   "--embedding-dim", str(args.embedding_dim), "--hidden", str(args.hidden)]
        if args.term_lexicon:
            common += ["--term-lexicon", str(args.term_lexicon)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = {"status": "VALIDATION_GRID_STARTED", "method": args.method, "split_manifest_sha256": parent_hash,
            "subset_manifest_sha256": digest(args.subset_dir / "manifest.json"), "sampling_seed": 101,
            "labeled_percent": 30, "learning_rates": rates, "weight_decays": decays,
            "standard_six_candidate_grid": standard_grid, "selection": "highest validation macro-F1; earliest listed candidate on tie",
            "test_file_accessed": False}
    dump(args.output_dir / "tuning_plan.json", plan)
    trials = []
    for index, (rate, decay) in enumerate(itertools.product(rates, decays), 1):
        trial_dir = args.output_dir / f"trial_{index:02d}"
        command = common + ["--run-dir", str(trial_dir), "--learning-rate", str(rate), "--weight-decay", str(decay)]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        with (args.output_dir / f"trial_{index:02d}_process.txt").open("x", encoding="utf-8") as handle:
            handle.write(process.stdout + "\n" + process.stderr)
        if process.returncode != 0:
            trials.append({"trial": index, "learning_rate": rate, "weight_decay": decay, "status": "FAILED",
                           "returncode": process.returncode})
            dump(args.output_dir / "tuning_summary.json", {"status": "FAILED_NO_SELECTION", "trials": trials,
                 "note": "A failed candidate was not silently omitted; use a new directory to rerun."})
            raise DataError("a tuning trial failed; no winner was selected")
        summary = read_json(trial_dir / "summary.json")
        if summary.get("status") != "COMPLETE":
            raise DataError("tuning trial did not finish")
        trials.append({"trial": index, "learning_rate": rate, "weight_decay": decay, "status": "COMPLETE",
                       "validation_macro_f1": summary["stages"][-1]["best_validation_macro_f1"],
                       "executed_optimizer_steps": summary["total_executed_steps_including_rolled_back"]})
    selected = max(trials, key=lambda item: (item["validation_macro_f1"], -item["trial"]))
    selected_dir = args.output_dir / f"trial_{selected['trial']:02d}"
    config, provenance = read_json(selected_dir / "config.json"), read_json(selected_dir / "provenance.json")
    names = TRANSFORMER_SETTINGS if is_transformer else PAIR_SETTINGS
    fixed = {name: config[name] for name in names}
    if not is_transformer:
        fixed["confidence"] = config["pseudo_label_confidence"]
        fixed["teacher_ema"] = config["teacher_ema"]
    reference_budget = (args.max_length == 128 and args.eval_every == 200 and batch == (8 if is_transformer else 16)
                        and accumulation == (4 if is_transformer else 2) and steps == default_steps
                        and (is_transformer or (args.stage2_steps == 2000 and args.embedding_dim == 300 and args.hidden == 256)))
    lock = {"status": "VALIDATION_TUNING_COMPLETE", "method": args.method, "split_manifest_sha256": parent_hash,
            "sampling_seed": 101, "labeled_percent": 30, "unlabeled_mode": child["unlabeled_mode"],
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
    p.add_argument("--subset-dir", type=Path, required=True, help="30-percent, seed-101 training subset")
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
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulation", type=int)
    p.add_argument("--stage1-steps", type=int)
    p.add_argument("--stage2-steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--embedding-dim", type=int, default=300)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--learning-rates", nargs="+", type=float)
    p.add_argument("--weight-decays", nargs="+", type=float)
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
