"""Apply a validation-selected hyperparameter lock to a subsequent run."""

from __future__ import annotations

import json
from pathlib import Path

from .data import DataError, digest


ROOT = Path(__file__).resolve().parent.parent


def apply_lock(args, method: str) -> dict | None:
    path = getattr(args, "tuning_lock", None)
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        lock = json.load(handle)
    if lock.get("status") != "VALIDATION_TUNING_COMPLETE" or lock.get("method") != method:
        raise DataError("tuning lock is incomplete or belongs to another method")
    if digest(args.split_dir / "manifest.json") != lock["split_manifest_sha256"]:
        raise DataError("tuning lock belongs to a different parent split")
    with (args.subset_dir / "manifest.json").open(encoding="utf-8") as handle:
        subset = json.load(handle)
    if subset.get("unlabeled_mode") != lock["unlabeled_mode"] or subset.get("unlabeled_cap") != lock["unlabeled_cap"]:
        raise DataError("U source or cap differs from the tuning reference")
    for key, expected in lock["fixed_arguments"].items():
        if getattr(args, key, None) != expected:
            raise DataError(f"training setting differs from tuning lock: {key}")
    expected_terms = lock.get("term_lexicon_sha256")
    if expected_terms and (not getattr(args, "term_lexicon", None) or digest(args.term_lexicon) != expected_terms):
        raise DataError("term lexicon differs from the tuning reference")
    if lock.get("pretrained_files_sha256") is not None:
        from .transformer_runner import artifact_hashes
        if method == "bert_domain_adaptation":
            with (args.checkpoint_dir / "adaptation_manifest.json").open(encoding="utf-8") as handle:
                adaptation = json.load(handle)
            actual_files = adaptation.get("base_checkpoint_files_sha256")
            if adaptation.get("recipe") != lock.get("adaptation_recipe"):
                raise DataError("domain adaptation recipe differs from tuning")
            if adaptation.get("implementation_sha256") != lock.get("adaptation_implementation_sha256"):
                raise DataError("domain adaptation code differs from tuning")
        else:
            actual_files = artifact_hashes(args.checkpoint_dir)
        if actual_files != lock["pretrained_files_sha256"]:
            raise DataError("pretrained checkpoint files differ from tuning")
    for relative, expected in lock["implementation_sha256"].items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT.resolve()) or not source.is_file() or digest(source) != expected:
            raise DataError("training code differs from the tuning implementation")
    args.learning_rate = lock["selected"]["learning_rate"]
    args.weight_decay = lock["selected"]["weight_decay"]
    return {"lock_sha256": digest(path), "protocol_compliant_grid_and_reference": lock["protocol_compliant"],
            "reference_sampling_seed": lock["sampling_seed"], "reference_labeled_percent": lock["labeled_percent"]}
