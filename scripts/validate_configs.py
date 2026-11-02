#!/usr/bin/env python3
"""Validate proposal consistency. Success does not mean training is ready."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def load(name: str) -> dict[str, Any]:
    with (CONFIGS / name).open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ValueError(f"{name}: invalid schema version or root")
    return result


def validate() -> list[str]:
    errors: list[str] = []
    data = load("datasets.json")
    protocol = load("protocol.json")
    model_file = load("models.json")
    baseline_file = load("baseline_protocols.json")
    matrix = load("experiment_matrix.json")
    analysis = load("analysis_plan.json")
    target = load("target_server.json")

    datasets = [item["id"] for item in data["datasets"]]
    methods = [item["id"] for item in model_file["models"]]
    if len(datasets) != len(set(datasets)) or len(methods) != len(set(methods)):
        errors.append("dataset or model IDs are duplicated")
    for ds in data["datasets"]:
        for key in ("folder", "train", "supervised_reference", "validation", "test"):
            path = Path(ds[key])
            if path.is_absolute() or ".." in path.parts:
                errors.append(f"{ds['id']}: unsafe relative {key}")
    for tier in ("smoke", "core", "full", "extensions"):
        plan = matrix[tier]
        for dataset in plan["datasets"]:
            if dataset not in datasets:
                errors.append(f"{tier}: unknown dataset {dataset}")
        for method in plan["methods"]:
            if method not in methods:
                errors.append(f"{tier}: unknown method {method}")
        if not set(plan["training_seeds"]) <= set(protocol["sampling"]["training_seeds"]):
            errors.append(f"{tier}: seed is not in protocol")
        if not set(plan["labeled_fractions"]) <= set(protocol["sampling"]["labeled_fractions"]):
            errors.append(f"{tier}: fraction is not in protocol")
    ablations = [entry["id"] for entry in matrix["ablation_conditions"]]
    if len(ablations) != len(set(ablations)):
        errors.append("duplicate ablation ID")
    for method in ("esim_self_training", "esim_consistency", "esim_mean_teacher", "pair_dropout_pretraining", "bert_domain_adaptation"):
        if method not in methods or method not in baseline_file:
            errors.append(f"optional baseline missing definition: {method}")
    fractions = protocol["split"]["target_fractions_train_validation_test"]
    if len(fractions) != 3 or not math.isclose(sum(fractions), 1.0):
        errors.append("target split fractions must sum to 1")
    samples = protocol["sampling"]
    if samples["tuning_seed"] in samples["training_seeds"]:
        errors.append("tuning seed overlaps reporting seeds")
    if not samples["unlabeled_no_original_labels_visible"]:
        errors.append("unlabeled labels must be hidden before model access")
    train = protocol["training"]
    if sum(train["two_stage_step_caps"]) > train["max_optimizer_steps"]:
        errors.append("phase caps exceed total maximum")
    if train["bilstm_micro_batch"] * train["bilstm_accumulation_steps"] != train["effective_optimizer_batch_size"]:
        errors.append("BiLSTM effective batch inconsistent")
    if train["transformer_micro_batch"] * train["transformer_accumulation_steps"] != train["effective_optimizer_batch_size"]:
        errors.append("Transformer effective batch inconsistent")
    if protocol["execution_ready"] is not False:
        errors.append("execution_ready must remain false until the real-data gates are verified")
    if any(not 0 < p <= 1 for p in analysis["mask_sensitivity"]["probabilities"]):
        errors.append("invalid masking probability")
    if analysis["mask_sensitivity"]["selection_split"] != "validation_only":
        errors.append("masking sensitivity must be selected on validation only")
    if analysis["tsne"]["perplexity"] >= analysis["tsne"]["sample_max"]:
        errors.append("t-SNE perplexity must be less than the sample maximum")
    if target["evidence"] != "user_reported_not_verified_on_this_execution_host":
        errors.append("target hardware provenance missing")
    return errors


def main() -> int:
    try:
        errors = validate()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Invalid configuration: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Configuration structure and consistency: PASS")
    print("Real-data readiness: NOT READY (rights, review, lexicon, pretrained weights and target environment remain unresolved).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
