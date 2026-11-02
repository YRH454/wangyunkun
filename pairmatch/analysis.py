"""Validated local prediction records and prespecified paired comparisons."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from .data import DataError, digest
from .metrics import calculate


METRICS = ("macro_f1", "accuracy", "positive_f1", "balanced_accuracy")


def metric_value(metrics: dict, name: str) -> float:
    if name == "positive_f1":
        return metrics["classes"]["1"]["f1"]
    if name == "balanced_accuracy":
        return metrics["balanced_accuracy_fixed_two_classes"]
    if name not in METRICS:
        raise DataError("unknown analysis metric")
    return metrics[name]


def load_result(metrics_path: Path) -> dict:
    with metrics_path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("status") != "COMPLETE" or result.get("split") not in ("validation", "test"):
        raise DataError("evaluation result is incomplete")
    prediction_path = metrics_path.parent / "predictions.jsonl"
    expected_prediction_hash = result.get("predictions_sha256")
    if expected_prediction_hash is not None and digest(prediction_path) != expected_prediction_hash:
        raise DataError("prediction file hash does not match evaluation record")
    predictions: dict[str, dict] = {}
    with prediction_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identifier, label, probability, prediction = (row.get(key) for key in
                ("sample_id", "gold_label", "probability_positive", "prediction"))
            if (not isinstance(identifier, str) or not identifier or identifier in predictions
                    or type(label) is not int or label not in (0, 1) or type(prediction) is not int
                    or prediction not in (0, 1) or type(probability) not in (int, float)
                    or not math.isfinite(probability) or not 0 <= probability <= 1
                    or prediction != int(probability >= 0.5)):
                raise DataError("invalid or duplicate prediction record")
            predictions[identifier] = row
    if not predictions or len(predictions) != result.get("count"):
        raise DataError("prediction count differs from evaluation result")
    if expected_prediction_hash is not None and digest(prediction_path) != expected_prediction_hash:
        raise DataError("prediction file changed while reading")
    ids = sorted(predictions)
    metrics = calculate([predictions[x]["gold_label"] for x in ids], [predictions[x]["probability_positive"] for x in ids])
    for name in METRICS:
        if not math.isclose(metric_value(metrics, name), metric_value(result["metrics"], name), rel_tol=1e-10, abs_tol=1e-10):
            raise DataError("saved metrics do not match predictions")
    if metrics["confusion_matrix_true_rows_predicted_columns"] != result["metrics"]["confusion_matrix_true_rows_predicted_columns"]:
        raise DataError("saved confusion matrix does not match predictions")
    config, provenance = result["training_config"], result["training_provenance"]
    if config.get("threshold") != 0.5 or config.get("seed") != provenance.get("sampling_seed"):
        raise DataError("prediction threshold or seed/sampling alignment is invalid")
    return {"result": result, "predictions": predictions, "ids": ids, "metrics": metrics,
            "metrics_sha256": digest(metrics_path), "predictions_sha256": digest(prediction_path)}


def u_information(config: dict, provenance: dict) -> tuple:
    condition = config["condition"]
    if condition in {"ce", "ce_scl", "A0", "A1", "bert_base_zh", "small_zh", "roberta_wwm_zh"}:
        return ("none",)
    if condition == "A5":
        return ("labeled_views", provenance["labeled_sha256"])
    return ("training_u", provenance["unlabeled_mode"], provenance["unlabeled_sha256_expected"],
            config.get("u_fraction", 1.0))


def wilcoxon_exact(differences: list[float]) -> dict:
    """Two-sided signed-rank test via exact sign enumeration with ties/zeros.

    Zero differences are discarded (Wilcox convention); equal absolute differences
    share average ranks. Values are rounded at 1e-12 solely to avoid float noise.
    """
    rounded = [round(value, 12) for value in differences]
    nonzero = [value for value in rounded if value != 0]
    if len(nonzero) > 50:
        raise DataError("too many seed pairs for exact signed-rank calculation")
    if not nonzero:
        return {"p_two_sided": 1.0, "nonzero_pairs": 0, "zero_pairs": len(rounded), "tied_absolute_values": False}
    sorted_values = sorted(set(abs(value) for value in nonzero))
    ranks: dict[float, int] = {}
    position = 1
    ties = False
    for value in sorted_values:
        count = sum(abs(x) == value for x in nonzero)
        ranks[value] = position + position + count - 1  # twice the average rank
        position += count
        ties = ties or count > 1
    rank_values = [ranks[abs(value)] for value in nonzero]
    total = sum(rank_values)
    positive = sum(rank for rank, value in zip(rank_values, nonzero) if value > 0)
    observed = min(positive, total - positive)
    distribution = {0: 1}
    for rank in rank_values:
        next_distribution: Counter[int] = Counter()
        for score, count in distribution.items():
            next_distribution[score] += count
            next_distribution[score + rank] += count
        distribution = dict(next_distribution)
    extreme = sum(count for score, count in distribution.items() if min(score, total - score) <= observed)
    return {"p_two_sided": min(1.0, extreme / (2 ** len(nonzero))), "nonzero_pairs": len(nonzero),
            "zero_pairs": len(rounded) - len(nonzero), "tied_absolute_values": ties,
            "statistic_W_min": observed / 2, "zero_method": "wilcox_drop_zeros",
            "calculation": "exact sign distribution of average tied ranks", "absolute_rounding_decimals": 12}


def load_groups(directory: Path, expected_split_sha: str, ids: list[str]) -> tuple[list[list[str]], dict]:
    with (directory / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    path = directory / "groups.jsonl"
    if manifest.get("status") != "COMPLETE" or manifest.get("split_sha256") != expected_split_sha or digest(path) != manifest.get("groups_sha256"):
        raise DataError("group file does not match the evaluation split")
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample, group = row.get("sample_id"), row.get("group_id")
            if not isinstance(sample, str) or sample in seen or not isinstance(group, str) or not group:
                raise DataError("invalid or duplicate group membership")
            seen.add(sample)
            groups.setdefault(group, []).append(sample)
    if seen != set(ids) or len(groups) != manifest["group_count"]:
        raise DataError("groups do not cover the same evaluation samples exactly")
    return [groups[key] for key in sorted(groups)], manifest


def bootstrap_difference(pairs: list[tuple[dict, dict]], ids: list[str], groups: list[list[str]],
                         metric: str, repetitions: int, seed: int) -> dict:
    if repetitions < 1 or len(groups) < 2:
        raise DataError("bootstrap requires positive repetitions and at least two independent units")
    rng = random.Random(seed)
    deltas = []
    missing_class = 0
    for _ in range(repetitions):
        sampled = [identifier for _ in range(len(groups)) for identifier in groups[rng.randrange(len(groups))]]
        labels = [pairs[0][0]["predictions"][identifier]["gold_label"] for identifier in sampled]
        missing_class += int(len(set(labels)) < 2)
        per_seed = []
        for left, right in pairs:
            l = calculate(labels, [left["predictions"][identifier]["probability_positive"] for identifier in sampled])
            r = calculate(labels, [right["predictions"][identifier]["probability_positive"] for identifier in sampled])
            per_seed.append(metric_value(l, metric) - metric_value(r, metric))
        deltas.append(statistics.mean(per_seed))
    ordered = sorted(deltas)
    def percentile(value: float) -> float:
        rank = (len(ordered) - 1) * value
        lo, hi = int(rank), min(int(rank) + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)
    return {"repetitions": repetitions, "seed": seed, "units": len(groups),
            "percentile_95_interval": [percentile(0.025), percentile(0.975)],
            "resamples_missing_a_class": missing_class,
            "estimand": "mean paired metric difference across the same training seeds; resample test units jointly",
            "limitation": "Test-sample uncertainty conditional on trained models; not cross-seed uncertainty."}


def paired(left_paths: list[Path], right_paths: list[Path], metric: str,
           allow_different_u: bool = False) -> tuple[dict, list[tuple[dict, dict]]]:
    if metric not in METRICS or not left_paths or not right_paths:
        raise DataError("a valid metric and at least one run on each side are required")
    def by_seed(paths: list[Path]) -> dict[int, dict]:
        outputs = [load_result(path) for path in paths]
        mapping = {item["result"]["training_config"]["seed"]: item for item in outputs}
        if len(mapping) != len(outputs):
            raise DataError("duplicate training seed on one comparison side")
        first = outputs[0]["result"]
        config = {k: v for k, v in first["training_config"].items() if k != "seed"}
        for item in outputs:
            r = item["result"]
            if ({k: v for k, v in r["training_config"].items() if k != "seed"} != config
                    or r["training_provenance"]["implementation_sha256"] != first["training_provenance"]["implementation_sha256"]):
                raise DataError("a comparison side mixes training settings or implementations")
        return mapping
    left, right = by_seed(left_paths), by_seed(right_paths)
    if set(left) != set(right):
        raise DataError("left and right do not contain the same paired seeds")
    pairs = []
    differences = []
    seed_results = []
    first = next(iter(left.values()))["result"]
    first_predictions = next(iter(left.values()))["predictions"]
    u_relations = set()
    for seed in sorted(left):
        a, b = left[seed], right[seed]
        la, rb = a["result"], b["result"]
        ap, bp = la["training_provenance"], rb["training_provenance"]
        if (la["split"] != rb["split"] or la["split_sha256"] != rb["split_sha256"]
                or la["split_sha256"] != first["split_sha256"]
                or ap["split_manifest_sha256"] != bp["split_manifest_sha256"]
                or ap["split_manifest_sha256"] != first["training_provenance"]["split_manifest_sha256"]
                or ap["labeled_percent"] != bp["labeled_percent"]
                or ap["labeled_percent"] != first["training_provenance"]["labeled_percent"]
                or ap["labeled_sha256"] != bp["labeled_sha256"]
                or ap["sampling_seed"] != bp["sampling_seed"] or a["ids"] != b["ids"]):
            raise DataError("pair does not share split, LDP, labeled IDs, sampling seed and evaluation IDs")
        if any(a["predictions"][x]["gold_label"] != b["predictions"][x]["gold_label"] for x in a["ids"]):
            raise DataError("paired predictions disagree on evaluation labels")
        if a["ids"] != sorted(first_predictions) or any(a["predictions"][x]["gold_label"] != first_predictions[x]["gold_label"] for x in a["ids"]):
            raise DataError("evaluation IDs or labels differ across seeds")
        ua, ub = u_information(la["training_config"], ap), u_information(rb["training_config"], bp)
        relation = "matched" if ua == ub else "different_training_information"
        if ua[0] != "none" and ub[0] != "none" and ua != ub and not allow_different_u:
            raise DataError("both methods use different U information; explicit comparison confirmation is required")
        u_relations.add(relation)
        av, bv = metric_value(a["metrics"], metric), metric_value(b["metrics"], metric)
        delta = av - bv
        differences.append(delta)
        seed_results.append({"seed": seed, "left": av, "right": bv, "difference_left_minus_right": delta,
                             "left_metrics_sha256": a["metrics_sha256"], "right_metrics_sha256": b["metrics_sha256"],
                             "left_predictions_sha256": a["predictions_sha256"], "right_predictions_sha256": b["predictions_sha256"]})
        pairs.append((a, b))
    report = {"status": "COMPLETE", "metric": metric, "split": first["split"], "split_sha256": first["split_sha256"],
              "split_manifest_sha256": first["training_provenance"]["split_manifest_sha256"],
              "labeled_percent": first["training_provenance"]["labeled_percent"],
              "left_method": pairs[0][0]["result"]["training_config"]["condition"],
              "right_method": pairs[0][1]["result"]["training_config"]["condition"],
              "seed_results": seed_results, "mean_difference_left_minus_right": statistics.mean(differences),
              "sample_sd_difference_ddof_1": statistics.stdev(differences) if len(differences) > 1 else None,
              "signed_rank_exact": wilcoxon_exact(differences), "training_information_relation": sorted(u_relations),
              "note": "The signed-rank test assumes exchangeable, symmetrically distributed independent seed differences; shared test samples remain shared."}
    return report, pairs


def holm(pvalues: list[float]) -> list[float]:
    if not pvalues or any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise DataError("invalid p-values")
    adjusted = [0.0] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=lambda i: (pvalues[i], i))):
        previous = max(previous, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted
