#!/usr/bin/env python3
"""Produce private, ID-linked errors and prespecified sentence-length summaries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pairmatch.analysis import load_result  # noqa: E402
from pairmatch.data import DataError, checked_rows, digest, read_json  # noqa: E402
from pairmatch.metrics import calculate  # noqa: E402
from pairmatch.runner import dump  # noqa: E402


BOUNDARIES = (0, 21, 31, 41, 51, float("inf"))


def length_group(length: int) -> int:
    return next(i for i in range(len(BOUNDARIES) - 1) if BOUNDARIES[i] <= length < BOUNDARIES[i + 1])


def exact_binomial_interval(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    interval = binomtest(successes, total).proportion_ci(confidence_level=0.95, method="exact")
    return [interval.low, interval.high]


def analyze(metrics_path: Path, split_dir: Path, output: Path, confirm: bool) -> dict:
    if not confirm:
        raise DataError("confirm that the error/length analysis was specified before inspecting test results")
    if output.exists():
        raise DataError("output directory exists; refusing to overwrite")
    if metrics_path.resolve().is_relative_to(output.resolve()) or split_dir.resolve().is_relative_to(output.resolve()):
        raise DataError("an input is inside the output directory")
    item = load_result(metrics_path)
    result = item["result"]
    manifest_path = split_dir / "manifest.json"
    if digest(manifest_path) != result["training_provenance"]["split_manifest_sha256"]:
        raise DataError("result and split manifest do not match")
    manifest = read_json(manifest_path)
    name = result["split"] + ".jsonl"
    if manifest["output_sha256"][name] != result["split_sha256"]:
        raise DataError("evaluation split hash mismatch")
    rows = checked_rows(split_dir / name, result["split_sha256"], True)
    by_id = {row["sample_id"]: row for row in rows}
    if set(by_id) != set(item["ids"]):
        raise DataError("evaluation samples do not match predictions")
    bins = [[] for _ in range(len(BOUNDARIES) - 1)]
    errors = []
    for identifier in item["ids"]:
        row, prediction = by_id[identifier], item["predictions"][identifier]
        if row["gold_label"] != prediction["gold_label"]:
            raise DataError("evaluation label differs from prediction record")
        length = len(row["sentence1"]) + len(row["sentence2"])
        group = length_group(length)
        bins[group].append(prediction)
        if prediction["gold_label"] != prediction["prediction"]:
            errors.append({"sample_id": identifier, "gold_label": prediction["gold_label"],
                           "prediction": prediction["prediction"], "probability_positive": prediction["probability_positive"],
                           "sum_original_character_length": length,
                           "error_type": "false_positive" if prediction["prediction"] == 1 else "false_negative"})
    summaries = []
    for index, group in enumerate(bins):
        positives = sum(row["gold_label"] == 1 for row in group)
        tp = sum(row["gold_label"] == 1 and row["prediction"] == 1 for row in group)
        metrics = calculate([r["gold_label"] for r in group], [r["probability_positive"] for r in group]) if group else None
        summaries.append({"left_inclusive": BOUNDARIES[index],
                          "right_exclusive": BOUNDARIES[index + 1] if math.isfinite(BOUNDARIES[index + 1]) else None,
                          "pair_count": len(group), "positive_count": positives, "true_positives": tp,
                          "positive_recall": tp / positives if positives else None,
                          "positive_recall_exact_95_iid_assumption": exact_binomial_interval(tp, positives),
                          "metrics": metrics})
    output.mkdir(parents=True, exist_ok=False)
    with (output / "errors.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in errors:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report = {"status": "COMPLETE", "metrics_sha256": item["metrics_sha256"],
              "predictions_sha256": item["predictions_sha256"], "split_sha256": result["split_sha256"],
              "length_unit": "sum of Unicode characters in the two original untruncated texts",
              "intervals": "left closed, right open", "bins": summaries, "error_count": len(errors),
              "errors_sha256": digest(output / "errors.jsonl"), "prespecified_user_attestation": True,
              "limitation": "Exact binomial intervals assume independent positive pairs; shared sources/sentences may require group bootstrap."
              " Human categorization requires authorized local review and is not inferred automatically."}
    dump(output / "length_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-prespecified-analysis", action="store_true")
    args = parser.parse_args()
    try:
        result = analyze(args.metrics, args.split_dir, args.output_dir, args.confirm_prespecified_analysis)
        print(f"Local length and error analysis complete: {result['error_count']} errors.")
        return 0
    except (DataError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Analysis failed ({type(exc).__name__}): {exc if isinstance(exc, DataError) else 'check inputs'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
