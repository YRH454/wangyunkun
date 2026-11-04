#!/usr/bin/env python3
"""Summarize independent seed results, rejecting mismatched data and settings."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("at least one metrics.json is required")
    results = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("status") != "COMPLETE":
            raise ValueError("an evaluation is incomplete")
        results.append(result)
    first = results[0]
    base_config = {k: v for k, v in first["training_config"].items() if k != "seed"}
    base_prov = first["training_provenance"]
    seeds = set()
    metrics = {"accuracy": [], "macro_f1": [], "positive_f1": [], "balanced_accuracy": []}
    for result in results:
        config, prov = result["training_config"], result["training_provenance"]
        if result["split"] != first["split"] or result["split_sha256"] != first["split_sha256"]:
            raise ValueError("evaluation splits do not match")
        if prov["split_manifest_sha256"] != base_prov["split_manifest_sha256"]:
            raise ValueError("parent splits do not match")
        if prov["implementation_sha256"] != base_prov["implementation_sha256"]:
            raise ValueError("training implementation differs")
        if prov["labeled_percent"] != base_prov["labeled_percent"] or prov["unlabeled_mode"] != base_prov["unlabeled_mode"]:
            raise ValueError("LDP or U source differs")
        for field in ("sampling_seed", "ldp_manifest_sha256", "labeled_sha256", "unlabeled_sha256_expected"):
            if prov[field] != base_prov[field]:
                raise ValueError("labeled subset or unlabeled data differs across runs")
        if {k: v for k, v in config.items() if k != "seed"} != base_config:
            raise ValueError("training configurations differ beyond seed")
        seed = config["seed"]
        if seed in seeds:
            raise ValueError("duplicate training seed")
        seeds.add(seed)
        m = result["metrics"]
        metrics["accuracy"].append(m["accuracy"])
        metrics["macro_f1"].append(m["macro_f1"])
        metrics["positive_f1"].append(m["classes"]["1"]["f1"])
        metrics["balanced_accuracy"].append(m["balanced_accuracy_fixed_two_classes"])
    return {"status": "COMPLETE", "condition": base_config["condition"], "split": first["split"],
            "labeled_percent": base_prov["labeled_percent"], "seeds": sorted(seeds),
            "missing_prespecified_reporting_seeds": sorted({11, 23, 37, 47, 59} - seeds),
            "metrics": {name: {"mean": statistics.mean(values),
                               "sample_sd_ddof_1": statistics.stdev(values) if len(values) > 1 else None}
                        for name, values in metrics.items()},
            "note": "Same split, condition and configuration; this is not a paired significance test."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local JSON file")
    args = parser.parse_args()
    try:
        result = summarize(args.metrics)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Summary written for {len(result['seeds'])} seed(s); results are local.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Summary failed ({type(exc).__name__}): {exc if isinstance(exc, ValueError) else 'check input files'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
