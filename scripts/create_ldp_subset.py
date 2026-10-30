#!/usr/bin/env python3
"""Create one nested stratified labeled fraction and a label-free training U pool."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_clean_split import PreparationError, file_digest, write_jsonl
from verify_clean_split import verify


PERCENTAGES = (10, 15, 20, 25, 30, 35, 40, 45, 50)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def create(split_dir: Path, output: Path, percent: int, seed: int, mode: str,
           unlabeled_cap: int = 15000, candidate_limit: int = 5000000) -> dict[str, Any]:
    if percent not in PERCENTAGES or mode not in {"external", "remaining"} or unlabeled_cap < 0:
        raise PreparationError("invalid percentage, unlabeled mode or cap")
    if output.exists():
        raise PreparationError("output directory already exists; refusing to overwrite")
    if split_dir.resolve().is_relative_to(output.resolve()):
        raise PreparationError("the input split is inside the output directory")
    verification = verify(split_dir, candidate_limit)
    manifest_hash = file_digest(split_dir / "manifest.json")
    if manifest_hash != verification["split_manifest_sha256"]:
        raise PreparationError("split manifest changed after verification")
    with (split_dir / "manifest.json").open(encoding="utf-8") as handle:
        parent_manifest = json.load(handle)
    train = read_rows(split_dir / "train.jsonl")
    external = read_rows(split_dir / "unlabeled.jsonl")
    for name in ("train.jsonl", "unlabeled.jsonl"):
        if file_digest(split_dir / name) != parent_manifest["output_sha256"][name]:
            raise PreparationError(f"parent file changed after verification: {name}")
    if file_digest(split_dir / "manifest.json") != manifest_hash:
        raise PreparationError("split manifest changed during sampling")
    by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in train:
        by_label[row["gold_label"]].append(row)
    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    actual_by_label: Counter[int] = Counter()
    for label in (0, 1):
        rows = sorted(by_label[label], key=lambda row: row["sample_id"])
        random.Random(f"ldp:{seed}:{label}").shuffle(rows)
        count = (len(rows) * percent + 50) // 100  # integer half-up rounding
        if count < 1 or count >= len(rows):
            raise PreparationError("fraction does not leave suitable examples in both classes")
        selected.extend(rows[:count])
        remaining.extend(rows[count:])
        actual_by_label[label] = count
    selected_ids = {row["sample_id"] for row in selected}
    if mode == "remaining":
        # Never let the original gold label enter U or its output metadata.
        candidates = [{key: value for key, value in row.items() if key != "gold_label"}
                      for row in remaining]
    else:
        candidates = external
        if not candidates:
            raise PreparationError("the split has no external unlabeled pool; use remaining if appropriate")
    candidates = sorted(candidates, key=lambda row: row["sample_id"])
    random.Random(f"unlabeled:{seed}:{mode}").shuffle(candidates)
    chosen_u = candidates[:unlabeled_cap]
    if selected_ids & {row["sample_id"] for row in chosen_u}:
        raise PreparationError("labeled/unlabeled pair overlap detected")
    if any("gold_label" in row for row in chosen_u):
        raise PreparationError("a gold label would be exposed to the unlabeled output")

    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "train_labeled.jsonl", sorted(selected, key=lambda row: row["sample_id"]))
    write_jsonl(output / "train_unlabeled.jsonl", sorted(chosen_u, key=lambda row: row["sample_id"]))
    report = {
        "schema_version": 1,
        "status": "COMPLETE",
        "python_version": platform.python_version(),
        "implementation_sha256": {name: file_digest(Path(__file__).with_name(name)) for name in
                                  ("create_ldp_subset.py", "verify_clean_split.py", "prepare_clean_split.py")},
        "split_manifest_sha256": manifest_hash,
        "parent_verification": verification["status"],
        "percent_of_locked_labeled_training_pool": percent,
        "sampling_seed": seed,
        "nested_sampling": "per_class_fixed_seed_shuffle_then_prefix_half_up_rounding",
        "eligible_labeled_training_count": len(train),
        "eligible_label_counts": {str(label): len(by_label[label]) for label in (0, 1)},
        "labeled_count": len(selected),
        "labeled_class_counts": {str(label): actual_by_label[label] for label in (0, 1)},
        "unlabeled_mode": mode,
        "unlabeled_candidate_count": len(candidates),
        "unlabeled_cap": unlabeled_cap,
        "unlabeled_count": len(chosen_u),
        "output_sha256": {name: file_digest(output / name) for name in ("train_labeled.jsonl", "train_unlabeled.jsonl")},
        "note": "Local derived data only; external and remaining are different training-information conditions.",
    }
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--percent", type=int, choices=PERCENTAGES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--unlabeled-mode", choices=("external", "remaining"), required=True)
    parser.add_argument("--unlabeled-cap", type=int, default=15000)
    parser.add_argument("--candidate-limit", type=int, default=5000000)
    args = parser.parse_args(argv)
    try:
        result = create(args.split_dir, args.output_dir, args.percent, args.seed,
                        args.unlabeled_mode, args.unlabeled_cap, args.candidate_limit)
        print(json.dumps({key: result[key] for key in ("status", "labeled_count", "labeled_class_counts", "unlabeled_count")}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, PreparationError) as exc:
        message = str(exc) if isinstance(exc, PreparationError) else "check files, encoding and manifest"
        print(f"LDP subset not created: {type(exc).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
