#!/usr/bin/env python3
"""Inspect connected sentence groups for a candidate new split, without creating a split.

Inputs are read-only JSONL files. Output is aggregate-only; neither texts nor IDs
are emitted. A small largest component is necessary, not sufficient, for the
desired split or label balance. Review rights and source independence separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_jsonl import EVALUATION_LABELS, Pair, load_file


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def diagnose(paths: list[Path], largest_target_fraction: float = 0.8) -> dict[str, Any]:
    if not 0 < largest_target_fraction < 1:
        raise ValueError("largest_target_fraction must be in (0,1)")
    pair_labels: dict[Pair, set[str]] = defaultdict(set)
    malformed = 0
    unlabeled_rows = 0
    valid_binary_rows = 0
    for path in paths:
        data = load_file(path, "train")
        malformed += data.invalid_lines
        unlabeled_rows += data.labels["-1"]
        valid_binary_rows += data.labels["0"] + data.labels["1"]
        for pair, labels in data.unordered_labels.items():
            known = labels & EVALUATION_LABELS
            if known:
                pair_labels[pair].update(known)

    graph = UnionFind()
    for left, right in pair_labels:
        graph.union(left, right)
    component_pairs: Counter[str] = Counter()
    for left, _ in pair_labels:
        component_pairs[graph.find(left)] += 1
    sizes = sorted(component_pairs.values(), reverse=True)
    unique_pairs = len(pair_labels)
    largest = sizes[0] if sizes else 0
    conflicting_pairs = sum({"0", "1"}.issubset(labels) for labels in pair_labels.values())
    largest_fraction = largest / unique_pairs if unique_pairs else None
    return {
        "schema_version": 1,
        "input_basenames": [path.name for path in paths],
        "method": "NFKC_whitespace_removed_unordered_pair_sentence_graph",
        "labelled_rows_before_deduplication": valid_binary_rows,
        "unlabeled_rows_not_used": unlabeled_rows,
        "malformed_rows": malformed,
        "unique_normalized_unordered_labelled_pairs": unique_pairs,
        "duplicate_labelled_rows": valid_binary_rows - unique_pairs,
        "conflicting_label_pairs_require_manual_review": conflicting_pairs,
        "distinct_normalized_sentences": len(graph.parent),
        "sentence_connected_components": len(sizes),
        "largest_component_pair_count": largest,
        "largest_component_fraction": largest_fraction,
        "target_largest_split_fraction": largest_target_fraction,
        "largest_component_alone_exceeds_target": (
            largest_fraction > largest_target_fraction if largest_fraction is not None else None
        ),
        "limitations": [
            "No split or file is created; no assignment or stratification is attempted.",
            "Necessary size check only: passing does not prove feasible class balance or exact proportions.",
            "Sentence normalization and symmetry assumptions require manual review.",
            "Source-level and semantic near-duplicate independence are not verified.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--largest-target-fraction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, help="New report path; no overwrites")
    args = parser.parse_args(argv)
    try:
        result = diagnose(args.files, args.largest_target_fraction)
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        else:
            sys.stdout.write(text)
        if (result["malformed_rows"] or result["conflicting_label_pairs_require_manual_review"]
                or not result["unique_normalized_unordered_labelled_pairs"]
                or result["largest_component_alone_exceeds_target"]):
            return 2
        return 0
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Diagnostic failed: {type(exc).__name__}. Check paths, encoding and arguments.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
