#!/usr/bin/env python3
"""Privacy-preserving, read-only audit for the supplied pair-classification JSONL format.

Only counts, file basenames, and file hashes are emitted; original text and labels
for individual records are never emitted. This tool does not alter input files,
create splits, execute a model, or assess semantic/source-level leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


Pair = tuple[str, str]
LABELS = frozenset({"-1", "0", "1"})
EVALUATION_LABELS = frozenset({"0", "1"})


def normalize_text(value: str) -> str:
    """NFKC + removal of Unicode whitespace; deliberately not semantic matching."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", value) if not ch.isspace())


def unordered(pair: Pair) -> Pair:
    return tuple(sorted(pair))  # type: ignore[return-value]


@dataclass
class DatasetSummary:
    path: Path
    sha256: str = ""
    nonempty_lines: int = 0
    invalid_lines: int = 0
    invalid_label_for_role: int = 0
    labels: Counter[str] = field(default_factory=Counter)
    ordered_counts: Counter[Pair] = field(default_factory=Counter)
    unordered_counts: Counter[Pair] = field(default_factory=Counter)
    ordered_labels: dict[Pair, set[str]] = field(default_factory=lambda: defaultdict(set))
    unordered_labels: dict[Pair, set[str]] = field(default_factory=lambda: defaultdict(set))
    sentences: set[str] = field(default_factory=set)
    labeled_ordered: set[Pair] = field(default_factory=set)
    unlabeled_ordered: set[Pair] = field(default_factory=set)

    @property
    def valid_rows(self) -> int:
        return sum(self.labels.values())

    def public_stats(self) -> dict[str, Any]:
        return {
            "basename": self.path.name,
            "sha256": self.sha256,
            "nonempty_lines": self.nonempty_lines,
            "valid_rows": self.valid_rows,
            "invalid_lines": self.invalid_lines,
            "labels_invalid_for_role": self.invalid_label_for_role,
            "label_counts": {label: self.labels[label] for label in ("0", "1", "-1")},
            "unique_ordered_pairs": len(self.ordered_counts),
            "unique_unordered_pairs": len(self.unordered_counts),
            "duplicate_rows_ordered": self.valid_rows - len(self.ordered_counts),
            "duplicate_rows_unordered": self.valid_rows - len(self.unordered_counts),
            "unique_normalized_sentences": len(self.sentences),
            "within_file_conflicting_unordered_pairs": sum(
                {"0", "1"}.issubset(labels) for labels in self.unordered_labels.values()
            ),
            "within_file_labeled_unlabeled_ordered_overlap": len(
                self.labeled_ordered & self.unlabeled_ordered
            ),
        }


def load_file(path: Path, role: str) -> DatasetSummary:
    result = DatasetSummary(path=path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_index, raw_line in enumerate(handle):
            digest.update(raw_line)
            line = raw_line.decode("utf-8-sig" if line_index == 0 else "utf-8")
            if not line.strip():
                continue
            result.nonempty_lines += 1
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record is not an object")
                s1, s2 = record["sentence1"], record["sentence2"]
                label = record["gold_label"]
                if not isinstance(s1, str) or not isinstance(s2, str) or isinstance(label, bool):
                    raise ValueError("wrong field type")
                label = str(label).strip()
                if label not in LABELS:
                    raise ValueError("unknown label")
                pair = (normalize_text(s1), normalize_text(s2))
                if not all(pair):
                    raise ValueError("empty normalized text")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                result.invalid_lines += 1
                continue

            result.labels[label] += 1
            if role in {"validation", "test", "supervised"} and label not in EVALUATION_LABELS:
                result.invalid_label_for_role += 1
            result.ordered_counts[pair] += 1
            result.unordered_counts[unordered(pair)] += 1
            result.ordered_labels[pair].add(label)
            result.unordered_labels[unordered(pair)].add(label)
            result.sentences.update(pair)
            (result.unlabeled_ordered if label == "-1" else result.labeled_ordered).add(pair)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise OSError("input changed during audit")
    result.sha256 = digest.hexdigest()
    return result


def compare(left: DatasetSummary, right: DatasetSummary) -> dict[str, int]:
    ordered_common = left.ordered_counts.keys() & right.ordered_counts.keys()
    unordered_common = left.unordered_counts.keys() & right.unordered_counts.keys()
    conflicting = sum(
        {"0", "1"}.issubset(left.unordered_labels[p] | right.unordered_labels[p])
        for p in unordered_common
    )
    return {
        "shared_unique_ordered_pairs": len(ordered_common),
        "shared_unique_unordered_pairs": len(unordered_common),
        "left_rows_with_shared_unordered_pair": sum(left.unordered_counts[p] for p in unordered_common),
        "right_rows_with_shared_unordered_pair": sum(right.unordered_counts[p] for p in unordered_common),
        "shared_unique_sentences": len(left.sentences & right.sentences),
        "shared_unordered_pairs_with_conflicting_labels": conflicting,
    }


def audit(paths: dict[str, Path], require_sentence_disjoint: bool = False) -> dict[str, Any]:
    data = {role: load_file(path, role) for role, path in paths.items()}
    summary = {role: item.public_stats() for role, item in data.items()}
    comparisons: dict[str, dict[str, int]] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        comparisons[f"{left}__{right}"] = compare(data[left], data[right])

    reasons: list[str] = []
    warnings: list[str] = []
    for role, item in summary.items():
        if item["valid_rows"] == 0:
            reasons.append(f"{role}: no valid records")
        if item["invalid_lines"] or item["labels_invalid_for_role"]:
            reasons.append(f"{role}: malformed records or labels invalid for role")
        if item["within_file_conflicting_unordered_pairs"]:
            reasons.append(f"{role}: conflicting binary labels for an unordered pair")
    for name, comparison in comparisons.items():
        if comparison["shared_unique_unordered_pairs"]:
            reasons.append(f"{name}: unordered pair overlap")
        if comparison["shared_unique_sentences"]:
            target = reasons if require_sentence_disjoint else warnings
            target.append(f"{name}: sentence overlap")

    supervised_check = None
    if "supervised" in data:
        sup = data["supervised"]
        train = data["train"]
        shared = sup.ordered_counts.keys() & train.ordered_counts.keys()
        shared_matching_label = sum(
            bool((sup.ordered_labels[p] & train.ordered_labels[p]) & EVALUATION_LABELS)
            for p in shared
        )
        supervised_check = {
            "unique_supervised_ordered_pairs": len(sup.ordered_counts),
            "present_in_train_ordered": len(shared),
            "present_in_train_with_matching_binary_label": shared_matching_label,
            "missing_from_train_ordered": len(sup.ordered_counts) - len(shared),
        }
        if shared_matching_label != len(sup.ordered_counts):
            reasons.append("supervised: not all ordered pairs have a matching labeled train record")

    return {
        "schema_version": 1,
        "method": {
            "text_normalization": "NFKC and removal of Unicode whitespace; case preserved",
            "pair_policy": "ordered and unordered exact equality",
            "semantic_near_duplicate_audit": "not performed",
            "source_level_audit": "not performed",
            "vocabulary_fit_audit": "not performed",
            "sentence_disjointness_required_for_gate": require_sentence_disjoint,
        },
        "files": summary,
        "cross_split": comparisons,
        "supervised_vs_train": supervised_check,
        "gate": {
            "status": "BLOCKED" if reasons else "EXACT_CHECKS_PASS",
            "reasons": reasons,
            "warnings": warnings,
            "interpretation": "Passing exact checks does not establish source or semantic independence.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("train", "validation", "test"):
        parser.add_argument(f"--{role}", type=Path, required=True)
    parser.add_argument("--supervised", type=Path, help="Optional labeled subset expected within train")
    parser.add_argument("--require-sentence-disjoint", action="store_true")
    parser.add_argument("--output", type=Path, help="New report path; existing files are not overwritten")
    args = parser.parse_args(argv)
    paths = {role: getattr(args, role) for role in ("train", "validation", "test")}
    if args.supervised is not None:
        paths["supervised"] = args.supervised

    try:
        result = audit(paths, require_sentence_disjoint=args.require_sentence_disjoint)
        serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
        else:
            sys.stdout.write(serialized)
        return 2 if result["gate"]["status"] == "BLOCKED" else 0
    except (OSError, UnicodeError) as exc:
        # Do not print record contents or a full private local path.
        print(f"Audit failed: {type(exc).__name__}. Check file paths and UTF-8 encoding.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
