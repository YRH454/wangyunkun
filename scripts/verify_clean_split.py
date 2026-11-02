#!/usr/bin/env python3
"""Re-read local split files and verify hashes and leakage rules."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from audit_jsonl import normalize_text, unordered
from prepare_clean_split import LexicalIndex, PreparationError, SPLITS, file_digest, sample_id
from component_diagnostic import UnionFind


def verify(directory: Path, candidate_limit: int = 5000000) -> dict[str, Any]:
    with (directory / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "COMPLETE_REQUIRES_INDEPENDENT_VERIFICATION":
        raise PreparationError("missing or incomplete split manifest")
    threshold = manifest["lexical_rule"]["threshold_percent"]
    if not isinstance(threshold, int) or not 1 <= threshold <= 100 or candidate_limit < 1:
        raise PreparationError("invalid verification threshold or candidate limit")
    if manifest.get("source_grouping") not in {"enabled", "not_available_not_verified"}:
        raise PreparationError("unknown source grouping status")
    filenames = [f"{name}.jsonl" for name in SPLITS] + ["unlabeled.jsonl", "quarantined_conflicts.jsonl"]
    review_enabled = "human_review" in manifest
    if review_enabled:
        filenames.extend(["reviewed_edges.jsonl", "quarantined_human_review.jsonl",
                          "resolved_conflict_observations.jsonl", "review_decisions.jsonl"])
    for name in filenames:
        if file_digest(directory / name) != manifest["output_sha256"][name]:
            raise PreparationError(f"file hash mismatch: {name}")

    texts: dict[str, set[str]] = {name: set() for name in (*SPLITS, "unlabeled")}
    sources: dict[str, set[str]] = {name: set() for name in texts}
    labels: dict[str, Counter[int]] = {name: Counter() for name in SPLITS}
    counts: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    labels_by_id: dict[str, int] = {}
    for name in texts:
        with (directory / f"{name}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                allowed = {"sample_id", "sentence1", "sentence2", "source_ids"}
                if name != "unlabeled":
                    allowed.add("gold_label")
                if not isinstance(row, dict) or not set(row).issubset(allowed):
                    raise PreparationError(f"unexpected fields in {name}")
                if not isinstance(row.get("sentence1"), str) or not isinstance(row.get("sentence2"), str):
                    raise PreparationError(f"invalid text field in {name}")
                pair = unordered((normalize_text(row["sentence1"]), normalize_text(row["sentence2"])))
                if not all(pair) or row.get("sample_id") != sample_id(pair):
                    raise PreparationError(f"invalid sample ID or normalized text in {name}")
                if pair in seen_pairs:
                    raise PreparationError("a normalized pair occurs more than once")
                seen_pairs.add(pair)
                texts[name].update(pair)
                if name != "unlabeled":
                    label = row.get("gold_label")
                    if type(label) is not int or label not in (0, 1):
                        raise PreparationError(f"invalid label in {name}")
                    labels[name][label] += 1
                    labels_by_id[row["sample_id"]] = label
                ids = row.get("source_ids", [])
                if not isinstance(ids, list) or any(not isinstance(item, str) or not item for item in ids):
                    raise PreparationError(f"invalid source IDs in {name}")
                if manifest["source_grouping"] == "enabled" and not ids:
                    raise PreparationError(f"missing source IDs in {name}")
                sources[name].update(ids)
                counts[name] += 1

    for name in SPLITS:
        if counts[name] != manifest["splits"][name]["count"] or not labels[name][0] or not labels[name][1]:
            raise PreparationError(f"split counts or class coverage invalid for {name}")
    if counts["unlabeled"] != manifest["unlabeled_kept"]:
        raise PreparationError("unlabeled count differs from manifest")
    comparisons = (("train", "validation"), ("train", "test"), ("validation", "test"),
                   ("unlabeled", "validation"), ("unlabeled", "test"))
    for left, right in comparisons:
        if texts[left] & texts[right]:
            raise PreparationError(f"exact sentence overlap: {left}/{right}")
        if manifest["source_grouping"] == "enabled" and sources[left] & sources[right]:
            raise PreparationError(f"source overlap: {left}/{right}")
    if review_enabled:
        graph = UnionFind()
        with (directory / "reviewed_edges.jsonl").open(encoding="utf-8") as handle:
            edge_rows = [json.loads(line) for line in handle if line.strip()]
        if len(edge_rows) != manifest["human_review"]["reviewed_group_edges"]:
            raise PreparationError("review edge count mismatch")
        for edge in edge_rows:
            if (not isinstance(edge, dict) or set(edge) != {"sentence_a", "sentence_b"}
                    or not all(isinstance(edge[key], str) and edge[key] and normalize_text(edge[key]) == edge[key] for key in edge)):
                raise PreparationError("invalid reviewed edge")
            graph.union(edge["sentence_a"], edge["sentence_b"])
        roots = {role: {graph.find(text) for text in values} for role, values in texts.items()}
        for left, right in comparisons:
            if roots[left] & roots[right]:
                raise PreparationError(f"human-reviewed group overlaps: {left}/{right}")
        with (directory / "quarantined_human_review.jsonl").open(encoding="utf-8") as handle:
            excluded_ids = {json.loads(line)["sample_id"] for line in handle if line.strip()}
        if len(excluded_ids) != manifest["human_review"]["quarantined_pair_ids"] or excluded_ids & {sample_id(pair) for pair in seen_pairs}:
            raise PreparationError("human-quarantined pairs present or count mismatch")
        with (directory / "resolved_conflict_observations.jsonl").open(encoding="utf-8") as handle:
            resolved = [json.loads(line) for line in handle if line.strip()]
        if len(resolved) != manifest["human_review"]["resolved_conflict_observations"]:
            raise PreparationError("resolved conflict observation count mismatch")
        resolved_ids: dict[str, int] = {}
        for row in resolved:
            identifier, label = row["sample_id"], row["reviewed_label"]
            if type(label) is not int or label not in (0, 1) or (identifier in resolved_ids and resolved_ids[identifier] != label):
                raise PreparationError("invalid reviewed conflict label")
            if identifier not in excluded_ids and labels_by_id.get(identifier) != label:
                raise PreparationError("resolved conflict label does not match the new split")
            resolved_ids[identifier] = label
        if len(resolved_ids) != manifest["human_review"]["resolved_conflict_pairs"]:
            raise PreparationError("resolved conflict pair count mismatch")

    all_texts = set().union(*texts.values())
    index = LexicalIndex(all_texts, threshold, candidate_limit)
    for text in sorted(texts["validation"]):
        index.add(text)
    for text in sorted(texts["test"]):
        if index.matching(text):
            raise PreparationError("lexical near-duplicate across validation/test")
    for text in sorted(texts["test"]):
        index.add(text)
    for name in ("train", "unlabeled"):
        for text in sorted(texts[name]):
            if index.matching(text):
                raise PreparationError(f"lexical near-duplicate between {name} and evaluation")
    # Detect ordinary concurrent edits after the initial hash checks or during parsing.
    for name in filenames:
        if file_digest(directory / name) != manifest["output_sha256"][name]:
            raise PreparationError(f"file changed during verification: {name}")
    return {
        "status": "EXACT_AND_CHOSEN_LEXICAL_CHECKS_PASS",
        "record_counts": {name: counts[name] for name in texts},
        "label_counts": {name: {str(label): labels[name][label] for label in (0, 1)} for name in SPLITS},
        "split_manifest_sha256": file_digest(directory / "manifest.json"),
        "python_version": platform.python_version(),
        "verification_implementation_sha256": {name: file_digest(Path(__file__).with_name(name)) for name in
                                             ("verify_clean_split.py", "prepare_clean_split.py", "audit_jsonl.py")},
        "lexical_threshold_percent": threshold,
        "lexical_candidate_comparisons": index.comparisons,
        "source_check": "passed_for_supplied_source_ids" if manifest["source_grouping"] == "enabled" else "not_available_not_verified",
        "human_review_groups_check": "passed_for_recorded_edges_and_exclusions" if review_enabled else "not_applied",
        "limitations": ["This lexical rule does not rule out all semantic near duplicates.",
                        "Generator exposure and data-use rights are not established."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--candidate-limit", type=int, default=5000000)
    parser.add_argument("--output", type=Path, help="Optional new aggregate-only local report")
    args = parser.parse_args(argv)
    try:
        report = verify(args.split_dir, args.candidate_limit)
        text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        print(text, end="")
        return 0
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, PreparationError) as exc:
        message = str(exc) if isinstance(exc, PreparationError) else "check manifest, files and encoding"
        print(f"Verification failed: {type(exc).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
