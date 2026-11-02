#!/usr/bin/env python3
"""Derive conservative sentence/source groups within an audited evaluation split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pairmatch.data import DataError, checked_rows, digest  # noqa: E402
from pairmatch.runner import dump  # noqa: E402
from audit_jsonl import normalize_text  # noqa: E402
from component_diagnostic import UnionFind  # noqa: E402
from prepare_clean_split import LexicalIndex, PreparationError, sample_id, write_jsonl  # noqa: E402
from verify_clean_split import verify  # noqa: E402


def build(split: Path, role: str, output: Path, candidate_limit: int) -> dict:
    if role not in {"validation", "test"} or output.exists():
        raise DataError("invalid role or output directory exists")
    if split.resolve().is_relative_to(output.resolve()):
        raise DataError("input is inside output directory")
    verification = verify(split, candidate_limit)
    with (split / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    filename = f"{role}.jsonl"
    data = checked_rows(split / filename, manifest["output_sha256"][filename], True)
    graph = UnionFind()
    texts = set()
    for row in data:
        left, right = (normalize_text(row[key]) for key in ("sentence1", "sentence2"))
        graph.union("text:" + left, "text:" + right)
        texts.update((left, right))
        for source in row.get("source_ids", []):
            graph.union("text:" + left, "source:" + source)
    lexical = LexicalIndex(texts, manifest["lexical_rule"]["threshold_percent"], candidate_limit)
    for text in sorted(texts):
        for other in lexical.matching(text):
            graph.union("text:" + text, "text:" + other)
        lexical.add(text)
    if "human_review" in manifest:
        with (split / "reviewed_edges.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    graph.union("text:" + row["sentence_a"], "text:" + row["sentence_b"])
    components: dict[str, list[str]] = defaultdict(list)
    for row in data:
        root = graph.find("text:" + normalize_text(row["sentence1"]))
        components[root].append(row["sample_id"])
    rows = [{"sample_id": identifier, "group_id": min(members)}
            for members in components.values() for identifier in members]
    rows.sort(key=lambda row: row["sample_id"])
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "groups.jsonl", rows)
    report = {"status": "COMPLETE", "split": role, "split_sha256": manifest["output_sha256"][filename],
              "split_manifest_sha256": verification["split_manifest_sha256"], "sample_count": len(data),
              "group_count": len(components), "largest_group_size": max(map(len, components.values())),
              "groups_sha256": digest(output / "groups.jsonl"),
              "grouping": "exact sentence, chosen lexical threshold, supplied source IDs, and reviewed edges",
              "limitation": "Unobserved sources and semantic dependence are not ruled out."}
    dump(output / "manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=5000000)
    args = parser.parse_args()
    try:
        result = build(args.split_dir, args.split, args.output_dir, args.candidate_limit)
        print(f"Created {result['group_count']} local evaluation groups; independence remains conditional on available metadata.")
        return 0
    except (DataError, PreparationError, OSError, ValueError, KeyError) as exc:
        print(f"Group creation failed ({type(exc).__name__}): {exc if isinstance(exc, (DataError, PreparationError)) else 'check inputs'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
