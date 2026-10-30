#!/usr/bin/env python3
"""Export local surface-near-duplicate candidates for human review, not auto-deletion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from audit_jsonl import normalize_text
from prepare_clean_split import LexicalIndex, PreparationError, file_digest, write_jsonl
from verify_clean_split import verify


def candidates(split_dir: Path, output: Path, minimum: int = 80,
               max_rows: int = 5000, candidate_limit: int = 5000000) -> dict[str, Any]:
    if output.exists():
        raise PreparationError("output directory already exists; refusing to overwrite")
    if split_dir.resolve().is_relative_to(output.resolve()):
        raise PreparationError("the input split is inside the output directory")
    if not 1 <= minimum < 100 or max_rows < 1 or candidate_limit < 1:
        raise PreparationError("invalid candidate-review parameters")
    verified = verify(split_dir, candidate_limit)
    high = verified["lexical_threshold_percent"]
    if minimum >= high:
        raise PreparationError("review threshold must be below the grouping threshold")
    roles = ("train", "validation", "test", "unlabeled")
    records: dict[str, dict[str, dict[str, Any]]] = {role: {} for role in roles}
    for role in roles:
        with (split_dir / f"{role}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("sentence1", "sentence2"):
                    normalized = normalize_text(row[key])
                    entry = records[role].setdefault(normalized, {"text": row[key], "sample_ids": set()})
                    entry["sample_ids"].add(row["sample_id"])
    manifest_hash = file_digest(split_dir / "manifest.json")
    if manifest_hash != verified["split_manifest_sha256"]:
        raise PreparationError("split manifest changed during candidate review")
    with (split_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for role in roles:
        name = f"{role}.jsonl"
        if file_digest(split_dir / name) != manifest["output_sha256"][name]:
            raise PreparationError(f"split file changed during candidate review: {name}")

    all_texts = set().union(*(set(records[role]) for role in roles))
    index = LexicalIndex(all_texts, minimum, candidate_limit)
    kept: list[dict[str, Any]] = []
    total = 0

    def compare(role: str, text: str, other_role: str, other: str) -> None:
        nonlocal total
        left_tokens, right_tokens = index.tokens[text], index.tokens[other]
        intersection, union = len(left_tokens & right_tokens), len(left_tokens | right_tokens)
        if 100 * intersection >= high * union:
            raise PreparationError("grouping-threshold violation detected during candidate review")
        total += 1
        if len(kept) < max_rows:
            left, right = records[role][text], records[other_role][other]
            kept.append({"role_a": role, "sentence_a": left["text"], "sample_ids_a": sorted(left["sample_ids"]),
                         "role_b": other_role, "sentence_b": right["text"], "sample_ids_b": sorted(right["sample_ids"]),
                         "jaccard_intersection": intersection, "jaccard_union": union,
                         "decision": "PENDING_HUMAN_REVIEW"})

    for text in sorted(records["validation"]):
        index.add(text)
    for text in sorted(records["test"]):
        for other in index.matching(text):
            compare("test", text, "validation", other)
    for text in sorted(records["test"]):
        index.add(text)
    for role in ("train", "unlabeled"):
        for text in sorted(records[role]):
            for other in index.matching(text):
                other_role = "validation" if other in records["validation"] else "test"
                compare(role, text, other_role, other)

    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "review_candidates.jsonl", kept)
    report = {
        "schema_version": 1,
        "status": "PENDING_HUMAN_REVIEW" if total else "NO_CANDIDATES_UNDER_THIS_RULE",
        "parent_manifest_sha256": manifest_hash,
        "comparison": "character_bigram_set_jaccard",
        "minimum_percent_inclusive": minimum,
        "grouping_percent_exclusive": high,
        "candidate_sentence_pairs": total,
        "rows_saved": len(kept),
        "truncated": total > len(kept),
        "candidate_comparisons": index.comparisons,
        "candidate_file_sha256": file_digest(output / "review_candidates.jsonl"),
        "limitations": ["This is not a semantic-equivalence classifier.",
                        "Truncated output is deterministic, not a random sample.",
                        "Texts and IDs must stay local unless sharing is authorized."],
    }
    with (output / "summary.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-percent", type=int, default=80)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--candidate-limit", type=int, default=5000000)
    args = parser.parse_args(argv)
    try:
        report = candidates(args.split_dir, args.output_dir, args.minimum_percent, args.max_rows, args.candidate_limit)
        print(json.dumps({key: report[key] for key in ("status", "candidate_sentence_pairs", "rows_saved", "truncated")}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, PreparationError) as exc:
        message = str(exc) if isinstance(exc, PreparationError) else "check files, encoding and manifest"
        print(f"Candidate review not created: {type(exc).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
