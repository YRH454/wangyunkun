#!/usr/bin/env python3
"""Prepare adjudication sheets and apply completed decisions to a NEW split.

The original split, candidate report and original inputs are never changed.
Human decisions are judgments, not automatic proof of semantic independence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from audit_jsonl import normalize_text
from prepare_clean_split import PreparationError, file_digest, prepare, write_jsonl
from verify_clean_split import verify


NEAR_DECISIONS = {"GROUP", "KEEP_SEPARATE", "EXCLUDE_A", "EXCLUDE_B", "EXCLUDE_BOTH"}
CONFLICT_DECISIONS = {"KEEP_QUARANTINED", "LABEL_0", "LABEL_1"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise PreparationError("expected a JSON object")
    return result


def load_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        result = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in result):
        raise PreparationError("expected JSON objects in review files")
    return result


def review_sources(split: Path, candidates: Path, candidate_limit: int) -> tuple[dict, list, list, str]:
    verification = verify(split, candidate_limit)
    parent_hash = file_digest(split / "manifest.json")
    if parent_hash != verification["split_manifest_sha256"]:
        raise PreparationError("parent split changed during verification")
    summary = load_json(candidates / "summary.json")
    candidate_file = candidates / "review_candidates.jsonl"
    if summary.get("parent_manifest_sha256") != parent_hash or file_digest(candidate_file) != summary.get("candidate_file_sha256"):
        raise PreparationError("candidate report is not bound to this split")
    if summary.get("truncated") is not False:
        raise PreparationError("candidate report is truncated; regenerate with a larger max-rows")
    near = load_lines(candidate_file)
    if len(near) != summary.get("rows_saved") or len(near) != summary.get("candidate_sentence_pairs"):
        raise PreparationError("candidate count mismatch")
    # Check the complete candidate-to-parent text and sample-ID mapping, without
    # treating the candidate's similarity as a human judgment.
    mapping: dict[str, dict[str, set[str]]] = {}
    parent = load_json(split / "manifest.json")
    for role in ("train", "validation", "test", "unlabeled"):
        path = split / f"{role}.jsonl"
        if file_digest(path) != parent["output_sha256"][path.name]:
            raise PreparationError("parent role file changed")
        by_text: dict[str, set[str]] = {}
        for row in load_lines(path):
            for key in ("sentence1", "sentence2"):
                by_text.setdefault(normalize_text(row[key]), set()).add(row["sample_id"])
        mapping[role] = by_text
    seen_edges: set[tuple[str, str, str, str]] = set()
    for row in near:
        try:
            role_a, role_b = row["role_a"], row["role_b"]
            a, b = normalize_text(row["sentence_a"]), normalize_text(row["sentence_b"])
            identity = (role_a, a, role_b, b)
            if (not a or not b or identity in seen_edges
                    or set(row["sample_ids_a"]) != mapping[role_a][a]
                    or set(row["sample_ids_b"]) != mapping[role_b][b]
                    or row["decision"] != "PENDING_HUMAN_REVIEW"):
                raise ValueError
            seen_edges.add(identity)
        except (KeyError, TypeError, ValueError) as exc:
            raise PreparationError("candidate row is not consistent with the parent split") from exc
    conflicts = sorted({row["sample_id"] for row in load_lines(split / "quarantined_conflicts.jsonl")})
    if len(conflicts) != parent["conflicting_pairs_quarantined"]:
        raise PreparationError("conflict count mismatch")
    if file_digest(split / "manifest.json") != parent_hash or file_digest(candidate_file) != summary["candidate_file_sha256"]:
        raise PreparationError("review inputs changed during checking")
    return parent, near, conflicts, parent_hash


def create_template(split: Path, candidates: Path, output: Path, candidate_limit: int = 5000000) -> dict:
    if output.exists():
        raise PreparationError("template directory exists; refusing to overwrite")
    if any(path.resolve().is_relative_to(output.resolve()) for path in (split, candidates)):
        raise PreparationError("an input is inside the output directory")
    _, near, conflicts, parent_hash = review_sources(split, candidates, candidate_limit)
    rows = ([{"kind": "near", "candidate_index": i, "decision": "PENDING", "reviewer": "", "reason": ""}
             for i in range(len(near))]
            + [{"kind": "conflict", "sample_id": identifier, "decision": "PENDING", "reviewer": "", "reason": ""}
               for identifier in conflicts])
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "decisions.jsonl", rows)
    report = {"schema_version": 1, "status": "PENDING_HUMAN_REVIEW", "parent_manifest_sha256": parent_hash,
              "candidate_file_sha256": file_digest(candidates / "review_candidates.jsonl"),
              "candidate_summary_sha256": file_digest(candidates / "summary.json"),
              "near_count": len(near), "conflict_count": len(conflicts),
              "decision_file_initial_sha256": file_digest(output / "decisions.jsonl"),
              "near_options": sorted(NEAR_DECISIONS), "conflict_options": sorted(CONFLICT_DECISIONS),
              "note": "Indices are zero-based in the immutable candidate file; edit only decisions.jsonl locally."}
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def apply(split: Path, candidates: Path, template: Path, inputs: list[Path], output: Path,
          confirm_before_test_analysis: bool, candidate_limit: int = 5000000) -> dict:
    if not confirm_before_test_analysis:
        raise PreparationError("confirm review and re-split were decided before inspecting test performance")
    if output.exists():
        raise PreparationError("output directory exists; refusing to overwrite")
    if any(path.resolve().is_relative_to(output.resolve()) for path in [split, candidates, template, *inputs]):
        raise PreparationError("an input is inside the output directory")
    parent, near, conflicts, parent_hash = review_sources(split, candidates, candidate_limit)
    template_manifest = load_json(template / "manifest.json")
    if (template_manifest.get("parent_manifest_sha256") != parent_hash
            or template_manifest.get("candidate_file_sha256") != file_digest(candidates / "review_candidates.jsonl")
            or template_manifest.get("candidate_summary_sha256") != file_digest(candidates / "summary.json")
            or template_manifest.get("near_count") != len(near)
            or template_manifest.get("conflict_count") != len(conflicts)):
        raise PreparationError("review template does not match the split and candidates")
    if [file_digest(path) for path in inputs] != [entry["sha256"] for entry in parent["inputs"]]:
        raise PreparationError("original inputs or their order do not match the parent manifest")
    decisions_path = template / "decisions.jsonl"
    decisions_hash = file_digest(decisions_path)
    decisions = load_lines(decisions_path)
    near_rows: dict[int, dict] = {}
    conflict_rows: dict[str, dict] = {}
    for row in decisions:
        if not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip() or not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise PreparationError("every decision needs a reviewer and a reason")
        if row.get("kind") == "near":
            index = row.get("candidate_index")
            if type(index) is not int or not 0 <= index < len(near) or index in near_rows or row.get("decision") not in NEAR_DECISIONS:
                raise PreparationError("invalid, pending or duplicated near-candidate decision")
            near_rows[index] = row
        elif row.get("kind") == "conflict":
            identifier = row.get("sample_id")
            if identifier not in conflicts or identifier in conflict_rows or row.get("decision") not in CONFLICT_DECISIONS:
                raise PreparationError("invalid, pending or duplicated conflict decision")
            conflict_rows[identifier] = row
        else:
            raise PreparationError("unknown review decision kind")
    if set(near_rows) != set(range(len(near))) or set(conflict_rows) != set(conflicts):
        raise PreparationError("all candidates and conflicts must have exactly one decision")
    edges: list[tuple[str, str]] = []
    excluded: set[str] = set()
    labels: dict[str, int] = {}
    prior_decisions: list[dict] = []
    if "human_review" in parent:
        edges.extend((row["sentence_a"], row["sentence_b"]) for row in load_lines(split / "reviewed_edges.jsonl"))
        excluded.update(row["sample_id"] for row in load_lines(split / "quarantined_human_review.jsonl"))
        for row in load_lines(split / "resolved_conflict_observations.jsonl"):
            identifier, value = row["sample_id"], row["reviewed_label"]
            if identifier in labels and labels[identifier] != value:
                raise PreparationError("previous conflict resolutions disagree")
            labels[identifier] = value
        prior_decisions = load_lines(split / "review_decisions.jsonl")
    for index, decision in near_rows.items():
        candidate = near[index]
        action = decision["decision"]
        if action == "GROUP":
            edges.append((normalize_text(candidate["sentence_a"]), normalize_text(candidate["sentence_b"])))
        if action in {"EXCLUDE_A", "EXCLUDE_BOTH"}:
            excluded.update(candidate["sample_ids_a"])
        if action in {"EXCLUDE_B", "EXCLUDE_BOTH"}:
            excluded.update(candidate["sample_ids_b"])
    labels.update({identifier: int(row["decision"][-1]) for identifier, row in conflict_rows.items()
                   if row["decision"] in {"LABEL_0", "LABEL_1"}})
    if file_digest(decisions_path) != decisions_hash:
        raise PreparationError("review decisions changed while reading")
    metadata = {"parent_manifest_sha256": parent_hash,
                "candidate_file_sha256": file_digest(candidates / "review_candidates.jsonl"),
                "candidate_summary_sha256": file_digest(candidates / "summary.json"),
                "template_manifest_sha256": file_digest(template / "manifest.json"),
                "decisions_sha256": decisions_hash,
                "action_counts_this_round": dict(Counter(row["decision"] for row in decisions)),
                "inherited_decision_count": len(prior_decisions),
                "pre_test_analysis_user_attestation": True}
    plan = {"edges": edges, "excluded_ids": excluded, "conflict_labels": labels,
            "metadata": metadata, "decisions": prior_decisions + [
                {**row, "review_round_parent_manifest_sha256": parent_hash} for row in decisions]}
    result = prepare(inputs, output, parent["seed"], parent["lexical_rule"]["threshold_percent"],
                     parent["source_field"], parent["maximum_absolute_fraction_error"],
                     parent["maximum_positive_rate_difference"], candidate_limit, plan)
    verification = verify(output, candidate_limit)
    with (output / "verification_report.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(verification, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return {"status": "NEW_SPLIT_VERIFIED_WITH_REVIEW_LIMITATIONS", "human_review": result["human_review"],
            "verification": verification["status"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    template = sub.add_parser("template")
    template.add_argument("--split-dir", required=True, type=Path)
    template.add_argument("--candidate-dir", required=True, type=Path)
    template.add_argument("--output-dir", required=True, type=Path)
    template.add_argument("--candidate-limit", type=int, default=5000000)
    run = sub.add_parser("apply")
    run.add_argument("--split-dir", required=True, type=Path)
    run.add_argument("--candidate-dir", required=True, type=Path)
    run.add_argument("--template-dir", required=True, type=Path)
    run.add_argument("--inputs", nargs="+", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--confirm-before-test-analysis", action="store_true")
    run.add_argument("--candidate-limit", type=int, default=5000000)
    args = parser.parse_args()
    try:
        if args.action == "template":
            result = create_template(args.split_dir, args.candidate_dir, args.output_dir, args.candidate_limit)
            print(json.dumps({key: result[key] for key in ("status", "near_count", "conflict_count")}, ensure_ascii=False))
        else:
            result = apply(args.split_dir, args.candidate_dir, args.template_dir, args.inputs, args.output_dir,
                           args.confirm_before_test_analysis, args.candidate_limit)
            print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, PreparationError) as exc:
        print(f"Review workflow failed ({type(exc).__name__}): {exc if isinstance(exc, PreparationError) else 'check local inputs'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
