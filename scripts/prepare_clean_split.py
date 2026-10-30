#!/usr/bin/env python3
"""Create a new local pair-classification split without exact/lexical leakage.

Inputs are read-only JSON Lines; outputs are private derived data. Recombining
an old test set creates a NEW experiment, not an official benchmark split.
The lexical rule is an exact check for one character-bigram similarity measure,
not a semantic or provenance guarantee. Never upload the outputs without rights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from audit_jsonl import normalize_text, unordered
from component_diagnostic import UnionFind


Pair = tuple[str, str]
SPLITS = ("train", "validation", "test")
FRACTIONS = (0.8, 0.1, 0.1)


class PreparationError(Exception):
    """An input or predeclared allocation constraint was not satisfied."""


@dataclass
class Observation:
    sentence1: str
    sentence2: str
    label: int
    source_ids: tuple[str, ...]
    input_index: int
    line_number: int


@dataclass
class PairGroup:
    pair: Pair
    observations: list[Observation] = field(default_factory=list)
    labels: set[int] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.labels.add(observation.label)
        self.sources.update(observation.source_ids)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id(pair: Pair) -> str:
    payload = json.dumps(pair, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def shingles(text: str) -> frozenset[str]:
    """Distinct Unicode character bigrams; one-character strings use themselves."""
    return frozenset(text[i:i + 2] for i in range(len(text) - 1)) if len(text) > 1 else frozenset({text})


def similar(left: frozenset[str], right: frozenset[str], threshold: int) -> bool:
    if 100 * min(len(left), len(right)) < threshold * max(len(left), len(right)):
        return False
    return 100 * len(left & right) >= threshold * len(left | right)


class LexicalIndex:
    """Prefix-filtered exact Jaccard join for a fixed bigram threshold.

    A common global token order and a prefix of n-ceil(t*n)+1 tokens guarantee
    that every pair satisfying the chosen SET-Jaccard threshold is considered.
    This guarantee says nothing about synonyms or semantic equivalence.
    """

    def __init__(self, texts: Iterable[str], threshold: int, limit: int) -> None:
        self.threshold = threshold
        self.limit = limit
        self.comparisons = 0
        self.tokens = {text: shingles(text) for text in set(texts)}
        self.frequency: Counter[str] = Counter()
        for values in self.tokens.values():
            self.frequency.update(values)
        self.postings: dict[str, list[str]] = defaultdict(list)

    def prefix(self, values: frozenset[str]) -> list[str]:
        length = len(values) - ((self.threshold * len(values) + 99) // 100) + 1
        return sorted(values, key=lambda token: (self.frequency[token], token))[:length]

    def matching(self, text: str, allowed: set[str] | None = None) -> list[str]:
        values = self.tokens.get(text)
        if values is None:
            values = shingles(text)
        candidates: set[str] = set()
        for token in self.prefix(values):
            candidates.update(self.postings.get(token, ()))
        if allowed is not None:
            candidates.intersection_update(allowed)
        self.comparisons += len(candidates)
        if self.comparisons > self.limit:
            raise PreparationError("lexical candidate limit reached; no output was created")
        return [other for other in sorted(candidates)
                if other != text and similar(values, self.tokens[other], self.threshold)]

    def add(self, text: str) -> None:
        for token in self.prefix(self.tokens[text]):
            self.postings[token].append(text)


def load_inputs(paths: list[Path], source_field: str | None) -> tuple[dict[Pair, PairGroup], dict[Pair, PairGroup], list[dict[str, Any]]]:
    labeled: dict[Pair, PairGroup] = {}
    unlabeled: dict[Pair, PairGroup] = {}
    input_stats: list[dict[str, Any]] = []
    resolved = [path.resolve(strict=True) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise PreparationError("the same input was provided more than once")
    for index, path in enumerate(resolved):
        before = path.stat()
        digest = hashlib.sha256()
        labels: Counter[int] = Counter()
        with path.open("rb") as handle:
            for number, raw in enumerate(handle, start=1):
                digest.update(raw)
                line = raw.decode("utf-8-sig" if number == 1 else "utf-8")
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError
                    left, right, label = row["sentence1"], row["sentence2"], row["gold_label"]
                    if not isinstance(left, str) or not isinstance(right, str) or isinstance(label, bool):
                        raise ValueError
                    if str(label).strip() not in {"-1", "0", "1"}:
                        raise ValueError
                    label = int(str(label).strip())
                    pair = unordered((normalize_text(left), normalize_text(right)))
                    if not all(pair):
                        raise ValueError
                    source_ids: tuple[str, ...] = ()
                    if source_field:
                        raw_source = row[source_field]
                        raw_sources = [raw_source] if isinstance(raw_source, str) else raw_source
                        if (not isinstance(raw_sources, list) or not raw_sources
                                or any(not isinstance(item, str) or not item.strip() for item in raw_sources)):
                            raise ValueError
                        source_ids = tuple(sorted({item.strip() for item in raw_sources}))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise PreparationError(f"invalid record in input {index + 1}, line {number}") from exc
                observation = Observation(left, right, label, source_ids, index, number)
                target = unlabeled if label == -1 else labeled
                target.setdefault(pair, PairGroup(pair)).add(observation)
                labels[label] += 1
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise PreparationError("an input changed while it was being read")
        input_stats.append({"basename": path.name, "sha256": digest.hexdigest(),
                            "label_counts": {str(k): labels[k] for k in (-1, 0, 1)}})
    return labeled, unlabeled, input_stats


def allocate(groups: list[list[PairGroup]], seed: int, size_tolerance: float,
             class_tolerance: float) -> tuple[dict[str, list[PairGroup]], dict[str, Any]]:
    totals: Counter[int] = Counter()
    for group in groups:
        for item in group:
            totals[next(iter(item.labels))] += 1
    count = sum(totals.values())
    if count == 0 or not totals[0] or not totals[1]:
        raise PreparationError("both binary classes must exist after conflict quarantine")
    if len(groups) < 3 or max(map(len, groups)) > FRACTIONS[0] * count:
        raise PreparationError("connected groups cannot satisfy the proposed split")
    rnd = random.Random(seed)
    # Stable tie-breaking does not depend on filesystem order or PYTHONHASHSEED.
    ordered = sorted(groups, key=lambda group: min(sample_id(item.pair) for item in group))
    rnd.shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    assigned: dict[str, list[PairGroup]] = {name: [] for name in SPLITS}
    current: dict[str, Counter[int]] = {name: Counter() for name in SPLITS}
    for group in ordered:
        group_counts = Counter(next(iter(item.labels)) for item in group)
        def incremental_score(name: str, fraction: float) -> float:
            previous = sum(current[name].values())
            desired = count * fraction
            change = ((previous + len(group) - desired) ** 2 - (previous - desired) ** 2) / max(desired, 1)
            for label in (0, 1):
                before = current[name][label]
                target = totals[label] * fraction
                change += ((before + group_counts[label] - target) ** 2 - (before - target) ** 2) / max(target, 1)
            return change
        ranked = [(incremental_score(name, fraction), rnd.random(), name)
                  for name, fraction in zip(SPLITS, FRACTIONS)]
        chosen = min(ranked)[2]
        assigned[chosen].extend(group)
        current[chosen].update(group_counts)

    stats: dict[str, Any] = {}
    global_rate = totals[1] / count
    for name, fraction in zip(SPLITS, FRACTIONS):
        actual = sum(current[name].values())
        if not actual or not current[name][0] or not current[name][1]:
            raise PreparationError(f"{name} lacks a class; no output was created")
        if abs(actual / count - fraction) > size_tolerance:
            raise PreparationError(f"{name} exceeds the predeclared size tolerance")
        if abs(current[name][1] / actual - global_rate) > class_tolerance:
            raise PreparationError(f"{name} exceeds the predeclared class-balance tolerance")
        stats[name] = {"count": actual, "fraction": actual / count,
                       "label_counts": {str(label): current[name][label] for label in (0, 1)}}
    return assigned, stats


def clean_row(group: PairGroup, include_label: bool) -> dict[str, Any]:
    first = group.observations[0]
    result: dict[str, Any] = {"sample_id": sample_id(group.pair),
                              "sentence1": first.sentence1, "sentence2": first.sentence2}
    if include_label:
        result["gold_label"] = next(iter(group.labels))
    if group.sources:
        result["source_ids"] = sorted(group.sources)
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare(paths: list[Path], output: Path, seed: int = 20261030,
            threshold: int = 90, source_field: str | None = None,
            size_tolerance: float = 0.02, class_tolerance: float = 0.05,
            candidate_limit: int = 5000000) -> dict[str, Any]:
    if not paths or not 1 <= threshold <= 100 or not 0 <= size_tolerance < 1 or not 0 <= class_tolerance < 1 or candidate_limit < 1:
        raise PreparationError("invalid arguments")
    target = output.resolve()
    if output.exists():
        raise PreparationError("output directory already exists; refusing to overwrite")
    if any(path.resolve().is_relative_to(target) for path in paths):
        raise PreparationError("an input is inside the output directory")
    labeled, unlabeled, input_stats = load_inputs(paths, source_field)
    conflicts = {pair: group for pair, group in labeled.items() if group.labels == {0, 1}}
    for pair in conflicts:
        del labeled[pair]
    if not labeled:
        raise PreparationError("no eligible labeled records")

    graph = UnionFind()
    text_node = lambda text: "text:" + text
    for pair, group in labeled.items():
        graph.union(text_node(pair[0]), text_node(pair[1]))
        for source in group.sources:
            graph.union(text_node(pair[0]), "source:" + source)
    texts = {text for pair in labeled for text in pair}
    index = LexicalIndex(texts, threshold, candidate_limit)
    lexical_edges = 0
    for text in sorted(texts):
        for other in index.matching(text):
            graph.union(text_node(text), text_node(other))
            lexical_edges += 1
        index.add(text)
    components: dict[str, list[PairGroup]] = defaultdict(list)
    for pair, group in labeled.items():
        components[graph.find(text_node(pair[0]))].append(group)
    assigned, split_stats = allocate(list(components.values()), seed, size_tolerance, class_tolerance)

    eval_items = assigned["validation"] + assigned["test"]
    eval_texts = {text for group in eval_items for text in group.pair}
    eval_sources = {source for group in eval_items for source in group.sources}
    removed: Counter[str] = Counter()
    kept_unlabeled: list[PairGroup] = []
    for pair, group in sorted(unlabeled.items()):
        if pair in conflicts:
            removed["conflicting_labeled_pair"] += 1
        elif pair in labeled:
            removed["duplicates_any_labeled_pair"] += 1
        elif set(pair) & eval_texts:
            removed["exact_evaluation_sentence"] += 1
        elif group.sources & eval_sources:
            removed["evaluation_source"] += 1
        elif any(index.matching(text, eval_texts) for text in pair):
            removed["lexical_near_evaluation_sentence"] += 1
        else:
            kept_unlabeled.append(group)

    # Assertions defend against programming mistakes before any outputs are created.
    split_texts = {name: {text for group in items for text in group.pair} for name, items in assigned.items()}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            if split_texts[left] & split_texts[right]:
                raise PreparationError("internal sentence-disjointness check failed")
    if any(set(group.pair) & eval_texts or group.sources & eval_sources for group in kept_unlabeled):
        raise PreparationError("internal unlabeled-disjointness check failed")

    output.mkdir(parents=True, exist_ok=False)
    for name in SPLITS:
        rows = sorted(assigned[name], key=lambda item: sample_id(item.pair))
        write_jsonl(output / f"{name}.jsonl", (clean_row(item, True) for item in rows))
    write_jsonl(output / "unlabeled.jsonl", (clean_row(item, False) for item in sorted(kept_unlabeled, key=lambda item: sample_id(item.pair))))
    conflict_rows = []
    for pair, group in sorted(conflicts.items()):
        for observation in group.observations:
            conflict_rows.append({"sample_id": sample_id(pair), "sentence1": observation.sentence1,
                                  "sentence2": observation.sentence2, "gold_label": observation.label,
                                  "input_index": observation.input_index, "line_number": observation.line_number})
    write_jsonl(output / "quarantined_conflicts.jsonl", conflict_rows)
    filenames = [f"{name}.jsonl" for name in SPLITS] + ["unlabeled.jsonl", "quarantined_conflicts.jsonl"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_REQUIRES_INDEPENDENT_VERIFICATION",
        "policy": "new_recombined_split_not_an_official_or_historical_test",
        "seed": seed,
        "python_version": platform.python_version(),
        "implementation_sha256": {name: file_digest(Path(__file__).with_name(name)) for name in
                                  ("prepare_clean_split.py", "audit_jsonl.py", "component_diagnostic.py")},
        "normalization": "NFKC_remove_unicode_whitespace_preserve_case",
        "pair_policy": "unordered_pair_assumes_symmetric_matching",
        "lexical_rule": {"features": "distinct_unicode_character_bigrams",
                         "similarity": "set_jaccard", "threshold_percent": threshold,
                         "labeled_sentence_edges": lexical_edges,
                         "candidate_comparisons": index.comparisons, "candidate_limit": candidate_limit},
        "source_grouping": "enabled" if source_field else "not_available_not_verified",
        "source_field": source_field,
        "inputs": input_stats,
        "labeled_rows": sum(sum(item["label_counts"][str(i)] for i in (0, 1)) for item in input_stats),
        "labeled_unique_pairs_after_quarantine": len(labeled),
        "conflicting_pairs_quarantined": len(conflicts),
        "conflicting_observations_quarantined": sum(len(group.observations) for group in conflicts.values()),
        "duplicate_labeled_observations_excluding_conflicts": (
            sum(sum(item["label_counts"][str(i)] for i in (0, 1)) for item in input_stats)
            - sum(len(group.observations) for group in conflicts.values()) - len(labeled)),
        "connected_components": len(components),
        "largest_component_pairs": max(map(len, components.values())),
        "target_fractions": dict(zip(SPLITS, FRACTIONS)),
        "maximum_absolute_fraction_error": size_tolerance,
        "maximum_positive_rate_difference": class_tolerance,
        "splits": split_stats,
        "unlabeled_input_rows": sum(item["label_counts"]["-1"] for item in input_stats),
        "unlabeled_input_unique_pairs": len(unlabeled),
        "unlabeled_excluded_unique_pairs_by_first_reason": dict(removed),
        "unlabeled_kept": len(kept_unlabeled),
        "output_sha256": {name: file_digest(output / name) for name in filenames},
        "limitations": ["Exact and chosen lexical similarity checks do not prove semantic independence.",
                        "Source independence is unverified when source IDs are unavailable.",
                        "Generator training/pretraining exposure and data-use rights remain unverified.",
                        "Quarantined conflicts require human review; no label was inferred."]}
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--lexical-similarity-percent", type=int, default=90)
    parser.add_argument("--source-field", help="Require this source field (string or list of strings) on every input record")
    parser.add_argument("--size-tolerance", type=float, default=0.02)
    parser.add_argument("--class-tolerance", type=float, default=0.05)
    parser.add_argument("--candidate-limit", type=int, default=5000000)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.inputs, args.output_dir, args.seed, args.lexical_similarity_percent,
                         args.source_field, args.size_tolerance, args.class_tolerance, args.candidate_limit)
        print(json.dumps({"status": result["status"], "splits": result["splits"],
                          "unlabeled_kept": result["unlabeled_kept"],
                          "conflicting_pairs_quarantined": result["conflicting_pairs_quarantined"]}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, UnicodeError, PreparationError) as exc:
        # Input paths and text are intentionally not included.
        print(f"Split not completed: {type(exc).__name__}: {exc if isinstance(exc, PreparationError) else 'check files and encoding'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
