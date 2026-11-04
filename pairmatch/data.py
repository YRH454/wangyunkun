"""Read locked local inputs without accessing the test file during training."""

from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


class DataError(ValueError):
    pass


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DataError("manifest/report must contain an object")
    return value


def normalized_pair(row: dict[str, Any]) -> tuple[str, str]:
    def normalize(text: str) -> str:
        return "".join(ch for ch in unicodedata.normalize("NFKC", text) if not ch.isspace())
    return tuple(sorted((normalize(row["sentence1"]), normalize(row["sentence2"]))))  # type: ignore[return-value]


def rows(path: Path, labeled: bool) -> list[dict[str, Any]]:
    output = []
    seen: set[str] = set()
    allowed = {"sample_id", "sentence1", "sentence2", "source_ids"}
    if labeled:
        allowed.add("gold_label")
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not set(row).issubset(allowed):
                    raise DataError("unexpected field")
                if not all(isinstance(row.get(key), str) for key in ("sentence1", "sentence2")):
                    raise DataError("invalid text")
                pair = normalized_pair(row)
                payload = json.dumps(pair, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                sample_id = hashlib.sha256(payload).hexdigest()
                if not all(pair) or row.get("sample_id") != sample_id or sample_id in seen:
                    raise DataError("invalid or duplicate sample ID")
                if labeled and (type(row.get("gold_label")) is not int or row["gold_label"] not in (0, 1)):
                    raise DataError("invalid label")
                sources = row.get("source_ids", [])
                if not isinstance(sources, list) or any(not isinstance(x, str) or not x for x in sources):
                    raise DataError("invalid source IDs")
                seen.add(sample_id)
                output.append(row)
            except (ValueError, TypeError, KeyError) as exc:
                raise DataError(f"invalid record in {path.name}, line {number}") from exc
    return output


def checked_rows(path: Path, expected: str, labeled: bool) -> list[dict[str, Any]]:
    if digest(path) != expected:
        raise DataError(f"hash mismatch: {path.name}")
    result = rows(path, labeled)
    if digest(path) != expected:
        raise DataError(f"file changed while reading: {path.name}")
    return result


def load_training_data(split: Path, subset: Path, report: Path, needs_external_u: bool) -> tuple[list, list, list, dict]:
    """Check the audited split/LDP relationship; never open the test file.

    A verification report is an integrity/provenance check, not proof of rights,
    completed manual review, absence of semantic leakage, or report authenticity.
    """
    manifest_path = split / "manifest.json"
    child_path = subset / "manifest.json"
    manifest_hash = digest(manifest_path)
    child_hash, report_hash = digest(child_path), digest(report)
    parent, child, verification = read_json(manifest_path), read_json(child_path), read_json(report)
    if parent.get("status") != "COMPLETE_REQUIRES_INDEPENDENT_VERIFICATION":
        raise DataError("incomplete parent split")
    if verification.get("status") != "EXACT_AND_CHOSEN_LEXICAL_CHECKS_PASS" or verification.get("split_manifest_sha256") != manifest_hash:
        raise DataError("verification report does not match the split manifest")
    if child.get("status") != "COMPLETE" or child.get("split_manifest_sha256") != manifest_hash:
        raise DataError("LDP subset is incomplete or belongs to a different split")
    if child.get("parent_verification") != verification["status"]:
        raise DataError("LDP verification status mismatch")
    labeled = checked_rows(subset / "train_labeled.jsonl", child["output_sha256"]["train_labeled.jsonl"], True)
    validation = checked_rows(split / "validation.jsonl", parent["output_sha256"]["validation.jsonl"], True)
    if not labeled or not validation or {r["gold_label"] for r in labeled} != {0, 1}:
        raise DataError("training L must contain both labels and validation must not be empty")
    if len(labeled) != child["labeled_count"]:
        raise DataError("labeled count mismatch")
    # The parent train pool is inspected only to check membership. Only selected L
    # labels are compared; the remaining labels never reach the model or U.
    parent_train = checked_rows(split / "train.jsonl", parent["output_sha256"]["train.jsonl"], True)
    train_by_id = {r["sample_id"]: r for r in parent_train}
    if any(train_by_id.get(r["sample_id"]) != r for r in labeled):
        raise DataError("labeled data does not match the parent train pool")
    unlabeled: list[dict[str, Any]] = []
    if needs_external_u:
        unlabeled = checked_rows(subset / "train_unlabeled.jsonl", child["output_sha256"]["train_unlabeled.jsonl"], False)
        if len(unlabeled) != child["unlabeled_count"]:
            raise DataError("unlabeled count mismatch")
        if child["unlabeled_mode"] == "external":
            available = checked_rows(split / "unlabeled.jsonl", parent["output_sha256"]["unlabeled.jsonl"], False)
        elif child["unlabeled_mode"] == "remaining":
            available = [{k: v for k, v in row.items() if k != "gold_label"} for row in parent_train]
        else:
            raise DataError("unknown unlabeled mode")
        available_by_id = {r["sample_id"]: r for r in available}
        if any(available_by_id.get(r["sample_id"]) != r for r in unlabeled):
            raise DataError("U does not match the declared training pool")
        if {r["sample_id"] for r in labeled} & {r["sample_id"] for r in unlabeled}:
            raise DataError("L/U overlap")
        if len(unlabeled) < 2:
            raise DataError("at least two distinct U pairs are required")
    if {r["sample_id"] for r in labeled + unlabeled} & {r["sample_id"] for r in validation}:
        raise DataError("training/validation sample overlap")
    if digest(manifest_path) != manifest_hash or digest(child_path) != child_hash or digest(report) != report_hash:
        raise DataError("a manifest or verification report changed while reading")
    meta = {
        "split_manifest_sha256": manifest_hash,
        "ldp_manifest_sha256": child_hash,
        "verification_report_sha256": report_hash,
        "validation_sha256": parent["output_sha256"]["validation.jsonl"],
        "test_sha256_expected_not_read": parent["output_sha256"]["test.jsonl"],
        "training_pool_sha256": parent["output_sha256"]["train.jsonl"],
        "labeled_sha256": child["output_sha256"]["train_labeled.jsonl"],
        "unlabeled_sha256_expected": child["output_sha256"]["train_unlabeled.jsonl"],
        "sampling_seed": child["sampling_seed"],
        "labeled_percent": child["percent_of_locked_labeled_training_pool"],
        "unlabeled_mode": child["unlabeled_mode"],
        "audit_status": verification["status"],
        "source_check": verification["source_check"],
    }
    return labeled, validation, unlabeled, meta


def vocabulary(training_rows: list[dict[str, Any]], max_length: int,
               max_characters: int | None = None) -> dict[str, int]:
    """Fit on training texts; an optional cap retains frequent characters deterministically."""
    counts = Counter(ch for row in training_rows for key in ("sentence1", "sentence2")
                     for ch in row[key][:max_length])
    if max_characters is not None:
        if type(max_characters) is not int or max_characters < 1:
            raise DataError("the character vocabulary size must be a positive integer")
        if len(counts) < max_characters:
            raise DataError(f"training texts contain {len(counts)} distinct characters; "
                            f"the requested character vocabulary contains {max_characters}")
        chars = sorted(counts, key=lambda ch: (-counts[ch], ch))[:max_characters]
    else:
        chars = counts
    return {token: i for i, token in enumerate(["<pad>", "<unk>", "<mask>"] + sorted(chars))}


def encode(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    return [vocab.get(ch, 1) for ch in text[:max_length]] or [1]


def truncation(rows_: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    lengths = [len(row[key]) for row in rows_ for key in ("sentence1", "sentence2")]
    total = sum(lengths)
    return {"sentence_count": len(lengths), "truncated_sentences": sum(x > max_length for x in lengths),
            "truncated_fraction": sum(x > max_length for x in lengths) / len(lengths) if lengths else None,
            "character_coverage": sum(min(x, max_length) for x in lengths) / total if total else None}


class BatchStream:
    """Seeded shuffle without repeated records inside any microbatch."""
    def __init__(self, data: list[dict[str, Any]], size: int, seed: int):
        if not data or size < 1:
            raise DataError("empty data or invalid microbatch size")
        self.data, self.size, self.rng = data, min(size, len(data)), random.Random(seed)
        self.order: list[int] = []
        self.position = 0

    def next(self, size: int | None = None) -> list[dict[str, Any]]:
        batch_size = self.size if size is None else min(size, len(self.data))
        if batch_size < 1:
            raise DataError("invalid microbatch size")
        if self.position + batch_size > len(self.order):
            self.order = list(range(len(self.data)))
            self.rng.shuffle(self.order)
            self.position = 0
        ids = self.order[self.position:self.position + batch_size]
        self.position += batch_size
        return [self.data[i] for i in ids]


class EpochBatchStream:
    """Shuffle each epoch and retain its final, potentially partial, microbatch."""
    def __init__(self, data: list[dict[str, Any]], size: int, seed: int):
        if not data or size < 1:
            raise DataError("empty data or invalid microbatch size")
        self.data, self.size, self.rng = data, size, random.Random(seed)
        self.order: list[int] = []
        self.position = 0

    def next(self) -> list[dict[str, Any]]:
        if self.position >= len(self.order):
            self.order = list(range(len(self.data)))
            self.rng.shuffle(self.order)
            self.position = 0
        ids = self.order[self.position:self.position + self.size]
        self.position += len(ids)
        return [self.data[i] for i in ids]


class FullUpdateEpochBatchStream:
    """Shuffle each epoch and keep only complete accumulated updates."""
    def __init__(self, data: list[dict[str, Any]], size: int, accumulation: int, seed: int):
        if not data or size < 1 or accumulation < 1:
            raise DataError("empty data or invalid microbatch/accumulation size")
        self.data, self.size, self.rng = data, size, random.Random(seed)
        effective_batch_size = size * accumulation
        self.usable = len(data) // effective_batch_size * effective_batch_size
        if not self.usable:
            raise DataError("labeled data must contain at least one complete optimizer batch")
        self.order: list[int] = []
        self.position = 0

    def next(self) -> list[dict[str, Any]]:
        if self.position >= len(self.order):
            order = list(range(len(self.data)))
            self.rng.shuffle(order)
            self.order = order[:self.usable]
            self.position = 0
        ids = self.order[self.position:self.position + self.size]
        self.position += self.size
        return [self.data[i] for i in ids]


def mask_view(row: dict[str, Any], terms: Iterable[str], rate: float, rng: random.Random,
              max_length: int, random_strength_matched: bool = False) -> tuple[list[str], list[str], dict[str, int]]:
    """Mask common character types; find protected spans on the original text."""
    originals = [row["sentence1"], row["sentence2"]]
    texts = [value[:max_length] for value in originals]
    protected: set[str] = set()
    for original in originals:
        for term in terms:
            start = 0
            while True:
                pos = original.find(term, start)
                if pos < 0:
                    break
                protected.update(original[pos:pos + len(term)])
                start = pos + 1
    common = set(texts[0]) & set(texts[1])
    eligible = sorted(common - protected)
    count = math.floor(rate * len(eligible) + 0.5)
    pool = sorted(common) if random_strength_matched else eligible
    chosen = set(rng.sample(pool, count))
    views = [["<mask>" if ch in chosen else ch for ch in text] for text in texts]
    return views[0], views[1], {"eligible_types": len(eligible), "masked_types": len(chosen),
                                "positions": sum(ch in chosen for text in texts for ch in text),
                                "total_positions": sum(map(len, texts))}
