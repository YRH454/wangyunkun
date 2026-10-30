from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import create_ldp_subset  # noqa: E402
import prepare_clean_split as split  # noqa: E402
import review_near_candidates  # noqa: E402
import verify_clean_split  # noqa: E402


def text(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:24]


def row(left: str, right: str, label: int, source: str | list[str] | None = None) -> dict[str, object]:
    result: dict[str, object] = {"sentence1": left, "sentence2": right, "gold_label": label}
    if source is not None:
        result["source_id"] = source
    return result


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_rows(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")

    def fixture(self, source: bool = False) -> Path:
        rows = []
        for i in range(160):
            group = (["shared-source", f"source-{i}"] if i in (4, 5) else f"source-{i}") if source else None
            rows.append(row(text(f"a{i}"), text(f"b{i}"), i % 2, group))
        # Same unordered pair with different binary labels; neither should be used.
        rows.append(row(text("b0"), text("a0"), 1, "source-0" if source else None))
        # Similar labeled sentences must remain in the same partition.
        rows.append(row("abcdefghijklmnopqrst", text("special-one"), 0, "source-special1" if source else None))
        rows.append(row("abcdefghijklmnopqrstu", text("special-two"), 1, "source-special2" if source else None))
        for i in range(160):
            rows.append(row(text(f"a{i}"), text(f"u{i}"), -1, f"u-source-{i}" if source else None))
        rows.append(row(text("a10"), text("b10"), -1, "u-source-duplicate" if source else None))
        path = self.root / "input.jsonl"
        self.write_rows(path, rows)
        return path

    def test_prefix_join_matches_bruteforce_for_the_defined_metric(self) -> None:
        randomizer = random.Random(42)
        alphabet = "abcdefghij"
        values = {"".join(randomizer.choices(alphabet, k=randomizer.randint(3, 12))) for _ in range(250)}
        values.update({"abcdefghijklmnopqrst", "abcdefghijklmnopqrstu", "aaaa", "aaaab"})
        for threshold in (50, 75, 90, 100):
            index = split.LexicalIndex(values, threshold, 1000000)
            observed: set[tuple[str, str]] = set()
            for value in sorted(values):
                observed.update(tuple(sorted((value, other))) for other in index.matching(value))
                index.add(value)
            ordered = sorted(values)
            expected = {(left, right) for i, left in enumerate(ordered) for right in ordered[i + 1:]
                        if split.similar(split.shingles(left), split.shingles(right), threshold)}
            self.assertEqual(observed, expected)

    def test_clean_split_quarantines_conflicts_and_is_reproducible(self) -> None:
        source = self.fixture()
        original_hash = split.file_digest(source)
        first = self.root / "first"
        second = self.root / "second"
        report = split.prepare([source], first)
        split.prepare([source], second)
        self.assertEqual(split.file_digest(source), original_hash)
        self.assertEqual(report["conflicting_pairs_quarantined"], 1)
        self.assertEqual(report["conflicting_observations_quarantined"], 2)
        self.assertEqual(report["unlabeled_excluded_unique_pairs_by_first_reason"]["duplicates_any_labeled_pair"], 1)
        self.assertGreater(report["unlabeled_excluded_unique_pairs_by_first_reason"].get("exact_evaluation_sentence", 0), 0)
        self.assertGreater(report["lexical_rule"]["labeled_sentence_edges"], 0)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "unlabeled.jsonl", "quarantined_conflicts.jsonl"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        verification = verify_clean_split.verify(first)
        self.assertEqual(verification["status"], "EXACT_AND_CHOSEN_LEXICAL_CHECKS_PASS")
        unlabeled = create_ldp_subset.read_rows(first / "unlabeled.jsonl")
        self.assertTrue(all("gold_label" not in item for item in unlabeled))

    def test_optional_source_groups_do_not_cross_splits(self) -> None:
        source = self.fixture(source=True)
        out = self.root / "source_split"
        report = split.prepare([source], out, source_field="source_id")
        self.assertEqual(report["source_grouping"], "enabled")
        verification = verify_clean_split.verify(out)
        self.assertEqual(verification["source_check"], "passed_for_supplied_source_ids")
        holders = [name for name in split.SPLITS if any("shared-source" in x["source_ids"]
                   for x in create_ldp_subset.read_rows(out / f"{name}.jsonl"))]
        self.assertEqual(len(holders), 1)

    def test_missing_source_id_malformed_rows_and_existing_output_fail_safely(self) -> None:
        source = self.fixture()
        out = self.root / "bad"
        with self.assertRaises(split.PreparationError):
            split.prepare([source], out, source_field="source_id")
        self.assertFalse(out.exists())
        source.write_text('{"sentence1": true}\n', encoding="utf-8")
        with self.assertRaises(split.PreparationError):
            split.prepare([source], out)
        self.assertFalse(out.exists())
        out.mkdir()
        marker = out / "existing.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(split.PreparationError):
            split.prepare([source], out)
        self.assertEqual(marker.read_text(), "preserve")

    def test_verifier_rejects_tampering_even_if_hash_is_recomputed(self) -> None:
        source = self.fixture()
        out = self.root / "tampered"
        split.prepare([source], out)
        unlabeled = out / "unlabeled.jsonl"
        rows = create_ldp_subset.read_rows(unlabeled)
        rows[0]["gold_label"] = 1
        self.write_rows(unlabeled, rows)
        with self.assertRaises(split.PreparationError):
            verify_clean_split.verify(out)
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output_sha256"]["unlabeled.jsonl"] = split.file_digest(unlabeled)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(split.PreparationError):
            verify_clean_split.verify(out)

    def test_verifier_detects_near_duplicate_after_forged_manifest_update(self) -> None:
        source = self.fixture()
        out = self.root / "lexical_tampering"
        split.prepare([source], out)
        train = create_ldp_subset.read_rows(out / "train.jsonl")
        validation_path = out / "validation.jsonl"
        validation = create_ldp_subset.read_rows(validation_path)
        validation[0]["sentence1"] = train[0]["sentence1"] + "x"
        pair = split.unordered((split.normalize_text(validation[0]["sentence1"]),
                                split.normalize_text(validation[0]["sentence2"])))
        validation[0]["sample_id"] = split.sample_id(pair)
        self.write_rows(validation_path, validation)
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output_sha256"]["validation.jsonl"] = split.file_digest(validation_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(split.PreparationError, "lexical near-duplicate"):
            verify_clean_split.verify(out)

    def test_nested_label_fractions_and_unlabeled_modes(self) -> None:
        source = self.fixture()
        out = self.root / "split"
        split.prepare([source], out)
        small = self.root / "small"
        large = self.root / "large"
        remaining = self.root / "remaining"
        create_ldp_subset.create(out, small, 10, 11, "external", 10)
        create_ldp_subset.create(out, large, 30, 11, "external", 10)
        create_ldp_subset.create(out, remaining, 30, 11, "remaining", 1000)
        ids = lambda path: {item["sample_id"] for item in create_ldp_subset.read_rows(path)}
        small_l = ids(small / "train_labeled.jsonl")
        large_l = ids(large / "train_labeled.jsonl")
        self.assertTrue(small_l < large_l)
        self.assertEqual(ids(small / "train_unlabeled.jsonl"), ids(large / "train_unlabeled.jsonl"))
        remainder_u = create_ldp_subset.read_rows(remaining / "train_unlabeled.jsonl")
        self.assertFalse(large_l & {item["sample_id"] for item in remainder_u})
        self.assertTrue(all("gold_label" not in item for item in remainder_u))

    def test_review_candidates_remain_pending_and_local(self) -> None:
        source = self.fixture()
        out = self.root / "candidate_split"
        split.prepare([source], out)
        validation = create_ldp_subset.read_rows(out / "validation.jsonl")
        original = validation[0]["sentence1"]
        near = original + "ghijk"
        self.assertTrue(split.similar(split.shingles(original), split.shingles(near), 80))
        self.assertFalse(split.similar(split.shingles(original), split.shingles(near), 90))
        pair = split.unordered((split.normalize_text(near), split.normalize_text(text("extra-u"))))
        unlabeled = out / "unlabeled.jsonl"
        rows = create_ldp_subset.read_rows(unlabeled)
        rows.append({"sample_id": split.sample_id(pair), "sentence1": near, "sentence2": text("extra-u")})
        self.write_rows(unlabeled, rows)
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output_sha256"]["unlabeled.jsonl"] = split.file_digest(unlabeled)
        manifest["unlabeled_kept"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        destination = self.root / "review"
        report = review_near_candidates.candidates(out, destination)
        self.assertEqual(report["status"], "PENDING_HUMAN_REVIEW")
        output_rows = create_ldp_subset.read_rows(destination / "review_candidates.jsonl")
        self.assertTrue(any(item["sentence_a"] == near or item["sentence_b"] == near for item in output_rows))
        self.assertTrue(all(item["decision"] == "PENDING_HUMAN_REVIEW" for item in output_rows))


if __name__ == "__main__":
    unittest.main()
