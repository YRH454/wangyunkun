from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import apply_review_decisions as review
import prepare_clean_split as split
import review_near_candidates
import verify_clean_split
from pairmatch.analysis import load_groups
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import build_bootstrap_groups


def fake(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:24]


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "synthetic.jsonl"
        records = []
        for i in range(160):
            a, b = fake(f"a{i}"), fake(f"b{i}")
            records.append({"sentence1": a, "sentence2": b, "gold_label": i % 2})
            records.append({"sentence1": a + "ghijk", "sentence2": fake(f"u{i}"), "gold_label": -1})
        records.append({"sentence1": fake("b0"), "sentence2": fake("a0"), "gold_label": 1})
        self.write(self.source, records)
        self.original_digest = split.file_digest(self.source)
        self.parent = self.root / "parent"
        split.prepare([self.source], self.parent)
        self.candidates = self.root / "candidates"
        result = review_near_candidates.candidates(self.parent, self.candidates)
        self.assertGreater(result["candidate_sentence_pairs"], 1)
        self.template = self.root / "template"
        review.create_template(self.parent, self.candidates, self.template)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def write(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def decide(self, template: Path, group_first: bool = True) -> None:
        path = template / "decisions.jsonl"
        rows = review.load_lines(path)
        near = 0
        for row in rows:
            row["reviewer"] = "synthetic-reviewer"
            row["reason"] = "synthetic decision for testing only"
            if row["kind"] == "near":
                row["decision"] = "GROUP" if group_first and near == 0 else "KEEP_SEPARATE"
                if group_first and near == 1:
                    row["decision"] = "EXCLUDE_A"
                near += 1
            else:
                row["decision"] = "LABEL_0"
        self.write(path, rows)

    def test_pending_and_test_exposure_attestation_block(self) -> None:
        out = self.root / "revised"
        with self.assertRaises(split.PreparationError):
            review.apply(self.parent, self.candidates, self.template, [self.source], out, True)
        self.assertFalse(out.exists())
        self.decide(self.template)
        with self.assertRaises(split.PreparationError):
            review.apply(self.parent, self.candidates, self.template, [self.source], out, False)
        self.assertFalse(out.exists())

    def test_apply_and_inherit_review_in_new_versions(self) -> None:
        self.decide(self.template)
        out = self.root / "new_v1"
        result = review.apply(self.parent, self.candidates, self.template, [self.source], out, True)
        self.assertEqual(result["verification"], "EXACT_AND_CHOSEN_LEXICAL_CHECKS_PASS")
        self.assertEqual(result["human_review"]["resolved_conflict_pairs"], 1)
        self.assertEqual(result["human_review"]["reviewed_group_edges"], 1)
        self.assertGreater(result["human_review"]["quarantined_pair_ids"], 0)
        self.assertEqual(split.file_digest(self.source), self.original_digest)
        self.assertEqual(verify_clean_split.verify(out)["human_review_groups_check"], "passed_for_recorded_edges_and_exclusions")
        later_candidates = self.root / "later_candidates"
        review_near_candidates.candidates(out, later_candidates)
        later_template = self.root / "later_template"
        review.create_template(out, later_candidates, later_template)
        self.decide(later_template, group_first=False)
        later = self.root / "new_v2"
        second = review.apply(out, later_candidates, later_template, [self.source], later, True)
        self.assertEqual(second["human_review"]["resolved_conflict_pairs"], 1)
        self.assertEqual(second["human_review"]["reviewed_group_edges"], 1)
        self.assertGreater(second["human_review"]["inherited_decision_count"], 0)

    def test_changed_input_or_candidate_rejected_without_output(self) -> None:
        self.decide(self.template)
        other = self.root / "different.jsonl"
        other.write_text("{}\n", encoding="utf-8")
        output = self.root / "bad"
        with self.assertRaises(split.PreparationError):
            review.apply(self.parent, self.candidates, self.template, [other], output, True)
        self.assertFalse(output.exists())
        with (self.candidates / "review_candidates.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaises(split.PreparationError):
            review.apply(self.parent, self.candidates, self.template, [self.source], output, True)
        self.assertFalse(output.exists())

    @unittest.skipUnless(HAS_TORCH, "torch is required for group builder")
    def test_bootstrap_groups_cover_exactly_the_evaluation_split(self) -> None:
        destination = self.root / "test_groups"
        report = build_bootstrap_groups.build(self.parent, "test", destination, 5000000)
        ids = [row["sample_id"] for row in review.load_lines(self.parent / "test.jsonl")]
        groups, info = load_groups(destination, report["split_sha256"], sorted(ids))
        self.assertEqual(sum(map(len, groups)), len(ids))
        self.assertEqual(len(groups), report["group_count"])
        self.assertEqual(info["split"], "test")
