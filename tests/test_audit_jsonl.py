from __future__ import annotations

import json
import hashlib
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import audit_jsonl  # noqa: E402
import component_diagnostic  # noqa: E402
import validate_configs  # noqa: E402


def row(a: str, b: str, label: object) -> dict[str, object]:
    return {"sentence1": a, "sentence2": b, "gold_label": label}


class AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, name: str, records: list[dict[str, object]]) -> Path:
        path = self.root / name
        path.write_text("".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")
        return path

    def basic_paths(self) -> dict[str, Path]:
        return {
            "train": self.write("train.txt", [row("alpha", "beta", "1"), row("gamma", "delta", -1)]),
            "validation": self.write("dev.txt", [row("epsilon", "zeta", 0)]),
            "test": self.write("test.txt", [row("eta", "theta", 1)]),
        }

    def test_clean_data_does_not_claim_semantic_or_source_verification(self) -> None:
        paths = self.basic_paths()
        report = audit_jsonl.audit(paths, require_sentence_disjoint=True)
        self.assertEqual(report["gate"]["status"], "EXACT_CHECKS_PASS")
        self.assertEqual(report["files"]["train"]["sha256"], hashlib.sha256(paths["train"].read_bytes()).hexdigest())
        self.assertEqual(report["method"]["source_level_audit"], "not performed")
        self.assertEqual(report["method"]["semantic_near_duplicate_audit"], "not performed")
        self.assertNotIn("alpha", json.dumps(report))

    def test_reversed_pairs_are_flagged_even_when_ordered_pair_is_new(self) -> None:
        paths = self.basic_paths()
        paths["test"] = self.write("test.txt", [row("beta", "alpha", 1)])
        report = audit_jsonl.audit(paths)
        comparison = report["cross_split"]["train__test"]
        self.assertEqual(comparison["shared_unique_ordered_pairs"], 0)
        self.assertEqual(comparison["shared_unique_unordered_pairs"], 1)
        self.assertEqual(report["gate"]["status"], "BLOCKED")

    def test_sentence_disjointness_is_an_explicit_stricter_policy(self) -> None:
        paths = self.basic_paths()
        paths["test"] = self.write("test.txt", [row("alpha", "new", 1)])
        pair_gate = audit_jsonl.audit(paths, require_sentence_disjoint=False)
        sentence_gate = audit_jsonl.audit(paths, require_sentence_disjoint=True)
        self.assertEqual(pair_gate["gate"]["status"], "EXACT_CHECKS_PASS")
        self.assertTrue(pair_gate["gate"]["warnings"])
        self.assertEqual(sentence_gate["gate"]["status"], "BLOCKED")

    def test_conflicting_labels_and_unlabeled_validation_block(self) -> None:
        paths = self.basic_paths()
        paths["train"] = self.write("train.txt", [row("a", "b", 0), row("b", "a", "1")])
        paths["validation"] = self.write("dev.txt", [row("c", "d", -1)])
        report = audit_jsonl.audit(paths)
        self.assertEqual(report["files"]["train"]["within_file_conflicting_unordered_pairs"], 1)
        self.assertEqual(report["files"]["validation"]["labels_invalid_for_role"], 1)
        self.assertEqual(report["gate"]["status"], "BLOCKED")

    def test_labeled_subset_mismatch_is_blocked(self) -> None:
        paths = self.basic_paths()
        paths["supervised"] = self.write("supervised.txt", [row("not", "present", 1)])
        report = audit_jsonl.audit(paths)
        self.assertEqual(report["supervised_vs_train"]["missing_from_train_ordered"], 1)
        self.assertEqual(report["gate"]["status"], "BLOCKED")

    def test_invalid_rows_and_unicode_whitespace_normalization(self) -> None:
        paths = self.basic_paths()
        paths["train"] = self.write("train.txt", [row("Ａ b", "c", True), row("Ａ b", "c", "0")])
        paths["test"] = self.write("test.txt", [row("Ab", "c", 0)])
        report = audit_jsonl.audit(paths)
        self.assertEqual(report["files"]["train"]["invalid_lines"], 1)
        self.assertEqual(report["cross_split"]["train__test"]["shared_unique_ordered_pairs"], 1)
        self.assertEqual(report["gate"]["status"], "BLOCKED")

    def test_report_does_not_overwrite_existing_file(self) -> None:
        paths = self.basic_paths()
        out = self.root / "report.json"
        out.write_text("keep", encoding="utf-8")
        args = ["--train", str(paths["train"]), "--validation", str(paths["validation"]),
                "--test", str(paths["test"]), "--output", str(out)]
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(audit_jsonl.main(args), 1)
        self.assertNotIn("keep", error.getvalue())
        self.assertEqual(out.read_text(encoding="utf-8"), "keep")

    def test_repository_configuration_is_internally_consistent(self) -> None:
        self.assertEqual(validate_configs.validate(), [])

    def test_component_diagnostic_deduplicates_without_exposing_text(self) -> None:
        path = self.write("pairs.txt", [row("a", "b", 0), row("b", "c", 1),
                                         row("b", "a", 0), row("d", "e", -1)])
        result = component_diagnostic.diagnose([path], 0.8)
        self.assertEqual(result["unique_normalized_unordered_labelled_pairs"], 2)
        self.assertEqual(result["duplicate_labelled_rows"], 1)
        self.assertEqual(result["unlabeled_rows_not_used"], 1)
        self.assertTrue(result["largest_component_alone_exceeds_target"])
        self.assertNotIn('"a"', json.dumps(result))

    def test_component_diagnostic_reports_conflicts_without_resolving_them(self) -> None:
        path = self.write("pairs.txt", [row("x", "y", 0), row("y", "x", 1)])
        result = component_diagnostic.diagnose([path])
        self.assertEqual(result["conflicting_label_pairs_require_manual_review"], 1)
        self.assertEqual(result["unique_normalized_unordered_labelled_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
