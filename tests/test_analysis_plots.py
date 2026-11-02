from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "matplotlib"))
HAS_SKLEARN = AVAILABLE and importlib.util.find_spec("sklearn") is not None
if AVAILABLE:
    import plot_results
    from pairmatch.analysis import bootstrap_difference, holm, load_result, paired, wilcoxon_exact
    from pairmatch.data import DataError
    from pairmatch.metrics import calculate
if HAS_SKLEARN:
    import numpy as np
    import plot_tsne
    from pairmatch.data import digest


@unittest.skipUnless(AVAILABLE, "torch and matplotlib are required")
class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def fixture(self, method: str, seed: int, fraction: int, probabilities: list[float],
                labeled_hash: str | None = None) -> Path:
        folder = self.root / f"{method}_{seed}_{fraction}"
        folder.mkdir()
        labels = [0, 0, 1, 1]
        records = [{"sample_id": f"sample-{i}", "gold_label": label, "probability_positive": prob,
                    "prediction": int(prob >= 0.5)} for i, (label, prob) in enumerate(zip(labels, probabilities))]
        (folder / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        info = {"status": "COMPLETE", "split": "test", "split_sha256": "eval-hash", "count": 4,
                "metrics": calculate(labels, probabilities),
                "training_config": {"condition": method, "seed": seed, "threshold": 0.5, "u_fraction": 1.0},
                "training_provenance": {"sampling_seed": seed, "split_manifest_sha256": "parent-hash",
                    "labeled_percent": fraction, "labeled_sha256": labeled_hash or f"l-{seed}-{fraction}",
                    "unlabeled_mode": "remaining", "unlabeled_sha256_expected": f"u-{seed}-{fraction}",
                    "labeled_count": 10, "u_count_used": 4 if method == "A3" else 0,
                    "implementation_sha256": {"runner": "synthetic"}}}
        path = folder / "metrics.json"
        path.write_text(json.dumps(info), encoding="utf-8")
        return path

    def test_exact_signed_rank_and_holm(self) -> None:
        self.assertAlmostEqual(wilcoxon_exact([1, 2, 3, 4, 5])["p_two_sided"], 0.0625)
        self.assertEqual(wilcoxon_exact([0, 0])["p_two_sided"], 1.0)
        tied = wilcoxon_exact([1, -1, 2, -2, 0])
        self.assertTrue(tied["tied_absolute_values"])
        self.assertEqual(tied["zero_pairs"], 1)
        self.assertEqual(tied["p_two_sided"], 1.0)
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_paired_and_group_bootstrap_are_reproducible(self) -> None:
        left = [self.fixture("A3", seed, 30, [0.1, 0.2, 0.8, 0.9]) for seed in (11, 23)]
        right = [self.fixture("ce", seed, 30, [0.1, 0.8, 0.8, 0.2]) for seed in (11, 23)]
        report, pairs = paired(left, right, "macro_f1")
        self.assertEqual(report["training_information_relation"], ["different_training_information"])
        self.assertGreater(report["mean_difference_left_minus_right"], 0)
        groups = [["sample-0", "sample-1"], ["sample-2", "sample-3"]]
        one = bootstrap_difference(pairs, pairs[0][0]["ids"], groups, "macro_f1", 50, 20261030)
        two = bootstrap_difference(pairs, pairs[0][0]["ids"], groups, "macro_f1", 50, 20261030)
        self.assertEqual(one, two)
        self.assertGreater(one["resamples_missing_a_class"], 0)
        with self.assertRaises(DataError):
            paired(left, right[:1], "macro_f1")

    def test_rejects_prediction_or_labeled_id_tampering(self) -> None:
        left = self.fixture("A3", 11, 30, [0.1, 0.2, 0.8, 0.9])
        wrong_l = self.fixture("ce", 11, 30, [0.1, 0.2, 0.8, 0.9], "different-labeled-hash")
        with self.assertRaises(DataError):
            paired([left], [wrong_l], "macro_f1")
        predictions = left.parent / "predictions.jsonl"
        records = [json.loads(line) for line in predictions.read_text().splitlines()]
        records[0]["prediction"] = 1
        predictions.write_text("".join(json.dumps(row) + "\n" for row in records))
        with self.assertRaises(DataError):
            load_result(left)

    def test_scientific_plots_export_editable_svg_and_source(self) -> None:
        left = self.fixture("A3", 11, 30, [0.1, 0.2, 0.8, 0.9])
        right = self.fixture("ce", 11, 30, [0.1, 0.8, 0.8, 0.2])
        plot_results.plot_ldp([left, right], "macro_f1", self.root / "ldp")
        plot_results.plot_confusion(left, self.root / "confusion")
        plot_results.plot_ablation([left, right], "macro_f1", self.root / "ablation")
        for name in ("ldp", "confusion", "ablation"):
            self.assertIn("<text", (self.root / name / "figure.svg").read_text(encoding="utf-8"))
            self.assertTrue((self.root / name / "figure.pdf").exists())
            self.assertTrue((self.root / name / "source_data.json").exists())
        with self.assertRaises(DataError):
            plot_results.plot_ldp([left, right], "macro_f1", self.root / "ldp")

    def test_paired_and_holm_command_line(self) -> None:
        left = self.fixture("A3", 11, 30, [0.1, 0.2, 0.8, 0.9])
        right = self.fixture("ce", 11, 30, [0.1, 0.8, 0.8, 0.2])
        paired_path = self.root / "paired.json"
        process = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "paired_statistics.py"),
            "--left", str(left), "--right", str(right), "--bootstrap-unit", "samples", "--bootstrap-repetitions", "20",
            "--output", str(paired_path), "--confirm-prespecified-analysis"], cwd=ROOT,
            text=True, capture_output=True, timeout=60)
        self.assertEqual(process.returncode, 0, process.stderr)
        corrected = self.root / "holm.json"
        process = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "holm_correction.py"),
            "--reports", str(paired_path), "--family-name", "synthetic", "--output", str(corrected),
            "--confirm-complete-prespecified-family"], cwd=ROOT, text=True, capture_output=True, timeout=60)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(corrected.read_text(encoding="utf-8"))
        self.assertEqual(result["tests"][0]["p_holm"], result["tests"][0]["p_raw"])

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required")
    def test_descriptive_tsne_uses_synthetic_embeddings(self) -> None:
        directory = self.root / "embeddings"
        directory.mkdir()
        rng = np.random.default_rng(42)
        np.savez_compressed(directory / "embeddings.npz", embeddings=rng.normal(size=(40, 4)),
                            labels=np.arange(40) % 2, sample_ids=np.asarray([f"synthetic-{i}" for i in range(40)]))
        manifest = {"status": "COMPLETE", "prespecified_analysis_user_attestation": True,
                    "embeddings_sha256": digest(directory / "embeddings.npz"), "sample_count": 40,
                    "split": "validation", "split_sha256": "synthetic", "condition": "synthetic"}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        report = plot_tsne.plot(directory, self.root / "tsne", perplexity=10, iterations=250)
        self.assertEqual(len(report["points"]), 40)
        self.assertIn("<text", (self.root / "tsne" / "figure.svg").read_text(encoding="utf-8"))
        with self.assertRaises(DataError):
            plot_tsne.plot(directory, self.root / "invalid_tsne", perplexity=40)
