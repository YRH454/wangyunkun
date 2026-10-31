from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

if importlib.util.find_spec("torch") is not None:
    import torch
    import create_ldp_subset
    import benchmark_pairmatch
    import prepare_clean_split
    import verify_clean_split
    import predict_pairmatch
    import summarize_results
    import train_pairmatch
    from pairmatch.data import BatchStream, DataError, load_training_data, mask_view
    from pairmatch.metrics import calculate
    from pairmatch.model import PairESIM, batch_encode
    from pairmatch.objectives import supervised_contrastive, unsupervised_contrastive
    from pairmatch.runner import CONDITIONS, make_model, pairs_for, train
    TORCH_AVAILABLE = True
else:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for training tests")
class PairMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        def fake(name: str) -> str:
            return hashlib.sha256(name.encode()).hexdigest()[:20]
        source = cls.root / "synthetic.jsonl"
        with source.open("w", encoding="utf-8") as handle:
            for i in range(160):
                handle.write(json.dumps({"sentence1": fake(f"left{i}"), "sentence2": fake(f"right{i}"),
                                         "gold_label": i % 2}) + "\n")
                handle.write(json.dumps({"sentence1": fake(f"left{i}"), "sentence2": fake(f"u{i}"),
                                         "gold_label": -1}) + "\n")
        cls.split = cls.root / "split"
        cls.subset = cls.root / "subset"
        cls.report = cls.root / "verification.json"
        cls.lexicon = cls.root / "synthetic_terms.txt"
        cls.lexicon.write_text("synthetic-nonoccurring-term\n", encoding="utf-8")
        prepare_clean_split.prepare([source], cls.split)
        verification = verify_clean_split.verify(cls.split)
        cls.report.write_text(json.dumps(verification), encoding="utf-8")
        create_ldp_subset.create(cls.split, cls.subset, 30, 11, "external", 40)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def arguments(self, output: Path, condition: str = "A3"):
        args = train_pairmatch.parser().parse_args([
            "--split-dir", str(self.split), "--subset-dir", str(self.subset),
            "--verification-report", str(self.report), "--run-dir", str(output),
            "--confirm-data-rights", "--confirm-reviewed-data", "--condition", condition,
            "--term-lexicon", str(self.lexicon), "--stage1-steps", "2", "--stage2-steps", "2",
            "--eval-every", "1", "--batch-size", "4", "--u-batch-size", "4", "--accumulation", "1",
            "--max-length", "12", "--embedding-dim", "6", "--hidden", "4", "--dropout", "0",
        ])
        return args

    def test_supervised_formula_and_no_positive_safeguard(self) -> None:
        z = torch.tensor([[1., 0.], [0.8, 0.6], [0., 1.], [-1., 0.]], requires_grad=True)
        y = torch.tensor([0, 0, 1, 1])
        loss, anchors = supervised_contrastive(z, y, 0.5)
        manual = []
        for i in range(4):
            similarities = [float(torch.nn.functional.cosine_similarity(z[i], z[j], dim=0).detach()) / 0.5 for j in range(4)]
            denom = sum(math.exp(similarities[j]) for j in range(4) if j != i)
            positive = next(j for j in range(4) if j != i and y[j] == y[i])
            manual.append(-math.log(math.exp(similarities[positive]) / denom))
        self.assertEqual(anchors, 4)
        self.assertAlmostEqual(loss.item(), sum(manual) / 4, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        skipped, anchors = supervised_contrastive(z[:2], torch.tensor([0, 1]), 0.5)
        self.assertEqual(anchors, 0)
        self.assertEqual(skipped.item(), 0.0)

    def test_unsupervised_masks_duplicate_false_negatives(self) -> None:
        a = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        b = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        loss, valid = unsupervised_contrastive(a, b, ["same", "same", "different"], 1.0)
        expected = (2 * math.log(1 + math.exp(-1)) + math.log(1 + 2 * math.exp(-1))) / 3
        self.assertTrue(valid)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        skipped, valid = unsupervised_contrastive(a[:2], b[:2], ["same", "same"], 1.0)
        self.assertFalse(valid)
        self.assertEqual(skipped.item(), 0.0)

    def test_mask_protects_term_characters_and_preserves_type_budget(self) -> None:
        row = {"sentence1": "番茄茎长", "sentence2": "番茄叶长"}
        a, b, info = mask_view(row, ["番茄"], 0.75, random.Random(2), 128)
        self.assertEqual(a, ["番", "茄", "茎", "<mask>"])
        self.assertEqual(b, ["番", "茄", "叶", "<mask>"])
        self.assertEqual(info["masked_types"], 1)
        _, _, random_info = mask_view(row, ["番茄"], 0.75, random.Random(2), 128, True)
        self.assertEqual(random_info["masked_types"], info["masked_types"])

    def test_metrics_confusion_macro_and_zero_division(self) -> None:
        result = calculate([0, 0, 1, 1], [0.1, 0.9, 0.6, 0.3])
        self.assertEqual(result["confusion_matrix_true_rows_predicted_columns"], [[1, 1], [1, 1]])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["macro_f1"], 0.5)
        no_positive = calculate([0, 0], [0.1, 0.2])
        self.assertEqual(no_positive["classes"]["1"]["f1"], 0.0)

    def test_padding_does_not_change_pair_prediction(self) -> None:
        torch.manual_seed(1)
        model = PairESIM(10, 6, 4, 0).eval()
        one = [([3, 4], [5])]
        padded = one + [([3, 4, 5, 6], [5, 6, 7])]
        with torch.inference_mode():
            a, _ = model(*batch_encode(one, torch.device("cpu")))
            b, _ = model(*batch_encode(padded, torch.device("cpu")))
        self.assertTrue(torch.allclose(a[0], b[0], atol=1e-6))

    def test_loader_rejects_unlabeled_gold_even_with_rehashed_manifest(self) -> None:
        copied = self.root / "tampered_subset"
        shutil.copytree(self.subset, copied)
        path = copied / "train_unlabeled.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        records[0]["gold_label"] = 1
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        manifest_path = copied / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(DataError):
            load_training_data(self.split, copied, self.report, True)

    def test_gates_and_output_overwrite_refusal(self) -> None:
        output = self.root / "gated"
        args = self.arguments(output)
        args.confirm_data_rights = False
        with self.assertRaises(DataError):
            train(args)
        self.assertFalse(output.exists())
        args.confirm_data_rights = True
        args.term_lexicon = None
        with self.assertRaises(DataError):
            train(args)
        output.mkdir()
        marker = output / "marker"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(DataError):
            train(args)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_all_ablation_code_paths_execute_on_synthetic_data(self) -> None:
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                args = self.arguments(self.root / f"condition_{condition}", condition)
                args.stage1_steps = args.stage2_steps = 1
                args.confidence = 0.5  # Synthetic test exercises acceptance, not the formal 0.95 setting.
                result = train(args)
                self.assertEqual(result["status"], "COMPLETE")
                self.assertEqual(len(result["stages"]), len(CONDITIONS[condition]))
                self.assertEqual(result["total_executed_steps_including_rolled_back"], len(CONDITIONS[condition]))
                self.assertTrue((args.run_dir / result["final_checkpoint"]).exists())
                self.assertTrue(all(x["labeled_exposures"] == 4 for x in result["stages"]))
                if any(u for _, u in CONDITIONS[condition]):
                    self.assertTrue(any(x["unlabeled_exposures"] == 4 for x in result["stages"]))

    def test_reproducible_two_stage_and_no_test_read_during_training(self) -> None:
        test = self.split / "test.jsonl"
        held = self.split / "temporarily_held_test.jsonl"
        test.rename(held)
        try:
            first = self.arguments(self.root / "repeat1")
            second = self.arguments(self.root / "repeat2")
            result1, result2 = train(first), train(second)
        finally:
            held.rename(test)
        self.assertEqual(result1["stages"], result2["stages"])
        a = torch.load(first.run_dir / result1["final_checkpoint"], map_location="cpu", weights_only=True)
        b = torch.load(second.run_dir / result2["final_checkpoint"], map_location="cpu", weights_only=True)
        self.assertTrue(all(torch.equal(a["model_state"][key], b["model_state"][key]) for key in a["model_state"]))
        self.assertEqual((first.run_dir / "train_log.jsonl").read_text(), (second.run_dir / "train_log.jsonl").read_text())

    def test_stage_two_starts_from_stage_one_best_weights(self) -> None:
        args = self.arguments(self.root / "transition", "A0")
        args.stage2_steps = 1
        train(args)
        checkpoint = torch.load(args.run_dir / "stage_1_best.pt", map_location="cpu", weights_only=True)
        model = make_model(checkpoint["config"], len(checkpoint["vocab"]))
        model.load_state_dict(checkpoint["model_state"])
        model.train()  # Synthetic setting has dropout=0.
        labeled, _, _, _ = load_training_data(self.split, self.subset, self.report, False)
        batch = BatchStream(labeled, args.batch_size, args.seed + 2000).next()
        logits, _ = model(*batch_encode(pairs_for(batch, checkpoint["vocab"], args.max_length), torch.device("cpu")))
        expected = torch.nn.functional.cross_entropy(logits, torch.tensor([row["gold_label"] for row in batch])).item()
        logs = [json.loads(line) for line in (args.run_dir / "train_log.jsonl").read_text().splitlines()]
        actual = next(row["ce"] for row in logs if row["stage"] == 2 and row["stage_step"] == 1)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_test_prediction_requires_locked_final_model(self) -> None:
        args = self.arguments(self.root / "for_prediction")
        result = train(args)
        prediction = type("Args", (), {"checkpoint": args.run_dir / result["final_checkpoint"],
                          "split_dir": self.split, "split": "test", "output_dir": self.root / "prediction",
                          "confirm_test_locked": False, "device": "cpu", "batch_size": 4, "cpu_threads": 1})()
        with self.assertRaises(DataError):
            predict_pairmatch.predict(prediction)
        prediction.confirm_test_locked = True
        result = predict_pairmatch.predict(prediction)
        self.assertEqual(result["status"], "COMPLETE")
        rows = [json.loads(line) for line in (prediction.output_dir / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), result["count"])
        self.assertTrue(all(set(row) == {"sample_id", "gold_label", "probability_positive", "prediction"} for row in rows))
        with self.assertRaises(DataError):
            predict_pairmatch.predict(prediction)

    def test_summary_rejects_duplicate_or_mismatched_runs(self) -> None:
        args = self.arguments(self.root / "for_summary", "ce")
        args.stage1_steps = 1
        trained = train(args)
        prediction = type("Args", (), {"checkpoint": args.run_dir / trained["final_checkpoint"],
                          "split_dir": self.split, "split": "validation", "output_dir": self.root / "for_summary_eval",
                          "confirm_test_locked": False, "device": "cpu", "batch_size": 4, "cpu_threads": 1})()
        one = predict_pairmatch.predict(prediction)
        first = prediction.output_dir / "metrics.json"
        with self.assertRaises(ValueError):
            summarize_results.summarize([first, first])
        one["training_config"]["seed"] = 23
        one["training_provenance"]["sampling_seed"] = 23
        second = self.root / "synthetic_second_metrics.json"
        second.write_text(json.dumps(one), encoding="utf-8")
        combined = summarize_results.summarize([first, second])
        self.assertEqual(combined["seeds"], [11, 23])
        self.assertEqual(combined["missing_prespecified_reporting_seeds"], [37, 47, 59])
        one["split_sha256"] = "different"
        second.write_text(json.dumps(one), encoding="utf-8")
        with self.assertRaises(ValueError):
            summarize_results.summarize([first, second])

    def test_cpu_benchmark_on_synthetic_validation_only(self) -> None:
        args = self.arguments(self.root / "for_benchmark", "ce")
        args.stage1_steps = 1
        result = train(args)
        options = type("Args", (), {"checkpoint": args.run_dir / result["final_checkpoint"],
                       "split_dir": self.split, "output": self.root / "benchmark.json",
                       "confirm_data_rights": True, "warmup": 2, "repeats": 3,
                       "sample_count": 2, "threads": 1})()
        report = benchmark_pairmatch.benchmark(options)
        self.assertEqual(report["selected_pairs"], 2)
        self.assertEqual(report["timed_repetitions_per_measurement"], 3)
        self.assertGreaterEqual(report["timings"]["model_only"]["median_ms"], 0)
        self.assertGreater(report["parameters_total_including_embeddings"], 0)
        with self.assertRaises(DataError):
            benchmark_pairmatch.benchmark(options)

    def test_command_line_train_and_predict_on_synthetic_data(self) -> None:
        run_dir = self.root / "cli_run"
        result_dir = self.root / "cli_eval"
        training = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "train_pairmatch.py"),
                   "--split-dir", str(self.split), "--subset-dir", str(self.subset),
                   "--verification-report", str(self.report), "--run-dir", str(run_dir),
                   "--condition", "ce", "--confirm-data-rights", "--confirm-reviewed-data",
                   "--stage1-steps", "1", "--eval-every", "1", "--batch-size", "4", "--accumulation", "1",
                   "--max-length", "12", "--embedding-dim", "6", "--hidden", "4"],
                  cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(training.returncode, 0, training.stderr)
        prediction = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "predict_pairmatch.py"),
                   "--checkpoint", str(run_dir / "stage_1_best.pt"), "--split-dir", str(self.split),
                   "--split", "validation", "--output-dir", str(result_dir)],
                  cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(prediction.returncode, 0, prediction.stderr)
        self.assertTrue((result_dir / "predictions.jsonl").exists())

    @unittest.skipUnless(TORCH_AVAILABLE and torch.cuda.is_available(), "CUDA is not available")
    def test_current_host_cuda_synthetic_smoke(self) -> None:
        args = self.arguments(self.root / "cuda_synthetic", "A3")
        args.device = "cuda"
        args.stage1_steps = args.stage2_steps = 1
        result = train(args)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertGreater(result["peak_gpu_allocated_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
