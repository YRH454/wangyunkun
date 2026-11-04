from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
if AVAILABLE:
    import auto_tune
    import create_ldp_subset
    import prepare_clean_split
    import train_pairmatch
    import train_transformer
    import verify_clean_split
    from pairmatch.data import DataError
    from pairmatch.runner import train


@unittest.skipUnless(AVAILABLE, "torch and transformers are required")
class AutoTuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        source = cls.root / "synthetic.jsonl"
        def fake(name: str) -> str:
            return hashlib.sha256(name.encode()).hexdigest()[:20]
        with source.open("w", encoding="utf-8") as handle:
            for i in range(160):
                handle.write(json.dumps({"sentence1": fake(f"a{i}"), "sentence2": fake(f"b{i}"),
                                         "gold_label": i % 2}) + "\n")
        cls.split = cls.root / "split"
        cls.tuning_subset = cls.root / "tuning_subset"
        cls.report_subset = cls.root / "report_subset"
        cls.report = cls.root / "verification.json"
        prepare_clean_split.prepare([source], cls.split)
        cls.report.write_text(json.dumps(verify_clean_split.verify(cls.split)), encoding="utf-8")
        create_ldp_subset.create(cls.split, cls.tuning_subset, 30, 101, "remaining", 40)
        create_ldp_subset.create(cls.split, cls.report_subset, 30, 11, "remaining", 40)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def tune_args(self, output: Path):
        return argparse.Namespace(method="ce", split_dir=self.split, subset_dir=self.tuning_subset,
                verification_report=self.report, output_dir=output, confirm_data_rights=True, confirm_reviewed_data=True,
                checkpoint_dir=None, confirm_model_rights_and_revision=False, allow_unsafe_pickle_weights=False,
                term_lexicon=None, device="cpu", cpu_threads=1, max_length=12, batch_size=4, accumulation=1,
                stage1_steps=1, stage2_steps=2000, eval_every=1, embedding_dim=6, hidden=4,
                learning_rates=[0.0003, 0.001], weight_decays=[0.0], allow_nonstandard_grid=True)

    def test_paper_learning_rate_and_effective_batch_grids(self) -> None:
        expected = {
            "ce": ([1e-3, 5e-4, 1e-4], [16, 32, 64], 16, 9),
            "bert_base_zh": ([1e-5, 2e-5, 3e-5], [16, 32], 8, 6),
            "small_zh": ([3e-5, 5e-5, 1e-4], [16, 32], 8, 6),
        }
        for method, (rates, batches, micro, count) in expected.items():
            with self.subTest(method=method):
                args = self.tune_args(self.root / "unused")
                args.method, args.batch_size, args.accumulation = method, None, None
                args.learning_rates, args.weight_decays, args.allow_nonstandard_grid = None, None, False
                grid = auto_tune.resolve_grid(args)
                self.assertEqual(grid["learning_rates"], rates)
                self.assertEqual(grid["effective_batches"], batches)
                self.assertEqual(grid["micro_batch"], micro)
                self.assertEqual(len(rates) * len(batches), count)
                self.assertTrue(grid["standard_grid"])

    def test_unstated_grid_and_conflicting_batch_settings_are_rejected(self) -> None:
        args = self.tune_args(self.root / "unused")
        args.method, args.learning_rates, args.accumulation = "A3", None, None
        with self.assertRaises(DataError):
            auto_tune.resolve_grid(args)
        args.learning_rates = [1e-4]
        args.effective_batch_sizes = [8]
        args.accumulation = 1
        with self.assertRaises(DataError):
            auto_tune.resolve_grid(args)

    def test_default_grid_commands_use_epoch_budget_and_training_seed(self) -> None:
        checkpoint = self.root / "placeholder_checkpoint"
        checkpoint.mkdir(exist_ok=True)
        for method, expected_trials in (("ce", 9), ("bert_base_zh", 6), ("small_zh", 6)):
            args = self.tune_args(self.root / f"mock_grid_{method}")
            args.method = method
            args.batch_size = args.accumulation = args.stage1_steps = args.stage2_steps = None
            args.learning_rates = args.weight_decays = None
            args.allow_nonstandard_grid = False
            args.checkpoint_dir = checkpoint if method != "ce" else None
            args.confirm_model_rights_and_revision = method != "ce"
            calls = []

            def fake_run(command, **kwargs):
                transformer = "--model" in command
                parsed = (train_transformer if transformer else train_pairmatch).parser().parse_args(command[3:])
                calls.append(parsed)
                parsed.run_dir.mkdir()
                names = auto_tune.TRANSFORMER_SETTINGS if transformer else auto_tune.PAIR_SETTINGS
                config = {name: getattr(parsed, name) for name in names}
                config.update({"training_mode": "epoch", "eval_every": 3})
                if transformer:
                    config.update({"steps": 96, "epochs": parsed.epochs, "local_checkpoint_files_sha256": {}})
                else:
                    config.update({"stage1_steps": 96, "stage2_steps": 48, "stage1_epochs": parsed.stage1_epochs,
                                   "stage2_epochs": parsed.stage2_epochs, "term_lexicon_sha256": None,
                                   "pseudo_label_confidence": parsed.confidence, "teacher_ema": parsed.teacher_ema})
                (parsed.run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
                (parsed.run_dir / "provenance.json").write_text(json.dumps({"implementation_sha256": {}}), encoding="utf-8")
                (parsed.run_dir / "summary.json").write_text(json.dumps({"status": "COMPLETE",
                    "stages": [{"best_validation_macro_f1": 0.5 + len(calls) / 100}],
                    "total_executed_steps_including_rolled_back": 1}), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.object(auto_tune.subprocess, "run", side_effect=fake_run):
                result = auto_tune.tune(args)
            self.assertEqual(len(result["trials"]), expected_trials)
            self.assertTrue(all(call.seed == 101 for call in calls))
            self.assertTrue(all((call.epochs == 32 and call.steps is None) if method != "ce"
                                else (call.stage1_epochs == 32 and call.stage2_epochs == 16
                                      and call.stage1_steps is None and call.stage2_steps is None)
                                for call in calls))
            lock = json.loads((args.output_dir / "selected_lock.json").read_text(encoding="utf-8"))
            self.assertNotIn("eval_every", lock["fixed_arguments"])
            self.assertEqual(result["selected"]["effective_batch_size"], 64 if method == "ce" else 32)

    def test_validation_only_grid_and_locked_followup(self) -> None:
        test_path = self.split / "test.jsonl"
        held = self.split / "held.jsonl"
        test_path.rename(held)
        try:
            result = auto_tune.tune(self.tune_args(self.root / "grid"))
            self.assertEqual(len(result["trials"]), 2)
            lock_path = self.root / "grid" / "selected_lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertFalse(lock["protocol_compliant"])
            args = train_pairmatch.parser().parse_args([
                "--split-dir", str(self.split), "--subset-dir", str(self.report_subset),
                "--verification-report", str(self.report), "--run-dir", str(self.root / "locked_run"),
                "--confirm-data-rights", "--confirm-reviewed-data", "--condition", "ce", "--seed", "11",
                "--tuning-lock", str(lock_path), "--stage1-steps", "1", "--eval-every", "1",
                "--batch-size", "4", "--accumulation", "1", "--max-length", "12",
                "--embedding-dim", "6", "--hidden", "4"])
            result = train(args)
            config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["learning_rate"], lock["selected"]["learning_rate"])
            self.assertEqual(config["weight_decay"], lock["selected"]["weight_decay"])
            self.assertFalse(config["tuning_lock"]["protocol_compliant_grid_and_reference"])
            self.assertEqual(result["status"], "COMPLETE")
            args.run_dir = self.root / "mismatch"
            args.batch_size = 3
            with self.assertRaises(DataError):
                train(args)
            self.assertFalse(args.run_dir.exists())
        finally:
            held.rename(test_path)

    def test_mismatched_verification_report_is_rejected(self) -> None:
        args = self.tune_args(self.root / "wrong")
        invalid_report = json.loads(self.report.read_text(encoding="utf-8"))
        invalid_report["split_manifest_sha256"] = "different-split"
        args.verification_report = self.root / "invalid_verification.json"
        args.verification_report.write_text(json.dumps(invalid_report), encoding="utf-8")
        with self.assertRaises(DataError):
            auto_tune.tune(args)
        self.assertFalse(args.output_dir.exists())

    def test_epoch_grid_and_lock_use_sampling_seed_independently(self) -> None:
        args = self.tune_args(self.root / "epoch_grid")
        args.subset_dir = self.report_subset  # fixed data sampling seed 11
        args.stage1_steps = args.stage2_steps = None
        args.stage1_epochs, args.stage2_epochs = 1, 1
        args.learning_rates = [0.0005]
        result = auto_tune.tune(args)
        self.assertEqual(len(result["trials"]), 1)
        plan = json.loads((args.output_dir / "tuning_plan.json").read_text(encoding="utf-8"))
        lock_path = args.output_dir / "selected_lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        config = json.loads((args.output_dir / "trial_01" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((plan["sampling_seed"], plan["tuning_seed"]), (11, 101))
        self.assertEqual((config["seed"], config["training_mode"], config["stage1_epochs"]), (101, "epoch", 1))
        self.assertEqual(len((args.output_dir / "trial_01" / "validation_log.jsonl").read_text(encoding="utf-8").splitlines()), 1)
        self.assertIsNone(lock["fixed_arguments"]["stage1_steps"])
        self.assertIsNone(lock["fixed_arguments"]["stage2_steps"])
        followup = train_pairmatch.parser().parse_args([
            "--split-dir", str(self.split), "--subset-dir", str(self.report_subset),
            "--verification-report", str(self.report), "--run-dir", str(self.root / "epoch_locked"),
            "--confirm-data-rights", "--confirm-reviewed-data", "--condition", "ce", "--seed", "23",
            "--tuning-lock", str(lock_path), "--stage1-epochs", "1", "--stage2-epochs", "1",
            "--batch-size", "4", "--accumulation", "1", "--max-length", "12",
            "--embedding-dim", "6", "--hidden", "4"])
        self.assertEqual(train(followup)["status"], "COMPLETE")
