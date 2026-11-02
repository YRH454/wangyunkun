from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
if AVAILABLE:
    import auto_tune
    import create_ldp_subset
    import prepare_clean_split
    import train_pairmatch
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

    def test_wrong_reference_subset_is_rejected(self) -> None:
        args = self.tune_args(self.root / "wrong")
        args.subset_dir = self.report_subset
        with self.assertRaises(DataError):
            auto_tune.tune(args)
        self.assertFalse(args.output_dir.exists())
