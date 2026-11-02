from __future__ import annotations

import hashlib
import argparse
import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

HAS_TRANSFORMERS = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
if HAS_TRANSFORMERS:
    import torch
    from transformers import BertConfig, BertModel, BertTokenizer
    import create_ldp_subset
    import benchmark_pairmatch
    import prepare_clean_split
    import verify_clean_split
    import predict_pairmatch
    import train_transformer
    import adapt_bert_mlm
    import auto_tune
    from pairmatch.data import DataError
    from pairmatch.transformer_runner import train


@unittest.skipUnless(HAS_TRANSFORMERS, "PyTorch and transformers are required")
class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        source = cls.root / "synthetic.jsonl"
        def fake(value: str) -> str:
            return hashlib.sha256(value.encode()).hexdigest()[:20]
        with source.open("w", encoding="utf-8") as handle:
            for i in range(160):
                handle.write(json.dumps({"sentence1": fake(f"a{i}"), "sentence2": fake(f"b{i}"),
                                         "gold_label": i % 2}) + "\n")
        cls.split = cls.root / "split"
        cls.subset = cls.root / "subset"
        cls.tuning_subset = cls.root / "tuning_subset"
        cls.report = cls.root / "verification.json"
        prepare_clean_split.prepare([source], cls.split)
        cls.report.write_text(json.dumps(verify_clean_split.verify(cls.split)), encoding="utf-8")
        create_ldp_subset.create(cls.split, cls.subset, 30, 11, "remaining", 40)
        create_ldp_subset.create(cls.split, cls.tuning_subset, 30, 101, "remaining", 40)
        cls.checkpoint = cls.root / "synthetic_tiny_bert"
        cls.checkpoint.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("0123456789abcdef")
        vocab += ["##" + c for c in "0123456789abcdef"]
        vocab_path = cls.checkpoint / "vocab.txt"
        vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        parameter = "vocab" if "vocab" in inspect.signature(BertTokenizer.__init__).parameters else "vocab_file"
        tokenizer = BertTokenizer(**{parameter: str(vocab_path)})
        tokenizer.save_pretrained(cls.checkpoint)
        config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=24, max_position_embeddings=32)
        BertModel(config).save_pretrained(cls.checkpoint, safe_serialization=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def arguments(self, output: Path):
        return train_transformer.parser().parse_args([
            "--split-dir", str(self.split), "--subset-dir", str(self.subset),
            "--verification-report", str(self.report), "--checkpoint-dir", str(self.checkpoint),
            "--run-dir", str(output), "--model", "bert_base_zh", "--confirm-data-rights",
            "--confirm-reviewed-data", "--confirm-model-rights-and-revision", "--steps", "2",
            "--eval-every", "1", "--batch-size", "4", "--accumulation", "1", "--max-length", "12"])

    def test_offline_transformer_training_and_locked_prediction(self) -> None:
        args = self.arguments(self.root / "bert_run")
        result = train(args)
        self.assertEqual(result["status"], "COMPLETE")
        checkpoint = torch.load(args.run_dir / result["final_checkpoint"], map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["format_version"], 2)
        self.assertEqual(checkpoint["provenance"]["u_count_used"], 0)
        self.assertTrue(checkpoint["provenance"]["test_was_not_read_during_training"])
        self.assertEqual(checkpoint["config"]["condition"], "bert_base_zh")
        prediction = type("Args", (), {"checkpoint": args.run_dir / result["final_checkpoint"],
                          "split_dir": self.split, "split": "test", "output_dir": self.root / "bert_test",
                          "confirm_test_locked": True, "device": "cpu", "batch_size": 4, "cpu_threads": 1})()
        tested = predict_pairmatch.predict(prediction)
        self.assertEqual(tested["status"], "COMPLETE")
        self.assertIn("encoded_token_coverage_including_special_tokens", tested["truncation"])
        options = type("Args", (), {"checkpoint": args.run_dir / result["final_checkpoint"],
                       "split_dir": self.split, "output": self.root / "bert_benchmark.json",
                       "confirm_data_rights": True, "warmup": 1, "repeats": 2,
                       "sample_count": 2, "threads": 1})()
        timing = benchmark_pairmatch.benchmark(options)
        self.assertEqual(timing["timed_repetitions_per_measurement"], 2)

    def test_rights_and_existing_output_are_required(self) -> None:
        output = self.root / "blocked"
        args = self.arguments(output)
        args.confirm_model_rights_and_revision = False
        with self.assertRaises(DataError):
            train(args)
        self.assertFalse(output.exists())
        output.mkdir()
        args.confirm_model_rights_and_revision = True
        with self.assertRaises(DataError):
            train(args)

    def test_u_only_mlm_then_supervised_finetuning(self) -> None:
        adapt_options = type("Args", (), {"split_dir": self.split, "subset_dir": self.subset,
            "verification_report": self.report, "checkpoint_dir": self.checkpoint,
            "output_dir": self.root / "adaptation", "confirm_data_rights": True,
            "confirm_reviewed_data": True, "confirm_model_rights_and_revision": True,
            "allow_unsafe_pickle_weights": False, "seed": 11, "device": "cpu", "cpu_threads": 1,
            "steps": 1, "batch_size": 4, "accumulation": 1, "max_length": 12,
            "learning_rate": 0.0001, "weight_decay": 0.01, "mask_probability": 0.15})()
        result = adapt_bert_mlm.adapt(adapt_options)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["training_u_pair_exposures"], 4)
        self.assertFalse(result["validation_or_test_text_used_for_mlm"])
        self.assertFalse(result["test_file_read_during_adaptation"])
        args = self.arguments(self.root / "domain_finetune")
        args.model = "bert_domain_adaptation"
        args.checkpoint_dir = adapt_options.output_dir / "adapted_model"
        fitted = train(args)
        self.assertEqual(fitted["status"], "COMPLETE")
        with (args.run_dir / "provenance.json").open(encoding="utf-8") as handle:
            provenance = json.load(handle)
        self.assertGreater(provenance["u_count_used"], 0)
        self.assertIsNotNone(provenance["adaptation_manifest_sha256"])

    def test_transformer_tuning_lock_is_applied_on_a_different_seed(self) -> None:
        options = argparse.Namespace(method="bert_base_zh", split_dir=self.split, subset_dir=self.tuning_subset,
            verification_report=self.report, output_dir=self.root / "tune_bert", confirm_data_rights=True,
            confirm_reviewed_data=True, checkpoint_dir=self.checkpoint, confirm_model_rights_and_revision=True,
            allow_unsafe_pickle_weights=False, term_lexicon=None, device="cpu", cpu_threads=1, max_length=12,
            batch_size=4, accumulation=1, stage1_steps=1, stage2_steps=2000, eval_every=1, embedding_dim=6, hidden=4,
            learning_rates=[0.00002], weight_decays=[0.0], allow_nonstandard_grid=True)
        tuned = auto_tune.tune(options)
        self.assertEqual(len(tuned["trials"]), 1)
        args = self.arguments(self.root / "bert_locked")
        args.steps = 1
        args.tuning_lock = options.output_dir / "selected_lock.json"
        result = train(args)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(args.learning_rate, 0.00002)
        self.assertEqual(args.weight_decay, 0.0)

    def test_domain_adaptation_tuning_lock_is_applied_across_seeds(self) -> None:
        def adapt_for(subset: Path, seed: int, output: Path) -> Path:
            options = argparse.Namespace(split_dir=self.split, subset_dir=subset,
                verification_report=self.report, checkpoint_dir=self.checkpoint, output_dir=output,
                confirm_data_rights=True, confirm_reviewed_data=True,
                confirm_model_rights_and_revision=True, allow_unsafe_pickle_weights=False,
                seed=seed, device="cpu", cpu_threads=1, steps=1, batch_size=4, accumulation=1,
                max_length=12, learning_rate=0.0001, weight_decay=0.01, mask_probability=0.15)
            self.assertEqual(adapt_bert_mlm.adapt(options)["status"], "COMPLETE")
            return output / "adapted_model"

        reference = adapt_for(self.tuning_subset, 101, self.root / "adapt_tune_101")
        options = argparse.Namespace(method="bert_domain_adaptation", split_dir=self.split,
            subset_dir=self.tuning_subset, verification_report=self.report,
            output_dir=self.root / "tune_domain", confirm_data_rights=True, confirm_reviewed_data=True,
            checkpoint_dir=reference, confirm_model_rights_and_revision=True,
            allow_unsafe_pickle_weights=False, term_lexicon=None, device="cpu", cpu_threads=1,
            max_length=12, batch_size=4, accumulation=1, stage1_steps=1, stage2_steps=2000,
            eval_every=1, embedding_dim=6, hidden=4, learning_rates=[0.00002], weight_decays=[0.0],
            allow_nonstandard_grid=True)
        self.assertEqual(auto_tune.tune(options)["status"], "COMPLETE")

        report_model = adapt_for(self.subset, 11, self.root / "adapt_report_11")
        args = self.arguments(self.root / "domain_locked")
        args.model = "bert_domain_adaptation"
        args.checkpoint_dir = report_model
        args.steps = 1
        args.tuning_lock = options.output_dir / "selected_lock.json"
        self.assertEqual(train(args)["status"], "COMPLETE")
        self.assertEqual(args.learning_rate, 0.00002)
        self.assertEqual(args.weight_decay, 0.0)

        manifest_path = report_model / "adaptation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["recipe"]["mask_probability"] = 0.25
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        args.run_dir = self.root / "domain_locked_wrong_recipe"
        with self.assertRaisesRegex(DataError, "domain adaptation recipe differs"):
            train(args)
        self.assertFalse(args.run_dir.exists())
