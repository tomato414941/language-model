import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import EntryNotFoundError
from tokenizers import Tokenizer, models, pre_tokenizers

from pretrain.hub import (
    download_checkpoint,
    prepare_checkpoint,
    publish_checkpoint,
    publish_training_run,
)
from pretrain.model import LanguageModel, ModelConfig
from pretrain.training import TrainConfig, load_model


class MemoryHub:
    def __init__(self):
        self.token = "test-hf-token"
        self.private = True
        self.sha = "0" * 40
        self.commits = {self.sha: {}}
        self.uploads = 0
        self.downloads = []
        self.change_after_manifest = False
        self.corrupt_download = None
        self.create_race = False

    def create_repo(self, **kwargs):
        self.created = kwargs

    def repo_info(self, repo_id, repo_type, revision=None):
        return SimpleNamespace(sha=revision or self.sha, private=self.private)

    def create_commit(
        self, repo_id, repo_type, operations, commit_message, parent_commit
    ):
        if self.create_race:
            self.sha = "f" * 40
            self.commits[self.sha] = copy.deepcopy(self.commits[parent_commit])
        if parent_commit != self.sha:
            raise RuntimeError("repository changed before committing")
        files = copy.deepcopy(self.commits[parent_commit])
        for operation in operations:
            if isinstance(operation, CommitOperationAdd):
                value = operation.path_or_fileobj
                files[operation.path_in_repo] = (
                    value if isinstance(value, bytes) else Path(value).read_bytes()
                )
            elif isinstance(operation, CommitOperationDelete):
                files.pop(operation.path_in_repo, None)
        self.uploads += 1
        self.sha = f"{self.uploads:040x}"
        self.commits[self.sha] = files
        return SimpleNamespace(oid=self.sha)

    def download(self, repo_id, repo_type, filename, revision, token, local_dir):
        self.downloads.append((filename, revision))
        if filename not in self.commits[revision]:
            raise EntryNotFoundError("file does not exist")
        data = self.commits[revision][filename]
        if self.change_after_manifest and filename.endswith("manifest.json"):
            self.sha = "e" * 40
            self.commits[self.sha] = {}
        if filename == self.corrupt_download:
            data += b"corrupt"
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)


class HubTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.hub = MemoryHub()
        self.api = patch("huggingface_hub.HfApi", return_value=self.hub)
        self.download = patch(
            "huggingface_hub.hf_hub_download", side_effect=self.hub.download
        )
        self.api.start()
        self.download.start()
        self.addCleanup(self.api.stop)
        self.addCleanup(self.download.stop)
        self.repo_id = "test-owner/language-model"
        tokenizer = Tokenizer(
            models.WordLevel(
                {"<|bos|>": 0, "<|eos|>": 1, "<|unk|>": 2, "hello": 3, "world": 4},
                unk_token="<|unk|>",
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        config = ModelConfig(width=8, heads=2, layers=1, context=8)
        torch.manual_seed(17)
        model = LanguageModel(tokenizer.get_vocab_size(), config)
        self.saved = {
            "version": 1,
            "model": model.state_dict(),
            "model_config": asdict(config),
            "train_config": asdict(TrainConfig()),
            "tokenizer": tokenizer.to_str(),
            "dataset_fingerprint": "a" * 64,
            "step": 7,
            "tokens_seen": 512,
        }
        self.checkpoint = self.root / "model.pt"
        torch.save(self.saved, self.checkpoint)

    def publish(self, **kwargs):
        return publish_checkpoint(
            self.checkpoint,
            self.repo_id,
            "bilingual-v2",
            token="test-hf-token",
            **kwargs,
        )

    def test_published_checkpoint_restores_weights_tokenizer_configuration_and_generation(
        self,
    ):
        published = self.publish(source_commit="b" * 40)
        restored = download_checkpoint(
            self.repo_id, "bilingual-v2", self.root / "restored", published["revision"]
        )
        original, original_tokenizer, original_saved = load_model(
            self.checkpoint, torch.device("cpu")
        )
        model, tokenizer, saved = load_model(
            restored["checkpoint"], torch.device("cpu")
        )
        self.assertTrue(restored["verified"])
        self.assertTrue(self.hub.created["private"])
        self.assertEqual(saved["model_config"], original_saved["model_config"])
        self.assertEqual(tokenizer.to_str(), original_tokenizer.to_str())
        for name, value in saved["model"].items():
            self.assertTrue(torch.equal(value, original_saved["model"][name]))
        self.assertEqual(
            model.generate(tokenizer, "hello", max_new_tokens=3, seed=9),
            original.generate(original_tokenizer, "hello", max_new_tokens=3, seed=9),
        )
        folder = Path(restored["checkpoint"]).parent
        self.assertEqual(
            json.loads((folder / "config.json").read_text())["training"],
            self.saved["train_config"],
        )
        self.assertEqual(
            Tokenizer.from_file(str(folder / "tokenizer.json")).to_str(),
            tokenizer.to_str(),
        )
        self.assertTrue((folder / "source/pretrain/model.py").is_file())
        self.assertIsNone(restored["resume_checkpoint"])

    def test_publishing_unchanged_files_reuses_the_existing_version(self):
        first = self.publish()
        second = self.publish()
        self.assertEqual(second["revision"], first["revision"])
        self.assertTrue(second["reused"])
        self.assertEqual(self.hub.uploads, 1)

    def test_download_uses_one_commit_when_the_branch_moves(self):
        published = self.publish()
        self.hub.downloads.clear()
        self.hub.change_after_manifest = True
        restored = download_checkpoint(
            self.repo_id, "bilingual-v2", self.root / "pinned"
        )
        self.assertTrue(restored["verified"])
        self.assertEqual(restored["revision"], published["revision"])
        self.assertTrue(
            all(revision == published["revision"] for _, revision in self.hub.downloads)
        )

    def test_download_detects_files_that_do_not_match_the_saved_checkpoint(self):
        self.publish()
        self.hub.corrupt_download = "runs/bilingual-v2/model.pt"
        with self.assertRaisesRegex(ValueError, "expected checksum"):
            download_checkpoint(self.repo_id, "bilingual-v2", self.root / "corrupt")

    def test_training_run_preserves_the_latest_optimizer_and_random_number_states(self):
        output = self.root / "training"
        output.mkdir()
        torch.save(self.saved, output / "best.pt")
        latest = {
            **self.saved,
            "step": 9,
            "tokens_seen": 1024,
            "optimizer": {"step": 9},
            "scaler": {},
            "sampling_rng": torch.get_rng_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": None,
        }
        torch.save(latest, output / "last.pt")
        with patch.dict("os.environ", {"HF_TOKEN": "test-hf-token"}):
            published = publish_training_run(output, self.repo_id, "continued")
        restored = download_checkpoint(
            self.repo_id, "continued", self.root / "resume", published["revision"]
        )
        saved = torch.load(restored["resume_checkpoint"], weights_only=True)
        self.assertEqual(saved["optimizer"], {"step": 9})
        self.assertEqual(saved["step"], 9)
        self.assertTrue(torch.equal(saved["sampling_rng"], latest["sampling_rng"]))
        self.assertEqual(
            torch.load(restored["checkpoint"], weights_only=True)["step"], 7
        )

    def test_training_cli_saves_its_completed_checkpoint_to_the_selected_repository(
        self,
    ):
        from pretrain.__main__ import main

        output = self.root / "completed"
        output.mkdir()
        latest = {
            **self.saved,
            "optimizer": {},
            "scaler": {},
            "sampling_rng": torch.get_rng_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": None,
        }
        torch.save(latest, output / "last.pt")
        config = self.root / "config.toml"
        config.write_text(
            "[model]\nwidth=8\nheads=2\nlayers=1\ncontext=8\n[training]\n"
        )
        with (
            patch("pretrain.__main__.train"),
            patch.dict("os.environ", {"HF_TOKEN": "test-hf-token"}),
            contextlib.redirect_stdout(io.StringIO()) as stream,
        ):
            main(
                [
                    "train",
                    "--data",
                    str(self.root / "data"),
                    "--output",
                    str(output),
                    "--config",
                    str(config),
                    "--device",
                    "cpu",
                    "--threads",
                    "1",
                    "--hub-repo",
                    self.repo_id,
                    "--hub-run-id",
                    "completed",
                ]
            )
        self.assertEqual(json.loads(stream.getvalue())["event"], "hub_saved")
        restored = download_checkpoint(
            self.repo_id, "completed", self.root / "cli-restored"
        )
        self.assertEqual(
            torch.load(restored["resume_checkpoint"], weights_only=True)["step"], 7
        )

    def test_repository_changes_are_reported_without_overwriting_the_newer_version(
        self,
    ):
        self.hub.create_race = True
        with self.assertRaisesRegex(RuntimeError, "repository changed"):
            self.publish()
        self.assertEqual(self.hub.sha, "f" * 40)

    def test_checkpoint_paths_are_kept_inside_the_selected_training_run(self):
        with self.assertRaisesRegex(ValueError, "single name"):
            prepare_checkpoint(self.checkpoint, "../other-run")


if __name__ == "__main__":
    unittest.main()
