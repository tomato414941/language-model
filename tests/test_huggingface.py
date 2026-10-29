import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pretrain.conversation import TURN_END, chat_prompt
from pretrain.data import BOS, EOS
from pretrain.huggingface import export_huggingface
from pretrain.model import LanguageModel, ModelConfig
from tests.test_benchmarks import byte_tokenizer


class HuggingFaceTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.tokenizer = byte_tokenizer(reserved_tokens=8)
        self.bos = self.tokenizer.token_to_id(BOS)
        self.ends = {self.tokenizer.token_to_id(TURN_END), self.tokenizer.token_to_id(EOS)}

    def export(self, **changes):
        config = ModelConfig(
            architecture="llama", width=32, heads=4, kv_heads=2, layers=2, context=64,
            qk_norm=True, attention_gate="headwise", document_start=self.bos, prediction_depth=1,
            **changes,
        )
        torch.manual_seed(0)
        native = LanguageModel(self.tokenizer.get_vocab_size(), config).eval()
        # Away from their starting values, so that every weight shows in the outputs.
        with torch.no_grad():
            for parameter in native.parameters():
                parameter.add_(0.1 * torch.randn_like(parameter))
        checkpoint = self.root / "last.pt"
        torch.save(
            {"version": 1, "model": native.state_dict(), "model_config": dataclasses.asdict(config),
             "tokenizer": self.tokenizer.to_str(), "step": 3, "tokens_seen": 10},
            checkpoint,
        )
        export_huggingface(checkpoint, self.root / "hf")
        loaded = AutoModelForCausalLM.from_pretrained(self.root / "hf", trust_remote_code=True).eval()
        return native, loaded

    def test_書き出したモデルは自前のモデルと同じ確率を出す(self):
        native, loaded = self.export()
        tokens = torch.tensor([[self.bos, 70, 71, 72, self.bos, 73, 74, 75, 76]])
        with torch.no_grad():
            expected = native(tokens)
            got = loaded(input_ids=tokens).logits
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    def test_左に詰め物をした一括の入力でも詰め物なしと同じ確率を出す(self):
        native, loaded = self.export()
        sequences = [[self.bos, 70, 71, 72, 73, 74], [self.bos, 75, 76]]
        pad = self.tokenizer.token_to_id(EOS)
        longest = max(map(len, sequences))
        tokens = torch.tensor([[pad] * (longest - len(row)) + row for row in sequences])
        mask = torch.tensor([[0] * (longest - len(row)) + [1] * len(row) for row in sequences])
        with torch.no_grad():
            got = loaded(input_ids=tokens, attention_mask=mask).logits
            for index, row in enumerate(sequences):
                expected = native(torch.tensor([row]))[0]
                torch.testing.assert_close(got[index, longest - len(row) :], expected, atol=1e-5, rtol=1e-5)

    def test_キャッシュを使った生成は自前の貪欲な生成と同じ語を書く(self):
        native, loaded = self.export()
        prompt = [self.bos, 70, 71, 72]
        expected = native.continue_greedily([prompt], 12, self.ends)[0]
        with torch.no_grad():
            written = loaded.generate(torch.tensor([prompt]), max_new_tokens=12, do_sample=False)[0, len(prompt) :]
        written = written.tolist()
        stop = next((index for index, token in enumerate(written) if token in self.ends), len(written))
        self.assertEqual(written[:stop], expected)

    def test_トークナイザは先頭にBOSを付けて自前と同じ番号を返し会話を自前と同じ書式で書く(self):
        self.export()
        loaded = AutoTokenizer.from_pretrained(self.root / "hf")
        self.assertEqual(loaded("雨が降る")["input_ids"], [self.bos, *self.tokenizer.encode("雨が降る").ids])
        messages = [
            {"role": "system", "content": "短く答える。"},
            {"role": "user", "content": "天気は？"},
            {"role": "assistant", "content": "雨です。"},
            {"role": "user", "content": "明日は？"},
        ]
        self.assertEqual(loaded.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), chat_prompt(messages))

    def test_書き出せない部品を持つモデルは書き出しを断る(self):
        with self.assertRaisesRegex(ValueError, "value_embeddings"):
            self.export(value_embeddings=True)


if __name__ == "__main__":
    unittest.main()
