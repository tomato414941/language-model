import secrets
import threading

import torch
from tokenizers.decoders import DecodeStream

from .training import load_model, select_device


class TextModel:
    def __init__(self, checkpoint, device="cpu", threads=1):
        if threads <= 0:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        self.model, self.tokenizer, saved = load_model(
            checkpoint, select_device(device)
        )
        self.model.eval()
        self.step = saved["step"]
        self.tokens_seen = saved["tokens_seen"]
        self.context = self.model.config.context

    def start(self, prompt, max_output_tokens, temperature, top_p):
        return TextGeneration(self, prompt, max_output_tokens, temperature, top_p)


class TextGeneration:
    def __init__(self, backend, prompt, max_output_tokens, temperature, top_p):
        self.tokenizer = backend.tokenizer
        self.input_tokens = 1 + len(
            self.tokenizer.encode(prompt, add_special_tokens=False).ids
        )
        if self.input_tokens + max_output_tokens > backend.context:
            raise ValueError(
                "Input and max_output_tokens exceed the model's context length "
                + str(backend.context)
                + ". Shorten the input or reduce max_output_tokens."
            )
        self.output_tokens = 0
        self.finished = False
        self.text = ""
        self.cancelled = threading.Event()
        self.eos = self.tokenizer.token_to_id("<|eos|>")
        self.tokens = backend.model.sample_tokens(
            self.tokenizer,
            prompt,
            max_output_tokens,
            temperature,
            top_k=0,
            seed=secrets.randbits(32),
            top_p=top_p,
            stop_requested=self.cancelled.is_set,
        )
        self.deltas = self._decode()

    def _decode(self):
        decoder = DecodeStream(skip_special_tokens=True)
        generated = []
        for token in self.tokens:
            self.output_tokens += 1
            if token == self.eos:
                self.finished = True
                break
            generated.append(token)
            delta = decoder.step(self.tokenizer, token)
            if delta:
                self.text += delta
                yield delta
        # Flush an unfinished byte sequence at the token limit in the same way as decode.
        complete = self.tokenizer.decode(generated, skip_special_tokens=True)
        if not complete.startswith(self.text):
            raise ValueError("Incremental decoding differs from the completed text")
        if tail := complete[len(self.text) :]:
            self.text = complete
            yield tail

    def __iter__(self):
        return self.deltas

    def close(self):
        self.cancelled.set()
        self.deltas.close()
        self.tokens.close()
