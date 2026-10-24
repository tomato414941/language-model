import secrets
import threading

import torch
from tokenizers.decoders import DecodeStream

from .conversation import THINK_END, reply_ends, require_markers
from .training import load_model, select_device


class TextModel:
    def __init__(self, checkpoint, device="cpu", threads=1):
        if threads <= 0:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        self.model, self.tokenizer, saved = load_model(
            checkpoint, select_device(device)
        )
        require_markers(self.tokenizer)
        self.model.eval()
        self.step = saved["step"]
        self.tokens_seen = saved["tokens_seen"]
        self.context = self.model.config.context

    def start(self, prompt, max_output_tokens, temperature, top_p, reasoning=False):
        return TextGeneration(self, prompt, max_output_tokens, temperature, top_p, reasoning)


class TextGeneration:
    """A reply's text as it is sampled, after its reasoning when the prompt opened one.

    Iterating yields ("reasoning", text) deltas until the reasoning closes,
    then ("text", text) deltas until the reply ends or reaches its limit.
    """

    def __init__(self, backend, prompt, max_output_tokens, temperature, top_p, reasoning=False):
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
        self.reasoning = reasoning
        self.reasoning_tokens = 0
        self.parts = {"reasoning": "", "text": ""}
        self.think_end = self.tokenizer.token_to_id(THINK_END)
        self.cancelled = threading.Event()
        self.ends = reply_ends(self.tokenizer)
        self.tokens = backend.model.sample_tokens(
            self.tokenizer,
            prompt,
            max_output_tokens,
            temperature,
            top_k=0,
            seed=secrets.randbits(32),
            top_p=top_p,
            stop_requested=self.cancelled.is_set,
            stops=self.ends,
        )
        self.deltas = self._decode()

    @property
    def text(self):
        return self.parts["text"]

    @property
    def reasoning_text(self):
        return self.parts["reasoning"]

    def _decode(self):
        phase = "reasoning" if self.reasoning else "text"
        decoder = DecodeStream(skip_special_tokens=True)
        generated = []
        for token in self.tokens:
            self.output_tokens += 1
            if token in self.ends:
                self.finished = True
                break
            if phase == "reasoning":
                self.reasoning_tokens += 1
                if token == self.think_end:
                    yield from self._flush(phase, generated)
                    phase, decoder, generated = "text", DecodeStream(skip_special_tokens=True), []
                    continue
            generated.append(token)
            delta = decoder.step(self.tokenizer, token)
            if delta:
                self.parts[phase] += delta
                yield phase, delta
        yield from self._flush(phase, generated)

    def _flush(self, phase, generated):
        """Yield what remains of a part, such as an unfinished byte sequence, as decode completes it."""
        complete = self.tokenizer.decode(generated, skip_special_tokens=True)
        if not complete.startswith(self.parts[phase]):
            raise ValueError("Incremental decoding differs from the completed text")
        if tail := complete[len(self.parts[phase]) :]:
            self.parts[phase] = complete
            yield phase, tail

    def __iter__(self):
        return self.deltas

    def close(self):
        self.cancelled.set()
        self.deltas.close()
        self.tokens.close()
