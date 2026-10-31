import secrets
import threading

import torch
from tokenizers.decoders import DecodeStream

from .conversation import CALL_END, CALL_START, THINK_END, chat_prompt, parse_call, reply_ends, require_markers
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

    def start(self, messages, max_output_tokens, temperature, top_p, reasoning=False, tools=(), single_call=False):
        prompt = chat_prompt(messages, reasoning)
        return TextGeneration(self, prompt, max_output_tokens, temperature, top_p, reasoning, tools, single_call)


class TextGeneration:
    """A reply's text as it is sampled, after its reasoning when the prompt opened one.

    Iterating yields ("reasoning", text) deltas until the reasoning closes,
    then ("text", text) deltas and a ("call", call) for each call of one of
    `tools`, until the reply ends or reaches its limit. A reply offered no
    tools cannot open a call; a call that names no offered tool, does not
    parse or is cut off stays in the reply's text. With `single_call` the
    reply ends at its first call.
    """

    def __init__(self, backend, prompt, max_output_tokens, temperature, top_p, reasoning=False, tools=(), single_call=False):
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
        self.tools, self.single_call = set(tools), single_call
        self.think_end, self.call_start, self.call_end = (
            self.tokenizer.token_to_id(marker) for marker in (THINK_END, CALL_START, CALL_END)
        )
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
            banned=() if self.tools else (self.call_start,),
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
        decoder, generated, written = DecodeStream(skip_special_tokens=True), [], ""
        for token in self.tokens:
            self.output_tokens += 1
            if token in self.ends:
                self.finished = True
                break
            if phase == "reasoning":
                self.reasoning_tokens += 1
                if token == self.think_end:
                    yield from self._flush(phase, generated, written)
                    phase, decoder, generated, written = "text", DecodeStream(skip_special_tokens=True), [], ""
                    continue
            elif phase == "text" and token == self.call_start:
                yield from self._flush(phase, generated, written)
                phase, generated = "call", []
                continue
            elif phase == "call":
                if token != self.call_end:
                    generated.append(token)
                    continue
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                call = parse_call(text)
                if call is None or call["name"] not in self.tools:
                    yield from self._words(text)
                else:
                    yield "call", call
                    if self.single_call:
                        self.finished = True
                        return
                phase, decoder, generated, written = "text", DecodeStream(skip_special_tokens=True), [], ""
                continue
            generated.append(token)
            delta = decoder.step(self.tokenizer, token)
            if delta:
                written += delta
                self.parts[phase] += delta
                yield phase, delta
        if phase == "call":
            yield from self._words(self.tokenizer.decode(generated, skip_special_tokens=True))
        else:
            yield from self._flush(phase, generated, written)

    def _flush(self, phase, generated, written):
        """Yield what remains of a part, such as an unfinished byte sequence, as decode completes it."""
        complete = self.tokenizer.decode(generated, skip_special_tokens=True)
        if not complete.startswith(written):
            raise ValueError("Incremental decoding differs from the completed text")
        if tail := complete[len(written) :]:
            self.parts[phase] += tail
            yield phase, tail

    def _words(self, text):
        """Yield a call that is not returned as one as the reply's text."""
        if text:
            self.parts["text"] += text
            yield "text", text

    def wait(self):
        """A reply sampled here has started already."""

    def __iter__(self):
        return self.deltas

    def close(self):
        self.cancelled.set()
        self.deltas.close()
        self.tokens.close()
