"""Serve a model in Hugging Face's format, as pretrain.serving serves our checkpoints.

The tokenizer's chat template writes the prompt, and its response template
tells a reply's reasoning, text and tool calls apart, so nothing here depends
on which model it serves. A tokenizer without a response template returns the
whole reply as text.
"""

import json
import queue
import secrets
import threading

import torch
from tokenizers.decoders import DecodeStream
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer


def template_form(messages):
    """The conversation and its tools, in the form that chat templates read."""
    conversation, tools = [], []
    for message in messages:
        message = dict(message)
        tools += [{"type": "function", "function": tool} for tool in message.pop("tools", ())]
        if "tool_calls" in message:
            message["tool_calls"] = [{"type": "function", "function": call} for call in message["tool_calls"]]
        conversation.append(message)
    return conversation, tools


def sampling(temperature, top_p):
    """The settings for `generate`; a temperature of zero always takes the likeliest token."""
    if temperature == 0:
        return {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    return {"do_sample": True, "temperature": temperature, "top_p": top_p}


class TokenQueue(BaseStreamer):
    """Hands the tokens `generate` writes to the thread reading them, without the prompt it puts first."""

    def __init__(self):
        self.tokens = queue.SimpleQueue()
        self.prompt_passed = False

    def put(self, value):
        if self.prompt_passed:
            self.tokens.put(value.reshape(-1).tolist()[0])
        self.prompt_passed = True

    def end(self):
        self.tokens.put(None)


class Cancelled(StoppingCriteria):
    """Stops `generate` at the next token once `event` is set."""

    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


class HuggingFaceModel:
    def __init__(self, directory, device="cpu", context=None, trust_remote_code=False):
        self.tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            directory, dtype="auto", device_map={"": device}, trust_remote_code=trust_remote_code
        ).eval()
        self.context = context or self.model.config.get_text_config().max_position_embeddings

    def start(self, messages, max_output_tokens, temperature, top_p, reasoning=False, tools=(), single_call=False):
        return HuggingFaceGeneration(self, messages, max_output_tokens, temperature, top_p, reasoning, tools, single_call)


class HuggingFaceGeneration:
    """A reply's reasoning, text and tool calls as `generate` writes them on another thread.

    Iterating yields ("reasoning", text) deltas when the prompt asked for
    reasoning, ("text", text) deltas and a ("call", call) for each call of one
    of `tools`, until the reply ends or reaches its limit. A call of anything
    else, or one that does not parse or is cut off, stays in the reply's text.
    With `single_call` the reply ends at its first call.
    """

    def __init__(self, backend, messages, max_output_tokens, temperature, top_p, reasoning=False, tools=(), single_call=False):
        self.tokenizer = backend.tokenizer
        self.reasoning, self.tools, self.single_call = reasoning, set(tools), single_call
        conversation, self.offered = template_form(messages)
        self.prompt = self.tokenizer.apply_chat_template(
            conversation, tools=self.offered or None, tokenize=False, add_generation_prompt=True, enable_thinking=reasoning
        )
        tokens = self.tokenizer(self.prompt, add_special_tokens=False)["input_ids"]
        bos = self.tokenizer.bos_token_id
        # Some templates write the first token themselves, and others leave it to the tokenizer.
        if bos is not None and tokens[:1] != [bos]:
            tokens = [bos, *tokens]
        self.input_tokens = len(tokens)
        if self.input_tokens + max_output_tokens > backend.context:
            raise ValueError(
                "Input and max_output_tokens exceed the model's context length "
                + str(backend.context)
                + ". Shorten the input or reduce max_output_tokens."
            )
        self.output_tokens = self.reasoning_tokens = 0
        self.finished = False
        self.parts = {"reasoning": "", "text": ""}
        self.thinking = False
        self.call_text = ""
        self.parser = self._new_parser()
        ends = backend.model.generation_config.eos_token_id
        self.ends = {ends} if isinstance(ends, int) else set(ends or ())
        self.cancelled = threading.Event()
        self.written = TokenQueue()
        self.failure = None
        torch.manual_seed(secrets.randbits(32))
        arguments = {
            "input_ids": torch.tensor([tokens], device=backend.model.device),
            "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long, device=backend.model.device),
            "max_new_tokens": max_output_tokens,
            "streamer": self.written,
            "stopping_criteria": StoppingCriteriaList([Cancelled(self.cancelled)]),
            **sampling(temperature, top_p),
        }
        self.thread = threading.Thread(target=self._generate, args=(backend.model, arguments), daemon=True)
        self.thread.start()
        self.deltas = self._decode()

    @property
    def text(self):
        return self.parts["text"]

    @property
    def reasoning_text(self):
        return self.parts["reasoning"]

    def _new_parser(self):
        if not getattr(self.tokenizer, "response_template", None):
            return None
        return self.tokenizer.get_response_parser(prefix=self.prompt, tools=self.offered or None)

    def _generate(self, model, arguments):
        try:
            model.generate(**arguments)
        except Exception as error:  # noqa: BLE001 - The reading thread reports it.
            self.failure = error
        finally:
            self.written.tokens.put(None)

    def _decode(self):
        decoder = DecodeStream(skip_special_tokens=self.parser is None)
        if self.parser:
            yield from self._read(self.parser.initial_events)
        while (token := self.written.tokens.get()) is not None:
            self.output_tokens += 1
            if token in self.ends:
                self.finished = True
                break
            piece = decoder.step(self.tokenizer.backend_tokenizer, token)
            if piece and not self.parser:
                yield from self._write("text", piece)
            elif piece:
                for delta in self._read(self._parse(piece)):
                    yield delta
                    if delta[0] == "call" and self.single_call:
                        self.finished = True
                        return
            self.reasoning_tokens += self.thinking
        if self.failure is not None:
            raise self.failure
        if self.parser:
            yield from self._read(self._parse())

    def _parse(self, piece=None):
        """The parser's events for a piece of the reply, or for its end when no piece is given."""
        try:
            return self.parser.feed(piece) if piece else self.parser.finalize()[1]
        except ValueError:
            # The parser stops at a call it cannot read, so the rest of the reply goes to a new one.
            text, self.call_text = self.call_text, ""
            self.parser = self._new_parser()
            return [{"type": "region_chunk", "field": "content", "text": text}]

    def _read(self, events):
        """The deltas in the parser's events."""
        for event in events:
            kind, region = event["type"], event["field"]
            if region == "thinking" and kind != "region_chunk":
                self.thinking = kind == "region_open"
            elif kind == "region_chunk" and region == "content":
                yield from self._write("text", event["text"])
            elif kind == "region_chunk" and region == "thinking":
                if self.reasoning:
                    yield from self._write("reasoning", event["text"])
            elif kind == "region_chunk" and region == "tool_calls":
                self.call_text += event["text"]
            elif kind == "region_close" and region == "tool_calls":
                self.call_text = ""
                call = event["value"]["function"]
                if call["name"] in self.tools:
                    yield "call", {"name": call["name"], "arguments": call["arguments"]}
                else:
                    yield from self._write("text", json.dumps(call, ensure_ascii=False))

    def _write(self, part, text):
        self.parts[part] += text
        yield part, text

    def __iter__(self):
        return self.deltas

    def close(self):
        self.cancelled.set()
        self.thread.join()
        self.deltas.close()
