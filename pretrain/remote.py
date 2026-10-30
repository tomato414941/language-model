"""Sample replies in one container and serve them from another.

The GPU container runs `relay` on a TextModel, and the API's container reads
its records through a RemoteTextModel, which starts replies as TextModel does.
"""


def relay(model, *arguments):
    """Sample a reply with `model.start(*arguments)` as records that RemoteGeneration reads.

    The first record is the input's token count, or why the model refused the
    input. The reply's deltas follow, and the last record holds whether it
    finished and how many tokens it used.
    """
    try:
        generation = model.start(*arguments)
    except ValueError as error:
        yield "refused", str(error)
        return
    try:
        yield "started", generation.input_tokens
        yield from generation
        yield "ended", (generation.finished, generation.output_tokens, generation.reasoning_tokens)
    finally:
        generation.close()


class RemoteTextModel:
    """A text model that samples elsewhere, through `sample`, which streams relay's records."""

    def __init__(self, sample, revision):
        self.sample, self.revision = sample, revision

    def start(self, prompt, max_output_tokens, temperature, top_p, reasoning=False, tools=(), single_call=False):
        records = self.sample(prompt, max_output_tokens, temperature, top_p, reasoning, list(tools), single_call)
        return RemoteGeneration(records, reasoning)


class RemoteGeneration:
    """A reply sampled elsewhere, with TextGeneration's deltas, parts and token counts."""

    def __init__(self, records, reasoning):
        self.records, self.reasoning = records, reasoning
        kind, value = next(records)
        if kind == "refused":
            raise ValueError(value)
        self.input_tokens = value
        self.output_tokens = self.reasoning_tokens = 0
        self.finished = False
        self.parts = {"reasoning": "", "text": ""}
        self.deltas = self._read()

    @property
    def text(self):
        return self.parts["text"]

    @property
    def reasoning_text(self):
        return self.parts["reasoning"]

    def _read(self):
        for kind, value in self.records:
            if kind == "ended":
                self.finished, self.output_tokens, self.reasoning_tokens = value
                return
            if kind != "call":
                self.parts[kind] += value
            yield kind, value

    def __iter__(self):
        return self.deltas

    def close(self):
        self.deltas.close()
        self.records.close()
