"""Sample replies in one container and serve them from another.

The GPU container runs `relay` on a TextModel, and the API's container reads
its records through a RemoteTextModel, which starts replies as TextModel does.
"""

import math
import time

DAY_SECONDS = 86400


class OutOfGPUTime(Exception):
    """Today's GPU time has run out, and returns in `retry_after` seconds."""

    def __init__(self, retry_after):
        super().__init__("Today's GPU time has run out")
        self.retry_after = retry_after


class GPUTime:
    """Stop asking the GPU for replies once it has been up for `limit` seconds in a UTC day.

    The GPU costs the same working or waiting, so this counts the time it is up,
    not the replies it writes. A reply keeps it up from when it is asked for until
    `idle` seconds after it ends, and time that replies share counts once. `store`,
    which outlasts the containers using it, holds each day's seconds and the
    moment they are counted up to.
    """

    def __init__(self, store, limit, idle, clock=time.time):
        self.store, self.limit, self.idle, self.clock = store, limit, idle, clock

    def begin(self):
        """Count the start of a reply and return when it started, or refuse it once the time has run out."""
        now = self.clock()
        if self.counted(now)[0] >= self.limit:
            raise OutOfGPUTime(math.ceil(DAY_SECONDS - now % DAY_SECONDS))
        self.count(now, now)
        return now

    def end(self, started):
        """Count the rest of the reply that began at `started`."""
        self.count(started, self.clock())

    def counted(self, now):
        return self.store.get(self.day(now), (0.0, 0.0))

    def count(self, started, now):
        seconds, counted = self.counted(now)
        up_until = now + self.idle
        self.store[self.day(now)] = (
            seconds + max(0.0, up_until - max(started, counted)),
            max(counted, up_until),
        )

    @staticmethod
    def day(now):
        return time.strftime("%Y-%m-%d", time.gmtime(now))


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

    def __init__(self, sample, revision, gpu_time):
        self.sample, self.revision, self.gpu_time = sample, revision, gpu_time

    def start(self, prompt, max_output_tokens, temperature, top_p, reasoning=False, tools=(), single_call=False):
        started = self.gpu_time.begin()
        try:
            records = self.sample(prompt, max_output_tokens, temperature, top_p, reasoning, list(tools), single_call)
            return RemoteGeneration(records, reasoning, lambda: self.gpu_time.end(started))
        except BaseException:
            # The GPU started to read even an input it refused.
            self.gpu_time.end(started)
            raise


class RemoteGeneration:
    """A reply sampled elsewhere, with TextGeneration's deltas, parts and token counts."""

    def __init__(self, records, reasoning, ended):
        self.records, self.reasoning, self.ended = records, reasoning, ended
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
        try:
            self.deltas.close()
            self.records.close()
        finally:
            self.ended()
