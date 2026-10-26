"""Measure how well a model recalls a number hidden in long filler text, in English and Japanese.

Each prompt repeats a few filler sentences to a chosen length and places a
five-digit pass key among them, the test of Mohtashami and Jaggi
(arXiv:2305.16300) that RULER's needle in a haystack builds on. The model
completes the closing question; the first number it writes must be the key.
The keys' depths spread evenly from the start of the filler to its end, so
a model that reads only the end of a long prompt misses most of them.
"""

import random
import re

from .conversation import reply_ends
from .data import BOS
from .training import autocast

# An introduction, the filler, the key's sentence and the question.
TEMPLATES = {
    "en": (
        "There is important information hidden inside a lot of irrelevant text. "
        "Find it and memorize it. I will quiz you about it.\n",
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ",
        "The pass key is {key}. Remember it. {key} is the pass key. ",
        "\nWhat is the pass key? The pass key is",
    ),
    "ja": (
        "たくさんの関係のない文章の中に、大切な情報が隠されています。"
        "それを見つけて覚えてください。あとで質問します。\n",
        "草は緑です。空は青いです。太陽は黄色いです。さあ、行きましょう。行って、帰ってきます。",
        "パスキーは{key}です。覚えておいてください。{key}がパスキーです。",
        "\nパスキーは何ですか？パスキーは",
    ),
}
# Enough for a space, the five digits and what follows them.
ANSWER_TOKENS = 8


def passkey_prompt(tokenizer, language, length, depth, key):
    """The token IDs of a prompt that leaves room for the answer within `length` tokens.

    The key's sentence sits `depth`, from 0 to 1, of the way through the filler.
    """
    introduction, filler, needle, question = TEMPLATES[language]
    needle = needle.format(key=key)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    room = length - ANSWER_TOKENS
    # Words merge across the joins of repeated filler, so each copy's share of
    # a long repetition sets how many fit; the loop removes any extra.
    share = len(encode(filler * 64)) / 64
    units = int((room - 1 - len(encode(introduction + needle + question))) / share)
    while units > 0:
        before = round(units * depth)
        text = introduction + filler * before + needle + filler * (units - before) + question
        tokens = [tokenizer.token_to_id(BOS), *encode(text)]
        if len(tokens) <= room:
            return tokens
        units -= 1
    raise ValueError(f"{length} tokens are too few for a pass key prompt")


def passkey_accuracy(model, tokenizer, lengths, samples, device, precision="float32", seed=0):
    """Each language's share of keys recalled at each length, with `samples` keys per length."""
    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if any(length > model.config.context for length in lengths):
        raise ValueError("pass key prompts must fit within the model's context")
    ends = reply_ends(tokenizer)
    accuracy = {}
    for language in TEMPLATES:
        generator = random.Random(f"{seed}:{language}")
        for length in lengths:
            recalled = 0
            for index in range(samples):
                key = generator.randrange(10_000, 100_000)
                prompt = passkey_prompt(tokenizer, language, length, index / max(1, samples - 1), key)
                with autocast(device, precision):
                    [continuation] = model.continue_greedily(
                        [prompt], ANSWER_TOKENS, ends, banned=[tokenizer.token_to_id(BOS)]
                    )
                answer = re.search(r"\d+", tokenizer.decode(continuation, skip_special_tokens=True))
                recalled += answer is not None and answer.group() == str(key)
            accuracy.setdefault(language, {})[str(length)] = recalled / samples
    return accuracy
