"""How well a model follows instructions, measured by a definition of language-model-evaluation, the language-model
projects' shared evaluation.

The definition (instructions-ja-en-1) asks IFEval's English prompts and M-IFEval's Japanese ones and scores the
replies by the authors' code: the share of prompts whose every instruction the reply follows, each language counting
equally. The model writes each reply as an assistant answering the prompt in its conversation format, greedily, up to
the definition's number of tokens, as the authors asked the models whose replies they published.

Replies follow some instructions by chance, whatever they answer. To read how much of a score is the model's following
the prompts, a result also scores the model's replies given to other prompts (`chance`), which neither the authors'
procedure nor the definition holds; the model's own replies compare with these as with another model's
(pretrain.comparison).
"""
import random
from pathlib import Path
from types import SimpleNamespace

from .conversation import chat_prompt
from .evaluation import NativeLM

DEFINITION = "instructions-ja-en-1"


def replies(model, tokenizer, device, precision, batch_tokens, asked, limit):
    """The model's reply to each prompt ({prompt: reply}), each prompt the user's message of a conversation."""
    backend = NativeLM(model, tokenizer, device, precision, batch_tokens)
    requests = [SimpleNamespace(args=(chat_prompt([{"role": "user", "content": row["prompt"]}]),
                                      {"max_gen_toks": limit, "until": []}))
                for row in asked]
    return {row["prompt"]: reply for row, reply in zip(asked, backend.generate_until(requests, disable_tqdm=True))}


def others(asked, replies, seed):
    """The replies given to other prompts ({prompt: reply}): the prompts in an order drawn by `seed`, each given the
    reply to the one after it and the last the first's, so that no prompt has its own and every reply is given once."""
    order = [row["prompt"] for row in asked]
    random.Random(seed).shuffle(order)
    return {prompt: replies[order[(at + 1) % len(order)]] for at, prompt in enumerate(order)}


def chance(language, replies, work, times=8, seed=0):
    """How often a language's prompts are followed by chance, by replies ({prompt: reply}) that answer other prompts:
    of each prompt, at each level, the share of `times` of the replies to others that follow its every instruction,
    {level: [share]}, in the authors' order, scored by their code as `score` scores the replies themselves."""
    from language_model_evaluation import instructions

    home, python = instructions.code(work), instructions.environment(work)
    asked = instructions.prompts(language, home)
    found = {level: [0.0] * len(asked) for level in instructions.LEVELS}
    for time in range(times):
        results = instructions.evaluated(asked, others(asked, replies, seed + time),
                                         Path(work) / "chance" / language / str(time), home, python)
        for level in instructions.LEVELS:
            for at, row in enumerate(results[level]):
                found[level][at] += row["follow_all_instructions"] / times
    return found


def measure(model, tokenizer, device, precision, work, batch_tokens=16384, chosen=DEFINITION):
    """A model's scores by a definition (its name, or one already loaded): each category's share of prompts followed
    at the definition's level, each language's shares at both levels, each reply with the authors' results, and each
    prompt's result at the definition's level (`items`); and under `chance`, the same categories and items of the
    replies given to other prompts. The authors' code and its environment are kept in the directory `work`."""
    from language_model_evaluation import definition, instructions

    chosen = definition.load(chosen) if isinstance(chosen, str) else chosen
    asked = chosen["instructions"]
    if asked["commit"] != instructions.COMMIT:
        raise ValueError(f"{chosen['name']} is scored by the authors' code at {asked['commit']}")
    home = instructions.code(work)
    languages = [language for names in chosen["languages"].values() for category in names
                 for language in chosen["categories"][category]["instructions"]]
    scored, by_chance = {}, {}
    for language in languages:
        written = replies(model, tokenizer, device, precision, batch_tokens, instructions.prompts(language, home),
                          asked["max_new_tokens"])
        scored[language] = instructions.score(language, written, work)
        by_chance[language] = chance(language, written, work)[asked["level"]]
    items = {"prompts": {language: [int(row["follow_all_instructions"]) for row in result["results"][asked["level"]]]
                         for language, result in scored.items()}}
    chance_items = {"prompts": by_chance}
    return {"definition": chosen["name"], "categories": definition.scored(chosen, items),
            "shares": {language: {level: result[level] for level in instructions.LEVELS}
                       for language, result in scored.items()},
            "results": {language: result["results"] for language, result in scored.items()},
            "items": items, "chance": {"categories": definition.scored(chosen, chance_items), "items": chance_items}}
