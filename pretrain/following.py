"""How well a model follows instructions, measured by a definition of language-model-evaluation, the language-model
projects' shared evaluation.

The definition (instructions-ja-en-1) asks IFEval's English prompts and M-IFEval's Japanese ones and scores the
replies by the authors' code: the share of prompts whose every instruction the reply follows, each language counting
equally. The model writes each reply as an assistant answering the prompt in its conversation format, greedily, up to
the definition's number of tokens, as the authors asked the models whose replies they published.
"""
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


def measure(model, tokenizer, device, precision, work, batch_tokens=16384, chosen=DEFINITION):
    """A model's scores by a definition (its name, or one already loaded): each category's share of prompts followed
    at the definition's level, each language's shares at both levels, and each reply with the authors' results. The
    authors' code and its environment are kept in the directory `work`."""
    from language_model_evaluation import definition, instructions

    chosen = definition.load(chosen) if isinstance(chosen, str) else chosen
    asked = chosen["instructions"]
    if asked["commit"] != instructions.COMMIT:
        raise ValueError(f"{chosen['name']} is scored by the authors' code at {asked['commit']}")
    home = instructions.code(work)
    languages = [language for names in chosen["languages"].values() for category in names
                 for language in chosen["categories"][category]["instructions"]]
    scored = {}
    for language in languages:
        written = replies(model, tokenizer, device, precision, batch_tokens, instructions.prompts(language, home),
                          asked["max_new_tokens"])
        scored[language] = instructions.score(language, written, work)
    shares = {language: result[asked["level"]] for language, result in scored.items()}
    return {"definition": chosen["name"], "categories": definition.followed(chosen, shares),
            "shares": {language: {level: result[level] for level in instructions.LEVELS}
                       for language, result in scored.items()},
            "results": {language: result["results"] for language, result in scored.items()}}
