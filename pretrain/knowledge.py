"""What a model knows, measured by a definition of language-model-evaluation, the language-model projects' shared
evaluation.

The definition (knowledge-ja-en-1) holds OLMES's English questions and Japanese ones, and Paloma's English text and
Japanese text. A model is measured by the bits per byte of each question's correct answer and of each text's
documents, which move with what it knows rather than with how it answers; the definition divides them into categories
and compares two models by the mean of each language's mean rate of improvement. The definition names its data by
what they hold, and `measure` reads them from a directory that holds them under those names, as the definition's
`make` and `make_japanese` leave them.
"""
from pathlib import Path

from .data import BOS, EOS
from .training import autocast

DEFINITION = "knowledge-ja-en-1"


def artifacts_of(chosen):
    """The names of the artifacts that a definition measures by: its selections of questions and its texts."""
    names = [chosen[section]["selection"] for section in ("olmes", "japanese") if section in chosen]
    return names + [chosen[section]["text"] for section in ("paloma", "japanese_text") if section in chosen]


def reader(model, tokenizer, device, precision, batch_tokens):
    """olmes-measure's reader over the model, which reads rows in windows of its context, with what gives texts'
    tokens and the BOS token."""
    import olmes_measure

    def logits(inputs):
        with autocast(device, precision):
            return model(inputs)

    read = olmes_measure.reader(logits, model.config.context, tokenizer.token_to_id(EOS), device, batch_tokens)
    encode = lambda texts: [encoding.ids for encoding in tokenizer.encode_batch(texts, add_special_tokens=False)]
    return read, encode, tokenizer.token_to_id(BOS)


def answers(read, encode, bos, selection):
    """Each task's bits per byte of a selection's correct answers, and the groups that the tasks are averaged in."""
    import olmes_measure

    groups, _, questions = olmes_measure.read_selection(selection)
    rows = olmes_measure.encode_rows(encode, bos, olmes_measure.pairs(questions))
    tasks, _ = olmes_measure.score(selection, read, rows)
    return tasks, groups


def text(read, encode, bos, artifact, context):
    """The bits per byte of each domain of an artifact of text, {SOURCE/DOMAIN: BPB}: each document read by itself
    after the BOS token, in pieces of the model's context."""
    from language_model_evaluation import paloma

    sizes = {f"{source}/{name}": sum(len(document.encode("utf-8")) for document in paloma.documents(paths))
             for (source, name), paths in paloma.files(artifact).items()}
    domains = {}
    for alias, rows in paloma.rows(artifact, encode, bos, context):
        name = alias.removeprefix(paloma.ALIAS)
        if rows:
            domains[name] = paloma.bits_per_byte(sum(read(rows)[0]), sizes[name])
    return domains


def measure(model, tokenizer, device, precision, artifacts, batch_tokens=16384, chosen=DEFINITION):
    """A model's scores by a definition (its name, or one already loaded): each category's bits per byte, and each
    task's and each domain's that make them, from the definition's artifacts in the directory `artifacts`."""
    import olmes_measure
    from language_model_evaluation import definition

    chosen = definition.load(chosen) if isinstance(chosen, str) else chosen
    artifacts = Path(artifacts)
    for name in artifacts_of(chosen):
        olmes_measure.check(artifacts / name)
    model.eval()
    read, encode, bos = reader(model, tokenizer, device, precision, batch_tokens)
    tasks, groups, domains = {}, {}, {}
    for section in ("olmes", "japanese"):
        if section in chosen:
            found, grouped = answers(read, encode, bos, artifacts / chosen[section]["selection"])
            tasks, groups = {**tasks, **found}, {**groups, **grouped}
    for section in ("paloma", "japanese_text"):
        if section in chosen:
            domains |= text(read, encode, bos, artifacts / chosen[section]["text"], model.config.context)
    categories = {**definition.answered(chosen, groups, tasks), **definition.read(chosen, domains)}
    return {"definition": chosen["name"], "artifacts": artifacts_of(chosen),
            "categories": {name: categories[name] for name in definition.categories(chosen)},
            "tasks": tasks, "domains": domains}


def improvement(result, standing):
    """By how much one model's result improves on another's by their definition's rule: positive when it is the
    better."""
    from language_model_evaluation import definition

    if result["definition"] != standing["definition"]:
        raise ValueError("results of different definitions do not compare")
    return definition.improvement(definition.load(result["definition"]), result["categories"],
                                  standing["categories"])
