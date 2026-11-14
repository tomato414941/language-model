"""What a model knows, measured by a definition of language-model-evaluation, the language-model projects' shared
evaluation.

The definition (knowledge-ja-en-1) holds OLMES's English questions and Japanese ones, and Paloma's English text and
Japanese text. A model is measured by the bits per byte of each question's correct answer and of each text's
documents, which move with what it knows rather than with how it answers; the definition divides them into categories
and compares two models by the mean of each language's mean rate of improvement. A result keeps each question's and
each document's measurement too, from which the comparison's standard errors come (pretrain.comparison). The
definition names its data by what they hold, and `measure` reads them from a directory that holds them under those
names, as the definition's `make` and `make_japanese` leave them.
"""
import math
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
    """Each task's questions' bits per byte of their correct answers, and the groups that the tasks are averaged in."""
    import olmes_measure

    groups, _, questions = olmes_measure.read_selection(selection)
    rows = olmes_measure.encode_rows(encode, bos, olmes_measure.pairs(questions))
    return olmes_measure.values(selection, read, rows), groups


def text(read, encode, bos, artifact, context):
    """Each document's bits and bytes, by domain ({SOURCE/DOMAIN: [[bits, bytes]]}): each document read by itself
    after the BOS token, in pieces of the model's context. Documents without text are left out."""
    from language_model_evaluation import paloma

    domains = {}
    for alias, documents in paloma.document_rows(artifact, encode, bos, context):
        documents = [(size, rows) for size, rows in documents if size]
        if not documents:
            continue
        totals, at, found = read([row for _, rows in documents for row in rows])[0], 0, []
        for size, rows in documents:
            found.append([-sum(totals[at:at + len(rows)]) / math.log(2), size])
            at += len(rows)
        domains[alias.removeprefix(paloma.ALIAS)] = found
    return domains


def measure(model, tokenizer, device, precision, artifacts, batch_tokens=16384, chosen=DEFINITION):
    """A model's scores by a definition (its name, or one already loaded): each category's bits per byte, each task's
    and each domain's that make them, and the measurement item by item from which they come, each question's bits per
    byte and each document's bits and bytes (`items`, with the selections' `groups`), from the definition's artifacts
    in the directory `artifacts`."""
    import olmes_measure
    from language_model_evaluation import definition

    chosen = definition.load(chosen) if isinstance(chosen, str) else chosen
    artifacts = Path(artifacts)
    for name in artifacts_of(chosen):
        olmes_measure.check(artifacts / name)
    model.eval()
    read, encode, bos = reader(model, tokenizer, device, precision, batch_tokens)
    items, groups = {}, {}
    for section in ("olmes", "japanese"):
        if section in chosen:
            found, grouped = answers(read, encode, bos, artifacts / chosen[section]["selection"])
            items.setdefault("questions", {}).update(found)
            groups |= grouped
    for section in ("paloma", "japanese_text"):
        if section in chosen:
            items.setdefault("documents", {}).update(
                text(read, encode, bos, artifacts / chosen[section]["text"], model.config.context))
    questions, documents = items.get("questions", {}), items.get("documents", {})
    return {"definition": chosen["name"], "artifacts": artifacts_of(chosen),
            "categories": definition.scored(chosen, items, groups),
            "tasks": {task: sum(values) / len(values) for task, values in questions.items()},
            "domains": {name: sum(bits for bits, _ in found) / sum(size for _, size in found)
                        for name, found in documents.items()},
            "groups": groups, "items": items}
