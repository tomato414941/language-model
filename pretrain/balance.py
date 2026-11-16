"""How evenly a model reads Japanese and English, measured by a definition of language-model-evaluation, the
language-model projects' shared evaluation.

The definition (balance-ja-en-1) holds FLORES+'s 1,012 devtest sentences, each written in English and translated by
people into Japanese. A model reads each sentence by itself after the BOS token, as the knowledge measure reads a
document (pretrain.knowledge), and its balance is the bits that it takes to read the Japanese sentences over those
that it takes to read the same sentences in English: above 1, Japanese costs it more for the same content. The
balance does not judge a generation; it tells how the two languages' data serve the model. A result keeps each
sentence's bits, from which the balance's standard error comes and by which two results compare
(pretrain.comparison).
"""
from pathlib import Path

from .knowledge import reader, text

DEFINITION = "balance-ja-en-1"


def artifacts_of(chosen):
    """The names of the artifacts that a definition of a balance measures by: its sentences."""
    return [chosen["flores"]["text"]]


def measure(model, tokenizer, device, precision, artifacts, batch_tokens=16384, chosen=DEFINITION):
    """A model's balance by a definition (its name, or one already loaded), with its standard error, each language's
    bits, bytes and bits per byte, and each sentence's bits (`items`), from the definition's artifacts in the directory
    `artifacts`."""
    import olmes_measure
    from language_model_evaluation import definition

    chosen = definition.load(chosen) if isinstance(chosen, str) else chosen
    sentences = Path(artifacts) / chosen["flores"]["text"]
    olmes_measure.check(sentences)
    model.eval()
    read, encode, bos = reader(model, tokenizer, device, precision, batch_tokens)
    documents = text(read, encode, bos, sentences, model.config.context)
    bits, languages = {}, {}
    for language in chosen["balance"]:
        found = documents[f"flores/{language}"]  # each sentence's bits and bytes
        bits[language] = [each for each, _ in found]
        size = sum(each for _, each in found)
        languages[language] = {"bits": sum(bits[language]), "bytes": size, "bits_per_byte": sum(bits[language]) / size}
    return {"definition": chosen["name"], "artifacts": artifacts_of(chosen), **definition.balance(chosen, bits),
            "languages": languages, "items": {"sentences": bits}}
