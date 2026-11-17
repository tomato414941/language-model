"""The training texts that overlap the evaluation (docs/evaluation.md), which nothing that learns may read: the
documents of the corpora and of the tokenizer's sample, and the conversations of the instruction and preference data.

A text overlaps when it holds a question or a stretch of a text of knowledge-ja-en-1, the whole that the part that
judges is drawn from, or a sentence of either split of FLORES+, which balance-ja-en-1 reads, in either language, as
the definitions' overlap artifacts tell them, or a prompt of instructions-ja-en-1, found by the middle of its letters
as a question is. The artifacts are kept by their names in one directory, as the measures read their own, and the
prompts are the authors', fetched with their code beside them.
"""
from pathlib import Path

# Where the evaluation's artifacts are kept by their names.
EVALUATION = Path("data/evaluation")


def artifacts():
    """The names of the overlap artifacts: the knowledge definition's, of its English questions and text and of its
    Japanese ones, and the balance definition's, of FLORES+'s sentences."""
    from language_model_evaluation import definition

    from . import balance, knowledge

    found, sentences = (definition.load(measure.DEFINITION)["overlap"] for measure in (knowledge, balance))
    return [found["artifact"], found["japanese"]["artifact"], sentences["artifact"]]


class EvaluationOverlap:
    """What tells a text that overlaps the evaluation, from the artifacts in `directory` and the authors' prompts,
    which are fetched with their code to directory/instructions the first time. `made_from` names what it tells by."""

    def __init__(self, directory=EVALUATION):
        from language_model_evaluation import definition, instructions, overlap

        from .following import DEFINITION

        directory, names = Path(directory), artifacts()
        chosen = definition.load(DEFINITION)
        asked = chosen["instructions"]
        if (asked["commit"], asked["revision"]) != (instructions.COMMIT, instructions.REVISION):
            raise ValueError(f"{chosen['name']} asks the authors' prompts of another version")
        home = instructions.code(directory / "instructions")
        keys = {key for category in chosen["categories"].values() for language in category["instructions"]
                for key in overlap.keys_of([row["prompt"] for row in instructions.prompts(language, home)], language)}
        self.texts = [overlap.load(directory / name) for name in names]
        self.prompts = overlap.Overlap(sorted(keys), b"")
        self.made_from = {"overlap": names, "instructions": instructions.REVISION}

    def holds(self, text):
        """Whether a text overlaps the evaluation."""
        from language_model_evaluation import overlap

        letters = overlap.normalized(text)
        return self.prompts.holds_question(text, letters) or any(
            found.holds_question(text, letters) or found.holds_paragraph(text) or found.holds_stretch(text, letters)
            for found in self.texts)
