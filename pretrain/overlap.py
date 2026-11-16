"""The training texts that overlap the evaluation that judges a generation (docs/evaluation.md), which nothing that
learns may read: the documents of the corpora and of the tokenizer's sample, and the conversations of the instruction
and preference data.

A text overlaps when it holds a question or a stretch of a text of knowledge-ja-en-1, the whole that the part that
judges is drawn from, in either language, as its overlap artifacts tell them, or a prompt of instructions-ja-en-1,
found by the middle of its letters as a question is. The artifacts are kept by their names in one directory, as the
knowledge measure reads its own, and the prompts are the authors', from their code fetched beside them.
"""
from pathlib import Path

# Where the evaluation's artifacts are kept by their names.
EVALUATION = Path("data/evaluation")


def artifacts():
    """The names of the knowledge definition's overlap artifacts: of its English questions and text, and of its
    Japanese ones."""
    from language_model_evaluation import definition

    from .knowledge import DEFINITION

    found = definition.load(DEFINITION)["overlap"]
    return [found["artifact"], found["japanese"]["artifact"]]


class EvaluationOverlap:
    """What tells a text that overlaps the evaluation, from the artifacts in `directory` and the authors' code, which
    is fetched to directory/instructions the first time. `made_from` names what it tells by."""

    def __init__(self, directory=EVALUATION):
        from language_model_evaluation import definition, instructions, overlap

        from .following import DEFINITION

        directory, names = Path(directory), artifacts()
        chosen = definition.load(DEFINITION)
        if chosen["instructions"]["commit"] != instructions.COMMIT:
            raise ValueError(f"{chosen['name']} asks the prompts of the authors' code at another commit")
        home = instructions.code(directory / "instructions")
        keys = {key for category in chosen["categories"].values() for language in category["instructions"]
                for key in overlap.keys_of([row["prompt"] for row in instructions.prompts(language, home)], language)}
        self.texts = [overlap.load(directory / name) for name in names]
        self.prompts = overlap.Overlap(sorted(keys), b"")
        self.made_from = {"overlap": names, "instructions": instructions.COMMIT}

    def holds(self, text):
        """Whether a text overlaps the evaluation."""
        from language_model_evaluation import overlap

        letters = overlap.normalized(text)
        return self.prompts.holds_question(text, letters) or any(
            found.holds_question(text, letters) or found.holds_paragraph(text) or found.holds_stretch(text, letters)
            for found in self.texts)
