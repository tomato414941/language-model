"""What several tests build: an evaluation directory that tells the text that overlaps the evaluation, as
pretrain.overlap reads one."""
import json
from pathlib import Path
from unittest.mock import patch


def evaluation(test, directory, questions=None, paragraphs=(), prompts=None):
    """`directory`, made to hold what tells the text that overlaps the evaluation for the rest of the test: an overlap
    artifact of these questions ({language: [question]}) and paragraphs, and of the paragraphs' stretches, in place of
    the knowledge definition's two, and the authors' code holding these prompts ({language: [prompt]}) in place of
    theirs."""
    from language_model_evaluation import overlap

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for language, asked in (questions or {}).items() for key in overlap.keys_of(asked, language)})
    artifact = overlap.write(directory, keys, overlap.digests_of(paragraphs), overlap.stretches_of(paragraphs))
    home = directory / "instructions" / "code"
    (home / "data").mkdir(parents=True, exist_ok=True)
    (home / "tasks" / "ifeval").mkdir(parents=True, exist_ok=True)
    (home / "tasks" / "ifeval" / "utils.py").write_text("")  # the code is there, so it is not fetched
    for language in ("en", "ja"):
        rows = [{"key": key, "prompt": prompt, "instruction_id_list": [], "kwargs": []}
                for key, prompt in enumerate((prompts or {}).get(language, []))]
        (home / "data" / f"ifeval_{language}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    patcher = patch("pretrain.overlap.artifacts", return_value=[artifact.name])
    patcher.start()
    test.addCleanup(patcher.stop)
    return directory
