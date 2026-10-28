"""M-IFEval-Ja's instructions are checked as M-IFEval's evaluation_main.py checks them.

The checkers are M-IFEval's own (lightblue-tech/M-IFEval, Apache-2.0), copied
beside this file; the instruction IDs and the strict and loose responses
follow its registry and evaluation_main.py at the same commit.
"""

import sys
from pathlib import Path

# The copied checkers import their utility module by its file's name.
sys.path.insert(0, str(Path(__file__).parent))
import m_ifeval_ja_instructions as ja  # noqa: E402

INSTRUCTIONS = {
    "ja:keywords:existence": ja.KeywordChecker,
    "ja:keywords:frequency": ja.KeywordFrequencyChecker,
    "ja:keywords:forbidden_words": ja.ForbiddenWords,
    "ja:keywords:letter_frequency": ja.LetterFrequencyChecker,
    "ja:language:response_language": ja.ResponseLanguageChecker,
    "ja:length_constraints:number_sentences": ja.NumberOfSentences,
    "ja:length_constraints:number_paragraphs": ja.ParagraphChecker,
    "ja:length_constraints:number_letters": ja.NumberOfLetters,
    "ja:length_constraints:nth_paragraph_first_word": ja.ParagraphFirstWordCheck,
    "ja:detectable_content:number_placeholders": ja.PlaceholderChecker,
    "ja:detectable_content:postscript": ja.PostscriptChecker,
    "ja:detectable_format:number_bullet_lists": ja.BulletListChecker,
    "ja:detectable_format:number_numbered_lists": ja.NumberedListChecker,
    "ja:detectable_format:constrained_response": ja.ConstrainedResponseChecker,
    "ja:detectable_format:number_highlighted_sections": ja.HighlightSectionChecker,
    "ja:detectable_format:multiple_sections": ja.SectionChecker,
    "ja:detectable_format:json_format": ja.JsonFormat,
    "ja:detectable_format:title": ja.TitleChecker,
    "ja:combination:two_responses": ja.TwoResponsesChecker,
    "ja:combination:repeat_prompt": ja.RepeatPromptThenAnswer,
    "ja:startend:end_checker": ja.EndChecker,
    "ja:startend:sentence_unified_end": ja.SentenceEndingUnification,
    "ja:detectable_format:nominal_ending": ja.NominalEndingChecker,
    "ja:letters:furigana": ja.FuriganaForKanji,
    "ja:letters:kansuuji": ja.KanjiNumberNotationChecker,
    "ja:letters:no_katakana": ja.NoKatakana,
    "ja:letters:katakana_only": ja.KatakanaOnly,
    "ja:letters:no_hiragana": ja.NoHiragana,
    "ja:letters:hiragana_only": ja.HiraganaOnly,
    "ja:letters:kanji": ja.KanjiLimit,
    "ja:punctuation:no_comma": ja.CommaChecker,
    "ja:punctuation:no_period": ja.PeriodChecker,
    "ja:startend:quotation": ja.QuotationChecker,
}


def loose_responses(response):
    """The response with its first or last line, or its asterisks, removed, as evaluation_main.py relaxes it."""
    lines = response.split("\n")
    without_first = "\n".join(lines[1:]).strip()
    without_last = "\n".join(lines[:-1]).strip()
    without_both = "\n".join(lines[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        without_first,
        without_last,
        without_both,
        without_first.replace("*", ""),
        without_last.replace("*", ""),
        without_both.replace("*", ""),
    ]


def followed(doc, responses):
    """Whether one of the responses follows each of the document's instructions."""
    results = []
    for index, name in enumerate(doc["instruction_id_list"]):
        instruction = INSTRUCTIONS[name](name)
        # The dataset's loader gives every document every argument's key,
        # filling the ones its instructions lack with None.
        instruction.build_description(**{key: value for key, value in doc["kwargs"][index].items() if value is not None})
        arguments = instruction.get_instruction_args()
        if arguments and "prompt" in arguments:
            instruction.build_description(prompt=doc["prompt"])
        results.append(any(response.strip() and instruction.check_following(response) for response in responses))
    return results


def process_results(doc, results):
    strict = followed(doc, [results[0]])
    loose = followed(doc, loose_responses(results[0]))
    return {
        "prompt_level_strict_acc": all(strict),
        "inst_level_strict_acc": strict,
        "prompt_level_loose_acc": all(loose),
        "inst_level_loose_acc": loose,
    }


def agg_inst_level_acc(items):
    flat = [item for sublist in items for item in sublist]
    return sum(flat) / len(flat)
