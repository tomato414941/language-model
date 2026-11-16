"""Compare two results by a definition of language-model-evaluation (pretrain.knowledge, pretrain.following) by the
definition's rule. A difference is read when it is beyond READABLE_ERRORS of its standard errors, which come from the
questions, documents and prompts drawn again.
"""

# Of two models that differ in nothing, about one comparison in twenty shows a
# difference beyond this many standard errors.
READABLE_ERRORS = 2


def compare_results(base, other):
    """How one model's result by a definition of language-model-evaluation compares with another's (`base`): the
    definition's improvement and each category's rate, positive when `other` is the better, each with its standard
    error and whether it is readable, beyond READABLE_ERRORS of them (None for a category whose score does not vary
    with the items measured). `base` may be a result's chance, {"definition", "categories", "items"}."""
    from language_model_evaluation import definition

    if base["definition"] != other["definition"]:
        raise ValueError("results of different definitions do not compare")
    chosen = definition.load(other["definition"])
    found = definition.compared(chosen, other["items"], base["items"], other.get("groups"))
    readable = lambda value, error: None if error is None else abs(value) > READABLE_ERRORS * error
    return {
        "definition": chosen["name"],
        "improvement": found["improvement"],
        "standard_error": found["standard_error"],
        "readable": readable(found["improvement"], found["standard_error"]),
        "categories": {
            category: {
                "base": base["categories"][category],
                "other": other["categories"][category],
                **compared,
                "readable": readable(compared["rate"], compared["standard_error"]),
            }
            for category, compared in found["categories"].items()
        },
    }
