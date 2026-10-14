"""Serialize the conversation format used for training and generation."""

ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


def conversation_text(messages):
    """Join turns and locate each assistant reply in the transcript."""
    parts, replies, length = [], [], 0
    for index, message in enumerate(messages):
        label = ("\n\n" if index else "") + ROLE_LABELS[message["role"]] + ":"
        content = " " + message["content"]
        if message["role"] == "assistant":
            replies.append((length + len(label), length + len(label) + len(content)))
        parts += [label, content]
        length += len(label) + len(content)
    return "".join(parts), replies


def chat_prompt(messages):
    """Return the transcript that asks the model for the next assistant reply."""
    text, _ = conversation_text(messages)
    return (text + "\n\n" if text else "") + "Assistant:"
