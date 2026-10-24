"""Serialize the conversation format used for training and generation.

Reserved tokens of the tokenizer mark where each turn starts and ends, the
assistant's reasoning and its tool calls, as ChatML-style formats do. Text
that spells a reserved token loses that spelling, so no speaker can forge
the structure. Every assistant turn opens a reasoning block, closed at once
when it answers directly, so the prompt chooses whether a reply reasons
first.
"""

import json
import re

TURN_START, TURN_END, THINK_START, THINK_END, CALL_START, CALL_END = (
    f"<|reserved_{index}|>" for index in range(6)
)
ROLES = ("system", "user", "assistant", "tool")
RESERVED = re.compile(r"<\|reserved_\d+\|>")


def plain(text):
    """The text without any spelling of a reserved token."""
    return RESERVED.sub("", text)


def opening(role, reasoning=False):
    """What starts a turn; an assistant's also opens its reasoning, closed unless it reasons."""
    text = TURN_START + role + "\n"
    if role == "assistant":
        text += THINK_START + ("" if reasoning else THINK_END)
    return text


def conversation_text(messages):
    """Join turns and locate each assistant reply, through its turn's end, in the transcript.

    An assistant message may carry its reasoning and the tool calls it ends
    with, each a name and its arguments; a tool turn returns a call's result.
    """
    parts, replies, length = [], [], 0
    for message in messages:
        reasoning = message.get("reasoning")
        head = opening(message["role"], bool(reasoning))
        body = plain(message["content"])
        if message["role"] == "assistant":
            if reasoning:
                body = plain(reasoning) + THINK_END + body
            body += "".join(
                CALL_START + plain(json.dumps(call, ensure_ascii=False)) + CALL_END
                for call in message.get("tool_calls", ())
            )
            replies.append((length + len(head), length + len(head) + len(body) + len(TURN_END)))
        piece = head + body + TURN_END
        parts.append(piece)
        length += len(piece)
    return "".join(parts), replies


def chat_prompt(messages, reasoning=False):
    """Return the transcript that asks for the next assistant reply, reasoning first if asked."""
    text, _ = conversation_text(messages)
    return text + opening("assistant", reasoning)


def require_markers(tokenizer):
    """Refuse a tokenizer without the reserved tokens that mark the conversation."""
    if any(tokenizer.token_to_id(marker) is None for marker in (TURN_START, TURN_END, THINK_START, THINK_END, CALL_START, CALL_END)):
        raise ValueError("the conversation format needs a tokenizer with at least six reserved tokens")


def reply_ends(tokenizer):
    """The token IDs that end a reply: the turn's end, and the end of the text."""
    return {token for token in (tokenizer.token_to_id(TURN_END), tokenizer.token_to_id("<|eos|>")) if token is not None}
