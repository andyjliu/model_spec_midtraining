"""Helpers for separating serialized model reasoning from final answers."""

import re


_THINK_BLOCK_RE = re.compile(
    r"<think\s*>.*?</think\s*>\s*", re.DOTALL | re.IGNORECASE
)
_STANDALONE_THINK_OPEN_RE = re.compile(
    r"(?im)^[ \t]*<think\s*>[ \t]*(?:\r?\n|$)"
)
_STANDALONE_THINK_CLOSE_RE = re.compile(
    r"(?im)^[ \t]*</think\s*>[ \t]*(?:\r?\n|$)"
)


def strip_think_content(text: str) -> str:
    """Remove serialized reasoning while preserving the final answer.

    Some Qwen serving templates begin reasoning outside the returned text, so
    the completion contains the reasoning body and ``</think>`` but no opening
    tag. In that case, the last standalone closing tag is the reliable boundary
    between reasoning and the final response.

    Inline literal discussion of ``<think>`` syntax is left alone; unmatched
    tags are treated as serialization markers only when they occupy a line.
    """
    stripped = _THINK_BLOCK_RE.sub("", text)
    unmatched_closes = list(_STANDALONE_THINK_CLOSE_RE.finditer(stripped))
    if unmatched_closes:
        stripped = stripped[unmatched_closes[-1].end():]
    stripped = _STANDALONE_THINK_OPEN_RE.sub("", stripped)
    stripped = _STANDALONE_THINK_CLOSE_RE.sub("", stripped)
    return stripped.strip()
