"""Canonical formatting for character-facing model output."""
import re


_PROTOCOL_PREFIXES = ("<|assistant|>", "assistant:", "assistant：", "助手:", "助手：")
_QUOTE_PAIRS = {"“": "”", '"': '"', "「": "」", "『": "』"}
_CLOSING_QUOTES = set(_QUOTE_PAIRS.values())
_ENGLISH_RUN = re.compile(r"[A-Za-z]+(?:[\s,.\-!?]+[A-Za-z]+){2,}")
_ROLE_LABEL = re.compile(r'^(?:妹妹|姐姐|哥哥|角色|助手|assistant)[：:]\s*', re.IGNORECASE)



# === 日志补接线（详细排障） ===
try:
    from app.logger import debug, info, success, warning, error, trace
except Exception:
    debug = info = success = warning = error = trace = lambda *a, **k: None

def _strip_protocol(text: str) -> str:
    value = re.sub(r"```(?:text|markdown)?\s*|```", "", text or "", flags=re.IGNORECASE).strip()
    value = re.sub(r"^(?:<\|im_start\|>|<\|assistant\|>|<\|endoftext\|>)+", "", value).strip()
    value = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)+$", "", value).strip()
    changed = True
    while changed:
        changed = False
        lowered = value.lower()
        for prefix in _PROTOCOL_PREFIXES:
            if lowered.startswith(prefix.lower()):
                value = value[len(prefix):].lstrip()
                changed = True
                break
    return value


def _strip_speaker_label(text: str) -> str:
    """Remove an explicit model-added speaker label, preserving dialogue colons."""
    return _ROLE_LABEL.sub("", text.strip(), count=1).strip()


def _collapse_parentheses(text: str) -> str:
    """Convert brackets and repair only mechanical bracket errors."""
    value = text.replace("(", "（").replace(")", "）")
    value = re.sub(r"（+", "（", value)
    value = re.sub(r"）+", "）", value)
    # Remove empty and directly nested pairs without changing prose.
    previous = None
    while value != previous:
        previous = value
        value = re.sub(r"（\s*）", "", value)
        value = re.sub(r"（\s*（", "（", value)
        value = re.sub(r"）\s*）", "）", value)
    # A lone opening/closing bracket is narration punctuation, not a pair.
    if value.count("（") > value.count("）"):
        value += "）"
    elif value.count("）") > value.count("（"):
        value = "（" + value
    return value.strip()


def _parenthesize_narration(text: str) -> str:
    value = text.strip()
    if not value:
        return ""
    value = _collapse_parentheses(value)
    if value.startswith("（") and value.endswith("）"):
        return value
    return f"（{value}）"


def _split_quoted_output(value: str) -> str:
    """Turn quoted speech into speech and leave surrounding narration parenthesized.

    This is deliberately a small lexer rather than a prose rewriter: unmatched quotes
    are treated as ordinary text, so formatting cannot invent or remove character speech.
    """
    parts = []
    cursor = 0
    index = 0
    while index < len(value):
        opener = value[index]
        if opener not in _QUOTE_PAIRS:
            index += 1
            continue
        closer = _QUOTE_PAIRS[opener]
        end = value.find(closer, index + 1)
        if end < 0:
            index += 1
            continue
        before = value[cursor:index]
        speech = value[index + 1:end].strip()
        if before.strip():
            parts.append(_parenthesize_narration(before))
        if speech:
            parts.append(speech)
        cursor = end + 1
        index = cursor
    if not parts:
        return value
    after = value[cursor:]
    if after.strip():
        parts.append(_parenthesize_narration(after))
    return "".join(parts).strip()


def _strip_english_sentences(text: str) -> str:
    return _ENGLISH_RUN.sub("", text or "").strip()


def normalize_character_output(text: str) -> str:
    """Return an idempotent character output with dialogue/narration separation."""
    debug("后处理", f"输出规范化开始: 长度={len(text)}字")
    value = _strip_english_sentences(_strip_speaker_label(_strip_protocol(text)))
    if not value:
        return ""
    value = _split_quoted_output(value)
    # A second mechanical pass makes mixed half/full-width input idempotent.
    value = _collapse_parentheses(value)
    return value.strip()


__all__ = ["normalize_character_output"]
