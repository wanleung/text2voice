"""Article text preparation: strip markup, split into paragraphs, detect language."""
import re
from dataclasses import dataclass

LANGUAGES = ("en", "yue", "zh")

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002ffff]")
_CJK_PUNCT = re.compile(r"[\u3000-\u303f\uff00-\uffef]")  # 【】《》，etc.
_LATIN_WORD = re.compile(r"[A-Za-z]+")
_END_PUNCT = re.compile(r"[.!?。！？…:：;；\"'”’」』】》)）]$")


@dataclass
class Segment:
    text: str
    language: str
    paragraph_end: bool = True  # insert a pause after this segment


def clean_markup(text: str) -> str:
    """Turn Markdown/HTML-ish article text into plain prose suitable for reading aloud."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```.*?```", "\n", text, flags=re.S)  # fenced code blocks
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>|</p>|</h\d>|</li>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>\n]+>", "", text)  # remaining HTML tags
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links -> link text
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # bare URLs
    text = re.sub(r"`([^`]*)`", r"\1", text)  # inline code
    # Line-anchored patterns use [ \t], not \s, so they never swallow the blank lines between blocks.
    text = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]*", "", text, flags=re.M)  # headings
    text = re.sub(r"^[ \t]*>[ \t]?", "", text, flags=re.M)  # blockquotes
    text = re.sub(r"^[ \t]*([-*+]|\d+[.)])[ \t]+", "", text, flags=re.M)  # list markers
    # Horizontal rules, Setext heading underlines and other divider lines (==========, ～～～, ───).
    text = re.sub(r"^[ \t]*([-*_=~#＝～─━—–·•][ \t]*){3,}$", "", text, flags=re.M)
    text = re.sub(r"^[ \t]*\|?([ \t]*:?-+:?[ \t]*\|)+[ \t]*$", "", text, flags=re.M)  # table separators
    text = text.replace("|", " ")
    text = re.sub(r"(\*\*|__|\*|~~)(\S.*?\S|\S)\1", r"\2", text)  # emphasis
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
            .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'"))
    return text


def detect_language(text: str, chinese_default: str = "yue") -> str:
    cjk = len(_CJK.findall(text))
    latin_words = len(_LATIN_WORD.findall(text))
    # One CJK character carries roughly as much speech as one English word.
    return chinese_default if cjk > 0 and cjk >= latin_words else "en"


def _cjk_join(left: str, right: str) -> bool:
    """Chinese text is joined without a space."""
    return bool((_CJK.search(left[-2:]) or _CJK_PUNCT.search(left[-1:])) and _CJK.search(right[:1]))


def split_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in block.split("\n")]
        # A line with nothing to say (a stray "】" or "..." after a line break) joins the line before.
        merged: list[str] = []
        for ln in lines:
            if re.search(r"\w", ln):
                merged.append(ln)
            elif ln and merged:
                merged[-1] += ln
        lines = merged
        if not lines:
            continue
        # Lines without terminal punctuation (headings, list items) get a pause.
        parts = []
        for ln in lines:
            if not _END_PUNCT.search(ln):
                ln += "。" if _CJK.search(ln[-1:]) or _CJK_PUNCT.search(ln[-1:]) else "."
            parts.append(ln)
        para = parts[0]
        for prev, ln in zip(parts, parts[1:]):
            para += ("" if _cjk_join(prev, ln) else " ") + ln
        paragraphs.append(para)
    return [p for p in paragraphs if re.search(r"\w", p)]


# Sentence ends: after CJK terminators, or after Latin ones (optionally closing a quote) that are
# followed by whitespace and a capital, digit, quote or CJK character. Over-splitting is harmless:
# prepare() re-joins adjacent sentences of the same language.
_SENTENCE_END = re.compile(
    r"(?<=[。！？])(?![。！？”」』】》\"'）)])"
    r"|(?<=[。！？][”」』】》）)])"
    r"|(?<=[.!?])\s+(?=[A-Z0-9\"“‘'(\u3400-\u9fff])"
    r"|(?<=[.!?][\"'”’)])\s+(?=[A-Z0-9\"“‘'(\u3400-\u9fff])"
)


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip()]


def prepare(text: str, language: str = "auto", chinese_default: str = "yue") -> list[Segment]:
    """Split an article into segments, each tagged with the language to speak it in.

    With language="auto", detection runs per sentence and consecutive sentences in the same
    language are merged, so a Cantonese article quoting an English sentence switches correctly.
    """
    segments = []
    for para in split_paragraphs(clean_markup(text)):
        if language != "auto":
            segments.append(Segment(para, language))
            continue
        runs: list[Segment] = []
        for sent in split_sentences(para):
            # A sentence with no words (a stray bracket) stays with the one before it.
            lang = detect_language(sent, chinese_default) if re.search(r"\w", sent) or not runs else runs[-1].language
            if runs and runs[-1].language == lang:
                runs[-1].text += ("" if _cjk_join(runs[-1].text, sent) else " ") + sent
            else:
                runs.append(Segment(sent, lang, paragraph_end=False))
        runs[-1].paragraph_end = True
        segments.extend(runs)
    return segments
