from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import WORDLIST_PATH

_LEET = str.maketrans(
    {
        "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "6": "g",
        "7": "t", "8": "b", "9": "g", "@": "a", "$": "s", "+": "u",
        "(": "c", ")": "c", "<": "c", ">": "c",
    }
)

_TAILS = (
    "s", "es", "ed", "d", "er", "ers", "est", "ing", "in", "yn", "y", "ier",
    "iest", "ish", "hole", "holes", "head", "heads", "face", "faces", "bag",
    "bags", "off", "out", "a", "z", "tard", "face",
)

_CENSOR = re.compile(r"\*+|_|[*_^#~=]{2,}")
_MASKS = re.compile(r"[*_]+")
_PUNCT = re.compile(r"[^a-z0-9'*]+")
_SPACES = re.compile(r"\s+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).casefold()
    text = text.replace("’", "'").replace("ʼ", "'")
    text = text.translate(_LEET)
    text = _CENSOR.sub("*", text)
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in _WS.split(normalize(text)) if token]


def _alternation(tails) -> str:
    return "|".join(
        re.escape(tail) for tail in sorted((t for t in tails if t), key=len, reverse=True)
    )


def _stem_pattern(stem: str) -> str:
    core = "".join(re.escape(character) + "+" for character in stem)
    tails = set(_TAILS)
    last = stem[-1]
    if last not in "aeiou":
        tails.update(last + tail for tail in ("ing", "ed", "er", "ers", "in"))
    alternation = _alternation(tails)
    body = core + (f"(?:{alternation})?" if alternation else "")
    return f"(?<![a-z]){body}(?![a-z])"


def _censor_pattern(stem: str) -> str:
    if len(stem) < 2:
        return ""
    body = stem[1:]
    pool = {body[i:] for i in range(len(body))} | set(_TAILS)
    parts: list[str] = []
    for prefix_length in (0, 1, 2):
        allowed = {tail for tail in pool if len(tail) >= 2 - prefix_length}
        if not allowed:
            continue
        lead = re.escape(stem[:prefix_length])
        parts.append(
            f"(?<![a-z]){lead}\\*+(?:{_alternation(allowed)})(?![a-z])"
        )
    if stem[0] not in "aeiou":
        ends = "|".join(re.escape(stem[:i]) for i in (1, 2))
        parts.append(f"(?<![a-z])(?:{ends})\\*+(?![a-z*])")
    return "|".join(parts)


@dataclass(frozen=True)
class Match:
    first: int
    last: int
    text: str


def _clean(words: Sequence[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for word in words:
        normalized = " ".join(tokenize(word))
        if normalized:
            cleaned.append(normalized)
    return tuple(dict.fromkeys(cleaned))


def _too_short_to_censor(token: str) -> bool:
    return len(token) < 4 and ("*" in token or "_" in token)


class ProfanityFilter:
    def __init__(
        self,
        stems: Sequence[str],
        extra: Sequence[str] = (),
        allow: Sequence[str] = (),
    ) -> None:
        self.stems = _clean(stems)
        self.extra = _clean(extra)
        self.allow = frozenset(
            [token for word in allow for token in tokenize(word)]
            + [" ".join(tokenize(word)) for word in allow]
        )
        patterns = [_stem_pattern(stem) for stem in self.stems]
        patterns += [
            pattern for pattern in (_censor_pattern(stem) for stem in self.stems) if pattern
        ]
        patterns += [re.escape(phrase) for phrase in self.extra]
        self._pattern = re.compile("|".join(patterns)) if patterns else None
        self._single = re.compile(r"\b[a-z]\b")

    @classmethod
    def load(cls, path: Path | None = None) -> "ProfanityFilter":
        target = path or WORDLIST_PATH
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        return cls(
            data.get("stems", ()),
            data.get("extra", ()),
            data.get("allow", ()),
        )

    def is_bad(self, text: str) -> bool:
        normalized = normalize(text)
        if not normalized or normalized in self.allow or self._pattern is None:
            return False
        if self._pattern.search(normalized):
            return True
        if "*" in normalized or "_" in normalized:
            stripped = _MASKS.sub("", normalized)
            if stripped and stripped != normalized and stripped not in self.allow:
                return self._pattern.search(stripped) is not None
        return False

    def scan(self, tokens: Sequence[str]) -> list[Match]:
        matches: list[Match] = []
        index = 0
        total = len(tokens)
        while index < total:
            run_end = index
            while run_end < total and self._single.fullmatch(tokens[run_end]):
                run_end += 1
            if run_end - index >= 3:
                joined = "".join(tokens[index:run_end])
                if self.is_bad(joined):
                    matches.append(Match(index, run_end - 1, joined))
                    index = run_end
                    continue
            if self.is_bad(tokens[index]) and not _too_short_to_censor(tokens[index]):
                last = index
                probe = index + 1
                while probe < total and self._single.fullmatch(tokens[probe]):
                    if not self.is_bad("".join(tokens[index : probe + 1])):
                        break
                    last = probe
                    probe += 1
                matches.append(Match(index, last, " ".join(tokens[index : last + 1])))
            index += 1
        return matches

    def find(self, text: str) -> list[str]:
        tokens = tokenize(text)
        return [" ".join(tokens[m.first : m.last + 1]) for m in self.scan(tokens)]
