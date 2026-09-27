from __future__ import annotations

import pytest

from app.profanity import ProfanityFilter, normalize, tokenize

SHOULD_FLAG = [
    "fuck", "fucking", "fucked", "FUCK", "fUcK", "fuuuuck", "sh1t", "shit", "shitty",
    "bitches", "ass", "asses", "asshole", "bastard", "damn", "goddamn", "wtf", "a$$hole",
    "fuck off", "motherfucking", "c0ck", "nigger", "retard", "dick", "dickhead",
    "jesus", "5h1t", "f4ggot", "dumbass", "stfu", "bollocks", "twat", "blowjob",
]

SHOULD_PASS = [
    "class", "bass", "pass", "grass", "assassin", "assist", "assign", "assess", "assume",
    "asset", "hello", "shell", "shoes", "sultan", "analysis", "analyst", "cocktail",
    "peacock", "dickens", "soccer", "cumulative", "document", "embarrass", "mass",
    "brass", "christmas", "christian", "guitar", "spectacular", "classic", "crass",
    "coarse", "penistone", "association", "assault", "spectre", "bassoon",
]


@pytest.fixture(scope="module")
def words() -> ProfanityFilter:
    return ProfanityFilter.load()


@pytest.mark.parametrize("text", SHOULD_FLAG)
def test_flags_profanity(words: ProfanityFilter, text: str) -> None:
    assert words.is_bad(text), f"expected {text!r} to be flagged"


@pytest.mark.parametrize("text", SHOULD_PASS)
def test_ignores_benign_words(words: ProfanityFilter, text: str) -> None:
    assert not words.is_bad(text), f"expected {text!r} to be ignored"


CENSORED = [
    "f***ing", "f**k", "f***ck", "f*ck", "s**t", "sh*t", "b*tch", "a$$", "c**k",
    "f***ing god", "what the f***ing", "fuuuuck", "F***ING", "s***", "sh*tty",
    "F_ck you!", "f_ck", "f_ucks", "f*ucks", "sh_t", "b_tch", "a**hole", "f_ck you",
]

CENSOR_SAFE = [
    "f**king my life", "1*2*3", "2 * 3 * 4", "a * b", "star**light",
    "C# programming", "let a*=2", "x = y", "c*0de", "if (a**b)",
]


@pytest.mark.parametrize("text", CENSORED)
def test_flags_censored_profanity(words: ProfanityFilter, text: str) -> None:
    assert words.is_bad(text), f"expected censored {text!r} to be flagged"


@pytest.mark.parametrize("text", CENSOR_SAFE)
def test_censor_patterns_do_not_catch_arithmetic(words: ProfanityFilter, text: str) -> None:
    assert not words.is_bad(text), f"expected {text!r} to be ignored"


def test_normalize_keeps_a_censor_marker(words: ProfanityFilter) -> None:
    assert normalize("oh my f***ing") == "oh my f*ing"
    assert normalize("f***ck") == "f*ck"


def test_normalize_folds_leetspeak_and_punctuation() -> None:
    assert normalize("F*U-C-K!") == "f*u c k"
    assert normalize("sh1t") == "shit"
    assert tokenize("Don't be a dick!") == ["don't", "be", "a", "dick"]


def test_finds_spaced_letters(words: ProfanityFilter) -> None:
    assert words.find("what the f u c k") == ["f u c k"]


def test_finds_inside_sentence(words: ProfanityFilter) -> None:
    found = words.find("this class is a masterpiece, absolutely brilliant")
    assert found == []


def test_scan_returns_token_spans() -> None:
    words = ProfanityFilter(["fuck", "dick"])
    matches = words.scan(tokenize("you are such a dick and a fuck"))
    assert [m.text for m in matches] == ["dick", "fuck"]
    assert matches[0].first == 4 and matches[1].first == 7


def test_allow_list_wins() -> None:
    custom = ProfanityFilter(["duck"], allow=["duck"])
    assert not custom.is_bad("duck")
    assert custom.is_bad("duuuuck")


def test_allow_list_blocks_a_stem_entirely() -> None:
    custom = ProfanityFilter(["duck"], allow=["ducks"])
    assert not custom.is_bad("ducks")
    assert custom.is_bad("duck")


def test_empty_filter_matches_nothing() -> None:
    empty = ProfanityFilter([])
    assert not empty.is_bad("fuck this")
    assert empty.find("fuck this") == []
