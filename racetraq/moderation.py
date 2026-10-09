"""Leaderboard name moderation: a small blocklist with leetspeak folding.

A public booth board shows whatever a visitor types. ``name_allowed`` catches
the obvious (the packaged ``data/name_blocklist.json`` plus a booth's own
``[leaderboard] blocklist``); anything it misses, an operator removes from
the board (``board`` message, booth machine only).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

BLOCKLIST_PATH = Path(__file__).resolve().parent / "data" / "name_blocklist.json"

_LEET = str.maketrans({"0": "o", "1": "i", "!": "i", "3": "e", "4": "a", "@": "a",
                       "5": "s", "$": "s", "7": "t", "8": "b", "9": "g"})
# plain endings a blocked word may carry ("shits", "fucker", "shitty")
SUFFIXES = ("", "s", "es", "er", "ers", "y", "ing", "ed")


def _fold(text: str, leet: bool = True) -> str:
    """Lowercase, umlauts to their two-letter spelling, and (``leet``) undo
    letter swaps."""
    text = text.lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(a, b)
    return text.translate(_LEET) if leet else text


@lru_cache(maxsize=1)
def packaged_words() -> tuple[str, ...]:
    try:
        data = json.loads(BLOCKLIST_PATH.read_text(encoding="utf-8"))
        return tuple(str(w) for w in data.get("words", []))
    except (OSError, ValueError):
        return ()


def name_allowed(name: str, extra_words: tuple[str, ...] | list[str] = ()) -> bool:
    """False when ``name`` contains a blocked word as a whole word (with a
    plain ending), or spells one with dots and spaces in between ("f.u c k").
    Whole words only, so real names stay allowed (Scunthorpe, Dickens,
    Cassie); what slips through, an operator removes from the board."""
    folded = _fold(name)
    # words with the swaps undone ("sh1t") and with digits as separators
    # ("Fotze123"), so both spellings are checked
    tokens = set(re.findall(r"[a-z]+", folded)) | set(re.findall(r"[a-z]+", _fold(name, False)))
    squashed = re.sub(r"[^a-z]", "", folded)
    for word in (*packaged_words(), *extra_words):
        w = re.sub(r"[^a-z]", "", _fold(str(word)))
        if not w:
            continue
        stems = (w, w + w[-1])  # "shit" -> "shitty"
        if squashed == w or any(stem + end in tokens for stem in stems for end in SUFFIXES):
            return False
    return True
