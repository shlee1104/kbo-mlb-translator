"""Korean player-name normalisation and matching keys.

This is the module that makes a KBO/MLB crosswalk possible at all. The same
player appears across sources as any of:

    이정후  /  Lee Jung-hoo  /  Jung Hoo Lee  /  Jung-hoo Lee  /  Lee Jeong-hu

Three separate problems are stacked on top of each other:

1. **Script.** Korean sources write Hangul; English sources romanise.
2. **Name order.** Korean order is surname-first; MLB sources usually flip it,
   but not always, and not consistently within a single source.
3. **Romanisation system.** South Korea switched from McCune-Reischauer to
   Revised Romanization in 2000, players choose their own spellings anyway,
   and clubs are inconsistent. Park/Bak, Lee/Yi/Rhee, Choi/Choe,
   Jung/Jeong/Chung and Kwang/Gwang all coexist in live data.

Design note: the keys produced here are deliberately **recall-oriented**.
Their job is to propose candidate pairs cheaply; the crosswalk then confirms
each pair against date of birth. A loose key that over-matches is safe because
of that second gate. A tight key that misses a real pair is not recoverable.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Hangul -> Revised Romanization
# ---------------------------------------------------------------------------
#
# Hangul syllables are composed arithmetically, so we can decompose them
# without a lookup table:
#     code = 0xAC00 + (initial * 21 + medial) * 28 + final
#
# We skip the inter-syllable assimilation rules of full RR (e.g. 신라 -> Silla).
# For name matching that is fine: the loose key below absorbs the difference.

_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3

_INITIALS = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss",
             "", "j", "jj", "ch", "k", "t", "p", "h"]
_MEDIALS = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae",
            "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
_FINALS = ["", "k", "k", "k", "n", "n", "n", "t", "l", "l", "l", "l", "l",
           "l", "l", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t",
           "p", "t"]


def has_hangul(text: str) -> bool:
    return any(_HANGUL_BASE <= ord(ch) <= _HANGUL_LAST for ch in text)


def romanize_hangul(text: str) -> str:
    """Transliterate Hangul to Revised Romanization, syllable by syllable.

    Non-Hangul characters pass through untouched, so mixed strings are safe.

    >>> romanize_hangul("이정후")
    'i jeong hu'
    """
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if _HANGUL_BASE <= code <= _HANGUL_LAST:
            offset = code - _HANGUL_BASE
            initial, rest = divmod(offset, 21 * 28)
            medial, final = divmod(rest, 28)
            out.append(
                _INITIALS[initial] + _MEDIALS[medial] + _FINALS[final]
            )
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


# ---------------------------------------------------------------------------
# Surname equivalence classes
# ---------------------------------------------------------------------------
#
# Each class maps every romanisation we have actually seen in KBO/MLB/NPB
# sources onto one canonical tag. Grouping surnames explicitly is far more
# reliable than trying to make a phonetic algorithm discover that Lee and Yi
# are the same syllable.

_SURNAME_CLASSES: dict[str, tuple[str, ...]] = {
    "LEE":   ("lee", "yi", "rhee", "ri", "li", "ree", "leigh", "i"),
    "KIM":   ("kim", "gim", "khim"),
    "PARK":  ("park", "bak", "pak", "bahk", "barg", "pahk"),
    "CHOI":  ("choi", "choe", "che", "chwe", "chey"),
    "JUNG":  ("jung", "jeong", "chung", "chong", "joung", "jeung"),
    "KANG":  ("kang", "gang", "khang"),
    "CHO":   ("cho", "jo", "joe", "jou"),
    "YOON":  ("yoon", "yun", "youn", "yune"),
    "JANG":  ("jang", "chang", "zang"),
    "LIM":   ("lim", "im", "rim", "yim", "leem"),
    "HAN":   ("han", "hahn"),
    "OH":    ("oh", "o", "oe", "au"),
    "SEO":   ("seo", "suh", "sur", "so", "서"),
    "SHIN":  ("shin", "sin", "shinn"),
    "KWON":  ("kwon", "gwon", "kweon", "quon"),
    "HWANG": ("hwang", "whang"),
    "AHN":   ("ahn", "an", "ann"),
    "SONG":  ("song", "soong"),
    "RYU":   ("ryu", "yu", "you", "lyu", "yoo", "rhyu", "ryoo", "yuh"),
    "JEON":  ("jeon", "jun", "chun", "chon", "jhun", "chon"),
    "HONG":  ("hong", "hoong"),
    "KO":    ("ko", "go", "koh", "goh", "kho"),
    "MOON":  ("moon", "mun", "mune", "mon"),
    "YANG":  ("yang", "ryang"),
    "SON":   ("son", "sohn", "sonn"),
    "BAE":   ("bae", "pae", "bai", "bay"),
    "BAEK":  ("baek", "paik", "back", "paek", "beak", "baik"),
    "HEO":   ("heo", "hur", "huh", "her", "hu", "huh"),
    "NAM":   ("nam", "nahm"),
    "SHIM":  ("shim", "sim", "sheem"),
    "NOH":   ("noh", "no", "roh", "ro", "nho"),
    "HA":    ("ha", "hah"),
    "KWAK":  ("kwak", "gwak", "kwack", "quack"),
    "SUNG":  ("sung", "seong", "song2"),
    "CHA":   ("cha", "char"),
    # 주 and 추 are different surnames in Hangul but collapse onto the same
    # Latin spellings (Joo/Ju/Chu/Choo), so we do not pretend to separate
    # them. Date of birth does that job downstream.
    "JOO":   ("joo", "ju", "chu", "zoo", "choo", "chou", "chew"),
    "WOO":   ("woo", "wu", "u", "ou"),
    "KOO":   ("koo", "ku", "gu", "goo", "kou"),
    "MIN":   ("min", "minn"),
    "CHAE":  ("chae", "che2", "chai"),
    "WON":   ("won", "weon", "wone"),
    "CHEON": ("cheon", "chun2", "chon2", "천"),
    "BANG":  ("bang", "pang", "bhang"),
    "GONG":  ("gong", "kong", "kohng"),
    "HYUN":  ("hyun", "hyeon", "hyon"),
    "HAM":   ("ham", "hahm"),
    "BYUN":  ("byun", "byeon", "pyun", "byon", "pyon"),
    "YEOM":  ("yeom", "yum", "youm", "yom"),
    "YEO":   ("yeo", "yuh", "yo", "yer"),
    "DO":    ("do", "doh", "to", "toh"),
    "NA":    ("na", "ra", "nah", "rah", "la"),
    "MA":    ("ma", "mah"),
    "JI":    ("ji", "chi", "jee", "gi2"),
    "JIN":   ("jin", "chin", "gin"),
    "EOM":   ("eom", "uhm", "um", "ohm"),
    "PYO":   ("pyo", "phyo", "byo"),
    "SEOK":  ("seok", "suk", "sok", "sek"),
    "SEON":  ("seon", "sun", "sunn"),
    "KYE":   ("kye", "gye", "kay"),
    "OK":    ("ok", "ock", "og"),
    "GIL":   ("gil", "kil", "khil"),
}

# Reverse index: spelling -> canonical tag.
_SURNAME_LOOKUP: dict[str, str] = {}
for _tag, _spellings in _SURNAME_CLASSES.items():
    for _s in _spellings:
        _SURNAME_LOOKUP.setdefault(_s, _tag)


def surname_class(token: str) -> str | None:
    """Canonical tag for a surname spelling, or None if unrecognised."""
    return _SURNAME_LOOKUP.get(_ascii_lower(token))


# Surnames that are *also* common given-name syllables, and so are weak
# evidence of where the surname sits. "Jung" in "Jung Ho Kang" is part of the
# given name; "Kim" essentially never is. Everything not listed here is
# treated as strong evidence.
_WEAK_SURNAMES = {
    "JUNG", "SUNG", "MIN", "JIN", "WON", "JI", "SEOK", "SEON", "HYUN",
    "DO", "NA", "MA", "YEO", "OK", "GIL", "CHEON", "BANG", "GONG", "HA",
    "SHIM", "CHAE", "JOO", "WOO", "SON", "OH", "HAN", "YANG",
}


def _surname_strength(tag: str) -> int:
    return 1 if tag in _WEAK_SURNAMES else 2


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _ascii_lower(text: str) -> str:
    """Lowercase, strip accents, drop everything that is not a letter."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z]", "", stripped).lower()


def clean_name(raw: str) -> str:
    """Strip the decorations Baseball-Reference adds to player names.

    Trailing '*' marks a left-handed batter and '#' a switch hitter; neither
    is part of the name. Parenthetical notes and stray whitespace also go.

    >>> clean_name("Sócrates Brito*")
    'Sócrates Brito'
    """
    text = raw.replace("\xa0", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("*", " ").replace("#", " ").replace("+", " ")
    return re.sub(r"\s+", " ", text).strip(" ,.")


def _tokenize_with_groups(raw: str) -> list[tuple[str, int]]:
    """Tokenise, tagging which tokens came from one hyphenated group.

    The hyphen is the single most reliable signal in romanised Korean names:
    "Ha-Seong Kim" and "Kim Ha-seong" both mark *Ha Seong* as the given name,
    whichever side of the surname it sits on. Group 0 means "not hyphenated".
    """
    text = clean_name(raw)
    if has_hangul(text):
        text = romanize_hangul(text)
    if "," in text:
        surname, _, given = text.partition(",")
        text = f"{given.strip()} {surname.strip()}"
    text = text.replace("'", "").replace(".", " ")

    tagged: list[tuple[str, int]] = []
    group = 0
    for chunk in re.split(r"\s+", text):
        if not chunk:
            continue
        parts = [p for p in chunk.split("-") if p]
        if len(parts) > 1:
            group += 1
            tagged.extend((p, group) for p in parts)
        else:
            tagged.extend((p, 0) for p in parts)

    return [(t, g) for t, g in tagged if _ascii_lower(t) not in _SUFFIXES]


def tokenize(raw: str) -> list[str]:
    """Split a name into comparable tokens, romanising Hangul first."""
    return [t for t, _ in _tokenize_with_groups(raw)]


def split_surname(raw: str) -> tuple[str | None, list[str]]:
    """Return (surname_class, given_tokens).

    The surname is found by *membership in the surname table*, not by
    position, so "Lee Jung-hoo" and "Jung Hoo Lee" resolve identically.
    Resolution order:

    1. A hyphenated group is the given name; the leftover token is the
       surname. This settles the common MLB spelling "Ha-Seong Kim".
    2. Otherwise, among tokens that are known surnames, prefer strong
       evidence over weak, and the outermost position on a tie. This stops
       "Jung Ho Kang" resolving to Jung.
    3. With no known surname (an import such as Merrill Kelly), return None
       and keep every token, so nothing is silently dropped from the key.
    """
    tagged = _tokenize_with_groups(raw)
    if not tagged:
        return None, []

    tokens = [t for t, _ in tagged]
    groups = [g for _, g in tagged]

    # 0. Written in Hangul: name order is not ambiguous. Korean puts the
    #    surname first, always, so take the first syllable and stop. Without
    #    this, 이정후 romanises to "i jeong hu" and the scorer below can be
    #    fooled by 'hu', which is also a spelling of the surname 허.
    if has_hangul(clean_name(raw)):
        return surname_class(tokens[0]), tokens[1:]

    # 1. Hyphenated group present, with exactly one token outside it.
    hyphenated = {g for g in groups if g}
    if len(hyphenated) == 1:
        outside = [i for i, g in enumerate(groups) if g == 0]
        if len(outside) == 1:
            idx = outside[0]
            cls = surname_class(tokens[idx])
            if cls:
                return cls, [t for i, t in enumerate(tokens) if i != idx]

    # 2. Score every known surname: strength first, then outermost position.
    known = [(i, c) for i, t in enumerate(tokens) if (c := surname_class(t))]
    if known:
        last = len(tokens) - 1
        best_i, best_c = max(
            known,
            key=lambda ic: (
                _surname_strength(ic[1]),
                1 if ic[0] in (0, last) else 0,
                ic[0],  # prefer the later of two equally strong edges
            ),
        )
        return best_c, [t for i, t in enumerate(tokens) if i != best_i]

    # 3. Not a recognisable Korean name: keep everything.
    return None, tokens


# ---------------------------------------------------------------------------
# Matching keys
# ---------------------------------------------------------------------------

# Applied in order. Vowel digraphs first, then consonant devoicing, so that
# e.g. "Kwang" and "Gwang" or "Paik" and "Baek" converge.
_LOOSE_RULES: tuple[tuple[str, str], ...] = (
    # vowel digraphs
    ("eo", "o"), ("eu", "u"), ("ae", "e"), ("oe", "e"), ("ui", "i"),
    ("ee", "i"), ("oo", "u"), ("ou", "u"), ("uh", "u"), ("yi", "i"),
    # doubled consonants
    ("kk", "g"), ("gg", "g"), ("tt", "d"), ("pp", "b"), ("ss", "s"),
    ("jj", "j"),
    # aspiration / voicing pairs
    ("sh", "s"), ("ch", "j"), ("kh", "g"), ("ph", "b"), ("th", "d"),
    ("k", "g"), ("p", "b"), ("t", "d"), ("c", "g"),
    ("r", "l"),
)


def _loose(token: str) -> str:
    s = _ascii_lower(token)
    for src, dst in _LOOSE_RULES:
        s = s.replace(src, dst)
    s = re.sub(r"h$", "", s)      # Noh -> No, Suh -> Su
    s = re.sub(r"(.)\1+", r"\1", s)  # collapse any remaining doubles
    s = s.replace("u", "o")        # Jung/Jeong, Hoo/Hu, Sung/Seong
    return s


def strict_key(raw: str) -> str:
    """Order-insensitive key with minimal phonetic change.

    Matches spelling variants of word order and hyphenation only.

    >>> strict_key("Lee Jung-hoo") == strict_key("Jung Hoo Lee")
    True
    """
    cls, given = split_surname(raw)
    given_part = "".join(sorted(_ascii_lower(t) for t in given))
    return f"{cls or ''}|{given_part}"


def loose_key(raw: str) -> str:
    """Aggressively phonetic key. Over-matches on purpose.

    Always confirm a loose-key match against date of birth.

    >>> loose_key("Lee Jung-hoo") == loose_key("Yi Jeong-hu")
    True
    """
    cls, given = split_surname(raw)
    given_part = "".join(sorted(_loose(t) for t in given))
    return f"{cls or ''}|{given_part}"


def name_keys(raw: str) -> dict[str, str]:
    """Both keys plus the cleaned display form, for writing to disk."""
    return {
        "name_clean": clean_name(raw),
        "name_strict_key": strict_key(raw),
        "name_loose_key": loose_key(raw),
    }


def is_probably_korean_name(raw: str) -> bool:
    """Heuristic: does this look like a Korean name rather than an import?

    Used to separate domestic players from the foreign imports, who take a
    different path through the model. Birth city from the roster table is the
    authoritative signal; this is the fallback when that is missing.
    """
    if has_hangul(raw):
        return True
    cls, given = split_surname(raw)
    if cls is None:
        return False
    # Korean given names are one or two short syllables.
    return all(len(_ascii_lower(t)) <= 6 for t in given) and len(given) <= 2
