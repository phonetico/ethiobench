"""Numeral equivalence: a number counts as equal written as words or as digits.

Three directions, applied identically to both sides. `to_digits` and
`to_words` use the parser below. `as_recorded` uses no parser: the reference
keeps the speaker's words and a hypothesis digit token is rewritten to them
only where it equals the digit form recorded for that span.

The parser reads place value, scales, teens, decimals, percent, ordinals and
an attached prefix or case suffix. Ge'ez numerals become digits first. Out of
scope: separator ranges (15-18) and digits fused inside words (አራ8ተኛ), neither
readable back without guessing.
"""

from __future__ import annotations

import re
from decimal import Decimal

GEEZ_DIGITS = {"፩": 1, "፪": 2, "፫": 3, "፬": 4, "፭": 5, "፮": 6, "፯": 7, "፰": 8, "፱": 9}
GEEZ_TENS = {"፲": 10, "፳": 20, "፴": 30, "፵": 40, "፶": 50, "፷": 60, "፸": 70, "፹": 80, "፺": 90}
GEEZ_100 = "፻"
GEEZ_10000 = "፼"


def geez_to_int(s: str) -> int | None:
    """Value of a Ge'ez numeral string, or None if it is not one."""
    if not s or any(ch not in GEEZ_DIGITS and ch not in GEEZ_TENS and ch not in (GEEZ_100, GEEZ_10000) for ch in s):
        return None
    total, group, small = 0, 0, 0
    for ch in s:
        if ch in GEEZ_DIGITS:
            small += GEEZ_DIGITS[ch]
        elif ch in GEEZ_TENS:
            small += GEEZ_TENS[ch]
        elif ch == GEEZ_100:
            group += (small or 1) * 100
            small = 0
        else:
            total += ((group + small) or 1) * 10000
            group = small = 0
    return total + group + small


LEXICONS = {
    "amh": {
        # Every attested spelling, because this step runs before the WER-N
        # fold and cannot rely on it: ሠ/ሥ/ሣ for ሰ/ስ/ሳ, ኃ/ሐ for ሃ, ሺሕ for ሺ.
        "units": {"ዜሮ": 0, "ዚሮ": 0, "አንድ": 1, "ሁለት": 2, "ሶስት": 3, "ሦስት": 3, "አራት": 4, "አምስት": 5, "ስድስት": 6,
                  "ሰባት": 7, "ሠባት": 7, "ስምንት": 8, "ሥምንት": 8, "ዘጠኝ": 9},
        "tens": {"አስር": 10, "አሥር": 10, "አስራ": 10, "አሥራ": 10, "ሃያ": 20, "ሀያ": 20,
                 "ሰላሳ": 30, "ሠላሳ": 30, "ሰላሣ": 30, "ሠላሣ": 30, "አርባ": 40,
                 "ሃምሳ": 50, "ሀምሳ": 50, "ኃምሳ": 50, "ሐምሳ": 50, "ሃምሣ": 50, "ሀምሣ": 50, "ኃምሣ": 50, "ሐምሣ": 50,
                 "ስልሳ": 60, "ሥልሳ": 60, "ስልሣ": 60, "ሥልሣ": 60, "ስድሳ": 60, "ሰባ": 70, "ሠባ": 70,
                 "ሰማንያ": 80, "ሠማንያ": 80, "ዘጠና": 90},
        "hundred": {"መቶ": 100},
        # million and billion: every spelling the loanword table rules one word
        "scales": {"ሺ": 1000, "ሺህ": 1000, "ሺሕ": 1000,
                   "ሚሊዮን": 10**6, "ሚልዮን": 10**6, "ሚሊየን": 10**6, "ሚልየን": 10**6, "ሚሊዬን": 10**6, "ሚልዬን": 10**6,
                   "ቢሊዮን": 10**9, "ቢልዮን": 10**9, "ቢሊየን": 10**9, "ቢልየን": 10**9, "ቢሊዬን": 10**9, "ቢልዬን": 10**9,
                   "ትሪሊዮን": 10**12, "ትሪልዮን": 10**12, "ትሪሊየን": 10**12, "ትሪልየን": 10**12},
        "ordinals": {"አንደኛ": 1, "ሁለተኛ": 2, "ሶስተኛ": 3, "ሦስተኛ": 3, "አራተኛ": 4, "አምስተኛ": 5, "ስድስተኛ": 6,
                     "ሰባተኛ": 7, "ስምንተኛ": 8, "ዘጠነኛ": 9, "አስረኛ": 10, "አሥረኛ": 10, "ሃያኛ": 20, "ሰላሳኛ": 30,
                     "አርባኛ": 40, "ሃምሳኛ": 50, "ስልሳኛ": 60, "ሰባኛ": 70, "ሰማንያኛ": 80, "ዘጠናኛ": 90,
                     "መቶኛ": 100, "ሺኛ": 1000},
        "ordinal_suffix": "ኛ",
        "point": {"ነጥብ"},
        "percent": {"በመቶ", "ፐርሰንት"},
        "prefixes": ("እስከ", "የ", "በ", "ለ", "ከ"),
        # Prefixes `as_recorded` forgives on a digit token (በ2014 for 2014).
        "digit_prefixes": ("የ", "በ", "ለ", "ከ"),
        "word_suffixes": (),
        "out": {"units": ["", "አንድ", "ሁለት", "ሶስት", "አራት", "አምስት", "ስድስት", "ሰባት", "ስምንት", "ዘጠኝ"],
                "tens": ["", "አስር", "ሃያ", "ሰላሳ", "አርባ", "ሃምሳ", "ስልሳ", "ሰባ", "ሰማንያ", "ዘጠና"],
                "teen": "አስራ", "hundred": "መቶ", "thousand": "ሺ", "million": "ሚሊዮን", "billion": "ቢሊዮን",
                "zero": "ዜሮ", "point": "ነጥብ", "percent": "በመቶ",
                "ordinal_units": ["", "አንደኛ", "ሁለተኛ", "ሶስተኛ", "አራተኛ", "አምስተኛ", "ስድስተኛ", "ሰባተኛ", "ስምንተኛ", "ዘጠነኛ"],
                "ordinal_tens": ["", "አስረኛ", "ሃያኛ", "ሰላሳኛ", "አርባኛ", "ሃምሳኛ", "ስልሳኛ", "ሰባኛ", "ሰማንያኛ", "ዘጠናኛ"]},
    },
    "tir": {
        "units": {"ዜሮ": 0, "ሓደ": 1, "ክልተ": 2, "ሰለስተ": 3, "ኣርባዕተ": 4, "ሓሙሽተ": 5, "ሽዱሽተ": 6, "ሸውዓተ": 7,
                  "ሸሞንተ": 8, "ትሽዓተ": 9},
        "tens": {"ዓሰርተ": 10, "ዕስራ": 20, "ሰላሳ": 30, "ኣርብዓ": 40, "ሓምሳ": 50, "ስሳ": 60, "ሰብዓ": 70,
                 "ሰማንያ": 80, "ተስዓ": 90},
        "hundred": {"ሚእቲ": 100},
        "scales": {"ሽሕ": 1000, "ሚልዮን": 10**6, "ሚሊዮን": 10**6, "ቢልዮን": 10**9},
        "ordinals": {"ቀዳማይ": 1, "ካልኣይ": 2, "ሳልሳይ": 3, "ራብዓይ": 4, "ሓምሻይ": 5, "ሻድሻይ": 6, "ሻብዓይ": 7,
                     "ሻምናይ": 8, "ታሽዓይ": 9, "ዓስራይ": 10},
        "ordinal_suffix": "ይ",
        "point": {"ነጥቢ"},
        "percent": {"ሚእታዊ", "ፐርሰንት"},
        "prefixes": ("ክሳብ", "ካብ", "ናብ", "ኣብ", "ብ", "ን"),
        "digit_prefixes": ("ክሳብ", "ካብ", "ናብ", "ኣብ", "ብ", "ን"),
        # Conjunction/case endings on number words: ዕስራን ሓሙሽተን ("twenty and five").
        "word_suffixes": ("ን",),
        "out": {"units": ["", "ሓደ", "ክልተ", "ሰለስተ", "ኣርባዕተ", "ሓሙሽተ", "ሽዱሽተ", "ሸውዓተ", "ሸሞንተ", "ትሽዓተ"],
                "tens": ["", "ዓሰርተ", "ዕስራ", "ሰላሳ", "ኣርብዓ", "ሓምሳ", "ስሳ", "ሰብዓ", "ሰማንያ", "ተስዓ"],
                "teen": "ዓሰርተ", "hundred": "ሚእቲ", "thousand": "ሽሕ", "million": "ሚልዮን", "billion": "ቢልዮን",
                "zero": "ዜሮ", "point": "ነጥቢ", "percent": "ሚእታዊ",
                "ordinal_units": ["", "ቀዳማይ", "ካልኣይ", "ሳልሳይ", "ራብዓይ", "ሓምሻይ", "ሻድሻይ", "ሻብዓይ", "ሻምናይ", "ታሽዓይ"],
                "ordinal_tens": ["", "ዓስራይ", "", "", "", "", "", "", "", ""]},
    },
}

DIRECTIONS = ("to_digits", "to_words", "as_recorded", "off")
_DIGIT_TOKEN = re.compile(r"^(?P<prefix>\D*?)(?P<num>\d+(?:[.,]\d+)*)(?P<pct>%)?(?P<suffix>\D*)$")

# `as_recorded`: a digit hypothesis matches only the digit form recorded in
# number_spans, or derived from a reference written in words (derive_spans).
# Any other digit token stays as written and misses.
# A differing prefix or fused suffix (2014 for በ2014) is one substitution, not
# a miss. Separators inside a digit form are removed by the base pass.
_EDGE_PUNCT = re.compile(r"^[\"“”'‘’(\[]+|[\"“”'‘’)\]።፣፤፥፦፧፨,.;:!?]+$")
_HAS_DIGIT = re.compile(r"[0-9]")
_SEPARATORS = re.compile(r"[/\-:]")
_FUSED_SUFFIX = re.compile(r"(?<=[0-9%])[ሀ-፿]+$")


def _prefix_pattern(language: str) -> re.Pattern:
    """A prefix attached to a digit token, from the language's `digit_prefixes`,
    longest first so ክሳብ is tried before ብ."""
    ps = sorted(LEXICONS[language]["digit_prefixes"], key=len, reverse=True)
    return re.compile("^(?:" + "|".join(map(re.escape, ps)) + ")")


def _digit_key(token: str) -> str:
    """A digit token with its group commas and edge punctuation removed and
    the Ethiopic colon read as a colon: 6,079,690፣ and 6079690 are one digit
    form, and so are 4፡30 and 4:30."""
    token = _EDGE_PUNCT.sub("", token.replace("፡", ":"))
    return re.sub(r"(?<=\d),(?=\d)", "", token)


def _pieces(key: str) -> list[str]:
    """A digit key split at its separators, the percent sign its own piece:
    274/1994 → [274, 1994]; 30% → [30, %]."""
    out = []
    for p in _SEPARATORS.split(key):
        if p.endswith("%") and len(p) > 1:
            out.extend([p[:-1], "%"])
        elif p:
            out.append(p)
    return out


def _replace_edge(words: str, token: str, first: bool) -> str:
    ws = words.split()
    if first:
        ws[0] = token
    else:
        ws[-1] = token
    return " ".join(ws)


def rewrite_recorded(text: str, spans: list[dict] | None, language: str = "amh") -> tuple[str, int, int, int]:
    """Rewrite matching digit tokens to their span's words.

    Returns (text, matched, matched up to an affix, digit tokens left). Spans
    are taken in order, each matching at most once.
    """
    tokens = text.split()
    out: list[str | None] = list(tokens)
    taken = [False] * len(tokens)
    keys = [_digit_key(t) for t in tokens]
    percent_words = LEXICONS[language]["percent"]
    prefix_re = _prefix_pattern(language)
    # the hypothesis as a flat list of pieces, each knowing its token
    pieces, owner = [], []
    for i, k in enumerate(keys):
        if k in percent_words and i > 0 and _HAS_DIGIT.search(keys[i - 1]):
            pieces.append("%")
            owner.append(i)
            continue
        for p in _pieces(k) or [k]:
            pieces.append(p)
            owner.append(i)
    starts = {}
    for idx, i in enumerate(owner):
        starts.setdefault(i, idx)
    ends = {i: idx for idx, i in enumerate(owner)}

    def find(want: list[str]) -> tuple[int, int] | None:
        """Token range whose pieces equal `want`, aligned to token edges, untaken."""
        n = len(want)
        for idx in range(len(pieces) - n + 1):
            if pieces[idx:idx + n] != want:
                continue
            i, j = owner[idx], owner[idx + n - 1]
            if starts[i] == idx and ends[j] == idx + n - 1 and not any(taken[i:j + 1]):
                return i, j
        return None

    def take(i: int, j: int, replacement: str) -> None:
        out[i] = replacement
        for k in range(i + 1, j + 1):
            out[k] = None
        for k in range(i, j + 1):
            taken[k] = True

    matched = partial = 0
    for span in spans or ():
        want = [p for t in span["digits"].split() for p in (_pieces(_digit_key(t)) or [_digit_key(t)])]
        if not want:
            continue
        hit = find(want)
        if hit is not None:
            take(*hit, span["words"])
            matched += 1
            continue
        # one substitution for a prefix or suffix that differs
        first, last = want[0], want[-1]
        for variant, edge in ((prefix_re.sub("", first), True), (_FUSED_SUFFIX.sub("", last), False)):
            alt = [variant] + want[1:] if edge else want[:-1] + [variant]
            if alt == want or not variant:
                continue
            hit = find(alt)
            if hit is not None:
                i, j = hit
                take(i, j, _replace_edge(span["words"], tokens[i] if edge else tokens[j], edge))
                partial += 1
                break
        else:
            # the hypothesis has a prefix letter or a fused suffix the reference does not
            for i, k in enumerate(keys):
                if taken[i] or not _HAS_DIGIT.search(k):
                    continue
                if prefix_re.match(k) and _pieces(prefix_re.sub("", k)) == want:
                    take(i, i, _replace_edge(span["words"], tokens[i], True))
                    partial += 1
                    break
                if _FUSED_SUFFIX.search(k) and _pieces(_FUSED_SUFFIX.sub("", k)) == want:
                    take(i, i, _replace_edge(span["words"], tokens[i], False))
                    partial += 1
                    break
    left = sum(1 for t, used in zip(tokens, taken) if not used and _HAS_DIGIT.search(t))
    return " ".join(t for t in out if t is not None), matched, partial, left


class _Word:
    """Classification of one token as a number word."""

    __slots__ = ("kind", "value", "prefix", "suffix", "terminal")

    def __init__(self, kind: str, value: int, prefix: str = "", suffix: str = ""):
        self.kind, self.value, self.prefix, self.suffix = kind, value, prefix, suffix
        self.terminal = False   # carried punctuation: the number ends here


_TRAILING_PUNCT = re.compile(r"^(.*?)([።፣፤፥፦፧፨,.;:!?]+)$")


def _split_punct(token: str) -> tuple[str, str]:
    m = _TRAILING_PUNCT.match(token)
    return (m.group(1), m.group(2)) if m and m.group(1) else (token, "")


def _classify(token: str, lex: dict, allow_prefix: bool) -> _Word | None:
    # This step precedes punctuation removal, so a number word arrives with its
    # clause punctuation (ስምንት፣): read through it, keep it, and end the number.
    token, punct = _split_punct(token)
    if token in lex["percent"] or token in lex["point"]:
        return None            # በመቶ is "percent" after a number, never the number 100
    word = _classify_bare(token, lex, allow_prefix)
    if word is not None and punct:
        word.suffix += punct
        word.terminal = True
    return word


def _classify_bare(token: str, lex: dict, allow_prefix: bool) -> _Word | None:
    candidates = [("", token)]
    if allow_prefix:
        candidates += [(p, token[len(p):]) for p in lex["prefixes"] if token.startswith(p) and len(token) > len(p)]
    for prefix, rest in candidates:
        forms = [(rest, "")] + [(rest[: -len(s)], s) for s in lex["word_suffixes"] if rest.endswith(s) and len(rest) > len(s)]
        for stem, suffix in forms:
            for kind in ("units", "tens", "hundred", "scales"):
                if stem in lex[kind]:
                    return _Word(kind, lex[kind][stem], prefix, suffix)
            if stem in lex["ordinals"]:
                return _Word("ordinal", lex["ordinals"][stem], prefix, suffix)
    return None


def _classify_digits(token: str, lex: dict) -> _Word | None:
    """A token already written in digits (20, 1,400, 1.4, በ2017), so that a
    scale or percent word after it joins it: 20 ሚሊዮን is one number."""
    core, punct = _split_punct(token)
    m = _DIGIT_TOKEN.match(core)
    if not m or m.group("pct") or m.group("suffix"):
        return None
    raw = m.group("num").replace(",", "")
    if raw.count(".") > 1:
        return None
    word = _Word("digits", Decimal(raw), m.group("prefix"), punct)
    word.terminal = bool(punct)
    return word


def _format(value: Decimal) -> str:
    """Digits, never scientific notation: 16.9 ቢሊዮን is 16900000000, 3.5 is 3.5."""
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


class _Run:
    """One number read from a run of tokens."""

    __slots__ = ("value", "consumed", "suffix", "ordinal", "ended", "last_kind", "words")

    def __init__(self):
        self.value = Decimal(0)
        self.consumed = 0
        self.suffix = ""
        self.ordinal = False
        self.ended = False
        self.last_kind = None
        self.words = 0


def _parse_run(tokens: list[str], start: int, lex: dict, *, allow_prefix: bool, fraction: bool = False) -> _Run | None:
    """Read one integer or ordinal from tokens[start:].

    Place rule: a word continues the number only when it fills a smaller place
    than the last. A unit after a unit, tens after tens, or anything after a
    written digit is an enumeration and starts a new number.
    """
    word = _classify(tokens[start], lex, allow_prefix=allow_prefix) or _classify_digits(tokens[start], lex)
    if word is None:
        return None
    run = _Run()
    total, current, last_scale = Decimal(0), Decimal(0), 0
    j = start
    while True:
        kind = word.kind
        if kind == "digits":
            if run.last_kind is not None:
                break
            current = word.value
        elif kind == "units":
            if run.last_kind in ("units", "digits", "ordinal"):
                break
            current += word.value
        elif kind == "tens":
            if run.last_kind in ("tens", "units", "digits"):
                break
            current += word.value
        elif kind == "hundred":
            if run.last_kind in ("hundred", "digits") or current >= 100:
                break
            current = (current or 1) * 100
        elif kind == "scales":
            # Scales descend within one number: a scale no smaller than one
            # already used starts a new number. After a decimal point the
            # scale belongs to the whole number (ስድስት ነጥብ ስድስት ቢልዮን), not
            # to the fraction.
            if fraction or (last_scale and word.value >= last_scale):
                break
            total += (current if current else 1) * word.value
            current = Decimal(0)
            last_scale = word.value
        elif kind == "ordinal":
            if run.last_kind == "digits" or (word.value < 10 and run.last_kind == "units") \
                    or (10 <= word.value < 100 and run.last_kind in ("tens", "units")):
                break
            current += word.value
            run.ordinal = True
        run.suffix = word.suffix
        run.ended = word.terminal
        run.last_kind = kind
        run.words += 1
        j += 1
        if run.ordinal or run.ended or j >= len(tokens):
            break
        word = _classify(tokens[j], lex, allow_prefix=False) or _classify_digits(tokens[j], lex)
        if word is None:
            break
    run.value = total + current
    run.consumed = j - start
    return run


def _fraction(tokens: list[str], start: int, lex: dict) -> tuple[str, int, str, bool] | None:
    """(digit string, tokens consumed, suffix, ended) for the words after a point.

    Unit words are digits read one by one (ዜሮ አምስት .05, አምስት ሶስት .53). A
    compound below 100 is one group (ሰባ አምስት .75), allowed first or after
    zeros only (ዜሮ ሰባ አምስት .075), and ends the fraction. None if no number
    word follows the point.
    """
    digits, j, suffix, ended = "", start, "", False
    while j < len(tokens) and not ended:
        frac = _parse_run(tokens, j, lex, allow_prefix=False, fraction=True)
        if frac is None or frac.ordinal or frac.last_kind == "digits":
            break
        if frac.words == 1 and frac.last_kind == "units":
            digits += str(int(frac.value))
        elif frac.value < 100 and set(digits) <= {"0"}:
            digits += str(int(frac.value))
            suffix, ended, j = frac.suffix, frac.ended, j + frac.consumed
            break
        else:
            break
        suffix, ended, j = frac.suffix, frac.ended, j + frac.consumed
    return (digits, j - start, suffix, ended) if digits else None


def _words_to_digits(tokens: list[str], lex: dict) -> list[str]:
    """Rewrite runs of number words to one digit string each (see _parse_run)."""
    return [replacement for _, _, replacement in _walk(tokens, lex)]


def _walk(tokens: list[str], lex: dict):
    """Yield (start, end, replacement) over `tokens`: a number run collapsed to
    its digit string, or a single token left as it is."""
    i = 0
    while i < len(tokens):
        run = _parse_run(tokens, i, lex, allow_prefix=True)
        if run is None:
            yield i, i + 1, tokens[i]
            i += 1
            continue
        first_kind = run.last_kind if run.words == 1 else None
        prefix = (_classify(tokens[i], lex, allow_prefix=True) or _classify_digits(tokens[i], lex)).prefix
        value, consumed, suffix, ended = run.value, i + run.consumed, run.suffix, run.ended
        # A point word followed by a number makes a decimal (ሶስት ነጥብ አምስት, ዜሮ
        # ነጥብ ሰላሳ አምስት), and a scale may follow the decimal (ስድስት ነጥብ ስድስት ቢልዮን).
        sequence = None
        if not run.ordinal and not ended and consumed + 1 < len(tokens) and tokens[consumed] in lex["point"]:
            frac = _fraction(tokens, consumed + 1, lex)
            if frac is not None:
                digits, f_consumed, suffix, ended = frac
                value = Decimal(f"{int(value)}.{digits}")
                consumed += 1 + f_consumed
                if not ended and consumed < len(tokens):
                    scale = _classify(tokens[consumed], lex, allow_prefix=False)
                    if scale is not None and scale.kind == "scales":
                        value = value * scale.value
                        suffix, ended = scale.suffix, scale.terminal
                        consumed += 1
        elif run.words == 1 and first_kind == "units" and value == 0 and not ended:
            # Digits read after a leading zero, one by one or in two-digit
            # blocks, are one digit string (ዜሮ ሓሙሽተ 05; ዜሮ ዘጠኝ አስራ አንድ 0911;
            # ዜሮ አስራ አንድ አስራ አምስት 0115). A number never starts with zero.
            sequence, j = "0", consumed
            while j < len(tokens):
                nxt = _parse_run(tokens, j, lex, allow_prefix=False)
                if nxt is None or nxt.ordinal:
                    break
                if nxt.words == 1 and nxt.last_kind == "units":
                    sequence += str(int(nxt.value))
                elif 10 <= nxt.value <= 99 and nxt.words <= 2 and nxt.last_kind in ("units", "tens"):
                    sequence += str(int(nxt.value))
                else:
                    break
                suffix, ended = nxt.suffix, nxt.ended
                j += nxt.consumed
                if ended:
                    break
            if len(sequence) == 1:
                sequence = None
            else:
                consumed = j
        number = sequence if sequence is not None else _format(value)
        if not run.ordinal and not ended and consumed < len(tokens):
            pct, punct = _split_punct(tokens[consumed])
            if pct in lex["percent"]:
                number += "%"
                suffix += punct
                consumed += 1
        if run.ordinal:
            number += lex["ordinal_suffix"]
        if first_kind == "digits" and consumed == i + 1:
            yield i, consumed, tokens[i]       # a digit token nothing joined: left exactly as written (05, 2013)
        else:
            yield i, consumed, f"{prefix}{number}{suffix}"
        i = consumed


def spans_for(reference_text: str, language: str, recorded: list[dict] | None) -> list[dict]:
    """Recorded pairs first, then derived pairs for the remaining number words.

    A derived pair repeating a recorded one is dropped: no double count.
    """
    recorded = list(recorded or ())
    tokens = reference_text.split()
    # Positions a recorded span covers: the parser's reading of those words
    # must not stand beside the recorded one.
    covered: set[int] = set()
    for s in recorded:
        n = len(s["words"].split())
        start = s.get("token_index")
        if start is None or tokens[start:start + n] != s["words"].split():
            ws = s["words"].split()
            start = next((i for i in range(len(tokens) - n + 1) if tokens[i:i + n] == ws), None)
        if start is not None:
            covered.update(range(start, start + n))
    out = recorded[:]
    for s in derive_spans(reference_text, language):
        if not covered.intersection(range(s["start"], s["end"])):
            out.append(s)
    return out


def derive_spans(text: str, language: str) -> list[dict]:
    """(digits, words) pairs the parser reads off a reference written in words.

    Marked derived; recorded pairs take precedence.
    """
    lex = LEXICONS[language]
    tokens = text.split()
    spans = []
    for start, end, replacement in _walk(tokens, lex):
        words = tokens[start:end]
        if any(_HAS_DIGIT.search(t) for t in words) or replacement == " ".join(words):
            continue
        spans.append({"digits": replacement, "words": " ".join(words), "start": start, "end": end, "derived": True})
    return spans


def _below_thousand(n: int, words: dict) -> list[str]:
    parts: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        parts += ([words["units"][hundreds]] if hundreds > 1 else []) + [words["hundred"]]
    tens, units = divmod(rest, 10)
    if tens == 1 and units:
        parts += [words["teen"], words["units"][units]]
    else:
        if tens:
            parts.append(words["tens"][tens])
        if units:
            parts.append(words["units"][units])
    return parts


def int_to_words(n: int, words: dict) -> list[str]:
    if n == 0:
        return [words["zero"]]
    parts: list[str] = []
    for scale, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        count, n = divmod(n, scale)
        if count:
            parts += (_below_thousand(count, words) if count > 1 else []) + [words[name]]
    return parts + _below_thousand(n, words)


def _ordinal_words(n: int, words: dict) -> list[str] | None:
    base = int_to_words(n, words)
    last_units = n % 10
    last_tens = n % 100 // 10
    if last_units and words["ordinal_units"][last_units]:
        return base[:-1] + [words["ordinal_units"][last_units]]
    if not last_units and last_tens and words["ordinal_tens"][last_tens]:
        return base[:-1] + [words["ordinal_tens"][last_tens]]
    return None


def _digits_to_words(tokens: list[str], lex: dict) -> list[str]:
    words = lex["out"]
    out: list[str] = []
    for token in tokens:
        m = _DIGIT_TOKEN.match(token)
        if not m:
            out.append(token)
            continue
        prefix, raw, pct, suffix = m.group("prefix"), m.group("num"), m.group("pct"), m.group("suffix")
        ordinal = suffix.startswith(lex["ordinal_suffix"]) and not pct
        if ordinal:
            suffix = suffix[len(lex["ordinal_suffix"]):]
        whole, _, frac = raw.replace(",", "").partition(".")
        if len(whole) > 12:
            out.append(token)
            continue
        value = int(whole)
        spoken = _ordinal_words(value, words) if ordinal else int_to_words(value, words)
        if spoken is None:
            out.append(token)
            continue
        if frac:
            spoken += [words["point"]] + [words["units"][int(d)] if d != "0" else words["zero"] for d in frac]
        if pct:
            spoken.append(words["percent"])
        spoken[0] = prefix + spoken[0]
        spoken[-1] = spoken[-1] + suffix
        out.extend(spoken)
    return out


def normalize_numerals(text: str, language: str, direction: str, *,
                       side: str = "reference", number_spans: list[dict] | None = None) -> str:
    """Rewrite numbers in `text` into the canonical form for `direction`.

    Under `as_recorded` the reference is returned as it is and the hypothesis
    has its recorded digit forms rewritten to the reference's words (see
    `rewrite_recorded`); `number_spans` is the manifest row's list.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"numeral direction must be one of {DIRECTIONS}")
    if direction == "off":
        return text
    if direction == "as_recorded":
        return text if side == "reference" else rewrite_recorded(text, number_spans, language)[0]
    lex = LEXICONS[language]
    tokens = []
    for token in text.split():
        m = re.fullmatch(r"(\D*?)([፩-፼]+)(\D*)", token)
        value = geez_to_int(m.group(2)) if m else None
        tokens.append(f"{m.group(1)}{value}{m.group(3)}" if m and value is not None else token)
    digits = _words_to_digits(tokens, lex)
    if direction == "to_digits":
        return " ".join(digits)
    # to_words goes through digits first, so spelling variants of number words
    # (ሺ/ሺህ, ሶስት/ሦስት) come out in one canonical spelling on both sides.
    return " ".join(_digits_to_words(digits, lex))
