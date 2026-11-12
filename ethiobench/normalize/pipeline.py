"""Apply a normalization contract to text.

The contract decides which steps run and in what order; this module only
performs each named step. Both sides pass through the same function, and the
one asymmetry (fluent mode drops fragments from the reference only) is a
step's `sides` in the contract, not coded here.

Layer order: base, the mode layer, then the WER-N fold. The fold is last so it
cannot turn a word into a filler token: a standalone ዐ is a word, and folding
it to አ before filler removal would delete it.
"""

from __future__ import annotations

import re
import unicodedata

from .config import MODES, SIDES, ContractError, EquivalenceSet, NormalizationContract
from .numerals import normalize_numerals

# Private-use character standing in for a fragment's trailing hyphen between
# mark_fragments and the mode layer. Unicode category Co, so punctuation removal
# leaves it alone.
FRAGMENT_MARK = ""

_ZERO_WIDTH = re.compile("[​-‏⁠﻿]")
_BRACKET_TAG = re.compile(r"\[(?:(?:unclear|overlap):\d+(?:\.\d+)?|laughter|cough|noise)\]")
# A hyphen ending a token: preceded by a non-space, followed by space, end of
# text, or punctuation that will itself be removed.
_FRAGMENT = re.compile(r"(?<=[^\s\-])-(?=\s|$|[፠-፨.,?!:;])")


def _remove_punctuation(text: str) -> str:
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        category = unicodedata.category(ch)
        if ch == FRAGMENT_MARK or not (category.startswith("P") or category.startswith("S")):
            out.append(ch)
            continue
        prev_digit = i > 0 and text[i - 1].isdigit()
        next_digit = i + 1 < n and text[i + 1].isdigit()
        if ch == "." and prev_digit and next_digit:
            out.append(ch)  # decimal point is part of the number as spoken
        elif ch == "," and prev_digit and next_digit:
            continue  # digit-group separator: 10,693 is one number
        elif ch in "'’" and 0 < i < n - 1 and text[i - 1].isascii() and text[i - 1].isalpha() \
                and text[i + 1].isascii() and text[i + 1].isalpha():
            continue  # word-internal apostrophe in embedded English: don't, speaker's
        else:
            out.append(" ")
    return "".join(out)


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _base_step(name: str, text: str) -> str:
    if name == "unicode_nfc":
        return unicodedata.normalize("NFC", text)
    if name == "strip_zero_width":
        return _ZERO_WIDTH.sub("", text)
    if name == "strip_bracket_tags":
        return _BRACKET_TAG.sub(" ", text)
    if name == "mark_fragments":
        return _FRAGMENT.sub(FRAGMENT_MARK, text)
    if name == "remove_punctuation":
        return _remove_punctuation(text)
    if name == "casefold":
        return text.casefold()
    if name == "collapse_whitespace":
        return _collapse(text)
    raise ContractError(f"base step {name!r} has no implementation")


def _mode_step(name: str, tokens: list[str], contract: NormalizationContract) -> list[str]:
    fillers = {t: members[0] for members in contract.filler_classes.values() for t in members}
    if name == "remove_fillers":
        return [t for t in tokens if t not in fillers]
    if name == "remove_fragments":
        return [t for t in tokens if FRAGMENT_MARK not in t]
    if name == "unmark_fragments":
        return [t.replace(FRAGMENT_MARK, "") for t in tokens]
    if name == "filler_equivalence":
        return [fillers.get(t, t) for t in tokens]
    raise ContractError(f"mode step {name!r} has no implementation")


def _default_orders(base: str) -> list[str]:
    """The seven vowel orders of a series, plus its labialized form if it has one."""
    start = ord(base)
    return [chr(start + k) for k in range(8)]


def _eighth_vowel(ch: str) -> str | None:
    """Vowel of an eighth-position syllable, read from its Unicode name.

    The eighth position is either a labialized -WA (ሷ SWA, ሗ HHWA) or an -OA
    form (ሇ HOA, ኇ XOA), and some rows leave it unassigned (ዐ, ኸ). The last
    two letters of the name tell these apart.
    """
    name = unicodedata.name(ch, "")
    return name[-2:] if name.startswith("ETHIOPIC SYLLABLE") else None


def _set_mapping(s: EquivalenceSet) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if s.orders:
        rows = [list(row) for row in s.orders]
        for row in rows[1:]:
            for k, ch in enumerate(row):
                if ch != rows[0][k]:
                    mapping[ch] = rows[0][k]
    else:
        target_rows = _default_orders(s.members[0])
        for member in s.members[1:]:
            for k, ch in enumerate(_default_orders(member)):
                target = target_rows[k]
                if k < 7 or (_eighth_vowel(ch) and _eighth_vowel(ch) == _eighth_vowel(target)):
                    mapping[ch] = target
    for ch in s.unfolded:
        if ch not in mapping:
            raise ContractError(
                f"equivalence set {s.name!r} lists {ch!r} as unfolded, but its expansion "
                f"never folds it.")
        del mapping[ch]
    return mapping


_FOLD_TABLES: dict[str, dict[int, str]] = {}


def _fold_table(contract: NormalizationContract) -> dict[int, str]:
    if contract.sha256 not in _FOLD_TABLES:
        _FOLD_TABLES[contract.sha256] = _build_fold_table(contract)
    return _FOLD_TABLES[contract.sha256]


def fold_mapping(contract: NormalizationContract) -> dict[str, str]:
    """The resolved fold, character to final form, for inspection and reporting."""
    return {chr(k): v for k, v in _fold_table(contract).items()}


def _build_fold_table(contract: NormalizationContract) -> dict[int, str]:
    mapping: dict[str, str] = {}
    for s in contract.equivalence_sets:
        for ch, target in _set_mapping(s).items():
            if mapping.get(ch, target) != target:
                raise ContractError(
                    f"{ch!r} folds to both {mapping[ch]!r} and {target!r}; equivalence "
                    f"sets must not disagree about a character.")
            mapping[ch] = target
    # Resolve chains so every character maps straight to its final form.
    for ch in list(mapping):
        seen = {ch}
        target = mapping[ch]
        while target in mapping:
            if mapping[target] in seen:
                raise ContractError(f"fold cycle through {ch!r}")
            seen.add(target)
            target = mapping[target]
        mapping[ch] = target
    return {ord(ch): target for ch, target in mapping.items()}


def normalize(text: str, contract: NormalizationContract, *, mode: str, side: str,
              wer_n: bool = False, numerals: str | None = None, loanword_profile: str | None = None,
              number_spans: list[dict] | None = None) -> str:
    """Normalize `text` under `contract` for one scoring mode and side.

    `numerals` overrides the declared direction and is recorded in every result.
    `number_spans` is the row's recorded (digits, words) pairs, used by
    `as_recorded` on the hypothesis side. `loanword_profile` applies the
    loanword table over the WER-N fold and needs both wer_n and a table.
    """
    if loanword_profile is not None:
        if not wer_n:
            raise ValueError("a loanword profile applies to the equivalence layer; pass wer_n=True")
        if contract.loanwords is None:
            raise ContractError(f"contract {contract.language} {contract.version} declares no loanword table")
        if loanword_profile not in contract.loanword_profiles:
            raise ContractError(f"profile {loanword_profile!r} is not declared by the contract "
                                f"({', '.join(contract.loanword_profiles)})")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")

    for step in contract.layers["base"]:
        if side not in step.sides:
            continue
        if step.name == "numerals":
            text = normalize_numerals(text, contract.language, numerals or step.direction,
                                      side=side, number_spans=number_spans)
        elif step.name == "expand_abbreviations":
            text = contract.abbreviations.expand(text)
        else:
            text = _base_step(step.name, text)

    tokens = text.split()
    for step in contract.layers[mode]:
        if side in step.sides:
            tokens = _mode_step(step.name, tokens, contract)
    # A marker on a side the mode layer did not act on (a hypothesis token in
    # fluent mode) is removed and the token kept, as the convention scores
    # hypothesis tokens as they stand.
    text = _collapse(" ".join(tokens).replace(FRAGMENT_MARK, ""))

    if wer_n and contract.equivalence_sets:
        text = text.translate(_fold_table(contract))
    # The loanword fold runs after the character fold, on whole tokens, so a
    # spelling reaches the table in the same folded form on both sides. The
    # table's own forms are written in the folded alphabet already; a table
    # entry using an unfolded character would never match, which the tests check.
    if loanword_profile is not None:
        text = contract.loanwords.fold(text, loanword_profile)
    return text
