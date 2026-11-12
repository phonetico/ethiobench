"""Code-switching metrics over the reference's word-level language spans.

Computed on the same fluent alignment as the reported WER, so every number here
decomposes a WER the report already carries: per-language WER, PIER over
points of interest (embedded-language tokens and their matrix neighbours),
PIER split by span length into insertion and alternation, and switch-point
accuracy.

CAVEATS, recorded in every report: sets without spans get no metrics; spans
carry only a language tag, so insertion and alternation are split by span
length.

Tagging runs on the raw reference, where span offsets are valid, then each raw
word is normalized alone and aligned to the whole-text normalization, so a
rewrite that joins words keeps its tag.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein

from ..normalize.config import NormalizationContract
from ..normalize.pipeline import normalize

# Tags that are not a switch into another language. `other` marks names and
# unclassified material, and is reported but not treated as a point of interest.
NON_SWITCH_TAGS = ("other",)
WORD = re.compile(r"\S+")
# Per-word normalization, keyed by contract hash and mode. Cleared when full, so
# memory stays bounded across many contracts or sets in one process.
_WORD_CACHE: dict[tuple, list[str]] = {}
_WORD_CACHE_MAX = 200_000


@dataclass
class Tally:
    """Error counts over one class of reference tokens."""

    errors: int = 0
    units: int = 0

    def add(self, other: "Tally") -> None:
        self.errors += other.errors
        self.units += other.units

    @property
    def rate(self) -> float | None:
        return self.errors / self.units if self.units else None


@dataclass
class SegmentCS:
    """Code-switching counts for one scored item."""

    spans: int = 0
    by_language: dict[str, Tally] = field(default_factory=dict)
    pier: Tally = field(default_factory=Tally)
    pier_insertion: Tally = field(default_factory=Tally)
    pier_alternation: Tally = field(default_factory=Tally)
    switch_points: Tally = field(default_factory=Tally)
    # PIER with established borrowings (tier Loanword in the contract's table)
    # taken out of the points of interest, and how many points that removed
    pier_switches: Tally = field(default_factory=Tally)
    borrowing_points: int = 0
    native_tags: int = 0        # English-tagged tokens the table rules Amharic (ቢያንስ): not switches

    def add(self, other: "SegmentCS") -> None:
        self.spans += other.spans
        for lang, t in other.by_language.items():
            self.by_language.setdefault(lang, Tally()).add(t)
        self.pier.add(other.pier)
        self.pier_insertion.add(other.pier_insertion)
        self.pier_alternation.add(other.pier_alternation)
        self.switch_points.add(other.switch_points)
        self.pier_switches.add(other.pier_switches)
        self.borrowing_points += other.borrowing_points
        self.native_tags += other.native_tags


def _utf16_to_index(text: str):
    """Map a UTF-16 code-unit offset into `text` to a Python string index."""
    starts = []  # starts[k]: UTF-16 offset at which Python character k begins
    unit = 0
    for ch in text:
        starts.append(unit)
        unit += 2 if ord(ch) > 0xFFFF else 1
    starts.append(unit)

    def convert(offset: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:  # first character starting at or after the offset
            mid = (lo + hi) // 2
            if starts[mid] < offset:
                lo = mid + 1
            else:
                hi = mid
        return lo

    return convert


ANCHOR_SLACK = 2   # characters a stored offset may be off and still be re-anchored


def _anchor(text: str, span: dict, to_py) -> dict | None:
    """Place a span in `text`, or return None when its text is not there.

    A stored offset a character or two off is re-anchored to span_text within
    ANCHOR_SLACK. Beyond that the span is dropped: an offset that no longer
    addresses its text would otherwise tag whatever word sits there.
    """
    start, end = to_py(span["char_start"]), to_py(span["char_end"])
    want = span.get("span_text")
    if want and text[start:end] != want:
        hits = [m.start() for m in re.finditer(re.escape(want), text) if abs(m.start() - start) <= ANCHOR_SLACK]
        if not hits:
            return None
        start = min(hits, key=lambda h: abs(h - start))
        end = start + len(want)
    return {**span, "start": start, "end": end}


def tag_words(text: str, spans: list[dict], matrix: str) -> list[tuple[str, str, int | None, int]]:
    """Split raw text on whitespace and tag each word.

    Returns (word, language, span index or None, span word count). A word takes
    the language of any span it overlaps. Span offsets are UTF-16 code units
    and are converted.
    """
    to_py = _utf16_to_index(text)
    # The platform export counts UTF-16 code units; the published and carried
    # spans count characters and say so in offset_unit.
    same = lambda v: v  # noqa: E731
    placed = (_anchor(text, s, same if s.get("offset_unit") == "char" else to_py) for s in spans)
    ordered = sorted((s for s in placed if s is not None), key=lambda s: s["start"])
    span_words = [len(WORD.findall(text[s["start"]:s["end"]])) or 1 for s in ordered]
    out = []
    for m in WORD.finditer(text):
        hit = next((i for i, s in enumerate(ordered)
                    if s["start"] < m.end() and m.start() < s["end"]), None)
        if hit is None:
            out.append((m.group(), matrix, None, 0))
        else:
            out.append((m.group(), ordered[hit]["language_code"], hit, span_words[hit]))
    return out


def tagged_reference(words: list[tuple[str, str, int | None, int]], normalized: list[str],
                     contract: NormalizationContract, *, mode: str, wer_n: bool,
                     numerals: str | None) -> list[tuple[str, int | None, int]]:
    """Carry word tags onto the tokens of the whole-text normalization.

    Returns one (language, span key, span word count) per normalized token.
    """
    pieces: list[str] = []
    tags: list[tuple[str, int | None, int]] = []
    for word, lang, span, n in words:
        key = (contract.sha256, word, mode, wer_n, numerals)
        if key not in _WORD_CACHE:
            if len(_WORD_CACHE) >= _WORD_CACHE_MAX:
                _WORD_CACHE.clear()
            _WORD_CACHE[key] = normalize(word, contract, mode=mode, side="reference", wer_n=wer_n,
                                         numerals=numerals).split()
        for tok in _WORD_CACHE[key]:
            pieces.append(tok)
            tags.append((lang, span, n))
    if pieces == normalized:
        return tags
    out: list[tuple[str, int | None, int] | None] = [None] * len(normalized)
    for op in Levenshtein.opcodes(pieces, normalized):
        if op.tag in ("equal", "replace"):
            src = list(range(op.src_start, op.src_end)) or [max(op.src_start - 1, 0)]
            for k, j in enumerate(range(op.dest_start, op.dest_end)):
                out[j] = tags[src[min(k, len(src) - 1)]] if tags else None
    for j in range(len(out)):
        if out[j] is None:
            neighbour = next((out[k] for k in range(j - 1, -1, -1) if out[k] is not None), None)
            out[j] = neighbour or (contract.language, None, 0)
    return out  # type: ignore[return-value]


def segment_counts(ref_tokens: list[str], ref_tags: list[tuple[str, int | None, int]], hyp_tokens: list[str],
                   matrix: str, span_count: int, tier_of=None, phrase_tiers=None) -> SegmentCS:
    """Attribute the alignment's errors to token classes.

    `tier_of(token)` gives the table's tier for an embedded token, or None;
    `phrase_tiers(tokens)` gives {index: tier} for tokens a multi-word entry
    covers, which take that tier instead. Native makes the token matrix
    language throughout; Loanword keeps it a point of interest for PIER but
    removes it from PIER-switches.
    """
    n = len(ref_tokens)
    wrong = [False] * n
    insertions_before = [0] * (n + 1)  # insertions_before[i]: inserted before ref token i
    for op in Levenshtein.editops(ref_tokens, hyp_tokens):
        if op.tag == "insert":
            insertions_before[op.src_pos] += 1
        else:
            wrong[op.src_pos] = True

    seg = SegmentCS(spans=span_count)
    langs = [t[0] for t in ref_tags]
    tiers = [None] * n
    if tier_of is not None:
        by_phrase = phrase_tiers(ref_tokens) if phrase_tiers is not None else {}
        for i in range(n):
            if langs[i] not in (matrix, *NON_SWITCH_TAGS):
                tiers[i] = by_phrase[i] if i in by_phrase else tier_of(ref_tokens[i])
                if tiers[i] == "Native":
                    langs[i] = matrix
                    seg.native_tags += 1
    for i, lang in enumerate(langs):
        seg.by_language.setdefault(lang, Tally()).add(Tally(int(wrong[i]), 1))
    # An insertion is attributed to the language of the token it follows, or the
    # first token when it precedes everything, so the classes partition WER.
    for i, count in enumerate(insertions_before):
        if count and n:
            owner = langs[i - 1] if i > 0 else langs[0]
            seg.by_language.setdefault(owner, Tally()).errors += count
        elif count:
            seg.by_language.setdefault(matrix, Tally()).errors += count

    def switched(i: int) -> bool:
        return langs[i] not in (matrix, *NON_SWITCH_TAGS)

    # Points of interest: embedded tokens and their matrix neighbours. A
    # neighbour belongs to the span class of the embedded token beside it.
    poi: dict[int, int] = {}
    # a point that only an established borrowing makes interesting is not a
    # point for PIER-switches; a neighbour shared with a switch stays
    switch_points_of_interest: set[int] = set()
    for i in range(n):
        if switched(i):
            poi[i] = ref_tags[i][2]
            for j in (i - 1, i + 1):
                if 0 <= j < n and not switched(j):
                    poi.setdefault(j, ref_tags[i][2])
            if tiers[i] != "Loanword":
                switch_points_of_interest.add(i)
                switch_points_of_interest.update(j for j in (i - 1, i + 1) if 0 <= j < n and not switched(j))
    borrowing_only = set(poi) - switch_points_of_interest
    for i, words in poi.items():
        errors = int(wrong[i])
        # insertions directly beside a point of interest count against it once
        errors += insertions_before[i] if (i == 0 or (i - 1) not in poi) else 0
        errors += insertions_before[i + 1]
        t = Tally(errors, 1)
        seg.pier.add(t)
        (seg.pier_insertion if words <= 1 else seg.pier_alternation).add(Tally(errors, 1))
        if i in borrowing_only:
            seg.borrowing_points += 1
        else:
            seg.pier_switches.add(Tally(errors, 1))

    # A switch is a change of language between neighbouring tokens, where
    # `other` counts as the matrix language (it is not a switch into one).
    def side(i: int) -> str:
        return langs[i] if switched(i) else matrix

    boundary = set()
    for i in range(n - 1):
        if side(i) != side(i + 1):
            boundary.update((i, i + 1))
    for j in sorted(boundary):
        seg.switch_points.add(Tally(int(not wrong[j]), 1))
    return seg


def score_item(reference_units: list[tuple[str, list[dict]]], normalized_reference: str, normalized_hypothesis: str,
               contract: NormalizationContract, *, mode: str, wer_n: bool, numerals: str | None) -> SegmentCS:
    """Code-switching counts for one manifest row.

    reference_units is the row's raw reference split into the pieces spans are
    anchored to: one (text, spans) for a single take, one per utterance for a
    panel region whose reference joins several takes with spaces.
    """
    words = []
    span_count = 0
    for k, (text, spans) in enumerate(reference_units):
        for word, lang, span, n in tag_words(text, spans, contract.language):
            words.append((word, lang, None if span is None else k * 10_000 + span, n))
        span_count += len(spans)
    ref_tokens = normalized_reference.split()
    tags = tagged_reference(words, ref_tokens, contract, mode=mode, wer_n=wer_n, numerals=numerals)
    tier_of = phrase_tiers = None
    if contract.loanwords is not None:
        table = contract.loanwords
        phrase_tiers = table.phrase_tiers

        def tier_of(token: str) -> str | None:
            entry = table.entry_of(token, affixed=True)
            return entry.tier if entry else None
    return segment_counts(ref_tokens, tags, normalized_hypothesis.split(), contract.language, span_count,
                          tier_of=tier_of, phrase_tiers=phrase_tiers)


def summarize(total: SegmentCS, *, items: int, items_with_spans: int) -> dict:
    return {
        "items": items,
        "items_with_spans": items_with_spans,
        "spans": total.spans,
        "language_wer": {lang: {"wer": t.rate, "errors": t.errors, "reference_tokens": t.units}
                         for lang, t in sorted(total.by_language.items())},
        "pier": {"rate": total.pier.rate, "errors": total.pier.errors, "points": total.pier.units},
        "pier_insertion": {"rate": total.pier_insertion.rate, "errors": total.pier_insertion.errors,
                           "points": total.pier_insertion.units},
        "pier_alternation": {"rate": total.pier_alternation.rate, "errors": total.pier_alternation.errors,
                             "points": total.pier_alternation.units},
        # PIER over switches only: points that an established borrowing (tier
        # Loanword in the contract table) alone made interesting are removed
        "pier_switches": {"rate": total.pier_switches.rate, "errors": total.pier_switches.errors,
                          "points": total.pier_switches.units, "borrowing_points_removed": total.borrowing_points,
                          "native_tags_reclassified": total.native_tags},
        # switch_points tallies correct tokens in its `errors` slot, so its rate is the accuracy
        "cs_point_accuracy": {"accuracy": total.switch_points.rate, "correct": total.switch_points.errors,
                              "points": total.switch_points.units},
    }


CAVEATS = [
    "Annotators mark the spans at word level in the transcription tool.",
    "No tier or cs_type is stored per span. PIER counts every English-tagged point; pier_switches removes the "
    "points that only an established borrowing made interesting, using the tier the contract's loanword table "
    "gives the word (bare or under known affixes), so a borrowing outside the table is still counted. A "
    "context-bound word (ወይ, ቼክ, ፕላን) is counted as a switch, since the scorer cannot tell its senses apart. "
    "The insertion/alternation split is by span length (one word vs two or more).",
    "MER scores each language in its natural unit, and Ge'ez is written with word spaces, so fidel takes words; "
    "every embedded span in the verified references is fidel, so MER reduces to WER and language_wer is its "
    "decomposition.",
    "A hypothesis that writes an English word in Latin where the reference has fidel is scored as wrong, pending a "
    "transliteration component.",
]
