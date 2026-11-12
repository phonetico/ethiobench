"""A contract's abbreviation table: written abbreviations rewritten as spoken.

A TSV beside the contract: `written` (a regex over the written forms of one
abbreviation), `spoken`, `note`. Each pattern matches inside word boundaries,
so a one-letter prefix may attach and stays attached to the expansion
(በዓ.ም → በዓመተ ምህረት) and no fidel may follow except a nominal suffix, keeping
ዓም out of ዓምድ. A suffix's vowel change is carried onto an expansion ending in
the same consonant (ት/ቤቱ → ትምህርት ቤቱ); otherwise the form is left as written
rather than half-expanded.

Both sides are rewritten, so a system writing ዓ.ም against a reference reading
ዓመተ ምህረት is not counted as two errors.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_FIDEL = "ሀ-፿"
_PREFIXES = "የበለከ"
# Nominal suffixes that ride on an expanded tail, longest first so ንም is
# tried before ን. Vowel-order suffixes (ቤት + ኡ → ቤቱ) are the carry rule's.
_SUFFIXES = ("ያቸው", "አቸው", "ዎች", "ዎቹ", "ችን", "ችሁ", "ውን", "ውም", "ንም",
             "ን", "ው", "ዉ", "ም", "ና", "ች", "ቹ")


def _row(ch: str) -> int:
    """Fidel row (consonant) of a syllable, or -1 for anything else."""
    o = ord(ch)
    return (o - 0x1200) // 8 if 0x1200 <= o <= 0x137F else -1


def _orders(ch: str) -> str:
    """Character class of every vowel order of `ch`'s consonant."""
    start = 0x1200 + _row(ch) * 8
    return "[" + "".join(chr(start + i) for i in range(8)) + "]"


class AbbreviationTable:
    """The rows of one abbreviation table, compiled to a single matcher."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.entries: list[tuple[str, str, str]] = []      # (written pattern, spoken, note)
        header = None
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.startswith("#"):
                continue
            cells = raw.rstrip("\n").split("\t")
            if header is None:
                header = cells
                if header[:2] != ["written", "spoken"]:
                    raise ValueError(f"{self.path}: expected columns written, spoken[, note]; got {header}")
                continue
            written, spoken = cells[0], cells[1]
            note = cells[2] if len(cells) > 2 else ""
            re.compile(written)                        # a malformed row fails at load, not at scoring
            self.entries.append((written, spoken, note))
        if not self.entries:
            raise ValueError(f"{self.path}: no abbreviations")
        self._carry: dict[str, str] = {}               # group name → expansion whose last letter takes the observed order
        alternatives = []
        for i, (written, spoken, _) in enumerate(self.entries):
            name = f"r{i}"
            last = written[-1]
            if _row(last) >= 0 and _row(last) == _row(spoken[-1]) and not written.endswith("\\" + last):
                written = written[:-1] + _orders(last)
                self._carry[name] = spoken
            alternatives.append(f"(?P<{name}>{written})")
        self._pattern = re.compile(
            f"(?<![{_FIDEL}])(?P<prefix>[{_PREFIXES}]?)(?:" + "|".join(alternatives) + ")"
            f"(?P<suffix>{'|'.join(_SUFFIXES)})?(?![{_FIDEL}])"
        )

    def expand(self, text: str) -> str:
        def repl(m: re.Match) -> str:
            name = next(g for g in m.groupdict() if g.startswith("r") and m.group(g) is not None)
            spoken = self.entries[int(name[1:])][1]
            if name in self._carry:
                spoken = spoken[:-1] + m.group(name)[-1]
            # A marker glued to its number (2008ዓም) is two spoken words; the
            # space lets the numerals step see the number on its own.
            glued = m.start() > 0 and text[m.start() - 1].isdigit() and not m.group("prefix")
            return (" " if glued else "") + (m.group("prefix") or "") + spoken + (m.group("suffix") or "")
        return self._pattern.sub(repl, text)

    def __len__(self) -> int:
        return len(self.entries)


def load(path: Path) -> AbbreviationTable:
    return AbbreviationTable(path)
