"""Loading and validation of per-language normalization contracts.

A contract is a YAML file declaring, in order, every transformation the scorer
applies, each with a justification. This module reads it and raises
ContractError on anything it does not recognize or that is incomplete: no
contract for the language, a step this toolkit does not implement, a rule
without a justification, a malformed layer, set or table, a pending rule
without `blocked_by`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Language tables (loanwords, abbreviations), shared by every task. A contract names
# a table by file name; it resolves here. Contracts belong to a task (ethiobench/asr/contracts).
TABLE_DIR = Path(__file__).parent / "tables"

LAYERS = ("base", "fluent", "verbatim")
MODES = ("fluent", "verbatim")
SIDES = ("reference", "hypothesis")

# Steps this toolkit implements, per place they may appear. A contract may name
# a subset in the order it declares; it may not name anything outside this.
KNOWN_STEPS: dict[str, frozenset[str]] = {
    "base": frozenset({
        "unicode_nfc", "strip_zero_width", "strip_bracket_tags", "mark_fragments",
        "expand_abbreviations", "numerals", "remove_punctuation", "casefold", "collapse_whitespace",
    }),
    "fluent": frozenset({"remove_fillers", "remove_fragments"}),
    "verbatim": frozenset({"unmark_fragments", "filler_equivalence"}),
    "wer_n": frozenset({"fold_orthography"}),
    "loanwords": frozenset({"normalize_loanwords"}),
}


class ContractError(Exception):
    """A normalization contract is missing, malformed, or unsupported."""


@dataclass(frozen=True)
class Step:
    """One declared transformation."""

    name: str
    justification: str
    sides: tuple[str, ...] = SIDES
    retains: tuple[str, ...] = ()
    direction: str | None = None  # numerals only: to_digits | to_words | off
    table: str | None = None      # expand_abbreviations only: a TSV in TABLE_DIR


@dataclass(frozen=True)
class EquivalenceSet:
    """Characters treated as interchangeable during WER-N alignment.

    `orders`, when given, lists each member's seven vowel orders explicitly and
    overrides the default of folding by code-point offset. `unfolded` names
    characters the expansion would fold that are ruled not to be: they stay
    distinct and a substitution involving them counts as an error.
    """

    name: str
    members: tuple[str, ...]
    justification: str
    orders: tuple[str, ...] = ()
    unfolded: tuple[str, ...] = ()
    open_question: str | None = None


@dataclass(frozen=True)
class PendingRule:
    """A committed rule the scorer does not yet apply."""

    name: str
    blocked_by: str
    justification: str
    note: str | None = None


@dataclass(frozen=True)
class NormalizationContract:
    """A language's complete, declared normalization contract."""

    rules_sha256: str           # identical across files that differ only in PROSE_KEYS
    language: str
    version: str
    sha256: str
    layers: dict[str, tuple[Step, ...]]
    wer_n_step: Step | None
    equivalence_sets: tuple[EquivalenceSet, ...]
    loanwords: "LoanwordTable | None"
    loanword_step: Step | None
    loanword_profiles: tuple[str, ...]
    abbreviations: "AbbreviationTable | None"
    filler_classes: dict[str, tuple[str, ...]]
    filler_inventory_version: str | None
    pending: tuple[PendingRule, ...]
    conforms_to: dict[str, Any]

    @property
    def stamp(self) -> dict[str, Any]:
        """Provenance recorded with every result computed under this contract."""
        return {
            "language": self.language,
            "version": self.version,
            "sha256": self.sha256,
            "rules_sha256": self.rules_sha256,
            "filler_inventory_version": self.filler_inventory_version,
            "loanwords": ({"table": self.loanwords.path.name, "sha256": self.loanwords.sha256,
                           "entries": len(self.loanwords.entries), "profiles": list(self.loanword_profiles)}
                          if self.loanwords else None),
            "abbreviations": ({"table": self.abbreviations.path.name, "sha256": self.abbreviations.sha256,
                               "entries": len(self.abbreviations)}
                              if self.abbreviations else None),
            "pending": [{"step": p.name, "blocked_by": p.blocked_by} for p in self.pending],
            "provisional": bool(self.pending),
        }

    def has_equivalence_sets(self) -> bool:
        """Whether WER-N differs from WER for this language.

        False when the contract declares no sets; WER-N then equals WER.
        """
        return bool(self.equivalence_sets)


def available_languages(contract_dir: Path) -> tuple[str, ...]:
    """Language codes with a declared contract in `contract_dir`."""
    return tuple(sorted(p.stem for p in contract_dir.glob("*.yaml")))


def load(language: str, contract_dir: Path, table_dir: Path = TABLE_DIR) -> NormalizationContract:
    """Load the contract for `language` from a task's `contract_dir`.

    Raises ContractError if none exists; there is no fallback to another language.
    """
    path = contract_dir / f"{language}.yaml"
    if not path.exists():
        known = ", ".join(available_languages(contract_dir)) or "none"
        raise ContractError(
            f"No normalization contract for language {language!r}. "
            f"Declared: {known}. "
            f"A contract must be written and reviewed by the language's "
            f"linguists before it can be scored; rules are not inherited "
            f"between languages, including within the Ge'ez script."
        )
    data = path.read_bytes()
    raw = yaml.safe_load(data.decode("utf-8"))
    return _parse(raw, source=path, sha256=hashlib.sha256(data).hexdigest(), table_dir=table_dir)


def _justification(entry: dict, what: str, source: Path) -> str:
    text = (entry.get("justification") or "").strip()
    if not text:
        raise ContractError(
            f"{source}: {what} has no justification. Every rule must state why "
            f"it exists; an unexplained rule cannot be argued with, which is the "
            f"failure this contract prevents."
        )
    return text


def _step(entry: Any, layer: str, index: int, source: Path) -> Step:
    if not isinstance(entry, dict) or "step" not in entry:
        raise ContractError(f"{source}: {layer} entry {index} must be a mapping with a 'step' key.")
    name = entry["step"]
    if name not in KNOWN_STEPS[layer]:
        raise ContractError(
            f"{source}: {layer} entry {index} names step {name!r}, which this toolkit "
            f"does not implement in {layer}. A contract may not declare a rule the "
            f"toolkit cannot apply; declare it under 'pending' with what blocks it."
        )
    sides = tuple(entry.get("sides") or SIDES)
    if not set(sides) <= set(SIDES):
        raise ContractError(f"{source}: step {name!r} has unknown sides {sides}.")
    direction = entry.get("direction")
    if name == "numerals":
        from .numerals import DIRECTIONS
        if direction not in DIRECTIONS:
            raise ContractError(f"{source}: numerals step needs a direction from {DIRECTIONS}, got {direction!r}.")
        if set(sides) != set(SIDES):
            raise ContractError(f"{source}: numerals must apply to both sides, or the sides cannot meet.")
    elif direction is not None:
        raise ContractError(f"{source}: step {name!r} takes no direction.")
    table = entry.get("table")
    if name == "expand_abbreviations":
        if not table:
            raise ContractError(f"{source}: expand_abbreviations names no table.")
        if set(sides) != set(SIDES):
            raise ContractError(f"{source}: expand_abbreviations must apply to both sides, or the sides cannot meet.")
    elif table is not None and name != "normalize_loanwords":
        raise ContractError(f"{source}: step {name!r} takes no table.")
    return Step(name=name, justification=_justification(entry, f"step {name!r}", source),
                sides=sides, retains=tuple(entry.get("retains") or ()), direction=direction, table=table)


# Keys that explain or label a rule without changing what it does. Excluded
# from rules_sha256, so an internal and a published contract that differ only
# in these stamp the same rules hash.
PROSE_KEYS = frozenset({"justification", "note", "blocked_by", "conforms_to", "governed_by",
                        "source", "version"})


def _rules_sha256(raw: dict, tables: list[Path]) -> str:
    """sha256 of the contract with PROSE_KEYS removed, plus each table's data
    rows (comment and blank lines excluded)."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in PROSE_KEYS}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    h = hashlib.sha256(json.dumps(strip(raw), ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for p in tables:
        rows = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        h.update(("\n".join(rows) + "\n").encode("utf-8"))
    return h.hexdigest()


def _parse(raw: Any, *, source: Path, sha256: str, table_dir: Path = TABLE_DIR) -> NormalizationContract:
    if not isinstance(raw, dict):
        raise ContractError(f"{source}: expected a mapping at the top level.")
    for required in ("language", "version", "pipeline"):
        if required not in raw:
            raise ContractError(f"{source}: missing required key {required!r}.")
    if not isinstance(raw["pipeline"], dict) or "base" not in raw["pipeline"]:
        raise ContractError(f"{source}: pipeline must be a mapping with at least a 'base' layer.")
    unknown = set(raw["pipeline"]) - set(LAYERS)
    if unknown:
        raise ContractError(f"{source}: unknown pipeline layers {sorted(unknown)}.")

    layers = {
        layer: tuple(_step(e, layer, i, source) for i, e in enumerate(raw["pipeline"].get(layer) or []))
        for layer in LAYERS
    }

    inventory = raw.get("filler_inventory") or {}
    filler_classes = {name: tuple(tokens) for name, tokens in (inventory.get("classes") or {}).items()}
    seen: set[str] = set()
    for name, tokens in filler_classes.items():
        if not tokens or seen & set(tokens):
            raise ContractError(f"{source}: filler class {name!r} is empty or shares a token with another class.")
        seen |= set(tokens)
    uses_fillers = any(s.name in ("remove_fillers", "filler_equivalence")
                       for layer in LAYERS for s in layers[layer])
    if uses_fillers and not filler_classes:
        raise ContractError(f"{source}: filler steps are declared but filler_inventory has no classes.")

    wer_n = raw.get("wer_n") or {}
    wer_n_step = None
    sets: list[EquivalenceSet] = []
    if wer_n:
        if not isinstance(wer_n, dict):
            raise ContractError(f"{source}: wer_n must be a mapping with 'step' and 'equivalence_sets'.")
        wer_n_step = _step(wer_n, "wer_n", 0, source)
        for index, entry in enumerate(wer_n.get("equivalence_sets") or []):
            for required in ("name", "members", "justification"):
                if required not in entry:
                    raise ContractError(f"{source}: equivalence set {index} missing {required!r}.")
            members = tuple(entry["members"])
            if len(members) < 2:
                raise ContractError(
                    f"{source}: equivalence set {entry['name']!r} needs at least two members.")
            orders = tuple(entry.get("orders") or ())
            if orders and (len(orders) != len(members) or len({len(o) for o in orders}) != 1):
                raise ContractError(
                    f"{source}: equivalence set {entry['name']!r} orders must give one "
                    f"equal-length row per member.")
            unfolded = tuple(entry.get("unfolded") or ())
            sets.append(EquivalenceSet(name=entry["name"], members=members,
                                       justification=_justification(entry, f"set {entry['name']!r}", source),
                                       orders=orders, unfolded=unfolded,
                                       open_question=entry.get("open_question")))

    loanword_step, table, profiles = None, None, ()
    lw = raw.get("loanwords") or {}
    if lw:
        from .loanwords import PROFILES, load as load_table
        if not isinstance(lw, dict) or "table" not in lw:
            raise ContractError(f"{source}: loanwords must be a mapping with 'step', 'table' and 'profiles'.")
        loanword_step = _step(lw, "loanwords", 0, source)
        profiles = tuple(lw.get("profiles") or ())
        unknown_profiles = set(profiles) - set(PROFILES)
        if not profiles or unknown_profiles:
            raise ContractError(f"{source}: loanword profiles must be a non-empty subset of {PROFILES}.")
        table_path = table_dir / lw["table"]
        if not table_path.exists():
            raise ContractError(f"{source}: loanword table {lw['table']!r} is not in {table_dir}.")
        table = load_table(table_path)

    abbreviations = None
    for step in layers["base"]:
        if step.name == "expand_abbreviations":
            from .abbreviations import load as load_abbreviations
            table_path = table_dir / step.table
            if not table_path.exists():
                raise ContractError(f"{source}: abbreviation table {step.table!r} is not in {table_dir}.")
            abbreviations = load_abbreviations(table_path)

    pending = []
    for index, entry in enumerate(raw.get("pending") or []):
        if "step" not in entry or "blocked_by" not in entry:
            raise ContractError(f"{source}: pending entry {index} needs 'step' and 'blocked_by'.")
        pending.append(PendingRule(name=entry["step"], blocked_by=entry["blocked_by"],
                                   justification=_justification(entry, f"pending {entry['step']!r}", source),
                                   note=entry.get("note")))

    rules_sha256 = _rules_sha256(raw, [t.path for t in (table, abbreviations) if t is not None])

    return NormalizationContract(
        rules_sha256=rules_sha256,
        language=raw["language"],
        version=str(raw["version"]),
        sha256=sha256,
        layers=layers,
        wer_n_step=wer_n_step,
        equivalence_sets=tuple(sets),
        loanwords=table,
        loanword_step=loanword_step,
        loanword_profiles=profiles,
        abbreviations=abbreviations,
        filler_classes=filler_classes,
        filler_inventory_version=inventory.get("version"),
        pending=tuple(pending),
        conforms_to=dict(raw.get("conforms_to") or {}),
    )
