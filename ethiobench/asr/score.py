"""Score stored hypotheses against a frozen manifest under a contract.

WER and CER per mode (fluent, verbatim) and layer (base, WER-N); CER counts
spaces, as jiwer does. Rates are corpus-level: total edits over total
reference units, not a mean of per-segment rates.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from . import codeswitch as cs
from ..normalize.config import MODES, NormalizationContract
from ..normalize.numerals import rewrite_recorded, spans_for
from ..normalize.pipeline import normalize

LANGUAGE_TAG = re.compile(r"\[([A-Z]{2,4})\]")
# Breakdowns reported where the manifest carries the field. gender and
# final_gender name the same field in different sets.
BREAKDOWN_FIELDS = ("gender", "final_gender", "overlapped")
# Resampling keys: internal name, then published name. Order is the protocol:
# session before speaker, because a panel region can hold several speakers.
SESSION_KEYS = ("panel_session_id", "session")
SPEAKER_KEYS = ("speaker_id", "speaker")
CLUSTER_KEYS = SESSION_KEYS + SPEAKER_KEYS
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260913


class ScoringError(Exception):
    """Hypotheses and manifest do not describe the same segments."""


@dataclass(frozen=True)
class Edits:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def word_edits(reference: str, hypothesis: str) -> Edits:
    ref, hyp = reference.split(), hypothesis.split()
    counts = Counter(op.tag for op in Levenshtein.editops(ref, hyp))
    return Edits(counts["replace"], counts["delete"], counts["insert"], len(ref))


def char_edits(reference: str, hypothesis: str) -> Edits:
    counts = Counter(op.tag for op in Levenshtein.editops(reference, hypothesis))
    return Edits(counts["replace"], counts["delete"], counts["insert"], len(reference))


def under_affix(reference: str, hypothesis: str, table) -> int:
    """Substitutions in the alignment that are a listed spelling difference inside an affixed token."""
    ref, hyp = reference.split(), hypothesis.split()
    return sum(1 for op in Levenshtein.editops(ref, hyp)
               if op.tag == "replace" and table.same_under_affix(ref[op.src_pos], hyp[op.dest_pos]))


def strip_language_tags(text: str) -> tuple[str, list[str]]:
    """Remove system language tags such as [AMH], returning the text and the tags."""
    tags = LANGUAGE_TAG.findall(text)
    return LANGUAGE_TAG.sub(" ", text), tags


def _mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def rate(edits: list[Edits]) -> float | None:
    units = sum(e.reference_units for e in edits)
    return sum(e.errors for e in edits) / units if units else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def segment_id(row: dict) -> str:
    """A row's segment key: `segment_recording_id` (internal) or `segment_id` (published)."""
    sid = row.get("segment_recording_id") or row.get("segment_id")
    if sid is None:
        raise ScoringError(f"record has neither segment_recording_id nor segment_id: {sorted(row)}")
    return sid


def load_hypotheses(path: Path) -> tuple[dict, dict[str, dict], dict | None]:
    """A runner output (typed records, a header naming the system), or a plain
    JSONL of {segment_id, hypothesis}, for which a header with no system is synthesised."""
    header, summary, records, untyped = None, None, {}, []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("record")
            if kind == "header":
                header = row
            elif kind == "summary":
                summary = row
            elif kind == "hypothesis":
                sid = segment_id(row)
                if sid in records:
                    raise ScoringError(f"{path}: two hypotheses for {sid}")
                records[sid] = row
            elif kind is None and "hypothesis" in row:
                untyped.append(row)
            else:
                raise ScoringError(f"{path}: record kind {kind!r} is not header, summary or hypothesis")
    if header is None:
        if records:
            raise ScoringError(f"{path}: typed hypothesis records with no header; not a runner output")
        if not untyped:
            raise ScoringError(f"{path}: no hypotheses")
        for row in untyped:
            sid = segment_id(row)
            if sid in records:
                raise ScoringError(f"{path}: two hypotheses for {sid}")
            records[sid] = row
        header = {"record": "header", "system": {"id": None, "revision": None}, "declared": None}
    elif untyped:
        raise ScoringError(f"{path}: {len(untyped)} untyped record(s) beside a header; "
                           "a runner output types every record")
    return header, records, summary


def load_spans(path: Path) -> dict[str, list[dict]]:
    """Language spans keyed by the row or utterance they annotate.

    Platform export rows: keyed by recording_id, UTF-16 offsets. Span files:
    keyed by segment_recording_id, character offsets into the frozen reference.
    """
    spans: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if "recording_id" in row:
                    spans[row["recording_id"]].append(row)
                else:
                    spans[segment_id(row)].append(
                        {"char_start": row["start"], "char_end": row["end"], "language_code": row["language"],
                         "span_text": row["text"], "offset_unit": "char"})
    return dict(spans)


def _published_spans(row: dict) -> list[dict]:
    """A published row's {start, end, language} spans as {char_start, char_end, language_code}."""
    return [{"char_start": s["start"], "char_end": s["end"], "language_code": s["language"],
             "span_text": row["reference_text"][s["start"]:s["end"]], "offset_unit": "char"}
            for s in (row.get("language_spans") or [])]


def _reference_units(row: dict, spans: dict[str, list[dict]]) -> list[tuple[str, list[dict]]]:
    """The raw texts a row's reference joins, each with its spans: the row's own
    spans on reference_text, else a panel row's utterance spans on the utterance texts."""
    own = spans.get(segment_id(row))
    utts = row.get("utterances") or []
    if own is None and any(segment_id(u) in spans for u in utts):
        return [(u["text"] or "", spans.get(segment_id(u), [])) for u in utts]
    if own is None and "language_spans" in row:
        own = _published_spans(row)
    return [(row["reference_text"], own or [])]


def score(manifest: list[dict], hypotheses: dict[str, dict], contract: NormalizationContract,
          *, strip_tags: bool, numerals: str | None = None,
          spans: dict[str, list[dict]] | None = None) -> tuple[dict, list[dict]]:
    manifest_ids = [segment_id(r) for r in manifest]
    # A spans file or the manifest's own language_spans turns the code-switching metrics on.
    if spans is None and any("language_spans" in r for r in manifest):
        spans = {}
    missing = [i for i in manifest_ids if i not in hypotheses]
    extra = sorted(set(hypotheses) - set(manifest_ids))
    if missing or extra:
        raise ScoringError(
            f"hypotheses and manifest disagree: {len(missing)} manifest segment(s) without a "
            f"hypothesis (e.g. {missing[:3]}), {len(extra)} hypothesis(es) for segments not in "
            f"the manifest (e.g. {extra[:3]}). A partial run is not scored as if it were whole.")

    per_segment = []
    edits: dict[str, list[Edits]] = defaultdict(list)
    profiles = contract.loanword_profiles if contract.loanwords is not None else ()
    affix_flags: Counter = Counter()
    cs_counts: dict[str, list[cs.SegmentCS]] = defaultdict(list)
    cs_has_spans: list[bool] = []
    tag_counts: Counter = Counter()
    # as_recorded: digit spans counted on the raw hypothesis with normalize()'s rewrite.
    direction = numerals or next((s.direction for s in contract.layers["base"] if s.name == "numerals"), None)
    number_counts = Counter()
    for row in manifest:
        sid = segment_id(row)
        record = hypotheses[sid]
        raw_hyp = record["hypothesis"] or ""
        hyp_text = raw_hyp
        if strip_tags:
            hyp_text, tags = strip_language_tags(raw_hyp)
            tag_counts.update(tags)
        seg = {"segment_recording_id": sid, **{f: row.get(f) for f in BREAKDOWN_FIELDS},
               "hit_token_cap": record.get("hit_token_cap"), "raw_hypothesis": raw_hyp}
        spans_here = None
        if direction == "as_recorded":
            recorded = row.get("number_spans") or []
            # Recorded pairs plus parser-derived pairs for unrecorded number words.
            spans_here = spans_for(row["reference_text"], contract.language, recorded)
            _, matched, partial, left = rewrite_recorded(hyp_text, spans_here, contract.language)
            number_counts["reference_spans_recorded"] += len(recorded)
            number_counts["reference_spans_derived"] += len(spans_here) - len(recorded)
            number_counts["hypothesis_digit_spans_matched"] += matched
            number_counts["hypothesis_digit_spans_matched_up_to_affix"] += partial
            number_counts["hypothesis_digit_tokens_unmatched"] += left
            seg["numbers"] = {"reference_spans_recorded": len(recorded), "reference_spans_derived": len(spans_here) - len(recorded),
                              "digit_spans_matched": matched, "digit_spans_matched_up_to_affix": partial,
                              "digit_tokens_unmatched": left}
        for mode in MODES:
            # Equivalence layer: off, character fold, fold plus loanwords per profile.
            for wer_n, profile in [(False, None), (True, None)] + [(True, p) for p in profiles]:
                key = f"{mode}{'_n' if wer_n else ''}{'_' + profile if profile else ''}"
                ref = normalize(row["reference_text"], contract, mode=mode, side="reference", wer_n=wer_n,
                                numerals=numerals, loanword_profile=profile)
                hyp = normalize(hyp_text, contract, mode=mode, side="hypothesis", wer_n=wer_n, numerals=numerals,
                                loanword_profile=profile, number_spans=spans_here)
                w, c = word_edits(ref, hyp), char_edits(ref, hyp)
                edits[f"{key}:word"].append(w)
                edits[f"{key}:char"].append(c)
                seg[key] = {"reference": ref, "hypothesis": hyp,
                            "word": [w.substitutions, w.deletions, w.insertions, w.reference_units],
                            "char": [c.substitutions, c.deletions, c.insertions, c.reference_units]}
                if profile == "lenient":
                    # Listed spellings inside affixed tokens: counted, not folded.
                    flagged = under_affix(ref, hyp, contract.loanwords)
                    affix_flags[mode] += flagged
                    seg[key]["spelling_under_affix"] = flagged
                if profile is not None:
                    continue
                if spans is not None and mode == "fluent":
                    units = _reference_units(row, spans)
                    counts = cs.score_item(units, ref, hyp, contract, mode=mode, wer_n=wer_n, numerals=numerals)
                    cs_counts[key].append(counts)
                    if not wer_n:
                        cs_has_spans.append(any(s for _, s in units))
        per_segment.append(seg)

    def summarize(indices: list[int]) -> dict:
        out = {}
        for mode in MODES:
            wer = rate([edits[f"{mode}:word"][i] for i in indices])
            wer_n = rate([edits[f"{mode}_n:word"][i] for i in indices])
            out[mode] = {
                "wer": wer, "wer_n": wer_n,
                "cer": rate([edits[f"{mode}:char"][i] for i in indices]),
                "cer_n": rate([edits[f"{mode}_n:char"][i] for i in indices]),
                "orthographic_gap": (wer - wer_n) if wer is not None and wer_n is not None else None,
            }
            # Per profile: WER-N with loanwords, and the gap the table alone makes.
            for p in profiles:
                v = rate([edits[f"{mode}_n_{p}:word"][i] for i in indices])
                out[mode][f"wer_n_{p}"] = v
                out[mode][f"loanword_gap_{p}"] = (wer_n - v) if wer_n is not None and v is not None else None
        return out

    everything = list(range(len(manifest)))
    totals = {}
    for key, rows in edits.items():
        totals[key] = {"substitutions": sum(e.substitutions for e in rows),
                       "deletions": sum(e.deletions for e in rows),
                       "insertions": sum(e.insertions for e in rows),
                       "reference_units": sum(e.reference_units for e in rows)}

    breakdowns = {}
    present = [f for f in BREAKDOWN_FIELDS if any(f in row for row in manifest)]
    for field in present:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(manifest):
            groups[str(row.get(field))].append(i)
        breakdowns[field] = {g: {"segments": len(idx), **summarize(idx)} for g, idx in sorted(groups.items())}

    # Resampling unit: the first CLUSTER_KEYS key every row carries, else the segment.
    unit = next((k for k in CLUSTER_KEYS if all(row.get(k) for row in manifest)), None)
    clusters: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(manifest):
        clusters[str(row[unit]) if unit else str(i)].append(i)
    cluster_ids = sorted(clusters)

    macro = None
    if unit in SPEAKER_KEYS:
        per_speaker = {c: summarize(idx) for c, idx in clusters.items()}
        macro = {mode: {m: _mean([per_speaker[c][mode][m] for c in cluster_ids])
                        for m in ("wer", "wer_n", "cer", "cer_n")} for mode in MODES}

    code_switching = None
    cs_series: dict[str, list[tuple[int, int]]] = {}
    if spans is not None:
        items_with_spans = sum(cs_has_spans)
        code_switching = {"caveats": cs.CAVEATS, "items_with_spans": items_with_spans}
        if items_with_spans == 0:
            code_switching["not_computed"] = "No reference in this set carries language spans."
        else:
            for key, rows in cs_counts.items():
                total = cs.SegmentCS()
                for c in rows:
                    total.add(c)
                code_switching[key] = cs.summarize(total, items=len(rows), items_with_spans=items_with_spans)
                cs_series[f"{key}_pier"] = [(c.pier.errors, c.pier.units) for c in rows]
                cs_series[f"{key}_pier_switches"] = [(c.pier_switches.errors, c.pier_switches.units) for c in rows]
                for lang in total.by_language:
                    if lang != contract.language:
                        cs_series[f"{key}_wer_{lang}"] = [
                            (c.by_language[lang].errors, c.by_language[lang].units) if lang in c.by_language
                            else (0, 0) for c in rows]

    rng = random.Random(BOOTSTRAP_SEED)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPS):
        chosen = [cluster_ids[rng.randrange(len(cluster_ids))] for _ in cluster_ids]
        sample = [i for c in chosen for i in clusters[c]]
        for name, series in cs_series.items():
            units = sum(series[i][1] for i in sample)
            if units:
                draws[f"cs_{name}"].append(sum(series[i][0] for i in sample) / units)
        for mode in MODES:
            for suffix in ("", "_n", *(f"_n_{p}" for p in profiles)):
                value = rate([edits[f"{mode}{suffix}:word"][i] for i in sample])
                if value is not None:
                    draws[f"{mode}_wer{suffix}"].append(value)
            if macro is not None:
                value = _mean([rate([edits[f"{mode}:word"][i] for i in clusters[c]]) for c in chosen])
                if value is not None:
                    draws[f"{mode}_wer_macro"].append(value)
    ci = {}
    for key, values in draws.items():
        values.sort()
        ci[key] = [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]

    speaker_caveat = "Speakers are resampled with their recordings (cluster bootstrap)."
    session_caveat = ("Sessions are resampled with their regions; a region can hold several speakers, "
                      "so the session is the smallest independent unit.")
    caveats = {k: speaker_caveat for k in SPEAKER_KEYS}
    caveats.update({k: session_caveat for k in SESSION_KEYS})
    caveats[None] = ("Segments are resampled independently. The manifest carries no speaker identity, "
                     "so between-speaker variance is not captured and the interval is narrower than a "
                     "speaker-level bootstrap would give.")
    n = len(manifest)
    report = {
        "segments": n,
        "metrics": summarize(everything),
        "metrics_macro_over_speakers": macro,
        "totals": totals,
        "ci95": ci,
        "bootstrap": {
            "unit": unit or "segment", "clusters": len(cluster_ids), "reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED, "caveat": caveats[unit],
        },
        "breakdowns": breakdowns,
        "loanwords": ({
            "profiles": list(profiles),
            # Lenient-profile substitutions that are a listed spelling inside an
            # affixed token (በሚኒስተር for በሚኒስቴር). Still scored as substitutions.
            "spelling_under_affix": {mode: {"substitutions_flagged": affix_flags[mode],
                                            "substitutions": totals[f"{mode}_n_lenient:word"]["substitutions"]}
                                     for mode in MODES},
        } if profiles else None),
        "code_switching": code_switching,
        "numbers": ({"direction": direction, **number_counts} if direction == "as_recorded" else None),
        "system_output": {
            "hit_token_cap": sum(1 for s in per_segment if s["hit_token_cap"]),
            "empty_hypotheses": sum(1 for s in per_segment if not s["fluent"]["hypothesis"]),
            "language_tags": dict(tag_counts),
        },
    }
    return report, per_segment
