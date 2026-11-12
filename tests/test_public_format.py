"""The published toolkit scores the published manifest and a plain hypothesis file."""

import json

import pytest

from ethiobench.asr import cli, load_contract
from ethiobench.asr import score as sc


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def published_row(sid, text, spans=None):
    row = {"segment_id": sid, "locale": "amh", "condition": "public_broadcast",
           "reference_text": text, "number_spans": []}
    if spans is not None:
        row["language_spans"] = spans
    return row


def test_segment_id_accepts_either_name():
    assert sc.segment_id({"segment_id": "a"}) == "a"
    assert sc.segment_id({"segment_recording_id": "b"}) == "b"
    with pytest.raises(sc.ScoringError):
        sc.segment_id({"text": "x"})


def test_plain_hypotheses_load_without_a_header(tmp_path):
    p = tmp_path / "h.jsonl"
    write_jsonl(p, [{"segment_id": "a", "hypothesis": "ሰላም"}, {"segment_id": "b", "hypothesis": "ነው"}])
    header, records, summary = sc.load_hypotheses(p)
    assert header["system"]["id"] is None
    assert set(records) == {"a", "b"} and summary is None


def test_typed_records_without_a_header_are_refused(tmp_path):
    p = tmp_path / "h.jsonl"
    write_jsonl(p, [{"record": "hypothesis", "segment_recording_id": "a", "hypothesis": "x"}])
    with pytest.raises(sc.ScoringError, match="no header"):
        sc.load_hypotheses(p)


def test_duplicate_plain_hypothesis_is_refused(tmp_path):
    p = tmp_path / "h.jsonl"
    write_jsonl(p, [{"segment_id": "a", "hypothesis": "x"}, {"segment_id": "a", "hypothesis": "y"}])
    with pytest.raises(sc.ScoringError, match="two hypotheses"):
        sc.load_hypotheses(p)


def test_published_manifest_scores_end_to_end(tmp_path):
    m, h, out = tmp_path / "m.jsonl", tmp_path / "h.jsonl", tmp_path / "out"
    write_jsonl(m, [published_row("s1", "ሰላም ነው"), published_row("s2", "እንዴት ናችሁ")])
    write_jsonl(h, [{"segment_id": "s1", "hypothesis": "ሰላም ነው"}, {"segment_id": "s2", "hypothesis": "እንዴት"}])
    code = cli.main(["score", "--language", "amh", "--manifest", str(m), "--hypotheses", str(h), "--out", str(out)])
    assert code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["stamp"]["system_declaration_checked"] is False
    assert report["metrics"]["fluent"]["wer"] == pytest.approx(1 / 4)


def test_an_output_made_against_another_set_version_is_refused(tmp_path):
    m, h, s = tmp_path / "m.jsonl", tmp_path / "h.jsonl", tmp_path / "systems.yaml"
    write_jsonl(m, [{**published_row("s1", "ሰላም"), "reference_revision": "1.1"}])
    s.write_text("systems:\n  - id: sys\n    revision: abc\n", encoding="utf-8")
    for made_for, code in (("1.0", 2), ("1.1", 0)):
        write_jsonl(h, [{"record": "header", "system": {"id": "sys", "revision": "abc"},
                         "reference_revision": made_for},
                        {"record": "hypothesis", "segment_id": "s1", "hypothesis": "ሰላም"},
                        {"record": "summary", "complete": True}])
        assert cli.main(["score", "--language", "amh", "--manifest", str(m), "--hypotheses", str(h),
                         "--systems", str(s), "--out", str(tmp_path / f"o{made_for}")]) == code


def test_systems_yaml_is_refused_for_a_headerless_file(tmp_path):
    m, h, s = tmp_path / "m.jsonl", tmp_path / "h.jsonl", tmp_path / "systems.yaml"
    write_jsonl(m, [published_row("s1", "ሰላም")])
    write_jsonl(h, [{"segment_id": "s1", "hypothesis": "ሰላም"}])
    s.write_text("systems: []\n", encoding="utf-8")
    assert cli.main(["score", "--language", "amh", "--manifest", str(m), "--hypotheses", str(h),
                     "--systems", str(s), "--out", str(tmp_path / "o")]) == 2


def test_leading_language_tag_is_an_error_unless_stripped(tmp_path):
    m, h = tmp_path / "m.jsonl", tmp_path / "h.jsonl"
    write_jsonl(m, [published_row("s1", "ሰላም ነው")])
    write_jsonl(h, [{"segment_id": "s1", "hypothesis": "[AMH] ሰላም ነው"}])
    wer = {}
    for strip in (False, True):
        out = tmp_path / f"o{strip}"
        argv = ["score", "--language", "amh", "--manifest", str(m), "--hypotheses", str(h), "--out", str(out)]
        assert cli.main(argv + (["--strip-language-tags"] if strip else [])) == 0
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        assert report["stamp"]["language_tags_stripped"] is strip
        wer[strip] = report["metrics"]["fluent"]["wer"]
    assert wer[True] == 0 and wer[False] > 0


def test_embedded_language_spans_enable_code_switching():
    amh = load_contract("amh")
    text = "ስብሰባው ላይ ሚቲንግ ነበር"
    start = text.index("ሚቲንግ")
    row = published_row("s1", text, [{"start": start, "end": start + len("ሚቲንግ"), "language": "eng"}])
    report, _ = sc.score([row], {"s1": {"hypothesis": text}}, amh, strip_tags=False)
    cs = report["code_switching"]["fluent"]
    assert report["code_switching"]["items_with_spans"] == 1
    assert cs["language_wer"]["eng"]["wer"] == 0


def test_a_manifest_without_spans_computes_no_code_switching():
    amh = load_contract("amh")
    report, _ = sc.score([published_row("s1", "ሰላም ነው")], {"s1": {"hypothesis": "ሰላም ነው"}}, amh,
                         strip_tags=False)
    assert report["code_switching"] is None
