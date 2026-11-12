"""Alignment counts and corpus aggregation."""

import itertools
import random

import pytest

from ethiobench.asr import load_contract
from ethiobench.asr import score as sc


def dp_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def test_word_edits_match_reference_dp():
    rng = random.Random(0)
    vocab = ["ሰላም", "ነው", "እንዴት", "ናችሁ", "ዛሬ"]
    for _ in range(300):
        ref = [rng.choice(vocab) for _ in range(rng.randrange(0, 9))]
        hyp = [rng.choice(vocab) for _ in range(rng.randrange(0, 9))]
        e = sc.word_edits(" ".join(ref), " ".join(hyp))
        assert e.errors == dp_distance(ref, hyp)
        assert e.reference_units == len(ref)


def test_counts_are_split_correctly():
    e = sc.word_edits("a b c d", "a x c d e")
    assert (e.substitutions, e.deletions, e.insertions) == (1, 0, 1)
    e = sc.word_edits("a b c", "a c")
    assert (e.substitutions, e.deletions, e.insertions) == (0, 1, 0)


def test_cer_counts_spaces():
    e = sc.char_edits("ab cd", "abcd")
    assert e.errors == 1 and e.reference_units == 5


def test_language_tags_stripped_even_when_fused():
    text, tags = sc.strip_language_tags("[AMH] ለዛሬ ሁለት አጀንዳዎች[AMH]")
    assert text.split() == ["ለዛሬ", "ሁለት", "አጀንዳዎች"] and tags == ["AMH", "AMH"]


def manifest_row(sid, text, gender="female"):
    return {"segment_recording_id": sid, "reference_text": text, "final_gender": gender,
            "parent_source_label": "p", "reference_producer": "human"}


def test_corpus_rate_weights_by_length():
    amh = load_contract("amh")
    manifest = [manifest_row("a", "ሰላም"), manifest_row("b", "አንድ ሁለት ሶስት አራት አምስት ስድስት ሰባት ስምንት ዘጠኝ")]
    hyps = {"a": {"hypothesis": "ነው"}, "b": {"hypothesis": "አንድ ሁለት ሶስት አራት አምስት ስድስት ሰባት ስምንት ዘጠኝ"}}
    report, _ = sc.score(manifest, hyps, amh, strip_tags=False)
    assert report["metrics"]["fluent"]["wer"] == pytest.approx(1 / 10)


def test_partial_run_is_refused():
    amh = load_contract("amh")
    with pytest.raises(sc.ScoringError, match="without a hypothesis"):
        sc.score([manifest_row("a", "ሰላም"), manifest_row("b", "ነው")], {"a": {"hypothesis": "ሰላም"}},
                 amh, strip_tags=False)


def test_wer_n_gap_is_spelling_only():
    amh = load_contract("amh")
    report, segs = sc.score([manifest_row("a", "ሐሙስ ነው")], {"a": {"hypothesis": "ሀሙስ ነው"}},
                            amh, strip_tags=False)
    fluent = report["metrics"]["fluent"]
    assert fluent["wer"] == 0.5 and fluent["wer_n"] == 0.0 and fluent["orthographic_gap"] == 0.5


def test_speaker_macro_average_and_cluster_bootstrap():
    amh = load_contract("amh")
    manifest = [
        {"segment_recording_id": "a", "reference_text": "ሰላም ነው", "speaker_id": "s1"},
        {"segment_recording_id": "b", "reference_text": "ሰላም ነው አሁን ደግሞ እዚህ", "speaker_id": "s1"},
        {"segment_recording_id": "c", "reference_text": "ሰላም", "speaker_id": "s2"},
    ]
    hyps = {"a": {"hypothesis": "ሰላም ነው"}, "b": {"hypothesis": "ሰላም ነው አሁን ደግሞ እዚህ"}, "c": {"hypothesis": "ነው"}}
    report, _ = sc.score(manifest, hyps, amh, strip_tags=False)
    assert report["metrics"]["fluent"]["wer"] == pytest.approx(1 / 8)          # micro: 1 error / 8 words
    assert report["metrics_macro_over_speakers"]["fluent"]["wer"] == pytest.approx(0.5)  # (0 + 1) / 2
    assert report["bootstrap"]["unit"] == "speaker_id" and report["bootstrap"]["clusters"] == 2
    assert "fluent_wer_macro" in report["ci95"]


def test_panel_clusters_on_session_and_segment_otherwise():
    amh = load_contract("amh")
    panel = [{"segment_recording_id": "r1", "reference_text": "ሰላም", "panel_session_id": "S1"},
             {"segment_recording_id": "r2", "reference_text": "ነው", "panel_session_id": "S2"}]
    report, _ = sc.score(panel, {"r1": {"hypothesis": "ሰላም"}, "r2": {"hypothesis": "ነው"}}, amh, strip_tags=False)
    assert report["bootstrap"]["unit"] == "panel_session_id" and report["metrics_macro_over_speakers"] is None
    plain = [manifest_row("a", "ሰላም")]
    report, _ = sc.score(plain, {"a": {"hypothesis": "ሰላም"}}, amh, strip_tags=False)
    assert report["bootstrap"]["unit"] == "segment"


def test_released_field_names_cluster_the_same_as_internal_ones():
    """Released rows say `speaker`, internal rows `speaker_id`; both cluster the bootstrap.

    A missed name falls back to segment resampling and drops the speaker macro-average.
    """
    amh = load_contract("amh")
    rows = [("a", "ሰላም ነው", "ሰላም ነው"), ("b", "ሰላም ነው አሁን ደግሞ እዚህ", "ሰላም ነው አሁን ደግሞ እዚህ"),
            ("c", "ሰላም", "ነው")]
    hyps = {i: {"hypothesis": h} for i, _, h in rows}
    out = {}
    for key, spk in (("speaker_id", {"a": "s1", "b": "s1", "c": "s2"}),
                     ("speaker", {"a": "amh-spk-1", "b": "amh-spk-1", "c": "amh-spk-2"})):
        manifest = [{"segment_recording_id": i, "reference_text": r, key: spk[i]} for i, r, _ in rows]
        out[key], _ = sc.score(manifest, hyps, amh, strip_tags=False)
        assert out[key]["bootstrap"]["unit"] == key
        assert out[key]["bootstrap"]["clusters"] == 2
    assert (out["speaker"]["metrics_macro_over_speakers"]
            == out["speaker_id"]["metrics_macro_over_speakers"])


def test_a_panel_row_clusters_on_its_session_not_its_speaker():
    """Panels resample sessions, other sets speakers.

    A panel region can hold several speakers, so the session is the independent unit.
    """
    amh = load_contract("amh")
    panel = [{"segment_recording_id": "r1", "reference_text": "ሰላም",
              "speaker": "amh-spk-1", "session": "amh-ses-1"},
             {"segment_recording_id": "r2", "reference_text": "ነው",
              "speaker": "amh-spk-2", "session": "amh-ses-1"}]
    hyps = {"r1": {"hypothesis": "ሰላም"}, "r2": {"hypothesis": "ነው"}}
    report, _ = sc.score(panel, hyps, amh, strip_tags=False)
    assert report["bootstrap"]["unit"] == "session"
    assert report["bootstrap"]["clusters"] == 1
    assert report["metrics_macro_over_speakers"] is None

    read = [{**r, "session": None} for r in panel]
    report, _ = sc.score(read, hyps, amh, strip_tags=False)
    assert report["bootstrap"]["unit"] == "speaker"
    assert report["bootstrap"]["clusters"] == 2


def test_fluent_ignores_suppressed_fillers_verbatim_counts_them():
    amh = load_contract("amh")
    report, _ = sc.score([manifest_row("a", "እ ሰላም ነው")], {"a": {"hypothesis": "ሰላም ነው"}},
                         amh, strip_tags=False)
    assert report["metrics"]["fluent"]["wer"] == 0.0
    assert report["metrics"]["verbatim"]["wer"] == pytest.approx(1 / 3)


def test_numbers_as_recorded_and_the_digit_count():
    amh = load_contract("amh")
    row = {**manifest_row("a", "አንቀጽ ዘጠኝ ሰማንያ አምስት ላይ ሁለት ሰዓት ተኩል"),
           "number_spans": [{"digits": "985", "words": "ዘጠኝ ሰማንያ አምስት"}, {"digits": "2:30", "words": "ሁለት ሰዓት ተኩል"}]}
    hyps = {"a": {"hypothesis": "አንቀጽ 985 ላይ ሁለት ከሰላሳ"}}
    report, segs = sc.score([row], hyps, amh, strip_tags=False)
    # 985 is the recorded digit form and matches; ሁለት ከሰላሳ is another reading
    # of 2:30 and scores as the substitutions it is
    assert report["metrics"]["fluent"]["wer"] == pytest.approx(2 / 8)
    assert report["numbers"] == {"direction": "as_recorded", "reference_spans_recorded": 2, "reference_spans_derived": 0,
                                 "hypothesis_digit_spans_matched": 1, "hypothesis_digit_spans_matched_up_to_affix": 0,
                                 "hypothesis_digit_tokens_unmatched": 0}
    assert segs[0]["numbers"]["digit_spans_matched"] == 1
    # a wrong value in digits stays a digit token (one substitution and two
    # deletions against the three words said) and is counted as unmatched
    report, _ = sc.score([row], {"a": {"hypothesis": "አንቀጽ 985 ላይ 2:33"}}, amh, strip_tags=False)
    assert report["metrics"]["fluent"]["wer"] == pytest.approx(3 / 8)
    assert report["numbers"]["hypothesis_digit_tokens_unmatched"] == 1
    # a reference written in words from the start: the parser's digit form
    # meets a digit hypothesis (derived span); a prefix that differs is one
    # substitution, as it would be between words
    row2 = manifest_row("b", "ከአምስት መቶ በላይ ወጣቶች")
    report, _ = sc.score([row2], {"b": {"hypothesis": "ከ500 በላይ ወጣቶች"}}, amh, strip_tags=False)
    assert report["metrics"]["fluent"]["wer"] == 0.0
    assert report["numbers"]["reference_spans_derived"] == 1 and report["numbers"]["hypothesis_digit_spans_matched"] == 1
    report, _ = sc.score([row2], {"b": {"hypothesis": "500 በላይ ወጣቶች"}}, amh, strip_tags=False)
    assert report["metrics"]["fluent"]["wer"] == pytest.approx(1 / 4)
    assert report["numbers"]["hypothesis_digit_spans_matched_up_to_affix"] == 1


def test_loanword_profiles_and_the_under_affix_flag():
    amh = load_contract("amh")
    rows = [manifest_row("a", "ፕሬዝዳንት መጣ"), manifest_row("b", "በሚኒስቴር ተባለ"), manifest_row("c", "ሚልየን ብር")]
    hyps = {"a": {"hypothesis": "ፕረዘዳንት መጣ"}, "b": {"hypothesis": "በሚኒስተር ተባለ"}, "c": {"hypothesis": "ሚልየን ብር"}}
    report, segs = sc.score(rows, hyps, amh, strip_tags=False)
    f = report["metrics"]["fluent"]
    # six reference words; ፕረዘዳንት and በሚኒስተር are the two errors under WER-N
    assert f["wer_n"] == pytest.approx(2 / 6)
    # lenient folds ፕረዘዳንት (generated by the pattern), not the affixed በሚኒስተር
    assert f["wer_n_lenient"] == pytest.approx(1 / 6)
    # strict folds neither: ፕረዘዳንት is not an established spelling
    assert f["wer_n_strict"] == pytest.approx(2 / 6)
    assert f["loanword_gap_lenient"] == pytest.approx(1 / 6) and f["loanword_gap_strict"] == 0
    flags = report["loanwords"]["spelling_under_affix"]["fluent"]
    assert flags == {"substitutions_flagged": 1, "substitutions": 1}
    assert report["ci95"]["fluent_wer_n_lenient"] is not None
