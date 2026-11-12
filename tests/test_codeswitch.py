"""Code-switching metrics over language spans."""

from ethiobench.asr import codeswitch as cs
from ethiobench.asr import load_contract
from ethiobench.asr import score as sc
from ethiobench.normalize.pipeline import normalize

AMH = load_contract("amh")


def span(text, word, lang="eng"):
    start = text.index(word)
    return {"char_start": start, "char_end": start + len(word), "language_code": lang, "span_text": word}


def item(text, spans, hyp, contract=AMH):
    ref = normalize(text, contract, mode="fluent", side="reference")
    h = normalize(hyp, contract, mode="fluent", side="hypothesis")
    return cs.score_item([(text, spans)], ref, h, contract, mode="fluent", wer_n=False, numerals=None)


def test_tags_follow_span_offsets():
    text = "ስብሰባው ላይ ሚቲንግ ነበር"
    words = cs.tag_words(text, [span(text, "ሚቲንግ")], "amh")
    assert [w[1] for w in words] == ["amh", "amh", "eng", "amh"]


def test_offsets_are_utf16_code_units():
    # an emoji is two UTF-16 code units, so the platform's offsets after it run one ahead
    text = "ሰላም 😀 ሚቲንግ ነበር"
    py = text.index("ሚቲንግ")
    spans = [{"char_start": py + 1, "char_end": py + 1 + len("ሚቲንግ"), "language_code": "eng"}]
    assert [w[1] for w in cs.tag_words(text, spans, "amh")] == ["amh", "amh", "eng", "amh"]


def test_stale_offsets_reanchor_on_span_text():
    text = "ሶስት አለ አንደርስታንድ ላይ"
    start = text.index("አንደርስታንድ") - 1  # one character early, as the data has it
    spans = [{"char_start": start, "char_end": start + len("አንደርስታንድ"), "language_code": "eng",
              "span_text": "አንደርስታንድ"}]
    assert [w[1] for w in cs.tag_words(text, spans, "amh")] == ["amh", "amh", "eng", "amh"]


def test_a_span_whose_text_is_gone_tags_nothing():
    # Span annotated on "ዙ 23 3ዲ 120"; the reference spells the numbers out, so
    # the stored offset lands inside ሶስት. Nothing may be tagged.
    text = "ዙ ሃያ ሶስት ሲሪ ዲ መቶ ሃያ"
    spans = [{"char_start": 5, "char_end": 7, "language_code": "eng", "span_text": "3ዲ"}]
    assert all(w[1] == "amh" for w in cs.tag_words(text, spans, "amh"))


def test_a_far_occurrence_is_not_taken_for_the_annotated_one():
    # The same word twice; a stored offset more than ANCHOR_SLACK from either
    # occurrence is not re-anchored to the nearer one.
    text = "ኦዲት ተደርጓል ከዚያም ብዙ ነገር ተባለ ኦዲት"
    spans = [{"char_start": 12, "char_end": 15, "language_code": "eng", "span_text": "ኦዲት"}]
    assert all(w[1] == "amh" for w in cs.tag_words(text, spans, "amh"))


def test_character_offsets_are_not_converted_as_utf16():
    # After an emoji a UTF-16 offset runs one ahead; a span counted in
    # characters must be read as it is.
    text = "ሰላም 😀 ሚቲንግ ነበር"
    py = text.index("ሚቲንግ")
    spans = [{"char_start": py, "char_end": py + len("ሚቲንግ"), "language_code": "eng", "offset_unit": "char"}]
    assert [w[1] for w in cs.tag_words(text, spans, "amh")] == ["amh", "amh", "eng", "amh"]


def test_carried_span_file_loads_keyed_by_row(tmp_path):
    p = tmp_path / "amh_spans.jsonl"
    p.write_text('{"segment_recording_id": "r1", "start": 4, "end": 9, "language": "eng", "text": "ሚቲንግ"}\n',
                 encoding="utf-8")
    spans = sc.load_spans(p)
    assert spans["r1"][0]["char_start"] == 4 and spans["r1"][0]["offset_unit"] == "char"


def test_row_keyed_spans_sit_on_the_panel_reference_not_its_utterances():
    # The utterance text is the uncorrected one; the row's reference is
    # corrected. Spans keyed by the row are read on the reference.
    row = {"segment_recording_id": "p1", "reference_text": "ኤአይም ጥሩ ነው",
           "utterances": [{"segment_recording_id": "u1", "text": "AIም ጥሩ ነው"}]}
    spans = {"p1": [{"char_start": 0, "char_end": 3, "language_code": "eng", "span_text": "ኤአይ",
                     "offset_unit": "char"}]}
    units = sc._reference_units(row, spans)
    assert units == [("ኤአይም ጥሩ ነው", spans["p1"])]


def test_language_wer_partitions_reported_wer():
    text = "ስብሰባው ላይ ሚቲንግ ነበር ትናንት"
    hyp = "ስብሰባው ሚቲን ነበር ዛሬ ትናንት"
    seg = item(text, [span(text, "ሚቲንግ")], hyp)
    total_errors = sum(t.errors for t in seg.by_language.values())
    total_units = sum(t.units for t in seg.by_language.values())
    w = sc.word_edits(normalize(text, AMH, mode="fluent", side="reference"),
                      normalize(hyp, AMH, mode="fluent", side="hypothesis"))
    assert (total_errors, total_units) == (w.errors, w.reference_units)


def test_pier_counts_span_and_neighbours():
    text = "አንድ ሁለት ሚቲንግ ሶስት አራት"
    seg = item(text, [span(text, "ሚቲንግ")], "አንድ ሁለት ሚቲን ሶስት አራት")
    assert (seg.pier.errors, seg.pier.units) == (1, 3)
    assert (seg.by_language["eng"].errors, seg.by_language["eng"].units) == (1, 1)
    assert (seg.by_language["amh"].errors, seg.by_language["amh"].units) == (0, 4)
    # an error on a word far from the span is not a point of interest
    seg = item(text, [span(text, "ሚቲንግ")], "አንዱ ሁለት ሚቲንግ ሶስት አራት")
    assert (seg.pier.errors, seg.pier.units) == (0, 3)


def test_insertion_beside_a_span_counts_once():
    text = "አንድ ሚቲንግ ሶስት"
    seg = item(text, [span(text, "ሚቲንግ")], "አንድ ሚቲንግ ኤክስትራ ሶስት")
    assert seg.pier.errors == 1
    assert sum(t.errors for t in seg.by_language.values()) == 1


def test_insertion_and_alternation_split_by_span_length():
    text = "አንድ ሚቲንግ ሁለት ሶስት ቤስት ፍሬንድ አራት"
    spans = [span(text, "ሚቲንግ"), span(text, "ቤስት ፍሬንድ")]
    seg = item(text, spans, "አንድ ሚቲን ሁለት ሶስት ቤስ ፍሬንድ አራት")
    assert (seg.pier_insertion.errors, seg.pier_insertion.units) == (1, 3)
    assert (seg.pier_alternation.errors, seg.pier_alternation.units) == (1, 4)


def test_switch_point_accuracy():
    text = "አንድ ሚቲንግ ሶስት አራት"
    seg = item(text, [span(text, "ሚቲንግ")], "አንዱ ሚቲንግ ሶስት አራት")
    # boundary tokens: አንድ, ሚቲንግ, ሶስት; አንድ is wrong
    assert (seg.switch_points.errors, seg.switch_points.units) == (2, 3)


def test_other_tag_is_not_a_switch():
    text = "አንድ ማርያም ሶስት"
    seg = item(text, [span(text, "ማርያም", "other")], "አንድ ማርያም ሶስት")
    assert seg.pier.units == 0 and seg.switch_points.units == 0
    assert seg.by_language["other"].units == 1


def test_tigrinya_amharic_same_script_span():
    tir = load_contract("tir")
    text = "ኣነ ግን እንደዚህ ኢለ"
    seg = item(text, [span(text, "እንደዚህ", "amh")], "ኣነ ግን ከምዚ ኢለ", contract=tir)
    assert (seg.by_language["amh"].errors, seg.by_language["amh"].units) == (1, 1)


def test_tags_survive_word_joining_normalization():
    # numerals join "ሁለት ሺህ" into one token; the English tag two words later stays put
    text = "ሁለት ሺህ ብር ሚቲንግ ላይ"
    ref = normalize(text, AMH, mode="fluent", side="reference")
    words = cs.tag_words(text, [span(text, "ሚቲንግ")], "amh")
    tags = cs.tagged_reference(words, ref.split(), AMH, mode="fluent", wer_n=False, numerals=None)
    assert len(tags) == len(ref.split())
    assert [t[0] for t in tags].count("eng") == 1
    assert tags[ref.split().index(normalize("ሚቲንግ", AMH, mode="fluent", side="reference"))][0] == "eng"


def test_panel_region_joins_utterance_spans():
    row = {"segment_recording_id": "r", "reference_text": "አንድ ሚቲንግ ሁለት ሶስት",
           "utterances": [{"segment_recording_id": "u1", "text": "አንድ ሚቲንግ"},
                          {"segment_recording_id": "u2", "text": "ሁለት ሶስት"}]}
    spans = {"u1": [span("አንድ ሚቲንግ", "ሚቲንግ")]}
    units = sc._reference_units(row, spans)
    ref = normalize(row["reference_text"], AMH, mode="fluent", side="reference")
    seg = cs.score_item(units, ref, ref, AMH, mode="fluent", wer_n=False, numerals=None)
    # the region's reference is one token stream, so u2's first word neighbours u1's span
    assert seg.spans == 1 and seg.by_language["eng"].units == 1 and seg.pier.units == 3


def test_score_reports_not_computed_without_spans():
    manifest = [{"segment_recording_id": "a", "reference_text": "ሰላም ነው"}]
    hyps = {"a": {"hypothesis": "ሰላም ነው"}}
    report, _ = sc.score(manifest, hyps, AMH, strip_tags=False, spans={})
    assert report["code_switching"]["not_computed"]
    report, _ = sc.score(manifest, hyps, AMH, strip_tags=False)
    assert report["code_switching"] is None


def test_pier_switches_removes_established_borrowings_only():
    # ሪፖርት has tier Loanword in the contract's table; ሚቲንግ is not in it
    text = "አንድ ሪፖርት ሁለት ሶስት ሚቲንግ አራት"
    seg = item(text, [span(text, "ሪፖርት"), span(text, "ሚቲንግ")], "አንዱ ሪፖርቱ ሁለት ሶስት ሚቲን አራት")
    assert (seg.pier.errors, seg.pier.units) == (3, 6)
    # the borrowing and its two neighbours leave PIER-switches; the switch keeps its three
    assert (seg.pier_switches.errors, seg.pier_switches.units) == (1, 3)
    assert seg.borrowing_points == 3


def test_a_neighbour_shared_with_a_switch_stays():
    text = "ሪፖርት ሚቲንግ ነው"
    seg = item(text, [span(text, "ሪፖርት"), span(text, "ሚቲንግ")], "ሪፖርት ሚቲንግ ነው")
    assert seg.pier.units == 3
    assert seg.pier_switches.units == 2 and seg.borrowing_points == 1


def test_borrowing_under_affix_is_recognized():
    text = "አንድ በሪፖርቱም ሁለት"
    seg = item(text, [span(text, "በሪፖርቱም")], "አንድ በሪፖርቱም ሁለት")
    assert seg.pier.units == 3 and seg.pier_switches.units == 0 and seg.borrowing_points == 3


def test_native_ruling_makes_the_tag_matrix():
    # ቢያንስ is an Amharic verb form inside an English span; the table gives it tier Native
    text = "አንድ ቢያንስ ሁለት"
    seg = item(text, [span(text, "ቢያንስ")], "አንድ ቢያንስ ሁለት")
    assert seg.native_tags == 1 and seg.pier.units == 0
    assert "eng" not in seg.by_language and seg.by_language["amh"].units == 3
