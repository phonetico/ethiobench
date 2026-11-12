"""The abbreviation step of the Amharic contract."""

import pytest

from ethiobench.asr import load_contract
from ethiobench.normalize.pipeline import normalize


@pytest.fixture(scope="module")
def amh():
    return load_contract("amh")


def n(text, contract, side="reference", wer_n=False):
    return normalize(text, contract, mode="fluent", side=side, wer_n=wer_n)


def test_contract_declares_the_table_and_stamps_it(amh):
    assert amh.abbreviations is not None
    stamp = amh.stamp["abbreviations"]
    assert stamp["table"] == "amh_abbreviations.tsv" and stamp["entries"] >= 24 and len(stamp["sha256"]) == 64
    assert "expand_abbreviations" not in {p["step"] for p in amh.stamp["pending"]}
    assert [s.name for s in amh.layers["base"]].index("expand_abbreviations") \
        < [s.name for s in amh.layers["base"]].index("numerals")


def test_every_written_form_of_the_era_marker_expands(amh):
    # every attested written form, less the dotless ዐም the table leaves alone
    for written in ("ዓ.ም", "ዓ.ም.", "ዓም", "ዓ/ም", "ዓም.", "ዓ/ም.", "ዐ.ም", "ዐ.ም.", "ዐ/ም", "ዓ. ም."):
        assert n(f"በ2015 {written} ተመረቀ", amh) == "በ2015 ዓመተ ምህረት ተመረቀ", written
    # ምሕረት reaches ምህረት under WER-N
    assert n("ዓመተ ምሕረት", amh, wer_n=True) == n("ዓ.ም", amh, wer_n=True)


def test_every_written_form_of_the_gregorian_marker_expands(amh):
    for written in ("እ.ኤ.አ", "እ.ኤ.አ.", "እ.አ.አ", "እኤአ", "እአአ", "እ.አ.አ.", "እ.ኣ.አ", "እ.ኤአ",
                    "እ.ኤ.ኣ.", "እ.ኤአ.", "እ/ኤ/አ", "እ.ኤ.ኣ", "እኤኣ", "ኢኤአ", "ኢ.ኤ.አ"):
        assert n(f"{written} በ2015", amh) == "እንደ ኤውሮፓውያን አቆጣጠር በ2015", written
    # The two spellings never meet under WER-N (ኤ folds only with ዔ, አ only with ዓ/ዐ),
    # so the table's spelling of the expansion changes the score.
    assert n("እንደ አውሮፓውያን አቆጣጠር", amh, wer_n=True) != n("እ.ኤ.አ", amh, wer_n=True)


def test_both_sides_meet_whichever_wrote_the_abbreviation(amh):
    spans = [{"digits": "2015", "words": "ሁለት ሺ አስራ አምስት"}]
    ref = n("እንደ ኤውሮፓውያን አቆጣጠር በማርች ሁለት ሺ አስራ አምስት", amh)
    assert normalize("እ.ኤ.አ. በማርች 2015", amh, mode="fluent", side="hypothesis", number_spans=spans) == ref
    assert normalize("እኤአ በማርች 2015ዓም", amh, mode="fluent", side="hypothesis", number_spans=spans) \
        == ref + " ዓመተ ምህረት"


def test_boundaries(amh):
    assert n("ዓምድ", amh) == "ዓምድ"                      # a word, not the marker
    assert n("ዓ ም ዐም", amh) == "ዓ ም ዐም"                 # a space is not a separator; dotless ዐም left alone
    assert n("2008ዓም ብሏል", amh) == "2008 ዓመተ ምህረት ብሏል"   # glued to its number
    assert n("የዓ.ም", amh) == "የዓመተ ምህረት"                 # prefix stays attached
    assert n("ጠ/ሚ ሚ/ር", amh) == "ጠሚ ሚር"                  # ambiguous ones not expanded, kept whole


def test_suffix_carried_onto_the_expansion(amh):
    assert n("ት/ቤቱ ት/ቤቶች ት/ቤትን", amh) == "ትምህርት ቤቱ ትምህርት ቤቶች ትምህርት ቤትን"
    assert n("ዶ/ሩ ም/ሉ ኮ/ሉ", amh) == "ዶክተሩ ምክትሉ ኮሎኔሉ"
    assert n("ወ/ሮ ወ/ሪት ወ/ት ሌ/ጄነራል ሌ/ኮ", amh) == "ወይዘሮ ወይዘሪት ወይዘሪት ሌተና ጄነራል ሌተና ኮሎኔል"


def test_an_ambiguous_abbreviation_stays_one_word(amh):
    # አ.አ split by punctuation removal would be two fillers, deleted in fluent mode.
    assert n("አ.አ ሄደ", amh) == "አአ ሄደ"
    assert n("በአ.አ ኖረ ጠ/ሚ ሚ/ር ረ/ፕ", amh) == "በአአ ኖረ ጠሚ ሚር ረፕ"
    assert n("ጠ/ሚር ረ/ፕሮፌሰር", amh) == "ጠቅላይ ሚኒስትር ረዳት ፕሮፌሰር"
    assert n("አ አ ሄደ", amh) == "ሄደ"


def test_a_table_row_must_be_a_valid_pattern(tmp_path):
    from ethiobench.normalize.abbreviations import load
    bad = tmp_path / "t.tsv"
    bad.write_text("written\tspoken\n[ዓ\tዓመተ ምህረት\n", encoding="utf-8")
    with pytest.raises(Exception):
        load(bad)
