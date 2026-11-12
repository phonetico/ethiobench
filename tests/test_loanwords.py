"""The loanword layer of the Amharic contract."""

import pytest

from ethiobench.asr import load_contract
from ethiobench.normalize import config
from ethiobench.normalize.pipeline import fold_mapping, normalize


@pytest.fixture(scope="module")
def amh():
    return load_contract("amh")


def n(text, contract, profile=None, side="reference"):
    return normalize(text, contract, mode="fluent", side=side, wer_n=True, loanword_profile=profile)


def test_contract_declares_the_table_and_stamps_it(amh):
    assert amh.loanwords is not None
    assert amh.loanword_profiles == ("lenient", "strict")
    stamp = amh.stamp["loanwords"]
    assert stamp["table"] == "amh_loanwords.tsv" and stamp["entries"] >= 131 and len(stamp["sha256"]) == 64
    # applied, not pending
    assert "normalize_loanwords" not in {p["step"] for p in amh.stamp["pending"]}


def test_lenient_folds_anything_the_pattern_generates(amh):
    # every spelling the pattern generates, attested or not
    for s in ("ፕረዝዳንት", "ፕረዚዳንት", "ፕረዝደንት", "ፕረዚደንት", "ፕረዘዳንት", "ፕሬዚዳንት", "ፕሬዝደንት"):
        assert n(s, amh, "lenient") == "ፕሬዝዳንት", s
    # million is a scale word, and the numerals step runs first and knows the
    # same spellings, so under to_digits every one becomes the number.
    for s in ("ሚሊዮን", "ሚሊየን", "ሚልዮን", "ሚልየን"):
        assert normalize(f"ሁለት {s}", amh, mode="fluent", side="reference", wer_n=True,
                         loanword_profile="lenient", numerals="to_digits") == "2000000", s
        assert n(f"ሁለት {s}", amh, "lenient") == "ሁለት ሚሊዮን", s


def test_strict_folds_the_established_spellings_only(amh):
    assert n("ፕሬዚዳንት", amh, "strict") == "ፕሬዝዳንት"      # a house style at Ethiopian outlets
    assert n("ፕሬዝደንት", amh, "strict") == "ፕሬዝደንት"      # not one; left unfolded
    assert n("ዩኒቨርስቲ", amh, "strict") == "ዩኒቨርሲቲ"
    assert n("ዩንቨርስቲ", amh, "strict") == "ዩንቨርስቲ"
    assert n("ዩንቨርስቲ", amh, "lenient") == "ዩኒቨርሲቲ"


def test_the_office_and_the_person_stay_apart(amh):
    # ሚኒስተር, ሚንስተር, ምንስተር, ምኒስቴር are the office; ሚኒስትር is the person
    assert n("ሚኒስተር ሚንስተር ምኒስቴር", amh, "lenient") == "ሚኒስቴር ሚኒስቴር ሚኒስቴር"
    assert n("ሚኒስትር", amh, "lenient") == "ሚኒስትር"
    assert n("ሚኒስትሪ", amh, "lenient") == "ሚኒስትሪ"          # ruled "depends": not folded at form level


def test_both_sides_fold_the_same(amh):
    assert n("ፕሬዚዳንት", amh, "lenient", side="hypothesis") == n("ፕሬዝዳንት", amh, "lenient", side="reference")


def test_affixed_forms_are_not_folded_but_are_flagged(amh):
    assert n("በሚኒስተር የፕሬዚዳንቱ", amh, "lenient") == "በሚኒስተር የፕሬዚዳንቱ"
    t = amh.loanwords
    assert t.same_under_affix("በሚኒስተር", "በሚኒስቴር")
    assert t.same_under_affix("የፕሬዚዳንቱ", "የፕሬዝዳንቱ")
    assert not t.same_under_affix("በሚኒስቴር", "በሚኒስትር")      # office against person: a real difference
    assert not t.same_under_affix("የፕሬዝዳንቱ", "የፕሬዝዳንታ")    # a different suffix vowel
    assert not t.same_under_affix("ሪፖርት", "ሪፖርቱ")


def test_tier_lookup_sees_through_affixes(amh):
    t = amh.loanwords
    assert t.entry_of("ለሪፖርቱ", affixed=True).tier == "Loanword"
    assert t.entry_of("ፕሮጀክቶቹን", affixed=True).tier == "Loanword"
    assert t.entry_of("ሪሌሽንሺፓችንን", affixed=True).tier == "Code-Switching"
    assert t.entry_of("ኢኮኖሚካሊ").tier == "Code-Switching"
    assert t.entry_of("ቢያንስ").tier == "Native"
    assert t.entry_of("ሶስቱም", affixed=True) is None
    assert t.entry_of("አንድ", affixed=True) is None


def test_a_profile_needs_wer_n_and_a_declared_table(amh):
    with pytest.raises(ValueError):
        normalize("ፕሬዚዳንት", amh, mode="fluent", side="reference", wer_n=False, loanword_profile="lenient")
    with pytest.raises(config.ContractError):
        n("ፕሬዚዳንት", amh, "loose")
    tir = load_contract("tir")
    with pytest.raises(config.ContractError):
        n("ፕሬዚዳንት", tir, "lenient")


def test_table_is_written_in_the_folded_alphabet(amh):
    # a table form using a character the WER-N fold rewrites could never match
    fm = fold_mapping(amh)
    touched = [f for f in amh.loanwords._lenient if any(ch in fm for ch in f)]
    assert touched == [], touched


def test_strict_keeps_each_word_of_a_multi_word_entry_itself(amh):
    # Social media is the table's one multi-word entry; its words are entries of their own
    for w in ("ሶሻል", "ሚዲያ"):
        assert amh.loanwords.fold_token(w, "strict") == w


def test_a_multi_word_entry_gives_its_tier_to_the_run_it_covers(amh):
    assert amh.loanwords.phrase_tiers("ስለ የሶሻል ሚዲያው ነው".split()) == {1: "Loanword", 2: "Loanword"}
    assert amh.loanwords.phrase_tiers("ሶሻል ኔትወርክ".split()) == {}


def test_only_a_final_consonant_fuses_with_a_suffix_vowel(amh):
    def tier_word(w):
        e = amh.loanwords.entry_of(w, affixed=True)
        return e.canonical if e else None
    assert tier_word("ሪፖርቱ") == "ሪፖርት"          # ት is sixth order: ት + ኡ = ቱ
    assert tier_word("ፖሊሶች") == "ፖሊስ"           # police, not policy
    assert tier_word("ፓርቶች") == "ፓርት"           # parts, not party
    assert tier_word("የፓርቲ") == "ፓርቲ"           # an unfused match wins
    assert tier_word("ነው") is None                # ኖ is not sixth order


_HEADER = "word\tcanonical\tpattern\talternative patterns\tstrict\ttier\n"


def test_a_strict_form_claimed_by_two_entries_is_refused(tmp_path):
    from ethiobench.normalize.loanwords import load
    p = tmp_path / "t.tsv"
    p.write_text(_HEADER + "A\tአ\tአ\t\tአ ኣ\tLoanword\nB\tበ\tበ\t\tበ ኣ\tLoanword\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strict form ኣ belongs to both"):
        load(p)
