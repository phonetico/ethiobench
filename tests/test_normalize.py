"""The Amharic contract does what it declares."""

import re

import pytest

from ethiobench.asr import CONTRACT_DIR, load_contract
from ethiobench.normalize import config
from ethiobench.normalize.pipeline import fold_mapping, normalize


@pytest.fixture(scope="module")
def amh():
    return load_contract("amh")


def n(text, contract, mode="fluent", side="reference", wer_n=False):
    return normalize(text, contract, mode=mode, side=side, wer_n=wer_n)


def test_rules_hash_ignores_prose_and_follows_rules(tmp_path):
    import shutil
    shutil.copy(CONTRACT_DIR / "tir.yaml", tmp_path / "tir.yaml")
    base = load_contract("tir").rules_sha256
    text = (tmp_path / "tir.yaml").read_text(encoding="utf-8")
    prose = text.replace("a filler matches another filler of the same language.",
                         "a filler counts as equal to another of the same language.")
    assert prose != text
    (tmp_path / "tir.yaml").write_text(prose, encoding="utf-8")
    import yaml
    parse = lambda p: config._parse(yaml.safe_load(p.read_text(encoding="utf-8")), source=p, sha256="x")
    assert parse(tmp_path / "tir.yaml").rules_sha256 == base
    raw = yaml.safe_load(prose)
    raw["filler_inventory"]["classes"]["eng"] = ["um"]          # a rule change: uh is no longer a filler
    assert config._rules_sha256(raw, []) != base


def test_no_contract_is_an_error():
    with pytest.raises(config.ContractError, match="No normalization contract"):
        load_contract("orm")


def test_tigrinya_never_folds_amharic_homophones():
    tir = load_contract("tir")
    assert not tir.has_equivalence_sets()
    assert n("ሐዚ ፀሓይ", tir, wer_n=True) == "ሐዚ ፀሓይ"
    # Filler removed; ሓደ ("one", also "a") becomes 1 on both sides.
    assert n("እ ሓደ ሰብ", tir) == "1 ሰብ"


def test_pending_rules_make_results_provisional(amh):
    assert amh.stamp["provisional"] is True
    pending = {p["step"] for p in amh.stamp["pending"]}
    assert pending == {"normalize_amharic_spelling"}
    # every pending entry says what blocks it, in words
    assert all(p["blocked_by"] and not re.fullmatch(r"[A-Z]\d+", p["blocked_by"])
               for p in amh.stamp["pending"])


def test_numerals_direction_declared_and_overridable(amh):
    # as_recorded is declared, so the reference keeps its words
    assert [s.direction for s in amh.layers["base"] if s.name == "numerals"] == ["as_recorded"]
    assert n("ሃያ ሁለት ቀን", amh) == "ሃያ ሁለት ቀን"
    # the parser directions remain available as overrides, recorded in every result
    assert normalize("ሃያ ሁለት ቀን", amh, mode="fluent", side="reference", numerals="to_digits") == "22 ቀን"
    assert normalize("22 ቀን", amh, mode="fluent", side="hypothesis", numerals="to_words") == "ሃያ ሁለት ቀን"
    assert normalize("ሃያ ሁለት 22", amh, mode="fluent", side="reference", numerals="off") == "ሃያ ሁለት 22"


def test_as_recorded_matches_only_the_recorded_digit_form(amh):
    spans = [{"digits": "985", "words": "ዘጠኝ ሰማንያ አምስት"}, {"digits": "6,079,690፣", "words": "ስድስት ሚሊዮን ሰባ ዘጠኝ ሺ ስድስት መቶ ዘጠና፣"}]
    h = lambda t: normalize(t, amh, mode="fluent", side="hypothesis", number_spans=spans)
    ref = n("አንቀጽ ዘጠኝ ሰማንያ አምስት በጀቱ ስድስት ሚሊዮን ሰባ ዘጠኝ ሺ ስድስት መቶ ዘጠና፣ ብር", amh)
    # the digit form of what was said, with or without group commas, matches
    assert h("አንቀጽ 985 በጀቱ 6079690 ብር") == ref
    assert h("አንቀጽ 985 በጀቱ 6,079,690፣ ብር") == ref
    # the words said match; another reading of the same value does not
    assert h("አንቀጽ ዘጠኝ ሰማንያ አምስት በጀቱ ስድስት ሚሊዮን ሰባ ዘጠኝ ሺ ስድስት መቶ ዘጠና ብር") == ref
    assert h("አንቀጽ ዘጠኝ መቶ ሰማንያ አምስት በጀቱ 6079690 ብር") != ref
    # a digit form that is not the recorded one stays as written and misses
    assert h("አንቀጽ 9 85 በጀቱ 6079690 ብር") == "አንቀጽ 9 85 በጀቱ ስድስት ሚሊዮን ሰባ ዘጠኝ ሺ ስድስት መቶ ዘጠና ብር"
    # the same digits twice: each recorded span matches once
    assert h("985 985") == "ዘጠኝ ሰማንያ አምስት 985"


def test_as_recorded_separators_affixes_and_derived_spans(amh):
    from ethiobench.normalize.numerals import rewrite_recorded, spans_for
    # separators inside a digit form are punctuation: 274 1994 meets 274/1994
    spans = [{"digits": "274/1994", "words": "ሁለት ሰባት አራት አስራ ዘጠኝ ዘጠና አራት"}, {"digits": "4፡30", "words": "አራት ተኩል"}]
    for t in ("274/1994 4:30", "274 1994 4፡30", "274-1994 4 30"):
        assert rewrite_recorded(t, spans)[0] == "ሁለት ሰባት አራት አስራ ዘጠኝ ዘጠና አራት አራት ተኩል", t
    # a prefix or fused suffix that differs is one substitution, not a miss
    spans = [{"digits": "በ2014", "words": "በሁለት ሺ አስራ አራት"}, {"digits": "13ቱ", "words": "አስራ ሶስቱ"}]
    text, matched, partial, left = rewrite_recorded("2014 13", spans)
    assert (text, matched, partial, left) == ("2014 ሺ አስራ አራት አስራ 13", 0, 2, 0)
    assert rewrite_recorded("የ2014", [{"digits": "2014", "words": "ሁለት ሺ አስራ አራት"}])[0] == "የ2014 ሺ አስራ አራት"
    # a reference written in words from the start gets the parser's pairs;
    # the review's pair covers its own words, so 9 85 cannot meet ዘጠኝ ሰማንያ አምስት
    ref = "አንቀጽ ዘጠኝ ሰማንያ አምስት ከአምስት መቶ በላይ"
    spans = spans_for(ref, "amh", [{"digits": "985", "words": "ዘጠኝ ሰማንያ አምስት", "token_index": 1}])
    assert [(s["digits"], s["words"]) for s in spans] == [("985", "ዘጠኝ ሰማንያ አምስት"), ("ከ500", "ከአምስት መቶ")]
    assert rewrite_recorded("አንቀጽ 9 85 ከ500 በላይ", spans)[0] == "አንቀጽ 9 85 ከአምስት መቶ በላይ"
    # percent: the sign and the word are one piece after a digit token
    spans = spans_for("ወደ ሰላሳ ፐርሰንት ዝቅ", "amh", [])
    assert [(s["digits"], s["words"]) for s in spans] == [("30%", "ሰላሳ ፐርሰንት")]
    for t in ("ወደ 30% ዝቅ", "ወደ 30 ፐርሰንት ዝቅ", "ወደ 30 በመቶ ዝቅ"):
        assert rewrite_recorded(t, spans)[0] == "ወደ ሰላሳ ፐርሰንት ዝቅ", t
    assert rewrite_recorded("ወደ 30 ዝቅ", spans)[3] == 1


def test_fluent_drops_fillers_both_sides(amh):
    assert n("እ ሰላም እም ነው አ", amh) == "ሰላም ነው"
    assert n("እ ሰላም እም ነው አ", amh, side="hypothesis") == "ሰላም ነው"


def test_fluent_drops_fragments_from_reference_only(amh):
    assert n("ተሰ- በተሰጠው", amh) == "በተሰጠው"
    assert n("ተሰ- በተሰጠው", amh, side="hypothesis") == "ተሰ በተሰጠው"


def test_verbatim_keeps_fragment_without_marker(amh):
    assert n("ተሰ- በተሰጠው", amh, mode="verbatim") == "ተሰ በተሰጠው"


def test_fragment_before_ethiopic_punctuation(amh):
    assert n("መ-፣ መስፈርቶች", amh) == "መስፈርቶች"


def test_verbatim_filler_equivalence_within_class_only(amh):
    assert n("እም", amh, mode="verbatim") == n("እ", amh, mode="verbatim")
    assert n("um", amh, mode="verbatim") == n("uh", amh, mode="verbatim")
    assert n("um", amh, mode="verbatim") != n("እ", amh, mode="verbatim")


def test_word_internal_hyphen_is_not_a_fragment(amh):
    assert n("ት-ቤት", amh, mode="verbatim") == "ት ቤት"
    assert n("ት-ቤት", amh) == "ት ቤት"


def test_punctuation_becomes_boundary_numbers_hold_together(amh):
    # under the to_digits parser: a number word with its clause punctuation
    # attached (ሁለተኛ፣) is read through the punctuation, so a hypothesis
    # without the comma meets it
    d = lambda t, side="reference": normalize(t, amh, mode="fluent", side=side, numerals="to_digits")
    assert d("ሁለተኛ፣ ተቋሙ። 10,693 (76%) 3.5 የ2013/14") == "2ኛ ተቋሙ 10693 76 3.5 የ2013 14"
    assert d("ሁለተኛ ተቋሙ", "hypothesis") == "2ኛ ተቋሙ"
    assert d("መቶ አርባ ስምንት፣ በሁለተኛ ዙር መቶ ሰላሳ አራት፣") == "148 በ2ኛ ዙር 134"
    assert d("ሰባ ስድስት በመቶ።") == "76"
    # under as_recorded the punctuation still becomes a boundary and the
    # number words stay
    assert n("ሁለተኛ፣ ተቋሙ። ሰባ ስድስት በመቶ።", amh) == "ሁለተኛ ተቋሙ ሰባ ስድስት በመቶ"


def test_bracket_tags_removed(amh):
    assert n("ሰላም [unclear:1.5] ነው [laughter]", amh, mode="verbatim") == "ሰላም ነው"


def test_latin_casefold_and_apostrophe(amh):
    assert n("ያንን Computer's ገዝቶ", amh) == "ያንን computers ገዝቶ"


def test_zero_width_and_nfc(amh):
    assert n("ሰ​ላም", amh) == "ሰላም"


def test_wer_n_folds_across_orders(amh):
    assert n("ሐሙስ ሑ ኀ ሠራ ፀሐይ ዐይን", amh, wer_n=True) == "ሀሙስ ሁ ሀ ሰራ ጸሀይ አይን"
    assert n("ሐሙስ", amh) == "ሐሙስ"  # base WER does not fold


def test_kha_first_order_is_not_folded(amh):
    # Published Amharic normalizers keep the first order distinct; orders 2, 3
    # and 5 to 7 fold to the Ha series.
    assert n("ኸ ኹ ኻ ኽ", amh, wer_n=True) == "ኸ ሁ ኻ ህ"
    assert "ኸ" not in fold_mapping(amh)


def test_wi_is_not_folded_to_we(amh):
    assert "ዉ" not in fold_mapping(amh)


def test_eighth_position_folds_only_on_same_vowel(amh):
    mapping = fold_mapping(amh)
    assert mapping["ሧ"] == "ሷ"          # SZWA -> SWA
    assert "ሗ" not in mapping           # HHWA has no HWA partner (ሇ is HOA)
    assert "ፇ" not in mapping           # TZOA vs TSWA


def test_fold_does_not_create_fillers_before_removal(amh):
    # A standalone ዐ is a word. Folding happens after filler removal, so it survives.
    assert n("ዐ ነው", amh, wer_n=True) == "አ ነው"


def test_full_fold_table_ruled_d1(amh):
    mapping = fold_mapping(amh)
    assert mapping["ሃ"] == "ሀ" and mapping["ኃ"] == "ሀ" and mapping["ኺ"] == "ሂ"
    assert mapping["ዓ"] == "አ" and mapping["ዐ"] == "አ"
    assert mapping["ጐ"] == "ጎ" and mapping["ቈ"] == "ቆ" and mapping["ኰ"] == "ኮ" and mapping["ኈ"] == "ሆ"
    assert "ጓ" not in mapping and "ዉ" not in mapping


def test_tigrinya_spellings_stay_unfolded(amh):
    # ሓ and ኣ are not Amharic spellings and ኻ is names and Tigrinya: none
    # folds, so writing one for ሃ or አ counts as an error.
    mapping = fold_mapping(amh)
    assert "ሓ" not in mapping and "ኣ" not in mapping and "ኻ" not in mapping
    assert n("ኻርቱም", amh, wer_n=True) != n("ሃርቱም", amh, wer_n=True)
    assert n("ሓሳብ", amh, wer_n=True) != n("ሃሳብ", amh, wer_n=True)
    assert n("ኣንድ", amh, wer_n=True) != n("አንድ", amh, wer_n=True)
    # The rest of each series still folds, across the order boundary.
    assert n("ሐሳብ ኃላፊ", amh, wer_n=True) == n("ሀሳብ ሀላፊ", amh, wer_n=True)
    assert n("ዓለም", amh, wer_n=True) == n("አለም", amh, wer_n=True)


def test_unfolded_must_name_a_folded_character(tmp_path):
    from ethiobench.normalize.config import EquivalenceSet
    from ethiobench.normalize.pipeline import _set_mapping
    with pytest.raises(config.ContractError, match="unfolded"):
        _set_mapping(EquivalenceSet(name="x", members=("ሀ", "ሐ"), justification="t", unfolded=("ከ",)))


def test_unknown_step_is_refused(tmp_path):
    (tmp_path / "xx.yaml").write_text(
        "language: xx\nversion: '0'\npipeline:\n  base:\n    - {step: made_up, justification: x}\n")
    with pytest.raises(config.ContractError, match="does not implement"):
        config.load("xx", tmp_path)


def test_unjustified_step_is_refused(tmp_path):
    (tmp_path / "xx.yaml").write_text(
        "language: xx\nversion: '0'\npipeline:\n  base:\n    - {step: unicode_nfc}\n")
    with pytest.raises(config.ContractError, match="no justification"):
        config.load("xx", tmp_path)
