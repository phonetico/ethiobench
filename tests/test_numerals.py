"""Numeral equivalence: both directions meet on one canonical form."""

import pytest

from ethiobench.normalize.numerals import geez_to_int, normalize_numerals as nn


@pytest.mark.parametrize("words,digits", [
    ("ሁለት ሺህ አስራ አምስት", "2015"),
    ("ሃያ ሁለት", "22"),
    ("መቶ ሃምሳ", "150"),
    ("ሁለት መቶ", "200"),
    ("አስር ሺ", "10000"),
    ("አስራ ሁለት ሺህ ሰባት መቶ", "12700"),
    ("አንድ ሚሊዮን ሁለት መቶ ሺ", "1200000"),
])
def test_amharic_words_to_digits(words, digits):
    assert nn(words, "amh", "to_digits") == digits


def test_place_value_splits_adjacent_numbers():
    # 2013/14 read aloud; punctuation removal turns the written form into "2013 14".
    assert nn("ሁለት ሺህ አስራ ሶስት አስራ አራት", "amh", "to_digits") == "2013 14"
    assert nn("ሁለት ሶስት", "amh", "to_digits") == "2 3"


def test_prefix_ordinal_decimal_percent():
    assert nn("በሃያ ሁለት ቀን", "amh", "to_digits") == "በ22 ቀን"
    assert nn("አስራ ስምንተኛ መደበኛ", "amh", "to_digits") == "18ኛ መደበኛ"
    assert nn("ሶስት ነጥብ አምስት", "amh", "to_digits") == "3.5"
    assert nn("ሰባ ስድስት በመቶ", "amh", "to_digits") == "76%"


def test_non_number_words_untouched():
    assert nn("ሰላም ነው ዛሬ", "amh", "to_digits") == "ሰላም ነው ዛሬ"


@pytest.mark.parametrize("text", ["2015", "በ22 ቀን", "18ኛ መደበኛ", "3.5", "76%", "10,000", "150", "12700"])
def test_amharic_round_trip_meets(text):
    assert nn(nn(text, "amh", "to_words"), "amh", "to_digits") == nn(text.replace(",", ""), "amh", "to_digits")


def test_amharic_to_words_forms():
    assert nn("2015", "amh", "to_words") == "ሁለት ሺ አስራ አምስት"
    assert nn("በ22", "amh", "to_words") == "በሃያ ሁለት"
    assert nn("18ኛ", "amh", "to_words") == "አስራ ስምንተኛ"
    assert nn("76%", "amh", "to_words") == "ሰባ ስድስት በመቶ"


def test_both_directions_equalize_a_mixed_pair():
    ref, hyp = "የ2015 በጀት ዓመት", "የሁለት ሺህ አስራ አምስት በጀት ዓመት"
    for direction in ("to_digits", "to_words"):
        assert nn(ref, "amh", direction) == nn(hyp, "amh", direction)


def test_tigrinya_words_and_suffixes():
    assert nn("ዕስራን ሓሙሽተን", "tir", "to_digits") == "25ን"
    assert nn("ክልተ ሽሕ ዓሰርተ ሸውዓተ", "tir", "to_digits") == "2017"
    assert nn("ሰላሳ ሚእታዊ", "tir", "to_digits") == "30%"
    ref, hyp = "2017", "ክልተ ሽሕ ዓሰርተ ሸውዓተ"
    for direction in ("to_digits", "to_words"):
        assert nn(ref, "tir", direction) == nn(hyp, "tir", direction)


def test_geez_numerals():
    assert geez_to_int("፲፱") == 19
    assert geez_to_int("፳፻") == 2000
    assert geez_to_int("፪፻፲") == 210
    assert nn("በ፲፱ ቀን", "amh", "to_digits") == "በ19 ቀን"


def test_off_is_identity():
    assert nn("ሁለት 2", "amh", "off") == "ሁለት 2"


def test_written_digits_join_a_scale_or_percent_word():
    # a digit token starts a number that only a scale may extend
    assert nn("20 ሚሊዮን ብር", "amh", "to_digits") == "20000000 ብር"
    assert nn("1.4 ሚሊዮን", "amh", "to_digits") == "1400000"
    assert nn("5 ቢሊዮን", "amh", "to_digits") == "5000000000"
    assert nn("በ35 በመቶ", "amh", "to_digits") == "በ35%"
    # a digit token nothing joins is left exactly as written
    assert nn("05 የ2015 2013/14", "amh", "to_digits") == "05 የ2015 2013/14"
    # both sides meet: ሃያ ሚሊዮን against 20 ሚሊዮን
    assert nn("ሃያ ሚሊዮን", "amh", "to_digits") == nn("20 ሚሊዮን", "amh", "to_digits")


def test_attested_spelling_variants_of_number_words():
    assert nn("ሁለት ሺሕ ሠላሳ ሠባት", "amh", "to_digits") == "2037"
    assert nn("ኃምሳ ሥምንት", "amh", "to_digits") == "58"
    assert nn("ዜሮ ነጥብ አምስት", "amh", "to_digits") == "0.5"
    assert nn("ዜሮ አንድ", "amh", "to_digits") == "01"
    assert nn("ሶስት ትሪሊዮን", "amh", "to_digits") == "3000000000000"


def test_fraction_digits_keep_their_leading_zeros():
    assert nn("ሶስት ነጥብ ዜሮ አምስት", "amh", "to_digits") == "3.05"
    assert nn("ዜሮ ነጥብ ዜሮ ዜሮ ሁለት", "amh", "to_digits") == "0.002"
    assert nn("ሁለት ነጥብ አምስት ሶስት", "amh", "to_digits") == "2.53"
    assert nn("ዜሮ ነጥብ ሶስት አምስት", "amh", "to_digits") == "0.35"
    assert nn("ዜሮ ነጥብ ዜሮ ሰባ አምስት", "amh", "to_digits") == "0.075"
    assert nn("ሰለስተ ነጥቢ ዜሮ ሓሙሽተ", "tir", "to_digits") == "3.05"
    assert nn("ሰለስተ ነጥቢ ሸውዓተ ሓሙሽተ ቢልዮን", "tir", "to_digits") == "3750000000"


def test_digits_after_a_leading_zero_are_one_digit_string():
    assert nn("ዜሮ ዘጠኝ አስራ አንድ", "amh", "to_digits") == "0911"
    assert nn("ዜሮ ሓሙሽተ", "tir", "to_digits") == "05"
    assert nn("ዜሮ ዘጠኝ አንድ መቶ ሰዎች", "amh", "to_digits") == "09 100 ሰዎች"
    assert nn("ዜሮ ሰዎች", "amh", "to_digits") == "0 ሰዎች"
    assert nn("ዜሮካስ", "tir", "to_digits") == "ዜሮካስ"


def test_a_unit_never_follows_a_unit():
    assert nn("ሁለት ሶስት ወረዳዎች", "amh", "to_digits") == "2 3 ወረዳዎች"
    assert nn("ሃያ ሶስት ሃያ አራት", "amh", "to_digits") == "23 24"
    assert nn("ስምንት ሰባት", "amh", "to_digits") == "8 7"


def test_decimal_then_scale_and_two_word_fractions():
    assert nn("ስድስት ነጥብ ስድስት ቢልዮን ብር", "amh", "to_digits") == "6600000000 ብር"
    assert nn("16.9 ቢሊዮን", "amh", "to_digits") == "16900000000"     # no scientific notation
    assert nn("ዜሮ ነጥብ ሰላሳ አምስት ፐርሰንት", "amh", "to_digits") == "0.35%"
    assert nn("ሶስት ነጥብ አምስት፣ ስድስት", "amh", "to_digits") == "3.5፣ 6"
