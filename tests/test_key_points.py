import pytest

pytest.importorskip("spacy")
pytest.importorskip("en_core_web_sm")

from statement_similarity_analysis import compare_key_points, extract_key_points  # noqa: E402


def test_negated_verb_is_tagged():
    assert "take[NEG]" in extract_key_points("I didn't take anything from the store.")["actions"]


def test_polarity_flip_is_one_sided_on_both_sides():
    comparison = compare_key_points(extract_key_points("I didn't take the phone."),
                                    extract_key_points("He took the phone."))
    assert "take[neg]" in comparison["actions"]["suspect_only"]
    assert "take" in comparison["actions"]["witness_only"]


def test_clock_time_is_extracted():
    assert any("8" in item for item in extract_key_points("I was at the store at 8 am.")["time"])
