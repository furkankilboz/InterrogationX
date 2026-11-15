import pytest

from rule_checks import clock_times, time_conflicts


def test_clock_times_parse_am_pm_and_named_times():
    assert [minutes for minutes, _ in clock_times("at 9 AM, 9:30 p.m. and around midnight, then noon")] == \
        [540, 1290, 0, 720]


def test_times_without_am_pm_are_ignored():
    assert clock_times("I left at 8 and came back at 10.") == []


def test_time_conflict_needs_times_on_both_sides_far_apart():
    assert time_conflicts("I arrived at 9 AM.", "He entered at 3 PM.") == ["9 AM vs 3 PM"]
    assert time_conflicts("I arrived at 9 AM.", "He entered at 9:40 am.") == []
    assert time_conflicts("I left at 11 PM.", "He was gone by midnight.") == []  # wraps around midnight
    assert time_conflicts("I arrived in the morning.", "He entered at 3 PM.") == []


class TestWithSpacy:
    @pytest.fixture(autouse=True)
    def nlp(self):
        pytest.importorskip("spacy")
        pytest.importorskip("en_core_web_sm")
        import models
        return models.get_nlp()

    def test_polarity_conflict(self, nlp):
        from rule_checks import polarity_conflicts
        assert polarity_conflicts(nlp("I didn't take the phone."), nlp("He took the phone.")) == ["take"]
        assert polarity_conflicts(nlp("I took the phone."), nlp("He took the phone.")) == []

    def test_negated_perception_is_not_a_conflict(self, nlp):
        from rule_checks import polarity_conflicts
        assert polarity_conflicts(nlp("I noticed the phone."), nlp("I didn't notice him.")) == []

    def test_number_conflict(self, nlp):
        from rule_checks import number_conflicts
        assert number_conflicts(nlp("I only had two beers."), nlp("He drank six beers.")) == ["beer: 2 vs 6"]
        assert number_conflicts(nlp("I only had two beers."), nlp("He drank 2 beers.")) == []

    def test_hedges(self, nlp):
        from rule_checks import find_hedges
        assert find_hedges(nlp("I couldn't see whether he took anything, but I'm not sure.")) == \
            ["could", "did not see", "not sure"]
        assert find_hedges(nlp("The suspect took the money at 8 AM.")) == []
