import subprocess
import sys

import pytest

import config
from statement_similarity_analysis import calculate_overlap_score, classify_consistency, compare_key_points


def test_compare_key_points_separates_shared_and_one_sided_elements():
    comparison = compare_key_points({"actions": ["take[NEG]", "talk"], "time": ["8 am"]},
                                    {"actions": ["take", "talk"], "time": ["8  AM "]})
    assert comparison["actions"] == {
        "matching": ["talk"],
        "different": ["take", "take[neg]"],
        "suspect_only": ["take[neg]"],
        "witness_only": ["take"],
    }
    assert comparison["time"]["matching"] == ["8 am"]
    assert comparison["location"] == {"matching": [], "different": [], "suspect_only": [], "witness_only": []}


def test_overlap_score():
    assert calculate_overlap_score(compare_key_points({"actions": ["a", "b"]}, {"actions": ["b", "c"]})) == \
        pytest.approx(1 / 3)
    assert calculate_overlap_score(compare_key_points({}, {})) == 0.0


@pytest.mark.parametrize("similarity, overlap, expected", [
    (0.9, 0.6, "consistent"),
    (0.5, 0.4, "partially_consistent"),
    (0.9, 0.2, "contradictory"),
    (0.3, 0.9, "contradictory"),
])
def test_classify_consistency(similarity, overlap, expected):
    assert classify_consistency(similarity, overlap) == expected
    assert expected in config.LABELS


def test_importing_analysis_does_not_load_models():
    code = ("import sys, statement_similarity_analysis, emotion_sentiment_analyzer, nli_consistency, rule_checks, evaluate, interrogation_analysis, report, sessions, observations, benchmarks, external_datasets; "
            "print(any(m in sys.modules for m in ('spacy', 'sentence_transformers', 'transformers')))")
    result = subprocess.run([sys.executable, "-c", code], cwd=config.PROJECT_DIR, capture_output=True, text=True,
                            check=True)
    assert result.stdout.strip() == "False"
