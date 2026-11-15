import pytest

pytest.importorskip("sklearn")

from benchmarks import binary_metrics, split_dev_test  # noqa: E402


def test_split_dev_test_is_a_disjoint_cover():
    splits = split_dev_test(list(range(7)))
    assert splits == {"dev": [0, 2, 4, 6], "test": [1, 3, 5]}


def test_binary_metrics():
    metrics = binary_metrics([True, True, False, False], [0.9, 0.4, 0.85, 0.1], threshold=0.8)
    assert metrics["auc"] == 0.75
    assert metrics["precision"] == 0.5 and metrics["recall"] == 0.5 and metrics["f1"] == 0.5


def test_dev_tuned_threshold_is_applied_to_test():
    from benchmarks import add_dev_tuned_result, best_threshold

    gold = {"dev": [True, True, False, False], "test": [True, False]}
    scores = {"dev": [0.6, 0.55, 0.3, 0.1], "test": [0.5, 0.45]}
    assert best_threshold(gold["dev"], scores["dev"]) == 0.35   # lowest threshold with a perfect dev F1
    system = {}
    add_dev_tuned_result(system, gold, scores)
    assert system["dev_tuned_threshold"] == 0.35
    assert system["test_at_dev_tuned_threshold"]["recall"] == 1.0
    assert system["test_at_dev_tuned_threshold"]["precision"] == 0.5
