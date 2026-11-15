import csv

import pytest

pytest.importorskip("sklearn")

import evaluate  # noqa: E402


def write_labels(path, labels):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "label"])
        writer.writerows(labels.items())


def test_metrics_perfect_and_confusion():
    gold = ["consistent", "contradictory", "partially_consistent", "contradictory"]
    m = evaluate.metrics(gold, gold)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["confusion_matrix"]["rows_gold_columns_predicted"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]


def test_read_rows_rejects_unknown_labels(tmp_path):
    path = tmp_path / "labels.csv"
    write_labels(path, {"001-W1": "true"})
    with pytest.raises(SystemExit):
        evaluate.read_rows(path)


def test_agreement_reports_kappa_and_disagreements(tmp_path, capsys):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_labels(a, {"1": "consistent", "2": "contradictory", "3": "contradictory", "4": ""})
    write_labels(b, {"1": "consistent", "2": "contradictory", "3": "partially_consistent", "4": "consistent"})
    evaluate.agreement(a, b)
    out = capsys.readouterr().out
    assert "Pairs labeled by both: 3" in out
    assert "Raw agreement: 0.67" in out
    assert "3: contradictory vs partially_consistent" in out


def test_fusion_predictions_are_out_of_fold():
    features, gold = [], []
    for i in range(12):
        label = evaluate.LABELS[i % 3]
        features.append({name: 0.0 for name in evaluate.FEATURES} | {"contradiction": (i % 3) / 2})
        gold.append(label)
    predicted, folds = evaluate.fusion_predictions(features, gold)
    assert len(predicted) == len(gold) and folds == 4


def test_fusion_needs_at_least_two_examples_per_class():
    features = [{name: 0.0 for name in evaluate.FEATURES}] * 3
    assert evaluate.fusion_predictions(features, ["consistent", "consistent", "contradictory"]) == (None, None)
