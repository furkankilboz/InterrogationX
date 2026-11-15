import csv

import pytest

import config
from data import iter_pairs, load_cases
from make_labeling_template import FIELDS, write_template


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_template_matches_dataset():
    rows = read_rows(config.LABELING_TEMPLATE)
    pairs = list(iter_pairs(load_cases()))
    assert [row["pair_id"] for row in rows] == [pair["pair_id"] for pair in pairs]
    for row, pair in zip(rows, pairs):
        assert row["suspect_statement"] == pair["suspect_statement"]
        assert row["witness_statement"] == pair["witness_statement"]


def test_template_labels_are_empty_or_valid():
    for row in read_rows(config.LABELING_TEMPLATE):
        assert row["label"] in ("",) + config.LABELS


def test_write_template_refuses_to_overwrite(tmp_path):
    path = tmp_path / "template.csv"
    write_template(path)
    assert list(read_rows(path)[0]) == FIELDS
    with pytest.raises(FileExistsError):
        write_template(path)
