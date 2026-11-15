"""Writes labels/labeling_template.csv: one row per suspect-witness pair with empty label columns.

Annotators copy the template (e.g. labels/annotator_A.csv) and fill `label` with one of config.LABELS,
following labels/GUIDELINES.md. An existing template is never overwritten unless --overwrite is given.
"""
import csv
import sys

import config
from data import iter_pairs, load_cases

FIELDS = ["pair_id", "case_id", "witness_index", "description", "suspect_statement", "witness_statement",
          "label", "notes"]


def write_template(path=config.LABELING_TEMPLATE, overwrite=False):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(exist_ok=True)
    # utf-8-sig so Excel shows quotes and non-ASCII characters correctly.
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for pair in iter_pairs(load_cases()):
            writer.writerow({**pair, "label": "", "notes": ""})
    return path


if __name__ == "__main__":
    print(f"Wrote {write_template(overwrite='--overwrite' in sys.argv)}")
