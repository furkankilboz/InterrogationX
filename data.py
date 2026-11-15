"""Loading the statement dataset and enumerating suspect-witness pairs."""
import json
import re

import config


def load_cases(path=config.DATA_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pair_id(case_id, witness_index):
    """Stable id of one suspect-witness pair, shared by predictions and human labels (e.g. "001-W2")."""
    return f"{case_id}-W{witness_index}"


def iter_pairs(cases):
    for case in cases:
        for i, witness in enumerate(case["witness_statements"], start=1):
            yield {
                "pair_id": pair_id(case["case_id"], i),
                "case_id": case["case_id"],
                "witness_index": i,
                "description": case["description"],
                "suspect_statement": case["suspect_statement"],
                "witness_statement": witness,
            }


def split_statements(text):
    """Split pasted text into statements separated by empty lines; whitespace inside a statement is collapsed."""
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text) if block.strip()]
