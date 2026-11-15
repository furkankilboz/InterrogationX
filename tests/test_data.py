from data import iter_pairs, load_cases, pair_id


def test_cases_have_required_fields():
    cases = load_cases()
    assert cases
    assert len({case["case_id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["description"].strip()
        assert case["suspect_statement"].strip()
        assert case["witness_statements"]
        assert all(witness.strip() for witness in case["witness_statements"])


def test_pair_ids_are_unique_and_stable():
    cases = load_cases()
    pairs = list(iter_pairs(cases))
    assert len(pairs) == sum(len(case["witness_statements"]) for case in cases)
    assert len({pair["pair_id"] for pair in pairs}) == len(pairs)
    assert pairs[0]["pair_id"] == pair_id(cases[0]["case_id"], 1) == f"{cases[0]['case_id']}-W1"


def test_split_statements_on_blank_lines():
    from data import split_statements
    text = "First witness,\n  over two lines.\n\n   \nSecond witness.\n\n"
    assert split_statements(text) == ["First witness, over two lines.", "Second witness."]
    assert split_statements("  \n\n ") == []
