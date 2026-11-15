import pytest

import config
from nli_consistency import claim_status, hybrid_verdict, nli_verdict, to_third_person

NO_RULES = {"polarity": [], "time": [], "number": []}
STATUSES = ("supported", "contradicted", "not_mentioned")
# A contradiction probability between the weak and the strong threshold.
WEAK = (config.NLI_CONTRADICTION_THRESHOLD + config.NLI_STRONG_CONTRADICTION) / 2


@pytest.mark.parametrize("text, expected", [
    ("I didn’t touch my phone.", "The suspect didn't touch the suspect's phone."),
    ("I'm sure I've seen him before.", "The suspect is sure the suspect has seen him before."),
    ("He told me to leave.", "He told the suspect to leave."),
    ("I don't drink.", "The suspect doesn't drink."),
])
def test_to_third_person(text, expected):
    assert to_third_person(text) == expected


def test_claim_status_ignores_unrelated_contradictions():
    contradiction = {"entailment": 0.0, "neutral": 0.1, "contradiction": 0.9}
    assert claim_status(contradiction, True, STATUSES) == "contradicted"
    assert claim_status(contradiction, False, STATUSES) == "not_mentioned"
    assert claim_status({"entailment": 0.8, "neutral": 0.2, "contradiction": 0.0}, False, STATUSES) == "supported"


@pytest.mark.parametrize("contradiction, expected", [
    (0.95, "contradictory"), (WEAK, "partially_consistent"), (0.1, "consistent")])
def test_nli_verdict(contradiction, expected):
    assert nli_verdict(contradiction) == expected


@pytest.mark.parametrize("contradiction, rules, hedges, expected", [
    (0.95, NO_RULES, [], "contradictory"),
    (WEAK, {**NO_RULES, "polarity": ["take"]}, [], "contradictory"),   # weak NLI + rule
    (WEAK, NO_RULES, [], "partially_consistent"),                      # weak NLI alone
    (0.1, {**NO_RULES, "time": ["9 AM vs 3 PM"]}, [], "partially_consistent"),  # rule without NLI support
    (0.95, NO_RULES, ["might"], "partially_consistent"),              # uncertain witness caps the verdict
    (0.1, NO_RULES, [], "consistent"),
])
def test_hybrid_verdict(contradiction, rules, hedges, expected):
    assert hybrid_verdict(contradiction, rules, hedges) == expected


def test_content_lemmas_skip_function_and_generic_words():
    pytest.importorskip("spacy")
    pytest.importorskip("en_core_web_sm")
    import models
    from nli_consistency import content_lemmas
    # spaCy treats number words like "two" as stop words.
    assert content_lemmas(models.get_nlp()("The suspect was drinking coffee at two tables.")) == \
        {"drink", "coffee", "table"}


def test_summarize_case_counts_claims_and_lists_shared_facts():
    from nli_consistency import summarize_case
    from statement_similarity_analysis import compare_key_points

    consistency = {
        "verdict": "partially_consistent",
        "suspect_claims": [{"status": "contradicted"}, {"status": "supported"}],
        "witness_claims": [{"status": "new_detail"}],
        "hedges": ["might"],
    }
    comparison = compare_key_points({"actions": ["brake"], "certainty": ["modal:might"]},
                                    {"actions": ["brake"], "certainty": ["modal:might"]})
    summary = summarize_case([{"witness_number": 2, "consistency": consistency, "comparison": comparison}])
    assert summary.splitlines() == [
        "Witness 2: Partially consistent",
        "  Suspect claims contradicted: 1/2; witness claims in conflict: 0/1",
        "  Witness is uncertain: might",
        "  Shared facts: brake",
    ]


def test_gate_ignores_unrelated_claims_but_not_denials(monkeypatch):
    pytest.importorskip("spacy")
    pytest.importorskip("en_core_web_sm")
    import nli_consistency
    # Pretend the NLI model calls every pair a contradiction, so only the gate decides.
    monkeypatch.setattr(nli_consistency, "nli_probabilities",
                        lambda pairs: [{"entailment": 0.0, "neutral": 0.0, "contradiction": 1.0}] * len(pairs))
    result = nli_consistency.analyze_consistency("I was drinking coffee. I never left the house.",
                                                 "I saw him at the bus stop at 9 PM.")
    assert [c["status"] for c in result["suspect_claims"]] == ["not_mentioned", "contradicted"]
    assert [c["status"] for c in result["witness_claims"]] == ["new_detail"]
    assert result["contradiction"] == result["contradiction_ungated"] == 1.0
