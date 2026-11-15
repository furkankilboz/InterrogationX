import pytest

pytest.importorskip("spacy")
pytest.importorskip("en_core_web_sm")

import interrogation_analysis  # noqa: E402
import models  # noqa: E402
from observations import make_observation  # noqa: E402
from report import render_report  # noqa: E402
from sessions import add_turn, new_session  # noqa: E402


def fake_nli(pairs):
    """Stands in for the NLI model: only the two conflicts written into the test session are contradictions."""
    results = []
    for premise, hypothesis in pairs:
        p, h = premise.lower(), hypothesis.lower()
        conflict = ("bus stop" in p and "never left" in h) or ("never left" in p and "left the house at" in h)
        results.append({"entailment": 0.0, "neutral": 0.03 if conflict else 1.0,
                        "contradiction": 0.97 if conflict else 0.0})
    return results


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.setattr(interrogation_analysis, "nli_probabilities", fake_nli)
    session = new_session("t1", "Test <session>", subject="Alex", baseline_turns=1,
                          evidence=[{"label": "Witness", "text": "Alex was at the bus stop at 9 PM."}])
    add_turn(session, "How are you today?", "I am fine, thank you for asking today.",
             [make_observation("posture", "upright")])
    add_turn(session, "Where were you that evening?",
             "I never left the house that evening. I watched a film with my sister.",
             [make_observation("posture", "leaning away"), make_observation("hands", "fidgeting with sleeve")])
    add_turn(session, "Did you go to the bus stop?", "Honestly, um, I don't remember. Why are you asking me that?")
    add_turn(session, "Did you go out at all?", "I left the house at eight to buy <b>cigarettes</b>.")
    return interrogation_analysis.analyze_session(session)


def test_verbal_cues():
    doc = models.get_nlp()("Honestly, um, I don't remember. Why are you asking me that? Maybe around 9 PM in Boston.")
    cues = interrogation_analysis.verbal_cues("Where were you?", doc)
    assert cues["evasion"] == ["asks a question back", "memory gap or not knowing",
                               "deflects or challenges the question"]
    assert cues["bolstering"] == ["honestly"] and cues["fillers"] == 1 and "maybe" in cues["hedges"]
    assert cues["details"]["places"] == 1 and cues["details"]["times"] >= 1


def test_yes_no_question_without_a_direct_answer():
    nlp = models.get_nlp()
    indirect = interrogation_analysis.verbal_cues("Did you take the money?", nlp("People take things all the time."))
    direct = interrogation_analysis.verbal_cues("Did you take the money?", nlp("No. I was at home."))
    assert "no direct yes/no to a yes/no question" in indirect["evasion"]
    assert direct["evasion"] == []


def test_claims_skip_questions_and_fragments():
    doc = models.get_nlp()("The tea? I made it after dinner that night. Why do you ask me about it?")
    assert [c.text for c in interrogation_analysis.extract_claims(doc)] == ["I made it after dinner that night."]


def test_evidence_contradiction_is_found_and_attributed(analysis):
    assert [(c["turn"], c["claim"], c["evidence_label"]) for c in analysis["evidence_contradictions"]] == [
        (1, "I never left the house that evening.", "Witness")]
    assert analysis["summary"]["evidence_contradictions"] == 1


def test_internal_contradiction_points_to_the_earlier_sentence(analysis):
    assert [(c["later_turn"], c["earlier_turn"], c["earlier_statement"]) for c in
            analysis["internal_contradictions"]] == [(3, 1, "I never left the house that evening.")]


def test_contested_turns_and_baseline_deviations(analysis):
    assert [turn["contested"] for turn in analysis["turns"]] == [False, True, False, True]
    assert analysis["turns"][0]["is_baseline"] and analysis["baseline"] == {"posture": ["upright"]}
    assert [o["cue"] for o in analysis["turns"][1]["deviations"]] == ["leaning away"]
    assert analysis["behavior"]["deviations"]["contested"] == {"turns": 2, "deviations": 1, "per_turn": 0.5}
    assert analysis["behavior"]["deviations"]["uncontested"]["deviations"] == 0


def test_report_contains_findings_and_escapes_session_text(analysis):
    html = render_report(analysis)
    assert "Test &lt;session&gt;" in html and "&lt;b&gt;cigarettes&lt;/b&gt;" in html
    assert "<b>cigarettes</b>" not in html
    for expected in ("Statement-evidence contradictions", "Contradictions with earlier answers",
                     "Behavioral observations", "differs from baseline", "leaning away", "baseline turn"):
        assert expected in html


def test_session_without_evidence_or_observations(monkeypatch):
    monkeypatch.setattr(interrogation_analysis, "nli_probabilities", fake_nli)
    session = new_session("t2", "Bare", baseline_turns=0)
    add_turn(session, "", "I was at home all evening with my family.")
    analysis = interrogation_analysis.analyze_session(session)
    assert analysis["summary"]["contested_turns"] == 0 and analysis["evidence"] == []
    html = render_report(analysis)
    assert "No evidence was entered" in html and "No behavioral observations were recorded" in html


def test_evidence_item_is_marked_when_a_claim_conflicts_with_it(analysis):
    assert [(item["label"], item["status"], item["turn"]) for item in analysis["evidence"]] == [
        ("Witness", "contradicted", 1)]
    assert analysis["summary"]["evidence_items_contradicted"] == 1


def test_evidence_conflict_index():
    index = interrogation_analysis.evidence_conflict_index
    assert index([]) is None
    assert index([{"contradiction": 0.9, "support": 0.0}, {"contradiction": 0.1, "support": 0.8}]) == \
        pytest.approx(0.1)
    assert index([{"contradiction": 0.0, "support": 1.0}]) == -1.0


def test_report_shows_the_conflict_index(analysis):
    assert analysis["summary"]["evidence_conflict_index"] == pytest.approx(0.97)
    assert "Evidence conflict index: <b>+0.97</b>" in render_report(analysis)
