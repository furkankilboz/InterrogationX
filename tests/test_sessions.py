import pytest

import config
from data import load_cases
from external_datasets import parse_alibi_conversation, split_stage_directions
from observations import make_observation
from sessions import (add_turn, evidence_from_text, evidence_to_text, load_session, new_session, save_session,
                      session_from_case, validation_problems)

CARD = """You are playing the character of Dana Reed in an interrogation scene.

## Character Profile
- **Name:** Dana Reed
- **Age:** 40

## Secret Knowledge (NEVER reveal directly)
- Dana took the cash box from the office at 9 PM
- Dana is GUILTY

## Alibi (LIE)
- Claims she left work at 6 PM
"""


def test_save_and_load_round_trip(tmp_path):
    session = new_session("s1", "Test", subject="Dana", evidence=[{"label": "CCTV", "text": "Seen at 9 PM."}])
    add_turn(session, " Where were you? ", " At home. ", [make_observation("eyes", "looks away")])
    path = tmp_path / "s1.json"
    save_session(session, path)
    assert load_session(path) == session
    assert session["turns"][0]["question"] == "Where were you?"


def test_validation_reports_problems(tmp_path):
    session = new_session("s1", "Test")
    session["turns"].append({"question": "Q", "answer": " ", "observations": [{"channel": "aura", "cue": "x"}]})
    problems = validation_problems(session)
    assert "turn 1: empty answer" in problems and "turn 1: unknown channel 'aura'" in problems
    with pytest.raises(ValueError):
        save_session(session, tmp_path / "bad.json")


def test_session_from_case_uses_witnesses_as_evidence():
    case = load_cases()[0]
    session = session_from_case(case)
    assert validation_problems(session) == []
    assert session["turns"][0]["answer"] == case["suspect_statement"]
    assert [item["text"] for item in session["evidence"]] == case["witness_statements"]


def test_evidence_text_round_trip():
    evidence = evidence_from_text("[Witness 1] He ran.\n\nNo label here.\n\n[CCTV] Seen at 9 PM.")
    assert evidence == [{"label": "Witness 1", "text": "He ran."}, {"label": "Evidence 2", "text": "No label here."},
                        {"label": "CCTV", "text": "Seen at 9 PM."}]
    assert evidence_from_text(evidence_to_text(evidence)) == evidence


def test_sample_sessions_are_valid_and_cover_both_kinds():
    sessions = [load_session(path) for path in sorted(config.SESSIONS_DIR.glob("alibi-*.json"))]
    assert {session["ground_truth"]["alibi"] for session in sessions} == {"lie", "truth"}
    assert all(session["turns"] and session["evidence"] for session in sessions)


def test_split_stage_directions():
    assert split_stage_directions("*looks away* I was home. *nervous laugh* All night.") == (
        "I was home. All night.", ["looks away", "nervous laugh"])


def test_parse_alibi_conversation():
    conversation = {"conversations": [
        {"from": "system", "value": CARD},
        {"from": "human", "value": "When did you leave?"},
        {"from": "gpt", "value": "*picks at nails* Around six. *voice drops* I went straight home."},
    ]}
    session = parse_alibi_conversation(conversation, 7)
    assert session["session_id"] == "alibi-0007" and session["subject"] == "Dana Reed"
    assert session["ground_truth"] == {"alibi": "lie"}
    # The line that gives the answer away ("is GUILTY") is not part of the evidence.
    assert session["evidence"] == [{"label": "Fact 1", "text": "Dana took the cash box from the office at 9 PM"}]
    assert session["turns"][0]["answer"] == "Around six. I went straight home."
    assert [(o["channel"], o["cue"]) for o in session["turns"][0]["observations"]] == [
        ("hands", "picks at nails"), ("voice", "voice drops")]
    assert validation_problems(session) == []


def test_parse_alibi_conversation_skips_unlabeled_cards():
    card = CARD.replace("## Alibi (LIE)", "## Alibi (PARTIAL LIE)")
    assert parse_alibi_conversation({"conversations": [{"from": "system", "value": card}]}, 0) is None
