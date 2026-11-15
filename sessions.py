"""Interrogation sessions: question-answer turns with manual behavioral observations, plus the evidence
(witness statements, known facts) the answers are checked against. Stored as JSON files.

    {"session_id": "...", "title": "...", "subject": "Jane Doe", "baseline_turns": 1,
     "evidence": [{"label": "Witness 1", "text": "..."}],
     "turns": [{"question": "...", "answer": "...",
                "observations": [{"channel": "hands", "cue": "fidgeting", "intensity": 2, "note": ""}]}]}
"""
import json
import re

from data import split_statements
from observations import CHANNELS, INTENSITIES

_LABELED = re.compile(r"^\[([^\]]{1,60})\] *(.+)$", re.DOTALL)


def new_session(session_id, title, subject="the suspect", evidence=None, baseline_turns=1):
    return {"session_id": session_id, "title": title, "subject": subject, "baseline_turns": baseline_turns,
            "evidence": list(evidence or []), "turns": []}


def add_turn(session, question, answer, observations=None):
    session["turns"].append({"question": question.strip(), "answer": answer.strip(),
                             "observations": list(observations or [])})


def session_from_case(case):
    """A dataset case as a one-turn session: the suspect statement is the answer, the witnesses are evidence."""
    session = new_session(f"case-{case['case_id']}", case["description"], baseline_turns=0,
                          evidence=[{"label": f"Witness {i}", "text": text}
                                    for i, text in enumerate(case["witness_statements"], start=1)])
    add_turn(session, "Please tell me in your own words what happened.", case["suspect_statement"])
    return session


def validation_problems(session):
    problems = []
    for key in ("session_id", "title", "subject", "evidence", "turns"):
        if key not in session:
            problems.append(f"missing field: {key}")
    if problems:
        return problems
    if not isinstance(session.get("baseline_turns", 0), int) or session.get("baseline_turns", 0) < 0:
        problems.append("baseline_turns must be a non-negative integer")
    for i, item in enumerate(session["evidence"], start=1):
        if not item.get("label") or not item.get("text", "").strip():
            problems.append(f"evidence {i}: needs a label and text")
    for i, turn in enumerate(session["turns"], start=1):
        if not turn.get("answer", "").strip():
            problems.append(f"turn {i}: empty answer")
        for observation in turn.get("observations", []):
            if observation.get("channel") not in CHANNELS:
                problems.append(f"turn {i}: unknown channel {observation.get('channel')!r}")
            if observation.get("intensity", 2) not in INTENSITIES:
                problems.append(f"turn {i}: intensity must be one of {sorted(INTENSITIES)}")
            if not observation.get("cue", "").strip():
                problems.append(f"turn {i}: observation without a cue")
    return problems


def load_session(path):
    with open(path, encoding="utf-8") as f:
        session = json.load(f)
    problems = validation_problems(session)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    session.setdefault("baseline_turns", 1)
    for turn in session["turns"]:
        turn.setdefault("question", "")
        turn.setdefault("observations", [])
    return session


def save_session(session, path):
    problems = validation_problems(session)
    if problems:
        raise ValueError("; ".join(problems))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)


def evidence_to_text(evidence):
    """Evidence items as editable text: one paragraph per item, written as "[Label] text"."""
    return "\n\n".join(f"[{item['label']}] {item['text']}" for item in evidence)


def evidence_from_text(text):
    """Inverse of evidence_to_text. Paragraphs without a "[Label]" get "Evidence 1", "Evidence 2", ..."""
    evidence = []
    for block in split_statements(text):
        match = _LABELED.match(block)
        label, body = (match.group(1).strip(), match.group(2).strip()) if match else ("", block)
        evidence.append({"label": label or f"Evidence {len(evidence) + 1}", "text": body})
    return evidence
