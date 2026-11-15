"""Analysis of an interrogation session (see sessions.py).

For every answer the analysis gives:
  - verbal cues: evasion, memory gaps, hedges, phrases like "honestly", filler words, amount of detail
  - evidence check: sentences that the evidence contradicts or supports
  - self-consistency: sentences that contradict an earlier answer
  - behavior: the manual observations and how they differ from the person's baseline turns

The two contradiction checks are the cues with the best research support (for example the "Strategic Use of
Evidence" work of Hartwig and Granhag). All results are hints for a human interviewer, never a verdict.
"""
import re

import config
import models
from nli_consistency import content_lemmas, nli_probabilities, to_third_person
from observations import baseline_cues, baseline_deviations, summarize_behavior
from rule_checks import find_hedges

_MEMORY_GAP = re.compile(
    r"\b(?:do not|don't|can't|cannot|couldn't|could not)\s+(?:really\s+|quite\s+|exactly\s+)?(?:remember|recall)\b"
    r"|\bno idea\b|\b(?:do not|don't)\s+know\b|\bnot sure\b", re.IGNORECASE)
_DEFLECTION = re.compile(
    r"\bwhy (?:are|do|would) you\b|\bwhat are you (?:implying|suggesting|saying|getting at)\b"
    r"|\bare you (?:accusing|saying|suggesting)\b|\bi want (?:a|my) lawyer\b"
    r"|\bam i (?:under arrest|a suspect|in trouble)\b", re.IGNORECASE)
_BOLSTERING = re.compile(
    r"\b(?:honestly|to be honest|frankly|i swear|believe me|trust me|to tell (?:you )?the truth|truthfully"
    r"|i promise)\b", re.IGNORECASE)
_FILLER = re.compile(r"\b(?:um+|uh+|erm?|i mean|you know)\b", re.IGNORECASE)
_DIRECT_ANSWER = re.compile(r"\b(?:yes|yeah|yep|no|nope|never|not|of course|absolutely|correct|right|sure)\b|n't",
                            re.IGNORECASE)
# A question starting with one of these words can be answered with yes or no.
_YES_NO_OPENERS = {"did", "do", "does", "were", "was", "are", "is", "have", "has", "had", "can", "could", "would",
                   "will", "should"}
# spaCy entity labels that count as a checkable detail.
_DETAIL_ENTITIES = {"times": {"TIME", "DATE"}, "places": {"GPE", "LOC", "FAC"}, "people": {"PERSON"},
                    "numbers": {"CARDINAL", "QUANTITY", "MONEY"}}
MIN_CLAIM_WORDS = 4


def verbal_cues(question, answer_doc):
    """Simple cues in one answer. `question` is plain text, `answer_doc` a spaCy doc."""
    answer = answer_doc.text.replace("’", "'")
    sentences = [sent.text.strip() for sent in answer_doc.sents if sent.text.strip()]

    evasion = []
    if any(sentence.endswith("?") for sentence in sentences):
        evasion.append("asks a question back")
    if _MEMORY_GAP.search(answer):
        evasion.append("memory gap or not knowing")
    if _DEFLECTION.search(answer):
        evasion.append("deflects or challenges the question")
    question_words = question.strip().lower().split()
    if question_words and question_words[0] in _YES_NO_OPENERS and sentences:
        if not _DIRECT_ANSWER.search(" ".join(sentences[:2])):
            evasion.append("no direct yes/no to a yes/no question")

    details = {}
    for name, labels in _DETAIL_ENTITIES.items():
        details[name] = 0
        for ent in answer_doc.ents:
            if ent.label_ in labels:
                details[name] += 1

    word_count = 0
    for tok in answer_doc:
        if tok.is_alpha:
            word_count += 1

    bolstering = set()
    for match in _BOLSTERING.finditer(answer):
        bolstering.add(match.group(0).lower())

    return {
        "evasion": evasion,
        "hedges": find_hedges(answer_doc),
        "bolstering": sorted(bolstering),
        "fillers": len(_FILLER.findall(answer)),
        "details": details,
        "word_count": word_count,
    }


def extract_claims(answer_doc):
    """The sentences that state something. Questions and very short fragments are left out."""
    claims = []
    for sent in answer_doc.sents:
        words = sum(1 for tok in sent if tok.is_alpha)
        if words >= MIN_CLAIM_WORDS and not sent.text.strip().endswith("?"):
            claims.append(sent)
    return claims


def _is_denial(span):
    return any(tok.dep_ == "neg" for tok in span)


def check_claims(session, answer_docs, evidence_docs):
    """Every claim against all the evidence together. Returns one list of claim results per turn."""
    subject = session["subject"]
    evidence_text = " ".join(doc.text.rstrip(".") + "." for doc in evidence_docs)
    evidence_lemmas = set()
    for doc in evidence_docs:
        evidence_lemmas |= content_lemmas(doc)

    results = []
    for doc in answer_docs:
        claims = extract_claims(doc)
        if evidence_docs:
            pairs = [(evidence_text, to_third_person(claim.text.strip(), subject)) for claim in claims]
            probabilities = nli_probabilities(pairs)
        else:
            probabilities = [{"entailment": 0.0, "contradiction": 0.0}] * len(claims)

        turn_results = []
        for claim, p in zip(claims, probabilities):
            # Same filter as in nli_consistency: the claim must share a word with the evidence or be a denial.
            related = bool(evidence_docs) and (bool(content_lemmas(claim) & evidence_lemmas) or _is_denial(claim))
            if related and p["contradiction"] >= config.SESSION_CONTRADICTION_THRESHOLD:
                status = "contradicted"
            elif p["entailment"] >= config.NLI_SUPPORT_THRESHOLD:
                status = "supported"
            else:
                status = "not_addressed"
            turn_results.append({"claim": claim.text.strip(), "status": status, "related": related,
                                 "support": p["entailment"], "contradiction": p["contradiction"]})
        results.append(turn_results)
    return results


def add_conflicting_item(session, claim_results, evidence_docs):
    """For every contradicted claim, find the evidence item it conflicts with most."""
    subject = session["subject"]
    for claims in claim_results:
        for claim in claims:
            if claim["status"] != "contradicted":
                continue
            pairs = [(doc.text, to_third_person(claim["claim"], subject)) for doc in evidence_docs]
            scores = [p["contradiction"] for p in nli_probabilities(pairs)]
            best = scores.index(max(scores))
            claim["evidence_label"] = session["evidence"][best]["label"]


def check_evidence_items(session, answer_docs, evidence_docs):
    """Every evidence item against every answer: does the account contradict it or acknowledge it?"""
    subject = session["subject"]
    answers = [to_third_person(doc.text, subject) for doc in answer_docs]
    answer_lemmas = [content_lemmas(doc) for doc in answer_docs]

    results = []
    for item, doc in zip(session["evidence"], evidence_docs):
        probabilities = nli_probabilities([(answer, doc.text) for answer in answers])
        item_lemmas = content_lemmas(doc)
        contradiction = 0.0
        contradiction_turn = None
        support = 0.0
        support_turn = None
        for turn, p in enumerate(probabilities):
            # A contradiction only counts when the answer talks about the same thing.
            if item_lemmas & answer_lemmas[turn] and p["contradiction"] > contradiction:
                contradiction = p["contradiction"]
                contradiction_turn = turn
            if p["entailment"] > support:
                support = p["entailment"]
                support_turn = turn

        if contradiction >= config.SESSION_CONTRADICTION_THRESHOLD:
            status = "contradicted"
            turn = contradiction_turn
        elif support >= config.NLI_SUPPORT_THRESHOLD:
            status = "acknowledged"
            turn = support_turn
        else:
            status = "not_addressed"
            turn = None
        results.append({"label": item["label"], "text": item["text"], "status": status, "turn": turn,
                        "contradiction": contradiction, "support": support})
    return results


def check_evidence(session, answer_docs, evidence_docs):
    """Both directions of the evidence check. Returns (claim results per turn, evidence results)."""
    claim_results = check_claims(session, answer_docs, evidence_docs)
    add_conflicting_item(session, claim_results, evidence_docs)
    evidence_results = check_evidence_items(session, answer_docs, evidence_docs)

    # If a claim was flagged against an item, the item counts as contradicted too.
    by_label = {item["label"]: item for item in evidence_results}
    for turn, claims in enumerate(claim_results):
        for claim in claims:
            item = by_label.get(claim.get("evidence_label"))
            if item and claim["contradiction"] > item["contradiction"]:
                item["status"] = "contradicted"
                item["turn"] = turn
                item["contradiction"] = claim["contradiction"]
    return claim_results, evidence_results


def evidence_conflict_index(evidence_results):
    """Average of (contradiction - support) over the evidence items.

    -1 means the account acknowledges every item, +1 means it contradicts every item; None without evidence.
    On the synthetic sessions of benchmarks.py this separates false from true alibis moderately well. The
    single strongest contradiction does not, because almost every long conversation contains one.
    """
    if not evidence_results:
        return None
    total = 0.0
    for item in evidence_results:
        total += item["contradiction"] - item["support"]
    return total / len(evidence_results)


def score_session_against_evidence(session):
    """Two session-level scores for the benchmark: the conflict index and the strongest contradiction."""
    nlp = models.get_nlp()
    answer_docs = [nlp(turn["answer"]) for turn in session["turns"]]
    evidence_docs = [nlp(item["text"]) for item in session["evidence"]]
    claim_results, evidence_results = check_evidence(session, answer_docs, evidence_docs)

    strongest = 0.0
    for claims in claim_results:
        for claim in claims:
            if claim["related"]:
                strongest = max(strongest, claim["contradiction"])
    for item in evidence_results:
        strongest = max(strongest, item["contradiction"])
    return {"conflict index": evidence_conflict_index(evidence_results), "strongest contradiction": strongest}


def check_self_consistency(session, answer_docs):
    """Claims that contradict a sentence of an earlier answer.

    Two sentences are only compared when they share a content word. For each later claim only the strongest
    conflict is kept.
    """
    subject = session["subject"]
    # Per turn: a list of (original text, third-person text, content words).
    claims_per_turn = []
    for doc in answer_docs:
        claims = []
        for claim in extract_claims(doc):
            text = claim.text.strip()
            claims.append((text, to_third_person(text, subject), content_lemmas(claim)))
        claims_per_turn.append(claims)

    candidates = []
    for later in range(len(claims_per_turn)):
        for new in claims_per_turn[later]:
            for earlier in range(later):
                for old in claims_per_turn[earlier]:
                    if old[2] & new[2]:
                        candidates.append((earlier, later, old, new))
    probabilities = nli_probabilities([(old[1], new[1]) for earlier, later, old, new in candidates])

    strongest = {}
    for (earlier, later, old, new), p in zip(candidates, probabilities):
        if p["contradiction"] < config.SESSION_CONTRADICTION_THRESHOLD:
            continue
        key = (later, new[0])
        if key not in strongest or p["contradiction"] > strongest[key]["contradiction"]:
            strongest[key] = {"earlier_turn": earlier, "later_turn": later, "claim": new[0],
                              "earlier_statement": old[0], "contradiction": p["contradiction"]}
    return sorted(strongest.values(), key=lambda result: result["contradiction"], reverse=True)


def analyze_session(session, with_emotion=False, progress=None):
    """The full analysis. `progress`, if given, is called with a number between 0 and 1."""
    def report(fraction):
        if progress:
            progress(fraction)

    nlp = models.get_nlp()
    answer_docs = [nlp(turn["answer"]) for turn in session["turns"]]
    evidence_docs = [nlp(item["text"]) for item in session["evidence"]]
    report(0.1)
    claim_results, evidence_results = check_evidence(session, answer_docs, evidence_docs)
    report(0.5)
    internal = check_self_consistency(session, answer_docs)
    report(0.85)

    # A turn is "contested" when one of its claims was flagged in any of the checks.
    contested_turns = set()
    evidence_contradictions = []
    for turn, claims in enumerate(claim_results):
        for claim in claims:
            if claim["status"] == "contradicted":
                contested_turns.add(turn)
                evidence_contradictions.append({"turn": turn, **claim})
    for item in evidence_results:
        if item["status"] == "contradicted":
            contested_turns.add(item["turn"])
    for result in internal:
        contested_turns.add(result["later_turn"])
    evidence_contradictions.sort(key=lambda claim: claim["contradiction"], reverse=True)

    baseline_turns = min(session.get("baseline_turns", 0), len(session["turns"]))
    baseline = baseline_cues(session["turns"], baseline_turns)
    turn_analyses = []
    for i, (turn, doc) in enumerate(zip(session["turns"], answer_docs)):
        is_baseline = i < baseline_turns
        analysis = {
            "index": i,
            "question": turn.get("question", ""),
            "answer": turn["answer"],
            "is_baseline": is_baseline,
            "claims": claim_results[i],
            "verbal": verbal_cues(turn.get("question", ""), doc),
            "observations": turn.get("observations", []),
            "deviations": [] if is_baseline else baseline_deviations(turn, baseline),
            "contested": i in contested_turns,
        }
        if with_emotion:
            from emotion_sentiment_analyzer import analyze_emotion_sentiment
            emotion = analyze_emotion_sentiment(turn["answer"])
            analysis["text_emotion"] = {"label": emotion["emotion"], "score": emotion["emotion_score"]}
        turn_analyses.append(analysis)
    report(1.0)

    items_contradicted = 0
    for item in evidence_results:
        if item["status"] == "contradicted":
            items_contradicted += 1
    evasive_turns = 0
    observation_count = 0
    for analysis in turn_analyses:
        if analysis["verbal"]["evasion"]:
            evasive_turns += 1
        observation_count += len(analysis["observations"])

    return {
        "session": {"session_id": session.get("session_id"), "title": session.get("title"),
                    "subject": session.get("subject"), "source": session.get("source")},
        "turns": turn_analyses,
        "evidence": evidence_results,
        "evidence_contradictions": evidence_contradictions,
        "internal_contradictions": internal,
        "baseline": {channel: sorted(cues) for channel, cues in baseline.items()},
        "behavior": summarize_behavior(turn_analyses, baseline_turns),
        "summary": {
            "turns": len(turn_analyses),
            "contested_turns": len(contested_turns),
            "evidence_contradictions": len(evidence_contradictions),
            "evidence_items_contradicted": items_contradicted,
            "evidence_conflict_index": evidence_conflict_index(evidence_results),
            "internal_contradictions": len(internal),
            "evasive_turns": evasive_turns,
            "observations": observation_count,
        },
    }
