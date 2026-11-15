"""Sentence-level consistency check with an NLI model (natural language inference), plus rule checks.

Both statements are split into sentences. Every suspect sentence is compared with the whole witness statement
and every witness sentence with the whole suspect statement. The NLI model answers one of:
  entailment    -> the sentence is supported
  contradiction -> the sentence is contradicted
  neutral       -> the other statement does not talk about it

One problem: NLI models assume both texts describe the same moment, so "he was drinking coffee" against
"he picked up the phone" comes out as a contradiction. To avoid this, a contradiction only counts when the
sentence shares a content word with the other statement, or when it is a denial ("I never left the house").
"""
import re

import config
import models
from rule_checks import find_hedges, find_rule_conflicts
from statement_similarity_analysis import FACT_CATEGORIES

# The suspect says "I", witnesses say "the suspect". The model works better when both sides use the same name,
# so first-person words are rewritten. Longer forms must come before the plain "I".
_FIRST_PERSON = [
    (r"\bI'm\b", "{s} is"),
    (r"\bI am\b", "{s} is"),
    (r"\bI've\b", "{s} has"),
    (r"\bI have\b", "{s} has"),
    (r"\bI haven't\b", "{s} hasn't"),
    (r"\bI don't\b", "{s} doesn't"),
    (r"\bI do\b", "{s} does"),
    (r"\bI'll\b", "{s} will"),
    (r"\bI'd\b", "{s} would"),
    (r"\bI\b", "{s}"),
    (r"\bmyself\b", "{s}"),
    (r"\b[Mm]y\b", "{s}'s"),
    (r"\b[Mm]ine\b", "{s}'s"),
    (r"\b[Mm]e\b", "{s}"),
]

_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}
_GENERIC_LEMMAS = {"suspect", "be", "have", "do", "get", "go", "make", "say", "seem"}


def content_lemmas(span):
    """The meaningful words of a text as a set of lemmas."""
    lemmas = set()
    for tok in span:
        lemma = tok.lemma_.lower()
        if tok.pos_ in _CONTENT_POS and not tok.is_stop and lemma not in _GENERIC_LEMMAS:
            lemmas.add(lemma)
    return lemmas


def to_third_person(text, subject="the suspect"):
    """Rewrite "I ..." as "the suspect ..." (or as the given name)."""
    text = text.replace("’", "'")
    for pattern, replacement in _FIRST_PERSON:
        # A function as replacement, so a name with a backslash or "\1" in it is not treated as a pattern.
        new_text = replacement.replace("{s}", subject)
        text = re.sub(pattern, lambda match: new_text, text)
    return text[:1].upper() + text[1:]


def nli_probabilities(pairs):
    """(premise, hypothesis) pairs -> one dict per pair with the keys entailment, neutral, contradiction."""
    if not pairs:
        return []
    model = models.get_nli_model()
    labels = [model.config.id2label[i].lower() for i in range(len(model.config.id2label))]
    scores = model.predict(pairs, apply_softmax=True, batch_size=32, show_progress_bar=False)
    results = []
    for row in scores:
        results.append({label: float(score) for label, score in zip(labels, row)})
    return results


def claim_status(probabilities, related, statuses):
    """statuses = (name when supported, name when contradicted, name otherwise)."""
    if related and probabilities["contradiction"] >= config.NLI_STRONG_CONTRADICTION:
        return statuses[1]
    if probabilities["entailment"] >= config.NLI_SUPPORT_THRESHOLD:
        return statuses[0]
    return statuses[2]


def nli_verdict(contradiction):
    """Verdict from the NLI model alone. `contradiction` is the strongest contradiction that was found."""
    if contradiction >= config.NLI_STRONG_CONTRADICTION:
        return "contradictory"
    if contradiction >= config.NLI_CONTRADICTION_THRESHOLD:
        return "partially_consistent"
    return "consistent"


def hybrid_verdict(contradiction, rule_conflicts, hedges):
    """NLI plus rules. A rule conflict alone is not enough for "contradictory", and an uncertain witness
    ("I couldn't see") gives at most "partially consistent" (see labels/GUIDELINES.md)."""
    has_rule_conflict = any(rule_conflicts.values())
    strong = contradiction >= config.NLI_STRONG_CONTRADICTION
    weak = contradiction >= config.NLI_CONTRADICTION_THRESHOLD
    if not hedges and (strong or (weak and has_rule_conflict)):
        return "contradictory"
    if weak or has_rule_conflict or hedges:
        return "partially_consistent"
    return "consistent"


def _sentences(doc):
    return [sent for sent in doc.sents if sent.text.strip()]


def _is_denial(sentence):
    return any(tok.dep_ == "neg" for tok in sentence)


def _check_sentences(sentences, probabilities, other_lemmas, statuses):
    results = []
    for sentence, p in zip(sentences, probabilities):
        related = bool(content_lemmas(sentence) & other_lemmas) or _is_denial(sentence)
        results.append({
            "claim": sentence.text.strip(),
            "status": claim_status(p, related, statuses),
            "related": related,
            "support": p["entailment"],
            "contradiction": p["contradiction"],
        })
    return results


def analyze_consistency(suspect, witness):
    nlp = models.get_nlp()
    suspect_doc = nlp(suspect)
    witness_doc = nlp(witness)
    suspect_sentences = _sentences(suspect_doc)
    witness_sentences = _sentences(witness_doc)

    # One call to the model for all pairs: first the suspect sentences, then the witness sentences.
    pairs = []
    for sentence in suspect_sentences:
        pairs.append((witness, to_third_person(sentence.text.strip())))
    suspect_text = to_third_person(suspect)
    for sentence in witness_sentences:
        pairs.append((suspect_text, sentence.text.strip()))
    probabilities = nli_probabilities(pairs)
    suspect_probabilities = probabilities[:len(suspect_sentences)]
    witness_probabilities = probabilities[len(suspect_sentences):]

    suspect_results = _check_sentences(suspect_sentences, suspect_probabilities, content_lemmas(witness_doc),
                                       ("supported", "contradicted", "not_mentioned"))
    witness_results = _check_sentences(witness_sentences, witness_probabilities, content_lemmas(suspect_doc),
                                       ("agrees", "conflicts", "new_detail"))

    # The strongest contradiction, with and without the shared-word filter.
    contradiction = 0.0
    contradiction_ungated = 0.0
    for result in suspect_results + witness_results:
        contradiction_ungated = max(contradiction_ungated, result["contradiction"])
        if result["related"]:
            contradiction = max(contradiction, result["contradiction"])

    supported = 0
    for result in suspect_results:
        if result["status"] == "supported":
            supported += 1
    support_ratio = supported / len(suspect_results) if suspect_results else 0.0

    rule_conflicts = find_rule_conflicts(suspect_doc, witness_doc)
    hedges = find_hedges(witness_doc)
    return {
        "suspect_claims": suspect_results,
        "witness_claims": witness_results,
        "contradiction": contradiction,
        "contradiction_ungated": contradiction_ungated,
        "support_ratio": support_ratio,
        "rule_conflicts": rule_conflicts,
        "hedges": hedges,
        "nli_ungated_verdict": nli_verdict(contradiction_ungated),
        "nli_verdict": nli_verdict(contradiction),
        "verdict": hybrid_verdict(contradiction, rule_conflicts, hedges),
    }


def summarize_case(witness_analyses, max_shared=6):
    """Short text summary of a case, built only from the analysis results. Each analysis has the baseline
    result ('comparison') and the sentence-level result under 'consistency'."""
    lines = []
    for analysis in witness_analyses:
        result = analysis["consistency"]
        contradicted = 0
        for claim in result["suspect_claims"]:
            if claim["status"] == "contradicted":
                contradicted += 1
        conflicting = 0
        for claim in result["witness_claims"]:
            if claim["status"] == "conflicts":
                conflicting += 1

        lines.append(f"Witness {analysis['witness_number']}: {config.LABEL_DISPLAY[result['verdict']]}")
        lines.append(f"  Suspect claims contradicted: {contradicted}/{len(result['suspect_claims'])}; "
                     f"witness claims in conflict: {conflicting}/{len(result['witness_claims'])}")
        if result["hedges"]:
            lines.append(f"  Witness is uncertain: {', '.join(result['hedges'])}")

        shared = []
        for category in FACT_CATEGORIES:
            shared.extend(analysis["comparison"][category]["matching"])
        if shared:
            lines.append(f"  Shared facts: {', '.join(shared[:max_shared])}")
        else:
            lines.append("  Shared facts: none found")
    return "\n".join(lines)
