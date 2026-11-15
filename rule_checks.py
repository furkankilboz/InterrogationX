"""Simple rule checks next to the NLI model, which is weak on clock times, numbers and negation.

The functions take spaCy docs (plain text for clock times).
"""
import re

import config
from statement_similarity_analysis import extract_key_points_from_doc, is_negated

# Verbs about seeing, knowing or intending. "I didn't notice" means the person is unsure, it is not a fact.
_NON_FACTUAL_VERBS = {"see", "notice", "hear", "tell", "confirm", "know", "remember", "realize", "think", "mean",
                      "intend", "want", "try"}
_HEDGE_VERBS = {"see", "notice", "hear", "tell", "confirm", "know", "remember"}
_HEDGE_MODALS = {"might", "may", "could"}
_HEDGE_WORDS = {"maybe", "perhaps", "possibly", "possible", "probably", "apparently", "unclear", "unsure"}
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                 "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50,
                 "hundred": 100}

# "9 AM", "9:30 p.m.", "11pm"
_CLOCK = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\b\.?", re.IGNORECASE)
_NAMED_TIME = re.compile(r"\b(noon|midnight)\b", re.IGNORECASE)
MINUTES_PER_DAY = 24 * 60


def clock_times(text):
    """Clock times in a text as (minutes after midnight, original text). "at 8" without am/pm is skipped."""
    times = []
    for match in _CLOCK.finditer(text):
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        is_pm = match.group(3).lower() == "p"
        if hour < 1 or hour > 12 or minute > 59:
            continue
        hour = hour % 12          # 12 AM is hour 0, 12 PM is hour 12
        if is_pm:
            hour += 12
        times.append((hour * 60 + minute, match.group(0).strip().rstrip(".")))
    for match in _NAMED_TIME.finditer(text):
        minutes = 12 * 60 if match.group(1).lower() == "noon" else 0
        times.append((minutes, match.group(0)))
    return times


def _minutes_apart(a, b):
    """Distance between two clock times; 11 PM and midnight are 60 minutes apart."""
    difference = abs(a - b) % MINUTES_PER_DAY
    return min(difference, MINUTES_PER_DAY - difference)


def time_conflicts(suspect_text, witness_text):
    """A conflict when both texts give clock times and no two of them are close to each other."""
    suspect_times = clock_times(suspect_text)
    witness_times = clock_times(witness_text)
    if not suspect_times or not witness_times:
        return []
    for suspect_minutes, _ in suspect_times:
        for witness_minutes, _ in witness_times:
            if _minutes_apart(suspect_minutes, witness_minutes) <= config.TIME_CONFLICT_MINUTES:
                return []
    suspect_part = ", ".join(text for _, text in suspect_times)
    witness_part = ", ".join(text for _, text in witness_times)
    return [f"{suspect_part} vs {witness_part}"]


def _action_polarity(doc):
    """(verbs that are stated, verbs that are denied) in a text."""
    positive = set()
    negative = set()
    for action in extract_key_points_from_doc(doc)["actions"]:
        if action.endswith("[NEG]"):
            negative.add(action[:-len("[NEG]")])
        else:
            positive.add(action)
    return positive - _NON_FACTUAL_VERBS, negative - _NON_FACTUAL_VERBS


def polarity_conflicts(suspect_doc, witness_doc):
    """Verbs that one side states and the other denies: "I didn't take it" vs "he took it"."""
    suspect_positive, suspect_negative = _action_polarity(suspect_doc)
    witness_positive, witness_negative = _action_polarity(witness_doc)
    return sorted((suspect_negative & witness_positive) | (suspect_positive & witness_negative))


def _number_value(text):
    text = text.lower().replace(",", "")
    if text.isdigit():
        return int(text)
    return _NUMBER_WORDS.get(text)


def _counts(doc):
    """noun -> set of numbers it is counted with ("two beers" -> {"beer": {2}})."""
    counts = {}
    for tok in doc:
        if tok.dep_ == "nummod" and tok.head.pos_ == "NOUN":
            value = _number_value(tok.text)
            if value is not None:
                noun = tok.head.lemma_.lower()
                if noun not in counts:
                    counts[noun] = set()
                counts[noun].add(value)
    return counts


def number_conflicts(suspect_doc, witness_doc):
    """The same noun counted differently: "two beers" vs "six beers"."""
    suspect_counts = _counts(suspect_doc)
    witness_counts = _counts(witness_doc)
    conflicts = []
    for noun in sorted(suspect_counts):
        if noun not in witness_counts:
            continue
        if suspect_counts[noun] & witness_counts[noun]:
            continue    # at least one number is the same
        suspect_numbers = ", ".join(str(n) for n in sorted(suspect_counts[noun]))
        witness_numbers = ", ".join(str(n) for n in sorted(witness_counts[noun]))
        conflicts.append(f"{noun}: {suspect_numbers} vs {witness_numbers}")
    return conflicts


def find_rule_conflicts(suspect_doc, witness_doc):
    return {
        "polarity": polarity_conflicts(suspect_doc, witness_doc),
        "time": time_conflicts(suspect_doc.text, witness_doc.text),
        "number": number_conflicts(suspect_doc, witness_doc),
    }


def find_hedges(witness_doc):
    """Words that show the speaker is unsure or did not see the important moment."""
    cues = set()
    for tok in witness_doc:
        lemma = tok.lemma_.lower()
        if lemma in _HEDGE_MODALS and tok.tag_ == "MD":
            cues.add(lemma)
        elif lemma in _HEDGE_WORDS:
            cues.add(lemma)
        elif lemma == "sure" and (is_negated(tok) or (tok.dep_ == "acomp" and is_negated(tok.head))):
            cues.add("not sure")    # "I'm not sure": the "not" hangs on "am", the head of "sure"
        elif lemma in _HEDGE_VERBS and tok.pos_ == "VERB" and is_negated(tok):
            cues.add(f"did not {lemma}")
    return sorted(cues)
