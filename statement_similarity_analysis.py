import re
from collections import defaultdict

import numpy as np

import config
import models
from data import load_cases


_PRON_STOP = {"i", "you", "he", "she", "we", "they", "it"}
_FILLERS = {"the", "a", "an", "this", "that", "there", "here"}
_LINKERS = {"and", "or", "but", "then", "so", "because", "while", "when"}
_GENERIC = {"thing", "things", "something", "someone", "anything", "anyone"}
_LIGHT_VERBS = {"be", "do", "have", "get", "go", "make"}
_AUX_DEPS = {"aux", "auxpass", "cop"}
_NEG_WORDS = {"no", "not", "never", "n't"}
_MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
_UNCERT_ADV = {"maybe", "possibly", "probably", "likely", "apparently", "roughly", "approximately", "around", "about"}
_PREP_LOC = {"at", "in", "on", "near", "inside", "outside", "by", "opposite", "behind", "before", "after", "beside"}

_TIME_PATTERNS = [
    r"\b(at|around|about)\s+\d{1,2}(:\d{2})?\s*(am|pm)?\b",
    r"\b(noon|midnight|dawn|dusk)\b",
    r"\b(morning|afternoon|evening|night)\b",
    r"\b(yesterday|today|tomorrow)\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(last|this|next)\s+(morning|afternoon|evening|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
]

_POSITIVE_LEX = {"good", "great", "excellent", "positive", "calm", "polite", "apologetic", "cooperative"}
_NEGATIVE_LEX = {"bad", "terrible", "negative", "awful", "angry", "furious", "shout", "threaten", "steal", "hit"}

KEY_POINT_CATEGORIES = ['time', 'location', 'actions', 'entities', 'modifiers', 'sentiment', 'temporal', 'certainty',
                        'context']
# Categories that describe facts of the event (used for the case summary).
FACT_CATEGORIES = ['actions', 'time', 'location', 'entities', 'context']


def _clean_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _keep_token(tok) -> bool:
    # keep numbers & content words, drop most noise
    if tok.pos_ in {"PUNCT", "SYM", "SPACE", "DET", "CCONJ", "SCONJ", "PART"}:
        return False
    t = tok.lemma_.lower().strip()
    if t in _PRON_STOP or t in _FILLERS or t in _LINKERS or t in _GENERIC:
        return False
    if tok.is_stop and not tok.like_num:
        return False
    return True


def is_negated(tok) -> bool:
    return any(ch.dep_ == "neg" or ch.lower_ in _NEG_WORDS for ch in tok.children) or tok.dep_ == "neg"


def _collect_time(doc):
    times = set()
    for ent in doc.ents:
        if ent.label_ in {"DATE", "TIME"}:
            times.add(_clean_text(ent.text))
    s = doc.text.lower()
    for pat in _TIME_PATTERNS:
        for m in re.finditer(pat, s):
            times.add(_clean_text(m.group(0)))
    return times


def _collect_location(doc):
    locs = set()
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC", "FAC"}:
            locs.add(_clean_text(ent.text))
    for tok in doc:
        if tok.dep_ == "prep" and tok.lemma_.lower() in _PREP_LOC:
            pobj = next((c for c in tok.children if c.dep_ == "pobj"), None)
            if pobj:
                span = doc[tok.i:pobj.i + 1]
                locs.add(_clean_text(span.text))
    return locs


def _collect_entities(doc):
    ents = set()
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG"}:
            ents.add(_clean_text(ent.text))
    return ents


def _collect_actions(doc):
    acts = set()
    for tok in doc:
        if tok.pos_ == "VERB" and tok.dep_ not in _AUX_DEPS:
            lemma = tok.lemma_.lower()
            if lemma in _LIGHT_VERBS:
                continue
            tag = f"{lemma}[NEG]" if is_negated(tok) else lemma
            acts.add(tag)
    return acts


def _collect_modifiers(doc):
    mods = set()
    for tok in doc:
        if tok.pos_ in {"ADJ", "ADV"} and _keep_token(tok):
            t = tok.lemma_.lower()
            if t in _NEG_WORDS:  # negation ayrı tutulacak
                continue
            mods.add(_clean_text(tok.text))
    return mods


def _collect_temporal_tokens(doc):
    tmp = set()
    for tok in doc:
        if tok.dep_ in {"advmod", "aux"} and tok.lemma_.lower() in {"before", "after", "during", "while"}:
            tmp.add(tok.lemma_.lower())
    return tmp


def _collect_certainty(doc):
    cues = set()
    for tok in doc:
        t = tok.lemma_.lower()
        if t in _MODALS:
            cues.add(f"modal:{t}")
        if t in _UNCERT_ADV:
            cues.add(f"uncert:{t}")
    return cues


def _collect_context(doc):
    nps = set()
    for chunk in doc.noun_chunks:
        words = [w.lemma_.lower() for w in chunk if _keep_token(w)]
        cand = " ".join(words).strip()
        if len(cand) >= 3 and any(ch.isalpha() for ch in cand):
            nps.add(_clean_text(cand))
    return nps


def _compute_sentiment(doc):
    score = 0
    for tok in doc:
        if tok.pos_ in {"ADJ", "ADV", "VERB"}:
            t = tok.lemma_.lower()
            if t in _POSITIVE_LEX:
                score += 1
            if t in _NEGATIVE_LEX:
                score -= 1
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def extract_key_points(text):
    return extract_key_points_from_doc(models.get_nlp()(text))


def extract_key_points_from_doc(doc):
    time_items = _collect_time(doc)
    loc_items = _collect_location(doc)
    ent_items = _collect_entities(doc)
    act_items = _collect_actions(doc)
    mod_items = _collect_modifiers(doc)
    tmp_items = _collect_temporal_tokens(doc)
    cert_items = _collect_certainty(doc)
    ctx_items = _collect_context(doc)

    sentiment_label = _compute_sentiment(doc)

    key_points = {
        'time': list(sorted(time_items)),
        'location': list(sorted(loc_items)),
        'actions': list(sorted(act_items)),
        'entities': list(sorted(ent_items)),
        'modifiers': list(sorted(mod_items)),
        'sentiment': [sentiment_label],
        'temporal': list(sorted(tmp_items)),
        'certainty': list(sorted(cert_items)),
        'context': list(sorted(ctx_items))
    }
    return key_points


def compare_key_points(suspect_points, witness_points):
    """Exact (normalized) string matching per category.

    'different' holds every element mentioned by only one side, so extra details and synonyms count
    here too; it is not a list of contradictions.
    """
    comparison = {}
    for category in KEY_POINT_CATEGORIES:
        s_set = set(_clean_text(s) for s in suspect_points.get(category, []))
        w_set = set(_clean_text(w) for w in witness_points.get(category, []))
        comparison[category] = {
            'matching': sorted(s_set & w_set),
            'different': sorted(s_set ^ w_set),
            'suspect_only': sorted(s_set - w_set),
            'witness_only': sorted(w_set - s_set),
        }
    return comparison


def calculate_overlap_score(comparison):
    """Share of key-point elements mentioned by both statements (a Jaccard-style overlap, not a confidence)."""
    total_points = 0
    matching_points = 0

    for category in comparison:
        total_points += len(comparison[category]['matching']) + len(comparison[category]['different'])
        matching_points += len(comparison[category]['matching'])

    if total_points == 0:
        return 0.0

    return matching_points / total_points


def classify_consistency(similarity, overlap):
    """Baseline verdict rule; returns one of config.LABELS. Thresholds are uncalibrated (see config)."""
    if similarity > config.STRONG_SIMILARITY and overlap > config.STRONG_OVERLAP:
        return "consistent"
    if similarity > config.PARTIAL_SIMILARITY and overlap > config.PARTIAL_OVERLAP:
        return "partially_consistent"
    return "contradictory"


def generate_statistics(analysis_results):
    stats = {
        'total_cases': len(analysis_results),
        'average_similarity': np.mean([r['similarity'] for r in analysis_results]) if analysis_results else 0.0,
        'average_overlap': np.mean([r['overlap'] for r in analysis_results]) if analysis_results else 0.0,
        'consistency_distribution': defaultdict(int)
    }

    for result in analysis_results:
        stats['consistency_distribution'][config.LABEL_DISPLAY[result['verdict']]] += 1

    return stats


def create_visualizations(analysis_results, case_id):
    import matplotlib.pyplot as plt

    config.VISUALIZATION_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    similarities = [r['similarity'] for r in analysis_results]
    overlaps = [r['overlap'] for r in analysis_results]
    plt.scatter(similarities, overlaps, alpha=0.6)
    plt.xlabel('Similarity Score')
    plt.ylabel('Key-point Overlap')
    plt.title('Statement Analysis: Similarity vs Key-point Overlap')
    plt.grid(True)
    plt.savefig(config.VISUALIZATION_DIR / f'case_{case_id}_scatter.png')
    plt.close()

    # Category comparison bar chart
    if analysis_results and analysis_results[0].get('comparison'):
        categories = list(analysis_results[0]['comparison'].keys())
        matching_counts = []
        different_counts = []

        for category in categories:
            matching = sum(len(r['comparison'][category]['matching']) for r in analysis_results)
            different = sum(len(r['comparison'][category]['different']) for r in analysis_results)
            matching_counts.append(matching)
            different_counts.append(different)

        x = np.arange(len(categories))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width / 2, matching_counts, width, label='Matching')
        plt.bar(x + width / 2, different_counts, width, label='Mentioned by one side only')
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.title('Statement Analysis: Category Comparison')
        plt.xticks(x, categories, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.VISUALIZATION_DIR / f'case_{case_id}_categories.png')
        plt.close()


def generate_detailed_explanation(analysis):
    explanation = []

    # Overall assessment
    if analysis['verdict'] == "consistent":
        explanation.append("The statements show strong consistency in their accounts.")
        explanation.append("Key points align well, and there are minimal contradictions in the details provided.")
    elif analysis['verdict'] == "partially_consistent":
        explanation.append("The statements show partial consistency.")
        explanation.append(
            "While some key points align, there are notable differences that require further investigation.")
    else:
        explanation.append("The statements show significant contradictions.")
        explanation.append(
            "Key elements of the accounts differ substantially, suggesting potential inconsistencies in the testimony.")
    explanation.append(
        "(Baseline rule on similarity + key-point overlap with uncalibrated thresholds; treat as a hint only.)")
    explanation.append("\nDetailed Analysis by Category:")

    for category, details in analysis['comparison'].items():
        if details['matching'] or details['different']:
            explanation.append(f"\n{category.upper()}:")
            if details['matching']:
                explanation.append(f"  • Matching elements: {', '.join(details['matching'])}")
            if details['suspect_only']:
                explanation.append(f"  • Only in suspect statement: {', '.join(details['suspect_only'])}")
            if details['witness_only']:
                explanation.append(f"  • Only in witness statement: {', '.join(details['witness_only'])}")

    return "\n".join(explanation)


def analyze_statements(suspect, witness):
    embeddings = models.get_similarity_model().encode(
        [config.SIMILARITY_PREFIX + suspect, config.SIMILARITY_PREFIX + witness], normalize_embeddings=True)
    similarity = float(np.dot(embeddings[0], embeddings[1]))

    suspect_points = extract_key_points(suspect)
    witness_points = extract_key_points(witness)
    comparison = compare_key_points(suspect_points, witness_points)
    overlap = calculate_overlap_score(comparison)

    return {
        'similarity': similarity,
        'comparison': comparison,
        'overlap': overlap,
        'verdict': classify_consistency(similarity, overlap),
    }


def analyze_new_statements(suspect_statement, witness_statements, case_description=""):
    results = {
        'case_description': case_description,
        'suspect_statement': suspect_statement,
        'witness_analyses': []
    }

    for i, witness in enumerate(witness_statements):
        analysis = analyze_statements(suspect_statement, witness)
        analysis['witness_statement'] = witness
        analysis['witness_number'] = i + 1
        analysis['detailed_explanation'] = generate_detailed_explanation(analysis)
        results['witness_analyses'].append(analysis)

    return results


def print_analysis_results(results):
    from tabulate import tabulate

    print(f"\n{'=' * 100}")
    if results['case_description']:
        print(f"📁 Case: {results['case_description']}")
    print(f"👤 Suspect Statement: {results['suspect_statement']}")
    print(f"{'=' * 100}")

    table_data = []
    for analysis in results['witness_analyses']:
        table_data.append([
            f"Witness {analysis['witness_number']}",
            f"{analysis['similarity']:.2f}",
            f"{analysis['overlap']:.2f}",
            config.LABEL_DISPLAY[analysis['verdict']]
        ])

    print("\n📊 Analysis Summary Table:")
    print(tabulate(table_data,
                   headers=['Witness', 'Similarity', 'Key-point Overlap', 'Assessment'],
                   tablefmt='grid'))

    for analysis in results['witness_analyses']:
        print(f"\n{'=' * 100}")
        print(f"🧾 Witness {analysis['witness_number']} Statement: {analysis['witness_statement']}")
        print(f"{'=' * 100}")

        print(f"\n📊 Analysis Results:")
        print(f"   🔍 Overall Similarity Score: {analysis['similarity']:.2f}")
        print(f"   🎯 Key-point Overlap: {analysis['overlap']:.2f}")

        print("\n📝 Detailed Explanation:")
        print(analysis['detailed_explanation'])

        print(f"\n{'-' * 100}")


if __name__ == "__main__":
    # Process all cases
    all_analysis_results = []
    for case in load_cases():
        case_analysis = [analyze_statements(case['suspect_statement'], witness)
                         for witness in case['witness_statements']]
        all_analysis_results.append(case_analysis)
        create_visualizations(case_analysis, case['case_id'])

    flat = [item for sublist in all_analysis_results for item in sublist]
    stats = generate_statistics(flat)
    print("\n📈 Overall Analysis Statistics:")
    print(f"Total Cases Analyzed: {stats['total_cases']}")
    print(f"Average Similarity Score: {stats['average_similarity']:.2f}")
    print(f"Average Key-point Overlap: {stats['average_overlap']:.2f}")
    print("\nConsistency Distribution:")
    for category, count in stats['consistency_distribution'].items():
        pct = (count / stats['total_cases'] * 100) if stats['total_cases'] else 0
        print(f"{category}: {count} cases ({pct:.1f}%)")
