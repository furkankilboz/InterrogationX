"""
Detection channels:
    1. Negation scope analysis
    2. Antonym detection
    3. Contradicting verb pairs
    4. Numerical discrepancy
    5. Temporal conflict
    6. Entity mismatch
    7. Quantifier opposition
    8. Property mismatch
"""

import re

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from constants import (
    ANTONYM_PAIRS,
    CHROMA_DIR,
    CONTRADICTING_VERB_PAIRS,
    NEGATION_WORDS,
    QUANTIFIER_OPPOSITES,
    THRESH_CORROBORATION,
    THRESH_PARTIAL,
    THRESH_RELATED,
    VERDICT_CONTRA,
    VERDICT_CORROB,
    VERDICT_STRONG_CONTRA,
    VERDICT_STRONG_CORROB,
    WORD_TO_NUMBER,
)

# Optional imports
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        HAS_LANGCHAIN = True
    except ImportError:
        HAS_LANGCHAIN = False

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


class StatementAnalyzer:

    def __init__(self):
        print("[StatementAnalyzer] Loading models ...")
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.tfidf = TfidfVectorizer(
            max_features=600,
            stop_words="english",
            ngram_range=(1, 2),
        )

        self._splitter = None
        if HAS_LANGCHAIN:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=220,
                chunk_overlap=30,
                separators=[".", "!", "?", "\n"],
            )

        self._vectorstore = None
        if HAS_LANGCHAIN and HAS_CHROMA:
            try:
                hf = HuggingFaceEmbeddings(
                    model_name="paraphrase-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                self._vectorstore = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=hf,
                    collection_name="forensic_stmts",
                )
            except Exception as exc:
                print(f"[VectorStore] skipped: {exc}")

        print("[StatementAnalyzer] Ready.")

    #  Sentence splitting

    def split_sentences(self, text):
        if self._splitter:
            parts = self._splitter.split_text(text)
            return [s.strip() for s in parts if len(s.strip()) >= 8]
        doc = self.nlp(text)
        return [s.text.strip() for s in doc.sents if len(s.text.strip()) >= 8]

    # Detection channels

    def _detect_negation(self, s1, s2):
        doc1 = self.nlp(s1.lower())
        doc2 = self.nlp(s2.lower())

        def verb_polarity(doc):
            result = {}
            for tok in doc:
                if tok.pos_ != "VERB" or tok.dep_ in ("aux", "auxpass"):
                    continue
                negated = any(
                    ch.dep_ == "neg" or ch.lower_ in NEGATION_WORDS
                    for ch in tok.children
                )
                if not negated:
                    negated = any(
                        ch.dep_ == "advmod" and ch.lower_ in NEGATION_WORDS
                        for ch in tok.children
                    )
                result[tok.lemma_] = negated
            return result

        vp1 = verb_polarity(doc1)
        vp2 = verb_polarity(doc2)

        shared = set(vp1) & set(vp2)
        for verb in shared:
            if vp1[verb] != vp2[verb]:
                return True, f"Verb '{verb}' is negated in one statement but not the other"

        # Fallback
        neg1 = any(t.lower_ in NEGATION_WORDS for t in doc1)
        neg2 = any(t.lower_ in NEGATION_WORDS for t in doc2)
        verbs1 = {t.lemma_ for t in doc1 if t.pos_ == "VERB" and t.dep_ not in ("aux", "auxpass")}
        verbs2 = {t.lemma_ for t in doc2 if t.pos_ == "VERB" and t.dep_ not in ("aux", "auxpass")}
        if (verbs1 & verbs2) and (neg1 != neg2):
            return True, "Opposing polarity: one statement negates shared actions"

        return False, None

    def _detect_antonyms(self, s1, s2):
        doc1 = self.nlp(s1)
        doc2 = self.nlp(s2)
        t1 = {tok.lemma_.lower() for tok in doc1} | {tok.lower_ for tok in doc1}
        t2 = {tok.lemma_.lower() for tok in doc2} | {tok.lower_ for tok in doc2}
        found = []
        for w in t1:
            ant = ANTONYM_PAIRS.get(w)
            if ant and ant in t2:
                found.append(f"{w} <> {ant}")
        if found:
            return True, f"Antonyms detected: {', '.join(found[:3])}"
        return False, None

    def _detect_verb_conflict(self, s1, s2):
        v1 = {t.lemma_.lower() for t in self.nlp(s1) if t.pos_ == "VERB"}
        v2 = {t.lemma_.lower() for t in self.nlp(s2) if t.pos_ == "VERB"}
        for a, b in CONTRADICTING_VERB_PAIRS:
            if (a in v1 and b in v2) or (b in v1 and a in v2):
                return True, f"Conflicting actions: {a} vs {b}"
        return False, None

    def _detect_numerical(self, s1, s2):
        # Detect conflicting numbers / quantities
        def extract_numbers(text):
            nums = [float(m) for m in re.findall(r'\b(\d+(?:\.\d+)?)\b', text)]
            for word, val in WORD_TO_NUMBER.items():
                if re.search(r'\b' + word + r'\b', text.lower()):
                    nums.append(float(val))
            return nums

        n1 = extract_numbers(s1)
        n2 = extract_numbers(s2)
        if n1 and n2:
            for a in n1:
                for b in n2:
                    if a != b and max(a, b) > 0:
                        ratio = abs(a - b) / max(a, b)
                        if ratio >= 0.15:
                            return True, f"Numerical discrepancy: {a} vs {b}"
        return False, None

    def _detect_temporal(self, s1, s2):
        # time references
        pattern = (
            r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)'
            r'|\d{1,2}\s*o\'?clock'
            r'|midnight|noon|dawn|dusk'
            r'|morning|afternoon|evening|night'
            r'|before|after|during|while)\b'
        )
        t1 = set(re.findall(pattern, s1.lower()))
        t2 = set(re.findall(pattern, s2.lower()))

        time_opposites = {
            "morning": {"night", "evening", "midnight"},
            "night": {"morning", "dawn", "noon"},
            "evening": {"morning", "dawn"},
            "before": {"after"},
            "after": {"before"},
        }

        if t1 and t2 and t1 != t2:
            for word in t1:
                opposites = time_opposites.get(word, set())
                if t2 & opposites:
                    return True, f"Temporal conflict: {word} vs {t2 & opposites}"
            times1 = set(re.findall(r'\d{1,2}(?::\d{2})?\s*(?:am|pm)', s1.lower()))
            times2 = set(re.findall(r'\d{1,2}(?::\d{2})?\s*(?:am|pm)', s2.lower()))
            if times1 and times2 and times1 != times2:
                return True, f"Different times stated: {times1} vs {times2}"
        return False, None

    def _detect_entity_mismatch(self, s1, s2):
        # Detect different named entities in related sentences
        doc1 = self.nlp(s1)
        doc2 = self.nlp(s2)

        persons1 = {e.text.lower() for e in doc1.ents if e.label_ == "PERSON"}
        persons2 = {e.text.lower() for e in doc2.ents if e.label_ == "PERSON"}
        locs1 = {e.text.lower() for e in doc1.ents if e.label_ in ("LOC", "GPE", "FAC")}
        locs2 = {e.text.lower() for e in doc2.ents if e.label_ in ("LOC", "GPE", "FAC")}

        conflicts = []
        if persons1 and persons2 and not (persons1 & persons2):
            conflicts.append(f"Different people: {persons1} vs {persons2}")
        if locs1 and locs2 and not (locs1 & locs2):
            conflicts.append(f"Different locations: {locs1} vs {locs2}")

        if conflicts:
            return True, "; ".join(conflicts)
        return False, None

    def _detect_quantifier(self, s1, s2):
        # Detect opposing quantifiers
        w1 = set(s1.lower().split())
        w2 = set(s2.lower().split())
        for quant, opposites in QUANTIFIER_OPPOSITES.items():
            if quant in w1 and (w2 & opposites):
                match = w2 & opposites
                return True, f"Opposing quantifiers: '{quant}' vs '{match.pop()}'"
            if quant in w2 and (w1 & opposites):
                match = w1 & opposites
                return True, f"Opposing quantifiers: '{match.pop()}' vs '{quant}'"
        return False, None

    def _detect_property_mismatch(self, s1, s2):
        doc1 = self.nlp(s1.lower())
        doc2 = self.nlp(s2.lower())

        def subject_attr_pairs(doc):
            pairs = set()
            for tok in doc:

                if tok.dep_ in ("acomp", "attr", "oprd") and tok.head.lemma_ == "be":
                    subj = next(
                        (c for c in tok.head.children if c.dep_ in ("nsubj", "nsubjpass")),
                        None,
                    )
                    if subj:
                        pairs.add((subj.lemma_, tok.lemma_))

                if tok.dep_ == "acomp" and tok.head.pos_ == "VERB":
                    subj = next(
                        (c for c in tok.head.children if c.dep_ in ("nsubj", "nsubjpass")),
                        None,
                    )
                    if subj:
                        pairs.add((subj.lemma_, tok.lemma_))

                if tok.dep_ == "amod" and tok.head.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "attr"):
                    pairs.add((tok.head.lemma_, tok.lemma_))
            return pairs

        p1 = subject_attr_pairs(doc1)
        p2 = subject_attr_pairs(doc2)

        for subj1, attr1 in p1:
            for subj2, attr2 in p2:
                if subj1 == subj2 and attr1 != attr2:
                    ant = ANTONYM_PAIRS.get(attr1)
                    if ant == attr2:
                        return True, f"Conflicting property of '{subj1}': {attr1} vs {attr2}"
                    if attr1 in ANTONYM_PAIRS or attr2 in ANTONYM_PAIRS:
                        return True, f"Different descriptions of '{subj1}': {attr1} vs {attr2}"
        return False, None

    def classify_pair(self, s1, s2, similarity):
        signals = []

        detectors = [
            (self._detect_negation, 0.35),
            (self._detect_antonyms, 0.25),
            (self._detect_verb_conflict, 0.25),
            (self._detect_numerical, 0.30),
            (self._detect_temporal, 0.30),
            (self._detect_entity_mismatch, 0.20),
            (self._detect_quantifier, 0.25),
            (self._detect_property_mismatch, 0.35),
        ]

        for detector, weight in detectors:
            found, reason = detector(s1, s2)
            if found:
                signals.append((reason, weight))

        if signals:
            signal_count = len(signals)
            total_weight = sum(w for _, w in signals)

            min_sim = 0.20 if signal_count >= 2 else 0.25

            if similarity >= min_sim:
                raw_conf = total_weight * (0.35 + similarity * 0.65)
                confidence = min(1.0, raw_conf)
                reasons = [r for r, _ in signals]
                return "CONTRADICTION", reasons, round(confidence, 3)

        if similarity >= THRESH_CORROBORATION:
            return "CORROBORATION", [], round(min(1.0, similarity), 3)
        if similarity >= THRESH_PARTIAL:
            return "PARTIAL_MATCH", [], round(similarity, 3)
        return "UNRELATED", [], 0.0


    def analyze(self, suspect_stmt, witness_stmts, case_id="unknown"):
        suspect_sents = self.split_sentences(suspect_stmt)

        all_texts = [suspect_stmt] + witness_stmts
        important_terms = set()
        try:
            tfidf_mat = self.tfidf.fit_transform(all_texts)
            features = self.tfidf.get_feature_names_out()
            suspect_scores = tfidf_mat[0].toarray().flatten()
            top_indices = suspect_scores.argsort()[-25:]
            important_terms = {features[i] for i in top_indices if suspect_scores[i] > 0}
        except Exception:
            pass

        witness_results = []

        for w_idx, w_stmt in enumerate(witness_stmts):
            w_sents = self.split_sentences(w_stmt)
            if not suspect_sents or not w_sents:
                continue

            s_embs = self.embedder.encode(suspect_sents, show_progress_bar=False)
            w_embs = self.embedder.encode(w_sents, show_progress_bar=False)
            sim_matrix = cosine_similarity(s_embs, w_embs)

            pairs = []
            for i, ss in enumerate(suspect_sents):
                for j, ws in enumerate(w_sents):
                    sim = float(sim_matrix[i][j])
                    if sim < THRESH_RELATED:
                        continue

                    rel, reasons, confidence = self.classify_pair(ss, ws, sim)

                    if rel == "CONTRADICTION" and important_terms:
                        pair_tokens = set(ss.lower().split()) | set(ws.lower().split())
                        key_overlap = pair_tokens & important_terms
                        if key_overlap:
                            boost = min(0.15, 0.05 * len(key_overlap))
                            confidence = min(1.0, confidence + boost)
                            confidence = round(confidence, 3)

                    if rel == "UNRELATED":
                        continue

                    pairs.append({
                        "suspect_sentence": ss,
                        "witness_sentence": ws,
                        "similarity": round(sim, 3),
                        "relationship": rel,
                        "reasons": reasons,
                        "confidence": confidence,
                    })

            contradictions = [p for p in pairs if p["relationship"] == "CONTRADICTION"]
            corroborations = [p for p in pairs if p["relationship"] == "CORROBORATION"]
            partial_matches = [p for p in pairs if p["relationship"] == "PARTIAL_MATCH"]

            total_conf = sum(p["confidence"] for p in pairs) or 1.0
            c_score = round(sum(p["confidence"] for p in contradictions) / total_conf, 3)
            s_score = round(sum(p["confidence"] for p in corroborations) / total_conf, 3)

            all_embs = np.vstack([s_embs, w_embs])
            if len(all_embs) > 2:
                n_clust = max(2, min(len(all_embs) // 3, len(all_embs) - 1))
                normed = normalize(all_embs)
                clusters = AgglomerativeClustering(
                    n_clusters=n_clust, metric="cosine", linkage="average"
                ).fit_predict(normed).tolist()
            else:
                clusters = list(range(len(all_embs)))

            verdict = self._compute_verdict(c_score, s_score, contradictions)

            witness_results.append({
                "witness_idx": w_idx + 1,
                "witness_statement": w_stmt,
                "suspect_sentences": suspect_sents,
                "witness_sentences": w_sents,
                "pairwise_analysis": pairs,
                "contradictions": contradictions,
                "corroborations": corroborations,
                "partial_matches": partial_matches,
                "contradiction_score": c_score,
                "corroboration_score": s_score,
                "verdict": verdict,
                "similarity_matrix": sim_matrix.tolist(),
                "topic_clusters": clusters,
            })

        if not witness_results:
            return {
                "case_id": case_id,
                "suspect_statement": suspect_stmt,
                "witness_analyses": [],
                "overall_contradiction_score": 0,
                "overall_corroboration_score": 0,
                "overall_verdict": "NO DATA",
            }

        oc = round(np.mean([r["contradiction_score"] for r in witness_results]), 3)
        os_ = round(np.mean([r["corroboration_score"] for r in witness_results]), 3)
        all_contras = [c for wr in witness_results for c in wr["contradictions"]]

        if self._vectorstore:
            try:
                all_sents = suspect_sents[:]
                meta = [
                    {"case_id": case_id, "role": "suspect", "idx": i}
                    for i in range(len(suspect_sents))
                ]
                for w_idx2, w_stmt2 in enumerate(witness_stmts):
                    ws = self.split_sentences(w_stmt2)
                    all_sents.extend(ws)
                    meta.extend([
                        {"case_id": case_id, "role": f"witness_{w_idx2 + 1}", "idx": j}
                        for j in range(len(ws))
                    ])
                ids = [f"{case_id}_{m['role']}_{m['idx']}" for m in meta]
                self._vectorstore.add_texts(texts=all_sents, metadatas=meta, ids=ids)
            except Exception:
                pass

        return {
            "case_id": case_id,
            "suspect_statement": suspect_stmt,
            "witness_analyses": witness_results,
            "overall_contradiction_score": oc,
            "overall_corroboration_score": os_,
            "overall_verdict": self._compute_verdict(oc, os_, all_contras),
        }

    def _compute_verdict(self, c_score, s_score, contradictions=None):
        has_strong = False
        if contradictions:
            has_strong = any(c["confidence"] >= 0.55 for c in contradictions)

        if c_score >= VERDICT_STRONG_CONTRA or (has_strong and c_score >= 0.20):
            return "HIGHLY CONTRADICTORY"
        if c_score >= VERDICT_CONTRA:
            return "CONTRADICTORY"
        if s_score >= VERDICT_STRONG_CORROB:
            return "STRONGLY CORROBORATING"
        if s_score >= VERDICT_CORROB:
            return "CORROBORATING"
        return "INCONCLUSIVE"
