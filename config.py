"""All settings of the project in one place: file paths, model names and thresholds."""
from pathlib import Path

# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "forensic_statements_data_en.json"
EMOTION_OUTPUT_DIR = PROJECT_DIR / "emotion_outputs"
VISUALIZATION_DIR = PROJECT_DIR / "analysis_visualizations"
RESULTS_DIR = PROJECT_DIR / "results"
SESSIONS_DIR = PROJECT_DIR / "sessions"
REPORTS_DIR = PROJECT_DIR / "reports"
CUE_VOCABULARY = PROJECT_DIR / "cue_vocabulary.json"
# Downloaded datasets are kept here. The folder is in .gitignore.
EXTERNAL_DATA_DIR = PROJECT_DIR / "external_data"

LABELS_DIR = PROJECT_DIR / "labels"
LABELING_TEMPLATE = LABELS_DIR / "labeling_template.csv"
CHALLENGE_SET = LABELS_DIR / "challenge_set.csv"
GOLD_LABELS = LABELS_DIR / "gold.csv"

# --- Models (all from Hugging Face, except the spaCy model) ---
SIMILARITY_MODEL = "intfloat/e5-base-v2"
# e5 models want a prefix before every text. "query: " is the one to use when comparing two texts.
SIMILARITY_PREFIX = "query: "
SPACY_MODEL = "en_core_web_sm"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
# Chosen with benchmarks.py: better than cross-encoder/nli-deberta-v3-base on Dialogue NLI, and it does not
# give almost every interrogation session a contradiction score of 1.0 like that model did.
NLI_MODEL = "tasksource/deberta-base-long-nli"

# --- Labels ---
LABELS = ("consistent", "partially_consistent", "contradictory")
LABEL_DISPLAY = {
    "consistent": "Consistent",
    "partially_consistent": "Partially consistent",
    "contradictory": "Contradictory",
}

# --- Baseline thresholds (similarity + key-point overlap) ---
# These are not tuned. e5 similarities are almost always above 0.7 and the overlap is rarely above 0.3, so
# the baseline calls nearly every pair "contradictory". It stays as the starting point to compare against.
STRONG_SIMILARITY = 0.6
STRONG_OVERLAP = 0.5
PARTIAL_SIMILARITY = 0.4
PARTIAL_OVERLAP = 0.3

# --- NLI thresholds ---
NLI_SUPPORT_THRESHOLD = 0.5
# A sentence counts as contradicted from this probability on. 0.40 gave the best F1 on the dev half of the
# Dialogue NLI benchmark and was then used unchanged on the test half.
NLI_STRONG_CONTRADICTION = 0.40
# A weaker signal still makes a pair "partially consistent". Half of the value above, not tuned.
NLI_CONTRADICTION_THRESHOLD = NLI_STRONG_CONTRADICTION / 2
# Sentences in an interrogation session are flagged at the same threshold.
SESSION_CONTRADICTION_THRESHOLD = NLI_STRONG_CONTRADICTION

# Two clock times further apart than this (in minutes) count as a time conflict.
TIME_CONFLICT_MINUTES = 60
