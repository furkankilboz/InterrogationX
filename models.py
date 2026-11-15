"""Lazy model loaders: nothing heavy is loaded at import time, and each model is loaded once."""
import os
from functools import lru_cache

import config

# This project only uses PyTorch; skip the (slow) TensorFlow import inside transformers.
os.environ.setdefault("USE_TF", "0")


@lru_cache(maxsize=None)
def get_nlp():
    import spacy
    return spacy.load(config.SPACY_MODEL)


@lru_cache(maxsize=None)
def get_similarity_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.SIMILARITY_MODEL)


@lru_cache(maxsize=None)
def get_sentiment_pipeline():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=config.SENTIMENT_MODEL)


@lru_cache(maxsize=None)
def get_emotion_pipeline():
    from transformers import pipeline
    return pipeline("text-classification", model=config.EMOTION_MODEL)


def get_nli_model(name=None):
    """The configured NLI cross-encoder, or another Hub model by name (used to compare models)."""
    return _load_nli_model(name or config.NLI_MODEL)


@lru_cache(maxsize=None)
def _load_nli_model(name):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name)
