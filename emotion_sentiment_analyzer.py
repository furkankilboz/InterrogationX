import models


def analyze_emotion_sentiment(text: str) -> dict:
    # truncation=True cuts at the model's 512-token limit instead of an arbitrary character count.
    sentiment_result = models.get_sentiment_pipeline()(text, truncation=True)[0]
    emotion_result = models.get_emotion_pipeline()(text, truncation=True, top_k=1)[0]

    return {
        "sentiment": sentiment_result["label"],
        "sentiment_score": sentiment_result["score"],
        "emotion": emotion_result["label"],
        "emotion_score": emotion_result["score"]
    }
