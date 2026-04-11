"""
Text-based emotion detection using a pre-trained DistilRoBERTa model.
The model is lazy-loaded on the first call to avoid blocking startup.
"""

_classifier = None


def _get_classifier():
    """Lazy-load the HuggingFace pipeline on first use."""
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline
            _classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=1
            )
            print("Text emotion model loaded successfully.")
        except Exception as e:
            print(f"Failed to load text emotion model: {e}")
    return _classifier


def get_text_emotion(text):
    """
    Predict the emotion from text.

    Returns:
        tuple (label, score) on success, or None if text is empty or model unavailable.
    """
    if not text or text.strip() == "":
        return None

    classifier = _get_classifier()
    if classifier is None:
        return None

    try:
        result = classifier(text)[0][0]
        label = result["label"].lower()
        score = float(result["score"])
        return (label, score)
    except Exception as e:
        print(f"Text emotion error: {e}")
        return None