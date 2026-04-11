"""
Late fusion of audio and text emotion predictions.

Both inputs should be either None or a (label, confidence) tuple.
Output is always None or a (label, confidence) tuple.
"""

# Maps text-emotion model labels → audio-emotion model labels
LABEL_MAP = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fearful",
    "surprise": "surprised",
}

# Audio is trained on acted speech (RAVDESS); text model works on spoken content
# → give text slightly more weight when it's available
AUDIO_WEIGHT = 0.4
TEXT_WEIGHT = 0.6


def normalize_label(label):
    """Map text-emotion labels to audio-emotion labels."""
    if not label:
        return None
    return LABEL_MAP.get(label.lower(), label.lower())


def fuse(audio_emotion, text_emotion):
    """
    Fuse audio and text emotion predictions.

    Args:
        audio_emotion: None or (label: str, confidence: float)
        text_emotion:  None or (label: str, confidence: float)

    Returns:
        None or (label: str, confidence: float)
    """
    # ── Neither source available ─────────────────────────────────────────
    if not audio_emotion and not text_emotion:
        return None

    # ── Only audio available ─────────────────────────────────────────────
    if audio_emotion and not text_emotion:
        label, score = audio_emotion
        return (normalize_label(label), score)

    # ── Only text available ──────────────────────────────────────────────
    if text_emotion and not audio_emotion:
        label, score = text_emotion
        return (normalize_label(label), score)

    # ── Both available — weighted fusion ─────────────────────────────────
    a_label, a_score = audio_emotion
    t_label, t_score = text_emotion

    a_label = normalize_label(a_label)
    t_label = normalize_label(t_label)

    # If both agree → combine confidences
    if a_label == t_label:
        combined = AUDIO_WEIGHT * a_score + TEXT_WEIGHT * t_score
        return (a_label, combined)

    # Conflict → weighted comparison
    a_weighted = AUDIO_WEIGHT * a_score
    t_weighted = TEXT_WEIGHT * t_score

    if a_weighted >= t_weighted:
        return (a_label, a_score)
    else:
        return (t_label, t_score)