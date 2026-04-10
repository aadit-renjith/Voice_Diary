def normalize_label(label):
    if not label:
        return None

    label = label.lower()

    mapping = {
        "joy": "happy",
        "sadness": "sad",
        "anger": "angry",
        "fear": "fearful",
        "surprise": "surprised"
    }

    return mapping.get(label, label)


def fuse(audio_emotion, text_emotion):
    if not audio_emotion and not text_emotion:
        return None

    if audio_emotion and not text_emotion:
        return audio_emotion

    if text_emotion and not audio_emotion:
        return text_emotion

    a_label, a_score = audio_emotion
    t_label, t_score = text_emotion

    t_label = normalize_label(t_label)
    a_label = normalize_label(a_label)

    # If both agree → combine confidence
    if a_label == t_label:
        return a_label, (a_score + t_score) / 2

    # Conflict → trust audio slightly more
    if a_score >= t_score:
        return a_label, a_score
    else:
        return t_label, t_score