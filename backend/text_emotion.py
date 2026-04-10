from transformers import pipeline

# Load once (heavy model)
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

def get_text_emotion(text):
    if not text or text.strip() == "":
        return None, 0.0

    try:
        result = classifier(text)[0][0]
        label = result["label"].lower()
        score = float(result["score"])
        return label, score
    except Exception as e:
        print(f"Text emotion error: {e}")
        return None, 0.0