# -*- coding: utf-8 -*-
"""Classify new SMS messages with the trained TF-IDF + Linear SVM model.

Usage:
    python src/predict.py "Congratulations! You won a free prize, call now"
    python src/predict.py          # runs a few built-in examples
"""

import sys
import unicodedata

import joblib

MODEL_PATH = "models/svm_spam.joblib"


def predict_message(text, model=None):
    """Return ("spam" | "ham", decision_score). Higher score = more spam-like."""
    model = model or joblib.load(MODEL_PATH)
    text = unicodedata.normalize("NFKC", text)  # same cleaning as training
    score = float(model.decision_function([text])[0])
    return ("spam" if score > 0 else "ham"), score


if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)
    messages = sys.argv[1:] or [
        "Congratulations! You won a free prize. Call now to claim.",
        "Are we still meeting for lunch tomorrow?",
    ]
    for msg in messages:
        label, score = predict_message(msg, model)
        print(f"[{label.upper():4}] ({score:+.2f})  {msg}")
