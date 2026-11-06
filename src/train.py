# -*- coding: utf-8 -*-
"""
Spam Detection - fair comparison of four models on the SMS Spam dataset.

Models: TF-IDF + Linear SVM, Dense NN (Embedding), GRU, LSTM
All models use the same stratified train/test split and are scored on the
same held-out test set. spam is the positive class.
"""

import os
import random
import joblib
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Embedding, Dense, Dropout, GRU, LSTM,
                                     GlobalAveragePooling1D, BatchNormalization)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------
# 1. Load data (Colab or local)
# ---------------------------------------------------------------
# Local default: data/spam.csv. Override with the SPAM_CSV environment variable.
CSV_PATH = os.environ.get("SPAM_CSV", "data/spam.csv")
try:  # Google Colab support (optional)
    from google.colab import drive
    drive.mount("/content/drive")
    CSV_PATH = os.environ.get(
        "SPAM_CSV", "/content/drive/MyDrive/Colab Notebooks/data seting/spam.csv")
except ImportError:
    pass

os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

df = pd.read_csv(CSV_PATH)
print("Raw shape:", df.shape)
print("Class counts:\n", df["Category"].value_counts())
print("Duplicate rows:", df.duplicated().sum())

# Duplicate messages would appear in both train and test, so drop them
df = df.drop_duplicates().reset_index(drop=True)
df["Message"] = df["Message"].apply(lambda t: unicodedata.normalize("NFKC", t))
print("Shape after removing duplicates:", df.shape)
print("Spam share (%):",
      round((df["Category"] == "spam").mean() * 100, 2))

le = LabelEncoder()                      # ham -> 0, spam -> 1
y = le.fit_transform(df["Category"])
X = df["Message"]

# ---------------------------------------------------------------
# 2. One stratified split shared by every model (70% / 30%)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ---------------------------------------------------------------
# 3. Tokenisation for the neural models (fitted on TRAIN only)
# ---------------------------------------------------------------
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
train_seq = tokenizer.texts_to_sequences(X_train)

# 95th percentile length instead of the longest message (less padding)
max_len = int(np.percentile([len(s) for s in train_seq], 95))
print("Sequence length (95th percentile):", max_len)

X_train_pad = pad_sequences(train_seq, maxlen=max_len, padding="post", truncating="post")
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test),
                           maxlen=max_len, padding="post", truncating="post")
vocab_size = len(tokenizer.word_index) + 1

results = {}
predictions = {}


def evaluate(name, y_pred):
    predictions[name] = y_pred
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    }
    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))


# ---------------------------------------------------------------
# 4. Model A: TF-IDF + Linear SVM
# ---------------------------------------------------------------
svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("svm", LinearSVC(C=1.0, random_state=SEED)),
])
svm_pipeline.fit(X_train, y_train)
evaluate("TF-IDF + Linear SVM", svm_pipeline.predict(X_test))

# Save the SVM pipeline so src/predict.py can classify new messages
joblib.dump(svm_pipeline, "models/svm_spam.joblib")

# ---------------------------------------------------------------
# 5. Neural models (same recipe: early stopping on a validation split)
# ---------------------------------------------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)


def train_and_eval(name, model):
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train_pad, y_train, epochs=15, batch_size=32, validation_split=0.2,
              callbacks=[early_stop], verbose=2)
    prob = model.predict(X_test_pad, verbose=0).ravel()
    evaluate(name, (prob > 0.5).astype(int))


# Dense NN (output layer is the LAST layer: nothing after the sigmoid)
dense_nn = Sequential([
    Input(shape=(max_len,)),
    Embedding(vocab_size, 32),
    GlobalAveragePooling1D(),
    Dense(128, activation="relu"), Dropout(0.2), BatchNormalization(),
    Dense(64, activation="relu"), Dropout(0.2), BatchNormalization(),
    Dense(1, activation="sigmoid"),
])
train_and_eval("Dense NN", dense_nn)

# GRU (default tanh activation)
gru = Sequential([
    Input(shape=(max_len,)),
    Embedding(vocab_size, 32, mask_zero=True),
    GRU(64), Dropout(0.2),
    Dense(1, activation="sigmoid"),
])
train_and_eval("GRU", gru)

# LSTM (default tanh activation)
lstm = Sequential([
    Input(shape=(max_len,)),
    Embedding(vocab_size, 32, mask_zero=True),
    LSTM(64), Dropout(0.2),
    Dense(1, activation="sigmoid"),
])
train_and_eval("LSTM", lstm)

# ---------------------------------------------------------------
# 6. Comparison + confusion matrix of the best model (by F1)
# ---------------------------------------------------------------
summary = pd.DataFrame(results).T.round(4).sort_values("F1", ascending=False)
print("\n===== Model comparison (test set, spam = positive class) =====")
print(summary)
summary.to_csv("results.csv")

best = summary.index[0]
print("\nBest model by F1:", best)
cm = confusion_matrix(y_test, predictions[best])
print("Confusion matrix (rows = actual, cols = predicted):\n", cm)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Confusion Matrix - {best}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=150)
plt.show()
