"""
train_model.py
--------------
Standalone training script for the complaint classifier.

Usage:
  python model/train_model.py
  python model/train_model.py --model naive_bayes
  python model/train_model.py --dataset path/to/custom_dataset.csv

This script:
  1. Loads the dataset CSV
  2. Preprocesses all complaint texts (tokenize, stopword removal)
  3. Trains a TF-IDF + Logistic Regression (or Naive Bayes) pipeline
  4. Evaluates accuracy on the held-out test split
  5. Saves the model as a pickle file to model/complaint_classifier.pkl
  6. Prints a detailed classification report
"""

import os
import sys
import csv
import argparse
import pickle
import logging
import numpy as np

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Handle both model/ and backend/ locations
if not os.path.exists(os.path.join(ROOT, "backend")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from backend.classifier import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET = os.path.join(ROOT, "dataset", "complaints_dataset.csv")
DEFAULT_MODEL_OUT = os.path.join(ROOT, "model", "complaint_classifier.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["complaint"].strip())
            labels.append(row["category"].strip())
    logger.info(f"Loaded {len(texts)} samples from {path}")
    return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(texts, labels, model_type="logistic", output_path=DEFAULT_MODEL_OUT):
    print("\n" + "═"*60)
    print("  ResolveAI  –  Model Training")
    print("═"*60)

    # ── Step 1: Preprocess ───────────────────────────────────────────────────
    print("\n[1/5] Preprocessing texts…")
    processed = [preprocess_text(t) for t in texts]
    print(f"      Sample: '{texts[0][:60]}…'")
    print(f"      → '{processed[0]}'")

    # ── Step 2: Train/test split ─────────────────────────────────────────────
    print("\n[2/5] Splitting dataset (80/20)…")
    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Category distribution
    from collections import Counter
    dist = Counter(labels)
    print("\n      Category distribution:")
    for cat, cnt in sorted(dist.items()):
        print(f"        {cat:<12} {cnt} samples")

    # ── Step 3: Build pipeline ───────────────────────────────────────────────
    print(f"\n[3/5] Building TF-IDF + {model_type.title()} pipeline…")

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
        sublinear_tf=True
    )

    if model_type == "naive_bayes":
        clf = MultinomialNB(alpha=0.5)
    else:
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="auto")

    pipeline = Pipeline([("tfidf", tfidf), ("clf", clf)])

    # ── Step 4: Train ────────────────────────────────────────────────────────
    print(f"\n[4/5] Training {model_type.title()} classifier…")
    pipeline.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(pipeline, processed, labels, cv=5, scoring="accuracy")
    print(f"      5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Step 5: Evaluate ─────────────────────────────────────────────────────
    print(f"\n[5/5] Evaluating on test set…")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n      Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("\n" + "─"*60)
    print("  Classification Report")
    print("─"*60)
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("  Confusion Matrix")
    print("─"*60)
    categories = sorted(set(labels))
    cm = confusion_matrix(y_test, y_pred, labels=categories)

    # Pretty-print confusion matrix
    col_width = max(len(c) for c in categories) + 2
    header = " " * col_width + "".join(c.ljust(col_width) for c in categories)
    print("  " + header)
    for row_label, row in zip(categories, cm):
        row_str = "  " + row_label.ljust(col_width) + "".join(str(v).ljust(col_width) for v in row)
        print(row_str)

    # ── Save model ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\n✓ Model saved to: {output_path}")
    print("═"*60 + "\n")

    return pipeline, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# TEST PREDICTIONS (DEMO)
# ─────────────────────────────────────────────────────────────────────────────

def demo_predictions(pipeline):
    test_complaints = [
        "My internet keeps cutting out every evening",
        "I was charged twice for my subscription this month",
        "The support agent was extremely rude and unhelpful",
        "What are your business hours?",
        "The app crashes when I try to login",
        "I never received my promised refund"
    ]

    print("\n  Demo Predictions")
    print("─"*60)
    print(f"  {'Complaint':<45} {'Predicted':>12}  {'Confidence':>10}")
    print("─"*60)

    for complaint in test_complaints:
        processed = preprocess_text(complaint)
        pred = pipeline.predict([processed])[0]
        prob = max(pipeline.predict_proba([processed])[0])
        print(f"  {complaint[:44]:<45} {pred:>12}  {prob:>9.1%}")

    print("─"*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResolveAI – Train complaint classifier")
    parser.add_argument("--model", choices=["logistic", "naive_bayes"], default="logistic",
                        help="Classifier type (default: logistic)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Path to dataset CSV (default: dataset/complaints_dataset.csv)")
    parser.add_argument("--output", default=DEFAULT_MODEL_OUT,
                        help="Output path for the saved model pickle")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset not found at {args.dataset}")
        sys.exit(1)

    texts, labels = load_dataset(args.dataset)
    pipeline, accuracy = train(texts, labels, model_type=args.model, output_path=args.output)
    demo_predictions(pipeline)

    print(f"Training complete! Model saved and ready to use.\n")
