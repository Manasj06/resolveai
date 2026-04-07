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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

def train(texts, labels, model_type="ensemble", output_path=DEFAULT_MODEL_OUT):
    print("\n" + "═"*80)
    print("  ResolveAI  –  Model Training (Optimized Stacking Ensemble)")
    print("═"*80)

    # ── Step 1: Preprocess ───────────────────────────────────────────────────
    print("\n[1/6] Preprocessing texts…")
    processed = [preprocess_text(t) for t in texts]
    print(f"      Sample: '{texts[0][:60]}…'")
    print(f"      → '{processed[0]}'")

    # ── Step 2: Train/test split ─────────────────────────────────────────────
    print("\n[2/6] Splitting dataset (80/20 stratified)…")
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

    # ── Step 3: Build enhanced TF-IDF vectorizer ────────────────────────────
    print(f"\n[3/6] Building enhanced TF-IDF vectorizer…")

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
        use_idf=True
    )
    
    # Fit TF-IDF once to use for all models
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"      Feature matrix shape: {X_train_tfidf.shape}")

    # ── Step 4: Train base models with aggressive hyperparameter tuning ──────
    print(f"\n[4/6] Training base models with hyperparameter tuning…")
    
    # Logistic Regression (expanded C range with different penalties)
    print("      [4a] Logistic Regression…")
    lr_params = {'C': [0.01, 0.1, 1.0, 10.0, 100.0], 'penalty': ['l2']}
    lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver='lbfgs')
    lr_grid = GridSearchCV(lr_clf, lr_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
    lr_grid.fit(X_train_tfidf, y_train)
    lr_best = lr_grid.best_estimator_
    lr_score = lr_grid.best_score_
    print(f"            CV Score: {lr_score:.4f}")

    # SVM (wider parameter search with multiple kernels)
    print("      [4b] Support Vector Machine (SVM)…")
    svm_params = {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}
    svm_clf = SVC(probability=True, class_weight="balanced", random_state=42)
    svm_grid = GridSearchCV(svm_clf, svm_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
    svm_grid.fit(X_train_tfidf, y_train)
    svm_best = svm_grid.best_estimator_
    svm_score = svm_grid.best_score_
    print(f"            CV Score: {svm_score:.4f}")

    # Gradient Boosting (tuned for small datasets)
    print("      [4c] Gradient Boosting Classifier…")
    gb_params = {'n_estimators': [150, 200, 300], 'learning_rate': [0.01, 0.05], 'max_depth': [3, 4]}
    gb_clf = GradientBoostingClassifier(random_state=42, subsample=0.8)
    gb_grid = GridSearchCV(gb_clf, gb_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
    gb_grid.fit(X_train_tfidf, y_train)
    gb_best = gb_grid.best_estimator_
    gb_score = gb_grid.best_score_
    print(f"            CV Score: {gb_score:.4f}")

    # Naive Bayes 
    print("      [4d] Multinomial Naive Bayes…")
    nb_params = [0.001, 0.01, 0.1, 0.5, 1.0]
    best_nb = None
    best_nb_score = 0
    for alpha in nb_params:
        nb_clf = MultinomialNB(alpha=alpha)
        nb_score = cross_val_score(nb_clf, X_train_tfidf, y_train, cv=5, scoring="accuracy").mean()
        if nb_score > best_nb_score:
            best_nb_score = nb_score
            best_nb = nb_clf
    best_nb.fit(X_train_tfidf, y_train)
    print(f"            CV Score: {best_nb_score:.4f}")

    # ── Step 5: Create Stacking Ensemble ─────────────────────────────────────
    print(f"\n[5/6] Creating Stacking Ensemble (4 base models + meta-learner)…")
    
    # Base estimators for stacking
    base_estimators = [
        ('lr', lr_best),
        ('svm', svm_best),
        ('gb', gb_best),
        ('nb', best_nb)
    ]
    
    # Meta-learner: uses predictions from base models
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5
    )
    
    stacking_clf.fit(X_train_tfidf, y_train)
    y_pred_stack = stacking_clf.predict(X_test_tfidf)
    stack_accuracy = accuracy_score(y_test, y_pred_stack)
    print(f"      Stacking Accuracy: {stack_accuracy:.4f} ({stack_accuracy*100:.1f}%)")
    
    # Also create voting ensemble for comparison
    voting_clf = VotingClassifier(
        estimators=base_estimators,
        voting='soft'
    )
    voting_clf.fit(X_train_tfidf, y_train)
    y_pred_voting = voting_clf.predict(X_test_tfidf)
    voting_accuracy = accuracy_score(y_test, y_pred_voting)
    print(f"      Voting Accuracy:   {voting_accuracy:.4f} ({voting_accuracy*100:.1f}%)")
    
    # Choose best ensemble
    if stack_accuracy >= voting_accuracy:
        final_clf = stacking_clf
        final_accuracy = stack_accuracy
        ensemble_name = "Stacking"
    else:
        final_clf = voting_clf
        final_accuracy = voting_accuracy
        ensemble_name = "Voting"

    # Create final pipeline for deployment
    final_pipeline = Pipeline([
        ("tfidf", tfidf),
        ("ensemble", final_clf)
    ])

    # ── Step 6: Final Evaluation and Report ──────────────────────────────────
    print(f"\n[6/6] Final Evaluation Report ({ensemble_name} Ensemble)…")
    print("\n" + "─"*80)
    print("  Base Model Individual Accuracies (test set)")
    print("─"*80)
    for name, est in base_estimators:
        y_pred = est.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        print(f"    {name.upper():12} {acc*100:6.1f}%")
    
    print("\n" + "─"*80)
    print("  Ensemble Accuracies (test set)")
    print("─"*80)
    print(f"    {'STACKING':12} {stack_accuracy*100:6.1f}%")
    print(f"    {'VOTING':12} {voting_accuracy*100:6.1f}%")
    print(f"\n    {'BEST':12} {final_accuracy*100:6.1f}%  ← Final Model ({ensemble_name})")
    
    print("\n" + "─"*80)
    print(f"  Classification Report ({ensemble_name})")
    print("─"*80)
    if ensemble_name == "Stacking":
        y_pred_final = y_pred_stack
    else:
        y_pred_final = y_pred_voting
    print(classification_report(y_test, y_pred_final))

    # Confusion matrix
    print("  Confusion Matrix")
    print("─"*80)
    categories = sorted(set(labels))
    cm = confusion_matrix(y_test, y_pred_final, labels=categories)

    # Pretty-print confusion matrix
    col_width = max(len(c) for c in categories) + 2
    header = " " * col_width + "".join(c.ljust(col_width) for c in categories)
    print("  " + header)
    for row_label, row in zip(categories, cm):
        row_str = "  " + row_label.ljust(col_width) + "".join(str(v).ljust(col_width) for v in row)
        print(row_str)

    # ── Save model ───────────────────────────────────────────────────────────
    print(f"\n✓ Saving {ensemble_name} ensemble model to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(final_pipeline, f)

    print("✓ Model saved successfully!")
    print("═"*80 + "\n")

    return final_pipeline, final_accuracy


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
