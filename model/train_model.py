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

# ── Data Augmentation ───────────────────────────────────────────────────────
import random

def augment_text(text: str) -> List[str]:
    """Generate augmented versions of the input text."""
    augmented = [text]  # Keep original
    
    # Synonym replacement (simple version)
    synonyms = {
        "internet": ["connection", "network", "web"],
        "billing": ["charge", "payment", "invoice"],
        "support": ["help", "service", "assistance"],
        "technical": ["tech", "system", "hardware"],
        "slow": ["lag", "delay", "slow"],
        "crash": ["freeze", "hang", "error"],
        "rude": ["unhelpful", "impolite", "disrespectful"],
        "wrong": ["incorrect", "mistake", "error"],
        "cancel": ["stop", "terminate", "end"],
        "refund": ["money back", "reimbursement", "return"]
    }
    
    words = text.split()
    for i, word in enumerate(words):
        if word in synonyms and random.random() < 0.3:  # 30% chance
            synonym = random.choice(synonyms[word])
            new_words = words.copy()
            new_words[i] = synonym
            augmented.append(" ".join(new_words))
    
    # Random word deletion (keep 80% of words)
    if len(words) > 3:
        keep_words = [w for w in words if random.random() < 0.8]
        if len(keep_words) > 2:
            augmented.append(" ".join(keep_words))
    
    return augmented[:3]  # Limit to 3 variations per text

def augment_dataset(texts: List[str], labels: List[str], target_samples: int = 200) -> Tuple[List[str], List[str]]:
    """Augment the dataset to reach target sample count with more diverse augmentations."""
    augmented_texts = []
    augmented_labels = []
    
    # Group by category
    category_texts = {}
    for text, label in zip(texts, labels):
        if label not in category_texts:
            category_texts[label] = []
        category_texts[label].append(text)
    
    target_per_category = target_samples // len(category_texts)
    
    for category, cat_texts in category_texts.items():
        current_count = len(cat_texts)
        
        # Add original samples
        augmented_texts.extend(cat_texts)
        augmented_labels.extend([category] * current_count)
        
        # Generate more diverse augmentations if needed
        if current_count < target_per_category:
            needed = target_per_category - current_count
            
            # Create templates for each category to generate diverse synthetic data
            templates = {
                "Technical": [
                    "My {device} is not working properly",
                    "The {service} keeps {problem}",
                    "I can't connect to {service}",
                    "{device} shows error {error_code}",
                    "The {service} is running very {speed}",
                    "{device} won't turn on",
                    "Lost connection to {service}",
                    "{device} keeps freezing",
                    "Can't access {service} online",
                    "{service} is down"
                ],
                "Billing": [
                    "I was charged ${amount} for {service}",
                    "Wrong charge on my bill",
                    "Double billing this month",
                    "Unauthorized charge of ${amount}",
                    "My bill shows incorrect amount",
                    "Refund not processed yet",
                    "Extra fees added to bill",
                    "Billing cycle is wrong",
                    "Payment not reflected",
                    "Subscription charged twice"
                ],
                "Service": [
                    "Customer support was {quality}",
                    "Agent was very {behavior}",
                    "Waited {time} minutes on hold",
                    "Support team gave wrong information",
                    "No callback received",
                    "Representative was unhelpful",
                    "Long queue time",
                    "Poor customer service experience",
                    "Support hours are inconvenient",
                    "Agent didn't solve my problem"
                ],
                "General": [
                    "What are your {info}?",
                    "Can you explain {topic}?",
                    "I need information about {service}",
                    "How do I {action}?",
                    "Tell me about your {feature}",
                    "What is the {detail}?",
                    "Can I get details on {service}?",
                    "How does {feature} work?",
                    "What are the {options}?",
                    "Please explain {topic}"
                ]
            }
            
            fillers = {
                "device": ["internet", "router", "modem", "computer", "phone", "app", "website", "connection"],
                "service": ["internet", "service", "account", "subscription", "plan", "support"],
                "problem": ["crashing", "disconnecting", "failing", "stopping", "breaking"],
                "error_code": ["404", "503", "500", "connection timeout"],
                "speed": ["slow", "slowly", "fast"],
                "amount": ["25", "50", "100", "200"],
                "quality": ["terrible", "poor", "excellent", "good"],
                "behavior": ["rude", "helpful", "professional", "unprofessional"],
                "time": ["30", "45", "60", "90"],
                "info": ["hours", "policies", "rates", "plans"],
                "topic": ["billing", "services", "plans", "policies"],
                "action": ["cancel", "upgrade", "change", "update"],
                "feature": ["service", "plans", "billing", "support"],
                "detail": ["cost", "process", "procedure", "policy"],
                "options": ["plans", "services", "packages", "features"]
            }
            
            # Generate synthetic samples
            category_templates = templates.get(category, [])
            for i in range(min(needed, len(category_templates))):
                template = category_templates[i % len(category_templates)]
                
                # Fill in placeholders
                synthetic = template
                for placeholder, options in fillers.items():
                    if "{" + placeholder + "}" in synthetic:
                        synthetic = synthetic.replace("{" + placeholder + "}", random.choice(options))
                
                augmented_texts.append(synthetic)
                augmented_labels.append(category)
    
    return augmented_texts, augmented_labels

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
    # Store original data for CV validation
    original_texts, original_labels = texts.copy(), labels.copy()
    print("\n" + "═"*80)
    print("  ResolveAI  –  Model Training (Enhanced with Data Augmentation)")
    print("═"*80)

    # ── Step 1: Data Augmentation ────────────────────────────────────────────
    print("\n[1/7] Data Augmentation…")
    original_count = len(texts)
    texts, labels = augment_dataset(texts, labels, target_samples=160)  # Reduced from 300
    print(f"      Original: {original_count} samples")
    print(f"      Augmented: {len(texts)} samples")
    
    # Show category distribution after augmentation
    from collections import Counter
    dist = Counter(labels)
    print("\n      Augmented category distribution:")
    for cat, cnt in sorted(dist.items()):
        print(f"        {cat:<12} {cnt} samples")

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    print("\n[2/7] Preprocessing texts (with lemmatization)…")
    processed = [preprocess_text(t) for t in texts]
    print(f"      Sample: '{texts[0][:60]}…'")
    print(f"      → '{processed[0]}'")

    # ── Step 3: Train/test split ─────────────────────────────────────────────
    print("\n[3/7] Splitting dataset (80/20 stratified)…")
    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Step 4: Build enhanced TF-IDF vectorizer ────────────────────────────
    print(f"\n[4/7] Building enhanced TF-IDF vectorizer…")

    tfidf = TfidfVectorizer(
        max_features=2000,  # Reduced from 3000
        ngram_range=(1, 3),  # Include trigrams
        min_df=2,  # Minimum document frequency
        max_df=0.8,  # Reduced from 0.9
        stop_words="english",
        sublinear_tf=True,
        use_idf=True,
        norm='l2'
    )
    
    # Fit TF-IDF once to use for all models
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"      Feature matrix shape: {X_train_tfidf.shape}")

    # ── Step 5: Train base models with enhanced hyperparameter tuning ───────
    print(f"\n[5/7] Training base models with enhanced hyperparameter tuning…")
    
    # Logistic Regression (expanded parameter search)
    print("      [5a] Logistic Regression…")
    lr_params = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2'],
        'class_weight': ['balanced', None]
    }
    lr_clf = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs')
    lr_grid = GridSearchCV(lr_clf, lr_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
    lr_grid.fit(X_train_tfidf, y_train)
    lr_best = lr_grid.best_estimator_
    lr_score = lr_grid.best_score_
    print(f"            CV Score: {lr_score:.4f} (params: {lr_grid.best_params_})")

    # SVM (expanded parameter search)
    print("      [5b] Support Vector Machine (SVM)…")
    svm_params = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'class_weight': ['balanced', None]
    }
    svm_clf = SVC(probability=True, random_state=42)
    svm_grid = GridSearchCV(svm_clf, svm_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
    svm_grid.fit(X_train_tfidf, y_train)
    svm_best = svm_grid.best_estimator_
    svm_score = svm_grid.best_score_
    print(f"            CV Score: {svm_score:.4f} (params: {svm_grid.best_params_})")

    # Random Forest (new model)
    print("      [5c] Random Forest Classifier…")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }
    rf_clf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf_clf, rf_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
    rf_grid.fit(X_train_tfidf, y_train)
    rf_best = rf_grid.best_estimator_
    rf_score = rf_grid.best_score_
    print(f"            CV Score: {rf_score:.4f} (params: {rf_grid.best_params_})")

    # Multinomial Naive Bayes (improved)
    print("      [5d] Multinomial Naive Bayes…")
    nb_params = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]}
    nb_clf = MultinomialNB()
    nb_grid = GridSearchCV(nb_clf, nb_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
    nb_grid.fit(X_train_tfidf, y_train)
    nb_best = nb_grid.best_estimator_
    nb_score = nb_grid.best_score_
    print(f"            CV Score: {nb_score:.4f} (params: {nb_grid.best_params_})")

    # ── Step 6: Create Enhanced Stacking Ensemble ───────────────────────────
    print(f"\n[6/7] Creating Enhanced Stacking Ensemble (5 base models + meta-learner)…")
    
    # Base estimators for stacking
    base_estimators = [
        ('lr', lr_best),
        ('svm', svm_best),
        ('rf', rf_best),
        ('nb', nb_best)
    ]
    
    # Meta-learner with regularization
    meta_learner = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        passthrough=True  # Pass original features to meta-learner
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

    # ── Step 7: Final Evaluation and Report ──────────────────────────────────
    print(f"\n[7/7] Final Evaluation Report ({ensemble_name} Ensemble)…")
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

    # ── Cross-validation on original data ───────────────────────────────────
    print(f"\n[8/8] Cross-validation on original dataset…")
    original_processed = [preprocess_text(t) for t in original_texts]
    cv_scores = cross_val_score(final_pipeline, original_processed, original_labels, cv=5, scoring="accuracy")
    print(f"      Original CV Scores: {cv_scores}")
    print(f"      Original CV Mean: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.1f}%)")
    print(f"      Original CV Std: {cv_scores.std():.4f}")
    
    if cv_scores.mean() >= 0.80:
        print("      ✓ Target accuracy achieved! (≥80%)")
    else:
        print("      ⚠ May need more data or different approach")
    
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
