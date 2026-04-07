"""
classifier.py
-------------
NLP Pipeline + ML Classification Module

Pipeline:
  1. Text Preprocessing  (tokenize → lowercase → remove stopwords → clean)
  2. TF-IDF Vectorization
  3. Logistic Regression / Naive Bayes Classifier
  4. Model save/load via pickle

The classifier outputs:
  - Predicted category (string)
  - Probability dict {category: float} for all categories
"""

import os
import re
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple

# NLP / ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── optional NLTK for richer stopwords ───────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    NLTK_AVAILABLE = True
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    NLTK_AVAILABLE = False
    LEMMATIZER = None
    # Fallback English stopwords
    STOP_WORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "they", "them", "their",
        "what", "which", "who", "this", "that", "these", "those", "am",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "shall", "can", "a", "an", "the", "and", "but",
        "or", "nor", "for", "so", "yet", "if", "then", "in", "on", "at",
        "by", "to", "of", "up", "out", "as", "with", "about", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "each", "other", "few", "more", "most", "such", "no", "not", "very",
        "also", "just", "now", "here", "there", "when", "where", "how"
    }

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "complaint_classifier.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
      1. Lowercase
      2. Remove special characters / numbers
      3. Tokenize
      4. Remove stopwords
      5. Lemmatize
      6. Rejoin tokens

    Parameters
    ----------
    text : str – raw complaint text

    Returns
    -------
    str – cleaned and preprocessed text
    """
    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove special chars, keep letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 3: Tokenize (split on whitespace)
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
    else:
        tokens = text.split()

    # Step 4: Remove stopwords and very short tokens
    tokens = [
        t for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]

    # Step 5: Lemmatize
    if NLTK_AVAILABLE and LEMMATIZER:
        try:
            tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
        except Exception:
            pass  # Skip lemmatization if it fails

    # Step 6: Rejoin
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ComplaintClassifier:
    """
    Wraps a scikit-learn Pipeline:
      TfidfVectorizer → LogisticRegression

    Methods
    -------
    train(texts, labels)   → trains and saves model
    predict(text)          → returns (category, probability_dict)
    load()                 → loads saved model from disk
    """

    CATEGORIES = ["Billing", "Technical", "Service", "General"]

    def __init__(self, model_type: str = "logistic"):
        """
        Parameters
        ----------
        model_type : "logistic" | "naive_bayes"
        """
        self.model_type = model_type
        self.pipeline = None
        self.is_trained = False

        # Try loading an existing model at startup
        self._try_load()

    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn Pipeline with TF-IDF + classifier."""
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),      # unigrams + bigrams
            min_df=1,
            stop_words="english",
            sublinear_tf=True        # apply log normalisation to TF
        )

        if self.model_type == "naive_bayes":
            clf = MultinomialNB(alpha=0.5)
        else:  # default: logistic regression
            clf = LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                multi_class="auto"
            )

        return Pipeline([("tfidf", tfidf), ("clf", clf)])

    def train(self, texts: List[str], labels: List[str]) -> dict:
        """
        Train the classifier on provided texts and labels.

        Returns a dict with accuracy and classification report.
        """
        # Preprocess all texts
        processed = [preprocess_text(t) for t in texts]

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            processed, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Build and train pipeline
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Save model to disk
        self.save()

        logger.info(f"Model trained. Accuracy: {accuracy:.3f}")
        return {
            "accuracy": round(accuracy, 4),
            "report": report,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Predict the category for a complaint text.

        Returns
        -------
        category : str              – predicted label
        proba_dict : dict           – {category: probability}
        """
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model is not trained yet. Call /train first.")

        processed = preprocess_text(text)

        # Predict class
        predicted_category = self.pipeline.predict([processed])[0]

        # Probability for each class
        classes = self.pipeline.classes_
        probabilities = self.pipeline.predict_proba([processed])[0]
        proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probabilities)}

        # Ensure all 4 categories are present (fill 0 if missing)
        for cat in self.CATEGORIES:
            proba_dict.setdefault(cat, 0.0)

        return predicted_category, proba_dict

    def save(self):
        """Pickle the trained pipeline to MODEL_PATH."""
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"Model saved to {MODEL_PATH}")

    def load(self):
        """Load a saved pipeline from MODEL_PATH."""
        with open(MODEL_PATH, "rb") as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True
        logger.info(f"Model loaded from {MODEL_PATH}")

    def _try_load(self):
        """Silently try to load a saved model; fail gracefully."""
        if os.path.exists(MODEL_PATH):
            try:
                self.load()
            except Exception as e:
                logger.warning(f"Could not load saved model: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instance
# ─────────────────────────────────────────────────────────────────────────────
classifier_instance = ComplaintClassifier(model_type="logistic")
