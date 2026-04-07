"""
search_algorithms.py
--------------------
Implements two AI search strategies:

1. Best-First Search  – selects the best complaint CATEGORY
   • Uses a max-priority queue (heapq) ordered by ML probability
   • Explores nodes (categories) from highest to lowest confidence
   • Returns the category with the highest probability first

2. A*-Inspired Response Selection – selects the best RESPONSE
   • g(n) = TF-IDF cosine similarity between complaint and response keywords
   • h(n) = heuristic usefulness weight stored in knowledge base
   • f(n) = g(n) + h(n)  → higher is better
   • Picks the response with the maximum f(n)
"""

import heapq
import math
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────────────────────
# 1. BEST-FIRST SEARCH  (Category Selection)
# ─────────────────────────────────────────────────────────────────────────────

class BestFirstSearch:
    """
    Uses a max-priority queue to explore category nodes.

    Each node represents a predicted category with its probability.
    We negate probabilities because Python's heapq is a min-heap.
    The node with the highest probability is always explored first.

    Node structure: (-probability, category_name)
    """

    def __init__(self, category_probabilities: Dict[str, float]):
        """
        Parameters
        ----------
        category_probabilities : dict
            {category_name: probability_score}  e.g. {"Billing": 0.87, ...}
        """
        self.probabilities = category_probabilities
        self.exploration_log = []   # log of nodes as they are explored

    def search(self) -> Tuple[str, float, List[dict]]:
        """
        Run Best-First Search and return the best category.

        Returns
        -------
        best_category : str
        best_probability : float
        exploration_log : list of explored nodes (for UI display)
        """
        # Build priority queue: negate probability for max-heap behaviour
        priority_queue = []
        for category, prob in self.probabilities.items():
            # Push (-prob, category) so highest prob is popped first
            heapq.heappush(priority_queue, (-prob, category))

        self.exploration_log = []
        best_category = None
        best_probability = 0.0

        # Explore nodes one by one (priority order)
        while priority_queue:
            neg_prob, category = heapq.heappop(priority_queue)
            prob = -neg_prob  # restore actual probability

            node_info = {
                "category": category,
                "probability": round(prob, 4),
                "explored": True,
                "is_best": False
            }
            self.exploration_log.append(node_info)

            # First node popped is always the best (max-heap)
            if best_category is None:
                best_category = category
                best_probability = prob
                node_info["is_best"] = True
                # We could break here for pure BFS, but we log all nodes
                # to show the full exploration trail in the UI

        return best_category, best_probability, self.exploration_log


# ─────────────────────────────────────────────────────────────────────────────
# 2. A*-INSPIRED RESPONSE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

class AStarResponseSelector:
    """
    Selects the best resolution response using A*-inspired scoring.

    For each candidate response node n:
        g(n) = TF-IDF cosine similarity between complaint text and response keywords
        h(n) = heuristic usefulness weight (pre-assigned in knowledge base)
        f(n) = g(n) + h(n)

    The response with the highest f(n) is selected.

    This mirrors A* where we want to minimize cost to goal;
    here we MAXIMIZE f(n) which represents "closeness to ideal resolution".
    """

    def __init__(self, complaint_text: str, responses: List[dict]):
        """
        Parameters
        ----------
        complaint_text : str   – the raw complaint
        responses : list       – candidate response dicts from knowledge_base
        """
        self.complaint_text = complaint_text
        self.responses = responses
        self.scored_nodes = []

    def _compute_g(self, response_keywords: List[str]) -> float:
        """
        g(n) – TF-IDF cosine similarity between complaint and response keywords.
        Measures how semantically similar the complaint is to the known keywords.
        Returns a float in [0, 1].
        """
        if not response_keywords:
            return 0.0

        keyword_text = " ".join(response_keywords)
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(
                [self.complaint_text, keyword_text]
            )
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            return 0.0

    def select_best_response(self) -> Tuple[dict, List[dict]]:
        """
        Score all response nodes and return the one with the highest f(n).

        Returns
        -------
        best_response : dict  – the chosen response entry
        scored_nodes  : list  – all nodes with g, h, f scores for logging
        """
        self.scored_nodes = []
        best_response = None
        best_f = -1.0

        for resp in self.responses:
            g = self._compute_g(resp.get("keywords", []))
            h = resp.get("h_score", 0.5)
            f = g + h   # A* cost function (maximising here)

            node = {
                "id": resp["id"],
                "title": resp["title"],
                "g_score": round(g, 4),   # similarity
                "h_score": round(h, 4),   # usefulness heuristic
                "f_score": round(f, 4),   # combined score
                "response": resp["response"]
            }
            self.scored_nodes.append(node)

            if f > best_f:
                best_f = f
                best_response = resp

        # Sort by f_score descending for logging
        self.scored_nodes.sort(key=lambda x: x["f_score"], reverse=True)

        return best_response, self.scored_nodes


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE THRESHOLD LOGIC
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.35  # Below this → create a support ticket; at/above → auto-resolve

def should_auto_resolve(confidence: float) -> bool:
    """Return True if confidence is high enough for auto-resolution."""
    return confidence >= CONFIDENCE_THRESHOLD
