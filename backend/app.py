"""
app.py – ResolveAI Flask Backend
---------------------------------
REST API endpoints:
  POST /train     – train the ML model on the dataset
  POST /predict   – classify + resolve a complaint
  GET  /resolve   – get complaint history
  GET  /tickets   – get all open support tickets
  GET  /analytics – get category analytics
  GET  /health    – health check

Run:
  cd resolveai
  python backend/app.py
"""

import os
import sys
import uuid
import logging
import csv
import sqlite3

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from backend.classifier import classifier_instance, preprocess_text
from backend.search_algorithms import BestFirstSearch, AStarResponseSelector, should_auto_resolve
from backend.knowledge_base import get_responses_for_category
from backend.database import (
    save_complaint, create_ticket, get_all_complaints,
    get_all_tickets, get_analytics, get_connection,
    create_user, authenticate_user, get_user_by_id
)

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_PATH = os.path.join(ROOT, "frontend")
app = Flask(__name__, static_folder=FRONTEND_PATH, static_url_path="")
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app, supports_credentials=True)  # Allow credentials for session management

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_id, email):
        self.id = user_id
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    user_data = get_user_by_id(user_id)
    if user_data:
        return User(user_data['id'], user_data['email'])
    return None

DATASET_PATH = os.path.join(ROOT, "dataset", "complaints_dataset.csv")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_trained": classifier_instance.is_trained,
        "service": "ResolveAI",
        "authenticated": current_user.is_authenticated if current_user else False
    })


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve the React/HTML frontend from the frontend directory."""
    if path != "" and os.path.exists(os.path.join(FRONTEND_PATH, path)):
        return send_from_directory(FRONTEND_PATH, path)
    return send_from_directory(FRONTEND_PATH, 'index.html')


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/auth/signup", methods=["POST"])
def signup():
    """Create a new user account."""
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long"}), 400

    if "@" not in email or "." not in email:
        return jsonify({"error": "Please provide a valid email address"}), 400

    user_id = create_user(email, password)
    if user_id:
        user = User(user_id, email)
        login_user(user)
        return jsonify({
            "success": True,
            "message": "Account created successfully!",
            "user": {"id": user_id, "email": email}
        })
    else:
        return jsonify({"error": "Email already exists"}), 409


@app.route("/auth/login", methods=["POST"])
def login():
    """Authenticate user and create session."""
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user_id = authenticate_user(email, password)
    if user_id:
        user = User(user_id, email)
        login_user(user)
        return jsonify({
            "success": True,
            "message": "Login successful!",
            "user": {"id": user_id, "email": email}
        })
    else:
        return jsonify({"error": "Invalid email or password"}), 401


@app.route("/auth/logout", methods=["POST"])
@login_required
def logout():
    """End user session."""
    logout_user()
    return jsonify({"success": True, "message": "Logged out successfully!"})


@app.route("/auth/status", methods=["GET"])
def auth_status():
    """Check current authentication status."""
    if current_user.is_authenticated:
        return jsonify({
            "authenticated": True,
            "user": {
                "id": current_user.id,
                "email": current_user.email
            }
        })
    else:
        return jsonify({"authenticated": False})


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/train", methods=["POST"])
def train():
    """
    Load the dataset CSV and train the ML classifier.
    The dataset must exist at dataset/complaints_dataset.csv
    """
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": "Dataset not found. Please check dataset/complaints_dataset.csv"}), 404

    texts, labels = [], []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["complaint"])
            labels.append(row["category"])

    if len(texts) < 10:
        return jsonify({"error": "Dataset too small (minimum 10 rows needed)"}), 400

    logger.info(f"Training on {len(texts)} complaints…")
    metrics = classifier_instance.train(texts, labels)

    return jsonify({
        "success": True,
        "message": "Model trained successfully!",
        "dataset_size": len(texts),
        "accuracy": metrics["accuracy"],
        "train_size": metrics["train_size"],
        "test_size": metrics["test_size"],
        "report": metrics["report"]
    })


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT / RESOLVE ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    """
    Main endpoint: classify complaint → search for best response → resolve/ticket.

    Request body (JSON):
      { "complaint": "My bill is wrong and I was charged twice" }

    Response (JSON):
      { category, confidence, probabilities, resolution, ticket_id?,
        bfs_log, astar_log, status, complaint_id }
    """
    body = request.get_json(silent=True) or {}
    complaint_text = (body.get("complaint") or "").strip()

    if not complaint_text:
        return jsonify({"error": "Please provide a complaint text."}), 400
    if len(complaint_text) < 5:
        return jsonify({"error": "Complaint text is too short."}), 400

    if not classifier_instance.is_trained:
        return jsonify({"error": "Model not trained. Call POST /train first."}), 503

    # ── Step 1: ML Classification ─────────────────────────────────────────────
    predicted_category, proba_dict = classifier_instance.predict(complaint_text)
    logger.info(f"Predicted: {predicted_category} | Probabilities: {proba_dict}")

    # ── Step 2: Best-First Search (Category Selection) ────────────────────────
    bfs = BestFirstSearch(proba_dict)
    best_category, best_confidence, bfs_log = bfs.search()

    final_category = best_category
    confidence = best_confidence

    # ── Step 3: Should we auto-resolve or create a ticket? ────────────────────
    auto_resolve = should_auto_resolve(confidence)
    ticket_id = None
    resolution_text = None
    astar_log = []

    if auto_resolve:
        # ── Step 4: A* Response Selection ────────────────────────────────────
        candidate_responses = get_responses_for_category(final_category)
        astar = AStarResponseSelector(complaint_text, candidate_responses)
        best_response, astar_log = astar.select_best_response()

        ref_id = uuid.uuid4().hex[:8].upper()
        resolution_text = best_response["response"].replace("{ticket_id}", ref_id)
        status = "resolved"
    else:
        status = "ticket_created"

    # ── Step 5: Save to database ──────────────────────────────────────────────
    complaint_id = save_complaint(
        text=complaint_text,
        category=final_category,
        confidence=confidence,
        status=status,
        user_id=current_user.id,
        response=resolution_text,
        ticket_id=ticket_id
    )

    if not auto_resolve:
        ticket_id = create_ticket(
            user_id=current_user.id,
            complaint_id=complaint_id,
            complaint_text=complaint_text,
            category=final_category,
            confidence=confidence
        )
        conn = get_connection()
        conn.execute("UPDATE complaints SET ticket_id=? WHERE id=?", (ticket_id, complaint_id))
        conn.commit()
        conn.close()

    # ── Step 6: Build response payload ───────────────────────────────────────
    preprocessed = preprocess_text(complaint_text)

    return jsonify({
        "complaint_id": complaint_id,
        "original_text": complaint_text,
        "preprocessed_text": preprocessed,
        "category": final_category,
        "confidence": round(confidence, 4),
        "probabilities": proba_dict,
        "status": status,
        "resolution": resolution_text,
        "ticket_id": ticket_id,
        "auto_resolved": auto_resolve,
        "bfs_exploration": bfs_log,
        "astar_scores": astar_log
    })


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY, TICKETS, ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/resolve", methods=["GET"])
@login_required
def resolve():
    complaints = get_all_complaints(current_user.id, limit=50)
    return jsonify({"complaints": complaints, "total": len(complaints)})


@app.route("/tickets", methods=["GET"])
@login_required
def tickets():
    all_tickets = get_all_tickets(current_user.id, limit=50)
    return jsonify({"tickets": all_tickets, "total": len(all_tickets)})


@app.route("/analytics", methods=["GET"])
@login_required
def analytics():
    data = get_analytics(current_user.id)
    return jsonify({"analytics": data})


@app.route("/resolve_ticket", methods=["POST"])
@login_required
def resolve_ticket():
    """
    Human reviewer endpoint: mark a ticket as resolved with notes.
    
    Request body (JSON):
      { "ticket_id": "TKT-XXXXX", "resolution_notes": "human review notes" }
    """
    body = request.get_json(silent=True) or {}
    ticket_id = body.get("ticket_id", "").strip()
    resolution_notes = body.get("resolution_notes", "").strip()

    if not ticket_id or not resolution_notes:
        return jsonify({"error": "ticket_id and resolution_notes required"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()
        ticket_row = cursor.execute(
            "SELECT complaint_id, category FROM tickets WHERE id = ?",
            (ticket_id,)
        ).fetchone()

        if not ticket_row:
            conn.close()
            return jsonify({"error": "Ticket not found"}), 404

        complaint_id = ticket_row["complaint_id"]
        category = ticket_row["category"]

        cursor.execute(
            "UPDATE tickets SET status=? WHERE id=?",
            ("resolved", ticket_id)
        )
        cursor.execute(
            "UPDATE complaints SET status=?, response=? WHERE ticket_id= ?",
            ("resolved", resolution_notes, ticket_id)
        )
        cursor.execute(
            "UPDATE analytics SET resolved = resolved + 1, tickets = CASE WHEN tickets > 0 THEN tickets - 1 ELSE 0 END WHERE category = ?",
            (category,)
        )

        conn.commit()
        conn.close()
        logger.info(f"Ticket {ticket_id} marked resolved by human reviewer")
        return jsonify({"success": True, "message": f"Ticket {ticket_id} resolved"})
    except Exception as e:
        logger.error(f"Error resolving ticket: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting ResolveAI backend on http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
