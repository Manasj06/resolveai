# 🧠 ResolveAI – Intelligent Complaint Classification & Auto-Resolution System

A complete full-stack AI system that classifies customer complaints using **Machine Learning** and resolves them using **Best-First Search** and **A\*-inspired** algorithms.

---

## 🗂 Project Structure

```
resolveai/
├── backend/
│   ├── __init__.py
│   ├── app.py                  ← Flask REST API (main entry point)
│   ├── classifier.py           ← NLP preprocessing + ML model
│   ├── search_algorithms.py    ← Best-First Search + A* logic
│   ├── knowledge_base.py       ← Predefined responses with h-scores
│   └── database.py             ← SQLite data layer
├── dataset/
│   └── complaints_dataset.csv  ← 100 labelled training samples
├── frontend/
│   └── index.html              ← Full web UI (open in browser)
├── model/
│   ├── __init__.py
│   ├── train_model.py          ← Standalone CLI training script
│   └── complaint_classifier.pkl← (auto-generated after training)
├── utils/
│   ├── __init__.py
│   └── helpers.py              ← Shared utility functions
├── requirements.txt
├── resolveai.db                ← (auto-generated SQLite database)
└── README.md
```

---

## 🚀 Quick Start (Step-by-Step)

### Step 1 – Navigate to the project folder

```bash
cd resolveai
```

### Step 2 – Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

> Optionally pre-download NLTK data for better tokenization:
> ```python
> import nltk
> nltk.download('punkt')
> nltk.download('stopwords')
> ```

### Step 4 – Start the Flask backend

```bash
python backend/app.py
```

Server starts on: **http://localhost:5000**

### Step 5 – Open the frontend

Open `frontend/index.html` directly in your browser (double-click or drag into Chrome/Firefox).

### Step 6 – Train the model

Click **"⚡ Train Model"** in the top bar — or use the API:
```bash
curl -X POST http://localhost:5000/train
```

Or use the standalone CLI training script:
```bash
python model/train_model.py
# With Naive Bayes:
python model/train_model.py --model naive_bayes
```

### Step 7 – Submit a complaint

Type a complaint in the UI and click **"Analyze Complaint"**. The system will:
1. Preprocess text (tokenize, remove stopwords, TF-IDF)
2. Classify it (Logistic Regression ML model)
3. Run Best-First Search to confirm the category
4. Run A* to select the best resolution response
5. Auto-resolve OR create a support ticket

---

## 🔌 API Endpoints

| Method | Endpoint      | Description                                      |
|--------|---------------|--------------------------------------------------|
| GET    | `/health`     | Check if backend and model are running           |
| POST   | `/train`      | Train the ML model on the dataset                |
| POST   | `/predict`    | Classify complaint + resolve or create ticket    |
| GET    | `/resolve`    | Get recent complaint history                     |
| GET    | `/tickets`    | Get all open support tickets                     |
| GET    | `/analytics`  | Get per-category analytics                       |

### Example `/predict` request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"complaint": "My internet keeps dropping and I cannot work from home"}'
```

---

## 🧠 AI / ML Architecture

### Classification Pipeline
1. **Text Preprocessing** – lowercase → clean → tokenize → remove stopwords → TF-IDF vectorize
2. **Logistic Regression** – trained on 100 labelled complaints (Billing / Technical / Service / General)
3. **Probability output** – each category gets a probability score

### Best-First Search (Category Selection)
- Uses Python `heapq` (min-heap with negated probabilities = max-heap)
- Explores category nodes in order of ML probability
- Returns the highest-probability category as the winner

### A* Response Selection
- For each candidate response in the knowledge base:
  - `g(n)` = TF-IDF cosine similarity between complaint and response keywords
  - `h(n)` = pre-assigned usefulness weight (0.7–0.95)
  - `f(n)` = g(n) + h(n)
- Selects the response with the **highest f(n)**

### Auto-Resolution Logic
- Confidence ≥ 55% → **Auto-resolve** with best A* response
- Confidence < 55% → **Create support ticket** for human review

---

## 📊 Dataset

Located at `dataset/complaints_dataset.csv` — 100 labelled complaints split evenly:
- 25 × Billing complaints
- 25 × Technical complaints  
- 25 × Service complaints
- 25 × General complaints

Format: `complaint,category`

---

## 🛠 Customization

- **Add more training data** → append rows to `dataset/complaints_dataset.csv` and retrain
- **Change confidence threshold** → edit `CONFIDENCE_THRESHOLD` in `backend/search_algorithms.py`
- **Add new response templates** → edit `backend/knowledge_base.py`
- **Switch to Naive Bayes** → run `python model/train_model.py --model naive_bayes`

---

## 🧪 Tech Stack

| Layer        | Technology                    |
|--------------|-------------------------------|
| Backend      | Python 3.10+, Flask 3.0       |
| ML           | scikit-learn (TF-IDF + LR)    |
| NLP          | NLTK, regex                   |
| Search       | heapq (BFS), cosine similarity (A*) |
| Database     | SQLite                        |
| Frontend     | Vanilla HTML/CSS/JS           |
| Fonts        | Sora, JetBrains Mono          |
