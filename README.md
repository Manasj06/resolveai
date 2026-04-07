# 🧠 ResolveAI – Intelligent Complaint Classification & Auto-Resolution System

A complete full-stack AI system that classifies customer complaints using **Machine Learning** and resolves them using **Best-First Search** and **A\*-inspired** algorithms. Features secure user authentication with data privacy protection.

---

## ✨ Key Features

- 🤖 **ML-Powered Classification**: 78.6% accuracy with ensemble model (Logistic Regression + SVM + Random Forest + Naive Bayes)
- 🔍 **Advanced Search Algorithms**: Best-First Search + A* Response Selection for optimal resolution
- 🔐 **User Authentication**: Secure login/signup with bcrypt password hashing and session management
- 🛡️ **Data Privacy**: Complete user data isolation - each user sees only their own complaints and analytics
- 🎯 **Auto-Resolution**: Automatic complaint resolution for high-confidence predictions
- 📊 **Analytics Dashboard**: Real-time analytics per user with category breakdowns
- 🎨 **Modern UI**: Responsive web interface with authentication overlay

---

## 🗂 Project Structure

```
resolveai/
├── backend/
│   ├── __init__.py
│   ├── app.py                  ← Flask REST API with authentication
│   ├── classifier.py           ← NLP preprocessing + ML ensemble model
│   ├── search_algorithms.py    ← Best-First Search + A* logic
│   ├── knowledge_base.py       ← Predefined responses with h-scores
│   └── database.py             ← SQLite data layer with user isolation
├── dataset/
│   └── complaints_dataset.csv  ← 140 labelled training samples (augmented)
├── frontend/
│   └── index.html              ← Full web UI with authentication
├── model/
│   ├── __init__.py
│   ├── train_model.py          ← Standalone CLI training script
│   └── complaint_classifier.pkl← Trained ensemble model (auto-generated)
├── utils/
│   ├── __init__.py
│   └── helpers.py              ← Shared utility functions
├── requirements.txt            ← Python dependencies
├── resolveai.db                ← SQLite database (auto-generated)
├── .gitignore                  ← Excludes database and test files
└── README.md
```

---

## 🚀 Quick Start (Step-by-Step)

### Prerequisites
- Python 3.8+
- pip package manager
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Step 1 – Clone & Navigate

```bash
git clone https://github.com/Manasj06/resolveai.git
cd resolveai
```

### Step 2 – Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

> **Optional**: Pre-download NLTK data for better tokenization:
> ```python
> python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
> ```

### Step 4 – Start the Backend Server

```bash
# Start Flask backend with authentication
python backend/app.py
```

**Backend Server**: http://localhost:5001
- ✅ Authentication endpoints: `/auth/signup`, `/auth/login`, `/auth/logout`
- ✅ Protected ML endpoints: `/predict`, `/analytics`, `/resolve`
- ✅ Health check: `/health`

### Step 5 – Start the Frontend Server

```bash
# In a new terminal, start frontend server
cd frontend
python -m http.server 3000
```

**Frontend UI**: http://localhost:3000

### Step 6 – Access the Application

1. **Open** http://localhost:3000 in your browser
2. **Create Account**: Click "Sign Up" and register with email/password
3. **Login**: Use your credentials to access the system
4. **Train Model**: Click "⚡ Train Model" to initialize the ML classifier
5. **Submit Complaints**: Type complaints and get AI-powered resolutions

---

## 🔧 API Endpoints

### Authentication Endpoints
```bash
# Create new account
POST /auth/signup
Content-Type: application/json
{"email": "user@example.com", "password": "securepass123"}

# Login
POST /auth/login
Content-Type: application/json
{"email": "user@example.com", "password": "securepass123"}

# Logout
POST /auth/logout

# Check auth status
GET /auth/status
```

### ML & Data Endpoints (Require Authentication)
```bash
# Train the model
POST /train

# Classify & resolve complaint
POST /predict
Content-Type: application/json
{"complaint": "My internet is very slow"}

# Get user complaints history
GET /resolve

# Get user support tickets
GET /tickets

# Get user analytics
GET /analytics

# Resolve ticket (human review)
POST /resolve_ticket
{"ticket_id": "TKT-XXXXX", "resolution_notes": "Issue resolved"}
```

### Public Endpoints
```bash
# Health check
GET /health
```

---

## 🧪 Testing the System

### 1. Health Check
```bash
curl http://localhost:5001/health
```

### 2. Create Test User
```bash
curl -X POST http://localhost:5001/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "testpass123"}'
```

### 3. Login & Get Session
```bash
curl -X POST http://localhost:5001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "testpass123"}' \
  -c cookies.txt
```

### 4. Submit Complaint (Authenticated)
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"complaint": "My internet keeps disconnecting randomly"}' \
  -b cookies.txt
```

### 5. View Analytics (Authenticated)
```bash
curl http://localhost:5001/analytics -b cookies.txt
```

---

## 🔧 Advanced Configuration

### Environment Variables
```bash
# Set production secret key
export SECRET_KEY="your-secure-random-key-here"
export FLASK_ENV="production"

# Database path (optional)
export DATABASE_PATH="/path/to/resolveai.db"
```

### Standalone Model Training
```bash
# Train with default ensemble model
python model/train_model.py

# Train with specific algorithm
python model/train_model.py --model naive_bayes
python model/train_model.py --model svm

# Use custom dataset
python model/train_model.py --dataset path/to/custom_dataset.csv
```

### Database Management
```bash
# View database schema
sqlite3 resolveai.db ".schema"

# Backup database
cp resolveai.db resolveai_backup.db

# Reset database (WARNING: deletes all data)
rm resolveai.db
python -c "from backend.database import init_db; init_db()"
```

---

## 🚀 Production Deployment

### 1. Environment Setup
```bash
# Set production environment
export SECRET_KEY="$(openssl rand -hex 32)"
export FLASK_ENV="production"

# Use production WSGI server
pip install gunicorn
gunicorn --bind 0.0.0.0:5001 backend.app:app
```

### 2. Web Server Configuration (nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static {
        alias /path/to/resolveai/frontend;
    }
}
```

### 3. SSL Configuration
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/key.pem;

    # ... proxy configuration
}
```

### 4. Process Management (systemd)
```ini
[Unit]
Description=ResolveAI Flask App
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/resolveai
Environment="SECRET_KEY=your-secret-key"
ExecStart=/path/to/resolveai/.venv/bin/gunicorn --bind 127.0.0.1:5001 backend.app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 📊 System Performance

- **Model Accuracy**: 78.6% (ensemble of 4 algorithms)
- **Training Data**: 140 samples (augmented from 100)
- **Response Time**: <500ms per complaint
- **Auto-Resolution Rate**: ~70% for high-confidence predictions
- **Categories**: Billing, Technical, Service, General

---

## 🛠 Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Kill process using port 5001
lsof -ti:5001 | xargs kill -9

# Kill process using port 3000
lsof -ti:3000 | xargs kill -9
```

**2. Database Errors**
```bash
# Reset database
rm resolveai.db
python -c "from backend.database import init_db; init_db()"
```

**3. Model Not Trained**
```bash
# Train model via API
curl -X POST http://localhost:5001/train

# Or use CLI
python model/train_model.py
```

**4. Authentication Issues**
```bash
# Check auth status
curl http://localhost:5001/auth/status

# Clear browser cookies and retry
```

**5. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 📞 Support

For issues, questions, or contributions:
- **GitHub Issues**: [Report bugs or request features](https://github.com/Manasj06/resolveai/issues)
- **Documentation**: This README and inline code comments
- **Community**: Open to discussions and improvements

---

**Happy resolving! 🎯**
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
