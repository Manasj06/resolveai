#!/bin/bash
# Render build script
# This runs during the build process to set up dependencies

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "📥 Downloading NLTK data..."
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print('✅ NLTK data downloaded successfully')
except Exception as e:
    print(f'⚠️  Warning: Could not download some NLTK data: {e}')
"

echo "🗄️ Initializing database schema..."
python -c "
import os
import sys
sys.path.insert(0, '.')
from backend.database import init_db
try:
    init_db()
    print('✅ Database schema initialized')
except Exception as e:
    print(f'⚠️  Warning: Database initialization had issues: {e}')
"

echo "✅ Build complete!"
