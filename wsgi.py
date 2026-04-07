"""
WSGI entry point for Render deployment.
This file allows Gunicorn to properly import and run the Flask app.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app
from backend.app import app

# This is what Gunicorn looks for: wsgi:app
if __name__ == "__main__":
    app.run()
