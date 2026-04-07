#!/usr/bin/env python3
"""
Database migration script for production deployment.
Run this after setting up PostgreSQL database on Render.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import init_db

if __name__ == "__main__":
    print("Initializing production database...")
    init_db()
    print("✅ Database initialized successfully!")

    # Optional: Load sample data or run migrations here
    print("🎯 Ready for deployment!")