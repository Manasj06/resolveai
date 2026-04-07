"""
utils/helpers.py
----------------
Shared utility functions for ResolveAI.
"""

import re
from datetime import datetime


def format_confidence(confidence: float) -> str:
    """Convert float confidence to percentage string."""
    return f"{confidence * 100:.1f}%"


def truncate_text(text: str, max_len: int = 100) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def get_priority_from_confidence(confidence: float) -> str:
    """Map confidence score to support ticket priority."""
    if confidence < 0.35:
        return "high"
    elif confidence < 0.45:
        return "medium"
    return "low"


def sanitize_input(text: str) -> str:
    """Basic input sanitization."""
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()


def timestamp_now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.utcnow().isoformat()
