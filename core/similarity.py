from __future__ import annotations

import difflib


def code_similarity_ratio(old_code: str, new_code: str) -> float:
    """Return a normalized similarity score between two code snippets."""
    return difflib.SequenceMatcher(None, old_code, new_code).ratio()
