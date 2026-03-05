"""
Pytest configuration and fixtures.
"""

import nltk


def pytest_configure(config):
    """Download required NLTK data before running tests."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
