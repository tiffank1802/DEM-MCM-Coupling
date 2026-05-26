"""
tests/conftest.py
=================
Pytest configuration and shared fixtures.
"""

import pytest
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def project_root_dir():
    """Get project root directory."""
    return project_root


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging between tests."""
    import logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    yield
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
