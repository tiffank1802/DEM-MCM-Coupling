"""
tests/conftest.py.
=================
Pytest configuration and shared fixtures.
"""

import os
import sys
from collections.abc import Generator

import pytest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def project_root_dir() -> str:
    """Get project root directory."""
    return project_root


@pytest.fixture(autouse=True)
def reset_logging() -> Generator[None, None, None]:
    """Reset logging between tests."""
    import logging

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    yield
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
