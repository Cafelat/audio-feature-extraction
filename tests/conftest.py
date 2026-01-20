"""Configuration for pytest."""

import pytest


@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate for tests."""
    return 16000
