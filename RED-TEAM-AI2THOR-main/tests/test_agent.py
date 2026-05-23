import pytest
from ai2thor_lab import Agent

def test_imports():
    """Verify that the module imports correctly."""
    assert Agent is not None
