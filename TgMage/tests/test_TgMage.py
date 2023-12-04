"""
Unit and regression test for the TgMage package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import TgMage


def test_TgMage_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "TgMage" in sys.modules
