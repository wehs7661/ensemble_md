"""
Unit and regression test for the ensemble_md package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import ensemble_md


def test_ensemble_md_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "ensemble_md" in sys.modules
