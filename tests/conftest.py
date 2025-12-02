"""Test configuration and fixtures for pytest."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def test_model_dir(tmp_path_factory):
    """Create temporary directory for test models."""
    return tmp_path_factory.mktemp("test_models")
