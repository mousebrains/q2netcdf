"""Pytest configuration and shared fixtures."""
import pytest
import json


@pytest.fixture
def sample_qfile_path(tmp_path):
    """
    Create a minimal valid Q-file for testing.

    TODO: This is a placeholder. Real Q-files have complex binary format
    defined in Rockland's TN-054. To enable comprehensive integration tests,
    we need actual Q-file samples with:
    - Valid v1.2 format file
    - Valid v1.3 format file
    - Files with various sensors/spectra combinations
    - Edge cases (single record, large files, etc.)

    For now, returns a path where test Q-file would be created.
    Tests should handle FileNotFoundError gracefully.
    """
    qfile = tmp_path / "sample.q"
    return qfile


@pytest.fixture
def sample_config_json(tmp_path):
    """Create a sample JSON configuration file for QReduce."""
    config = {
        "channels": ["pressure", "temperature_JAC"],
        "spectra": ["sh_1", "T_1"],
        "config": ["diss_length", "fft_length"],
    }
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config))
    return config_file


@pytest.fixture
def empty_file(tmp_path):
    """Create an empty file for error testing."""
    empty = tmp_path / "empty.q"
    empty.touch()
    return empty


@pytest.fixture
def corrupt_file(tmp_path):
    """Create a file with invalid content for error testing."""
    corrupt = tmp_path / "corrupt.q"
    corrupt.write_bytes(b"This is not a valid Q-file\x00\x01\x02")
    return corrupt
