"""Pytest configuration and shared fixtures."""

import struct
import pytest
import json
import numpy as np
from pathlib import Path

from q2netcdf.QRecordType import RecordType
from q2netcdf.QVersion import QVersion

# Real MRI files from the RIOT deployment
MRI_DIR = Path(__file__).parent.parent.parent / "RIOT" / "osu685" / "from-glider"

# Pick a file with data records (2KB, 198 records)
MRI_FILE = MRI_DIR / "02880000.mri"
# A smaller file (453 bytes, 39 records)
MRI_FILE_SMALL = MRI_DIR / "02830000.mri"


def _have_mri_files() -> bool:
    return MRI_FILE.exists()


@pytest.fixture
def mri_file():
    """Path to a real MRI file with data records."""
    if not MRI_FILE.exists():
        pytest.skip("MRI test files not found")
    return MRI_FILE


@pytest.fixture
def mri_file_small():
    """Path to a small real MRI file."""
    if not MRI_FILE_SMALL.exists():
        pytest.skip("MRI test files not found")
    return MRI_FILE_SMALL


@pytest.fixture
def synthetic_v13_qfile(tmp_path):
    """Create a synthetic v1.3 Q-file with channels, config, and data records."""
    qfile_path = tmp_path / "synthetic_v13.q"

    header = bytearray()
    header += struct.pack("<H", RecordType.HEADER.value)
    header += struct.pack("<f", QVersion.v13.value)

    dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
    header += struct.pack("<Q", dt_ms)

    Nc, Ns, Nf = 3, 0, 0
    header += struct.pack("<HHH", Nc, Ns, Nf)
    # Channel idents: epsilon_2 (0xA12), pressure (0x160), epsilon_1 (0xA11)
    header += struct.pack("<HHH", 0xA12, 0x160, 0xA11)

    config_str = '{"diss_length":16,"fft_length":4}'
    header += struct.pack("<H", len(config_str))
    header += config_str.encode("utf-8")

    data_size = 2 + 2 + (Nc * 2)  # ident + stime + channels
    header += struct.pack("<H", data_size)

    data_records = bytearray()
    for i in range(10):
        data_records += struct.pack("<H", RecordType.DATA.value)
        data_records += struct.pack("<e", float(i))
        data_records += struct.pack("<eee", 1e-8 + i * 1e-9, 100.0 + i, 2e-8 + i * 1e-9)

    qfile_path.write_bytes(header + data_records)
    return qfile_path


@pytest.fixture
def sample_qfile_path(tmp_path):
    """Create a minimal valid Q-file path for testing."""
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
