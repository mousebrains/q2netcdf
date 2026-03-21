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


def _build_v13_segment(
    dt_ms: int,
    Nc: int,
    channel_idents: list[int],
    config_str: str,
    data_values: list[list[float]],
) -> bytes:
    """Build a single v1.3 header+data segment."""
    Ns, Nf = 0, 0
    header = bytearray()
    header += struct.pack("<H", RecordType.HEADER.value)
    header += struct.pack("<f", QVersion.v13.value)
    header += struct.pack("<Q", dt_ms)
    header += struct.pack("<HHH", Nc, Ns, Nf)
    for ident in channel_idents:
        header += struct.pack("<H", ident)
    header += struct.pack("<H", len(config_str))
    header += config_str.encode("utf-8")
    data_size = 2 + 2 + (Nc * 2)  # ident + stime + channels
    header += struct.pack("<H", data_size)

    data_records = bytearray()
    for row in data_values:
        data_records += struct.pack("<H", RecordType.DATA.value)
        data_records += struct.pack("<e", row[0])  # stime
        for val in row[1:]:
            data_records += struct.pack("<e", val)

    return bytes(header + data_records)


@pytest.fixture
def synthetic_multiheader_qfile(tmp_path):
    """Create a synthetic v1.3 Q-file with two header/data segments."""
    channel_idents = [0xA12, 0x160, 0xA11]  # e_2, pressure, e_1
    config_str = '{"diss_length":16,"fft_length":4}'
    dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))

    # Segment 1: 5 records, surface (pressure ~ 0, epsilon = NaN)
    seg1_data = []
    for i in range(5):
        seg1_data.append([float(i * 16), float("nan"), -0.24 + i * 0.01, float("nan")])
    seg1 = _build_v13_segment(dt_ms, 3, channel_idents, config_str, seg1_data)

    # Segment 2: 10 records, profiling (valid epsilon and pressure)
    dt_ms2 = dt_ms + 100_000
    seg2_data = []
    for i in range(10):
        seg2_data.append(
            [float(i * 16), -8.0 - i * 0.1, 500.0 - i * 10, -7.5 - i * 0.2]
        )
    seg2 = _build_v13_segment(dt_ms2, 3, channel_idents, config_str, seg2_data)

    qfile_path = tmp_path / "multiheader.q"
    qfile_path.write_bytes(seg1 + seg2)
    return qfile_path


@pytest.fixture
def synthetic_multi_mri_files(tmp_path):
    """Create three synthetic v1.3 Q-files for testing multi-file merge.

    File 1: 3 records, base time 2025-01-01T00:00:00, surface (epsilon NaN)
    File 2: 4 records, base time 2025-01-01T01:00:00, profiling
    File 3: 2 records, base time 2025-01-01T02:00:00, profiling (multiheader)
             + 3 records in a second segment at 2025-01-01T02:05:00
    """
    channel_idents = [0xA12, 0x160, 0xA11]  # e_2, pressure, e_1
    config_str = '{"diss_length":16,"fft_length":4}'

    base1 = int(
        np.datetime64("2025-01-01T00:00:00").astype("datetime64[ms]").astype(int)
    )
    base2 = int(
        np.datetime64("2025-01-01T01:00:00").astype("datetime64[ms]").astype(int)
    )
    base3a = int(
        np.datetime64("2025-01-01T02:00:00").astype("datetime64[ms]").astype(int)
    )
    base3b = int(
        np.datetime64("2025-01-01T02:05:00").astype("datetime64[ms]").astype(int)
    )

    # File 1: surface, epsilon = NaN, pressure near 0
    f1_data = [
        [0.0, float("nan"), -0.5, float("nan")],
        [16.0, float("nan"), -0.5, float("nan")],
        [32.0, float("nan"), -0.5, float("nan")],
    ]
    f1 = _build_v13_segment(base1, 3, channel_idents, config_str, f1_data)

    # File 2: profiling, all valid
    f2_data = [
        [0.0, -8.0, 500.0, -8.5],
        [16.0, -9.0, 490.0, -9.5],
        [32.0, -10.0, 480.0, -10.0],
        [48.0, -8.5, 470.0, -9.0],
    ]
    f2 = _build_v13_segment(base2, 3, channel_idents, config_str, f2_data)

    # File 3: two segments (multiheader)
    f3a_data = [
        [0.0, float("nan"), 100.0, float("nan")],
        [16.0, float("nan"), 200.0, float("nan")],
    ]
    f3b_data = [
        [0.0, -7.0, 300.0, -7.5],
        [16.0, -8.0, 400.0, -8.0],
        [32.0, -9.0, 500.0, -9.0],
    ]
    f3 = _build_v13_segment(
        base3a, 3, channel_idents, config_str, f3a_data
    ) + _build_v13_segment(base3b, 3, channel_idents, config_str, f3b_data)

    p1 = tmp_path / "file1.mri"
    p2 = tmp_path / "file2.mri"
    p3 = tmp_path / "file3.mri"
    p1.write_bytes(f1)
    p2.write_bytes(f2)
    p3.write_bytes(f3)
    return [p1, p2, p3]


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
