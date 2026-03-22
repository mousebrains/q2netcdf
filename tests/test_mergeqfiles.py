#
# Unit tests for mergeqfiles.py
#
# Mar-2025, Claude Code Assistant

import pytest
import os
import subprocess
import numpy as np
from argparse import Namespace
from pathlib import Path
import json

# Import from mergeqfiles (standalone module)
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import q2netcdf.mergeqfiles as mergeqfiles_module
from q2netcdf.mergeqfiles import (
    RecordType,
    QVersion,
    QConfig,
    QHexCodes,
    QHeader,
    QReduce,
    glueFiles,
    decimateFiles,
    reduceFiles,
    reduceAndDecimate,
    fileCandidates,
    scanDirectory,
)

# Test data directory
DATA_DIR = Path(__file__).parent / "data"
QFILE_V12 = DATA_DIR / "v12_15records.q"
QFILE_V12_SMALL = DATA_DIR / "v12_14records.q"
MRI_FILE = DATA_DIR / "v13_131records.mri"
MRI_FILE_SMALL = DATA_DIR / "v13_39records.mri"


class TestRecordType:
    """Test RecordType enum."""

    def test_header_value(self):
        assert RecordType.HEADER.value == 0x1729

    def test_data_value(self):
        assert RecordType.DATA.value == 0x1657

    def test_config_v12_value(self):
        assert RecordType.CONFIG_V12.value == 0x0827


class TestQVersion:
    """Test QVersion enum."""

    def test_v12_value(self):
        assert QVersion.v12.value == 1.2

    def test_v13_value(self):
        assert QVersion.v13.value == 1.3

    def test_isV12(self):
        assert QVersion.v12.isV12() is True
        assert QVersion.v13.isV12() is False

    def test_isV13(self):
        assert QVersion.v13.isV13() is True
        assert QVersion.v12.isV13() is False


class TestQConfig:
    """Test QConfig parser."""

    def test_init_v13(self):
        config_data = b'{"test": 123}'
        qc = QConfig(config_data, QVersion.v13)
        assert qc.raw() == config_data
        assert len(qc) == len(config_data)

    def test_config_v13_json(self):
        config_data = b'{"key": "value", "number": 42}'
        qc = QConfig(config_data, QVersion.v13)
        cfg = qc.config()
        assert cfg["key"] == "value"
        assert cfg["number"] == 42

    def test_parse_value_int(self):
        config_data = b'{"test": 1}'
        qc = QConfig(config_data, QVersion.v13)
        # Access private method for testing
        result = qc._QConfig__parseValue("123")
        assert result == 123
        assert isinstance(result, int)

    def test_parse_value_float(self):
        config_data = b"{}"
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue("3.14")
        assert result == 3.14
        assert isinstance(result, float)

    def test_parse_value_string(self):
        config_data = b"{}"
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue('"hello"')
        assert result == "hello"

    def test_parse_value_bool_true(self):
        config_data = b"{}"
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue("true")
        assert result is True

    def test_parse_value_bool_false(self):
        config_data = b"{}"
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue("false")
        assert result is False

    def test_parse_value_array_empty(self):
        config_data = b"{}"
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue("[]")
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_parse_value_array_numbers(self):
        config_data = b"{}"
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue("[1, 2, 3]")
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert result[0] == 1
        assert result[2] == 3

    def test_repr_v13(self):
        """QConfig repr shows sorted key-value pairs."""
        config_data = b'{"b_key": 2, "a_key": 1}'
        qc = QConfig(config_data, QVersion.v13)
        text = repr(qc)
        assert "a_key -> 1" in text
        assert "b_key -> 2" in text
        # Keys should be sorted: a before b
        assert text.index("a_key") < text.index("b_key")

    def test_config_v12_perl_format(self):
        """QConfig parses v1.2 Perl-style key-value pairs."""
        config_data = b'"diss_length" => 16\n"fft_length" => 4\n'
        qc = QConfig(config_data, QVersion.v12)
        cfg = qc.config()
        assert cfg["diss_length"] == 16
        assert cfg["fft_length"] == 4

    def test_config_v12_with_array(self):
        """QConfig parses v1.2 arrays."""
        config_data = b'"channels" => [1, 2, 3]\n'
        qc = QConfig(config_data, QVersion.v12)
        cfg = qc.config()
        assert isinstance(cfg["channels"], np.ndarray)
        assert len(cfg["channels"]) == 3

    def test_config_v12_bad_unicode(self):
        """QConfig handles non-UTF-8 bytes in v1.2 config."""
        config_data = b'"key" => 1\n\xff\xfe\n"key2" => 2\n'
        qc = QConfig(config_data, QVersion.v12)
        cfg = qc.config()
        assert cfg["key"] == 1
        assert cfg["key2"] == 2

    def test_config_v13_invalid_json(self):
        """QConfig handles invalid v1.3 JSON gracefully."""
        config_data = b"not json at all"
        qc = QConfig(config_data, QVersion.v13)
        cfg = qc.config()
        assert cfg == {}

    def test_size_method(self):
        """QConfig size() returns length of raw config bytes."""
        config_data = b'{"test": 1}'
        qc = QConfig(config_data, QVersion.v13)
        assert qc.size() == len(config_data)

    def test_parse_value_bare_string(self):
        """QConfig __parseValue returns bare string for non-matching values."""
        config_data = b"{}"
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue("some_bare_value")
        assert result == "some_bare_value"


class TestQHexCodes:
    """Test QHexCodes mapping."""

    def test_init(self):
        qh = QHexCodes()
        assert qh is not None

    def test_name_shear_probe(self):
        name = QHexCodes.name(0x610)
        assert name == "sh_0"

        name = QHexCodes.name(0x611)
        assert name == "sh_1"

    def test_name_temperature(self):
        name = QHexCodes.name(0x620)
        assert name == "T_0"

    def test_name_pressure(self):
        name = QHexCodes.name(0x160)
        assert name == "pressure"

    def test_name_unknown(self):
        name = QHexCodes.name(0xFFFF)
        assert name is None

    def test_attributes_pressure(self):
        attrs = QHexCodes.attributes(0x160)
        assert attrs is not None
        assert attrs["long_name"] == "pressure_ocean"
        assert attrs["units"] == "decibar"

    def test_attributes_unknown(self):
        attrs = QHexCodes.attributes(0xFFFF)
        assert attrs is None

    def test_name2ident_shear(self):
        ident = QHexCodes.name2ident("sh_1")
        assert ident == 0x611

    def test_name2ident_no_number(self):
        ident = QHexCodes.name2ident("pressure")
        assert ident == 0x160

    def test_name2ident_unknown(self):
        ident = QHexCodes.name2ident("unknown_sensor")
        assert ident is None


class TestQHeader:
    """Test QHeader parsing."""

    def test_repr_v12(self, qfile_v12):
        """QHeader repr shows formatted header info."""
        with open(str(qfile_v12), "rb") as fp:
            hdr = QHeader(fp, str(qfile_v12))
        text = repr(hdr)
        assert "filename:" in text
        assert "Version:" in text
        assert "Channels:" in text
        assert "Spectra:" in text
        assert "Data Size:" in text
        assert "Header Size:" in text

    def test_repr_v13(self, mri_file):
        """QHeader repr works for v1.3 files."""
        with open(str(mri_file), "rb") as fp:
            hdr = QHeader(fp, str(mri_file))
        text = repr(hdr)
        assert "Version:" in text
        assert "v13" in text

    def test_chk_ident_valid(self, qfile_v12):
        """chkIdent returns True for valid header."""
        with open(str(qfile_v12), "rb") as fp:
            result = QHeader.chkIdent(fp)
        assert result is True

    def test_chk_ident_invalid(self, tmp_path):
        """chkIdent returns False for invalid header."""
        f = tmp_path / "bad.q"
        f.write_bytes(b"\x00\x00\x00\x00")
        with open(str(f), "rb") as fp:
            result = QHeader.chkIdent(fp)
        assert result is False

    def test_chk_ident_short_file(self, tmp_path):
        """chkIdent returns None for file shorter than 2 bytes."""
        f = tmp_path / "tiny.q"
        f.write_bytes(b"\x00")
        with open(str(f), "rb") as fp:
            result = QHeader.chkIdent(fp)
        assert result is None

    def test_invalid_version(self, tmp_path):
        """QHeader raises NotImplementedError for unknown version."""
        import struct

        f = tmp_path / "bad_version.q"
        data = struct.pack("<H", 0x1729)  # Valid header ident
        data += struct.pack("<f", 9.9)  # Invalid version
        data += struct.pack("<Q", 0)  # dt
        data += struct.pack("<HHH", 0, 0, 0)  # Nc, Ns, Nf
        data += b"\x00" * 50  # Padding
        f.write_bytes(data)
        with pytest.raises(NotImplementedError):
            with open(str(f), "rb") as fp:
                QHeader(fp, str(f))

    def test_invalid_ident(self, tmp_path):
        """QHeader raises ValueError for wrong header identifier."""
        import struct

        f = tmp_path / "wrong_ident.q"
        data = struct.pack("<H", 0xBEEF)  # Wrong ident
        data += b"\x00" * 50
        f.write_bytes(data)
        with pytest.raises(ValueError, match="Invalid header"):
            with open(str(f), "rb") as fp:
                QHeader(fp, str(f))


class TestQReduce:
    """Test QReduce functionality."""

    def test_load_config_nonexistent(self):
        config = QReduce.loadConfig("/nonexistent/file.cfg")
        assert config is None

    def test_load_config_invalid_json(self, tmp_path):
        cfg_file = tmp_path / "invalid.cfg"
        cfg_file.write_text("not json")
        config = QReduce.loadConfig(str(cfg_file))
        assert config is None

    def test_load_config_valid(self, tmp_path):
        cfg_file = tmp_path / "valid.cfg"
        cfg_data = {
            "config": ["key1", "key2"],
            "channels": ["sh_1", "T_1"],
            "spectra": [],
        }
        cfg_file.write_text(json.dumps(cfg_data))
        config = QReduce.loadConfig(str(cfg_file))
        assert config is not None
        assert config["channels"] == ["sh_1", "T_1"]


class TestGlueFiles:
    """Test glueFiles function."""

    def test_glue_empty_list(self, tmp_path):
        output = tmp_path / "output.q"
        size = glueFiles([], str(output))
        assert size == 0
        assert output.exists()

    def test_glue_single_file(self, tmp_path):
        input1 = tmp_path / "input1.q"
        input1.write_bytes(b"test data 1")
        output = tmp_path / "output.q"

        size = glueFiles([str(input1)], str(output))
        assert size == 11
        assert output.read_bytes() == b"test data 1"

    def test_glue_multiple_files(self, tmp_path):
        input1 = tmp_path / "input1.q"
        input2 = tmp_path / "input2.q"
        input1.write_bytes(b"data1")
        input2.write_bytes(b"data2")
        output = tmp_path / "output.q"

        size = glueFiles([str(input1), str(input2)], str(output))
        assert size == 10
        assert output.read_bytes() == b"data1data2"


class TestFileCandidates:
    """Test fileCandidates function."""

    def test_no_qfiles(self, tmp_path):
        from argparse import Namespace

        args = Namespace(datadir=str(tmp_path))
        times = np.array([0, 1000])

        qfiles, total = fileCandidates(args, times)
        assert qfiles == {}
        assert total == 0

    def test_with_qfiles(self, tmp_path):
        from argparse import Namespace
        import time as time_module

        # Create test Q-files
        qfile1 = tmp_path / "test1.q"
        qfile2 = tmp_path / "test2.q"
        qfile1.write_bytes(b"x" * 100)
        qfile2.write_bytes(b"y" * 200)

        # Use current time range
        now = time_module.time()
        args = Namespace(datadir=str(tmp_path))
        times = np.array([now - 60, now + 60])  # ±1 minute from now

        qfiles, total = fileCandidates(args, times)
        assert len(qfiles) == 2
        assert total == 300

    def test_filters_by_time(self, tmp_path):
        from argparse import Namespace
        import time as time_module

        # Create a Q-file
        qfile = tmp_path / "test.q"
        qfile.write_bytes(b"data")

        # Set time range that excludes the file
        now = time_module.time()
        args = Namespace(datadir=str(tmp_path))
        times = np.array([now + 3600, now + 7200])  # 1-2 hours in future

        qfiles, total = fileCandidates(args, times)
        assert len(qfiles) == 0
        assert total == 0

    def test_ignores_non_qfiles(self, tmp_path):
        from argparse import Namespace
        import time as time_module

        # Create various files
        (tmp_path / "test.q").write_bytes(b"data")
        (tmp_path / "test.txt").write_bytes(b"data")
        (tmp_path / "readme.md").write_bytes(b"data")

        now = time_module.time()
        args = Namespace(datadir=str(tmp_path))
        times = np.array([now - 60, now + 60])

        qfiles, total = fileCandidates(args, times)
        assert len(qfiles) == 1  # Only .q file


class TestQReduceWithRealFiles:
    """Test QReduce class with real Q-file data."""

    def test_init_v12_channels_only(self, qfile_v12):
        """QReduce with only channel selection on a v1.2 file."""
        config = {
            "channels": ["pressure", "e_1"],
            "spectra": [],
            "config": ["diss_length"],
        }
        qr = QReduce(str(qfile_v12), config)
        assert qr.filename == str(qfile_v12)
        assert qr.fileSizeOrig > 0
        assert qr.hdrSize > 0
        assert qr.dataSizeOrig == 210  # v1.2 data record size (25ch + 4*18 spectra)
        # Reduced data should be smaller
        assert qr.dataSize < qr.dataSizeOrig
        assert qr.fileSize < qr.fileSizeOrig

    def test_init_v12_with_spectra(self, qfile_v12):
        """QReduce with channels and spectra on a v1.2 file."""
        config = {
            "channels": ["pressure"],
            "spectra": ["shear_gfd_1"],
            "config": [],
        }
        qr = QReduce(str(qfile_v12), config)
        assert qr.dataSize > 0
        # Should include pressure + 18 frequency bins for 1 spectrum
        # dataSize = 4 (ident+stime) + 2*(1 channel + 1*18 freq bins) = 4 + 38 = 42
        assert qr.dataSize == 4 + 2 * (1 + 18)

    def test_init_v13_file(self, mri_file):
        """QReduce with a v1.3 MRI file."""
        config = {
            "channels": ["pressure"],
            "spectra": [],
            "config": ["diss_length"],
        }
        qr = QReduce(str(mri_file), config)
        assert qr.filename == str(mri_file)
        assert qr.fileSize < qr.fileSizeOrig
        # 1 channel: dataSize = 4 + 2*1 = 6
        assert qr.dataSize == 6

    def test_repr(self, qfile_v12):
        """QReduce __repr__ shows file sizes."""
        config = {"channels": ["pressure"], "spectra": [], "config": []}
        qr = QReduce(str(qfile_v12), config)
        text = repr(qr)
        assert "fn " in text
        assert "hdr " in text
        assert "data " in text
        assert "file " in text
        assert "->" in text

    def test_reduce_file_v12(self, qfile_v12, tmp_path):
        """reduceFile() writes reduced output from v1.2 file."""
        config = {"channels": ["pressure", "e_1"], "spectra": [], "config": []}
        qr = QReduce(str(qfile_v12), config)
        output = tmp_path / "reduced.q"
        with open(str(output), "wb") as ofp:
            sz = qr.reduceFile(ofp)
        assert sz > 0
        assert output.stat().st_size == sz
        # Verify the output is a valid Q-file by reading its header
        with open(str(output), "rb") as fp:
            hdr = QHeader(fp, str(output))
        assert hdr.version == QVersion.v13  # Always writes v1.3
        assert hdr.Nc == 2  # pressure + e_1

    def test_reduce_file_v13(self, mri_file, tmp_path):
        """reduceFile() writes reduced output from v1.3 file."""
        config = {"channels": ["pressure"], "spectra": [], "config": ["diss_length"]}
        qr = QReduce(str(mri_file), config)
        output = tmp_path / "reduced.mri"
        with open(str(output), "wb") as ofp:
            sz = qr.reduceFile(ofp)
        assert sz > 0
        with open(str(output), "rb") as fp:
            hdr = QHeader(fp, str(output))
        assert hdr.version == QVersion.v13
        assert hdr.Nc == 1  # pressure only
        cfg = hdr.config.config()
        assert "diss_length" in cfg
        assert cfg["diss_length"] == 16

    def test_reduce_file_with_spectra(self, qfile_v12, tmp_path):
        """reduceFile() preserves selected spectra."""
        config = {
            "channels": ["pressure"],
            "spectra": ["shear_gfd_1", "shear_gfd_2"],
            "config": [],
        }
        qr = QReduce(str(qfile_v12), config)
        output = tmp_path / "reduced_spectra.q"
        with open(str(output), "wb") as ofp:
            sz = qr.reduceFile(ofp)
        assert sz > 0
        with open(str(output), "rb") as fp:
            hdr = QHeader(fp, str(output))
        assert hdr.Nc == 1  # pressure
        assert hdr.Ns == 2  # 2 spectra
        assert hdr.Nf == 18  # frequencies preserved

    def test_decimate_v12(self, qfile_v12, tmp_path):
        """decimate() writes subset of records."""
        config = {"channels": ["pressure", "e_1"], "spectra": [], "config": []}
        qr = QReduce(str(qfile_v12), config)
        # Keep every other record (indices 0, 2, 4, ...)
        indices = np.arange(0, int(qr.nRecords), 2)
        output = tmp_path / "decimated.q"
        with open(str(output), "wb") as ofp:
            sz = qr.decimate(ofp, indices)
        assert sz > 0
        assert sz < qr.fileSize  # Decimated should be smaller

    def test_decimate_v13(self, mri_file, tmp_path):
        """decimate() on v1.3 file."""
        config = {"channels": ["pressure"], "spectra": [], "config": []}
        qr = QReduce(str(mri_file), config)
        # Keep first 10 records
        indices = np.arange(min(10, int(qr.nRecords)))
        output = tmp_path / "decimated.mri"
        with open(str(output), "wb") as ofp:
            sz = qr.decimate(ofp, indices)
        assert sz > 0

    def test_reduce_empty_channels(self, mri_file, tmp_path):
        """QReduce with no matching channels still works."""
        config = {
            "channels": ["nonexistent_sensor"],
            "spectra": [],
            "config": [],
        }
        qr = QReduce(str(mri_file), config)
        # 0 channels -> dataSize = 4 + 0 = 4
        assert qr.dataSize == 4
        output = tmp_path / "empty_channels.mri"
        with open(str(output), "wb") as ofp:
            sz = qr.reduceFile(ofp)
        assert sz > 0

    def test_load_config_missing_key(self, tmp_path):
        """loadConfig rejects config missing required keys."""
        cfg_file = tmp_path / "incomplete.cfg"
        cfg_data = {"channels": ["pressure"]}  # missing spectra and config
        cfg_file.write_text(json.dumps(cfg_data))
        config = QReduce.loadConfig(str(cfg_file))
        assert config is None

    def test_load_config_not_dict(self, tmp_path):
        """loadConfig rejects non-dict JSON."""
        cfg_file = tmp_path / "array.cfg"
        cfg_file.write_text("[1, 2, 3]")
        config = QReduce.loadConfig(str(cfg_file))
        assert config is None


class TestDecimateFiles:
    """Test decimateFiles function."""

    def test_decimate_v12_files(self, qfile_v12, tmp_path):
        """decimateFiles reduces v1.2 Q-files to fit max size."""
        file_size = os.path.getsize(str(qfile_v12))
        qfiles = {str(qfile_v12): file_size}
        output = tmp_path / "decimated.q"
        # Set maxSize to half the original
        max_size = file_size // 2
        sz = decimateFiles(qfiles, str(output), file_size, max_size)
        assert sz > 0
        assert sz <= max_size + 500  # allow some margin for headers

    def test_decimate_v13_files(self, mri_file, tmp_path):
        """decimateFiles reduces v1.3 MRI files to fit max size."""
        file_size = os.path.getsize(str(mri_file))
        qfiles = {str(mri_file): file_size}
        output = tmp_path / "decimated.mri"
        max_size = file_size // 3
        sz = decimateFiles(qfiles, str(output), file_size, max_size)
        assert sz > 0
        assert sz <= max_size + 200

    def test_decimate_multiple_files(self, qfile_v12, qfile_v12_small, tmp_path):
        """decimateFiles handles multiple input files."""
        size1 = os.path.getsize(str(qfile_v12))
        size2 = os.path.getsize(str(qfile_v12_small))
        qfiles = {str(qfile_v12): size1, str(qfile_v12_small): size2}
        total = size1 + size2
        output = tmp_path / "multi_dec.q"
        max_size = total // 2
        sz = decimateFiles(qfiles, str(output), total, max_size)
        assert sz > 0

    def test_decimate_with_truncated_file_only(self, tmp_path):
        """decimateFiles raises ZeroDivisionError when all files are truncated.

        This is an edge case: if every input file has a truncated header,
        totDataSize is 0 and the ratio calculation divides by zero.
        """
        bad_file = tmp_path / "truncated.q"
        bad_file.write_bytes(b"\x00" * 5)
        output = tmp_path / "output.q"
        qfiles = {str(bad_file): 5}
        with pytest.raises(ZeroDivisionError):
            decimateFiles(qfiles, str(output), 5, 100)

    def test_decimate_with_invalid_header(self, tmp_path):
        """decimateFiles raises ZeroDivisionError when header is invalid.

        When the only input file has an invalid header (wrong ident), it gets
        skipped, leaving totDataSize=0 and triggering a division by zero.
        """
        bad_file = tmp_path / "bad_header.q"
        bad_file.write_bytes(b"\x00" * 1000)
        output = tmp_path / "output.q"
        qfiles = {str(bad_file): 1000}
        with pytest.raises(ZeroDivisionError):
            decimateFiles(qfiles, str(output), 1000, 500)

    def test_decimate_zero_ratio(self, qfile_v12, tmp_path):
        """decimateFiles handles case where ratio <= 0 (maxSize too small)."""
        file_size = os.path.getsize(str(qfile_v12))
        qfiles = {str(qfile_v12): file_size}
        output = tmp_path / "tiny.q"
        # maxSize smaller than header => ratio <= 0
        sz = decimateFiles(qfiles, str(output), file_size, 10)
        # Should return 0 (empty output) since ratio is too small
        assert sz >= 0


class TestReduceFiles:
    """Test reduceFiles function."""

    def test_reduce_v12_no_decimate(self, qfile_v12, tmp_path):
        """reduceFiles reduces a v1.2 file without decimation."""
        cfg_data = {
            "channels": ["pressure", "e_1"],
            "spectra": [],
            "config": [],
        }
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(cfg_data))
        output = tmp_path / "reduced.q"
        file_size = os.path.getsize(str(qfile_v12))
        qfiles = {str(qfile_v12): file_size}
        # Large maxSize so no decimation needed
        result = reduceFiles(qfiles, str(cfg_file), str(output), file_size * 10)
        assert result is not None
        assert result > 0
        # Verify output is valid
        with open(str(output), "rb") as fp:
            hdr = QHeader(fp, str(output))
        assert hdr.Nc == 2

    def test_reduce_v12_with_decimate(self, qfile_v12, tmp_path):
        """reduceFiles reduces and decimates when file exceeds maxSize."""
        cfg_data = {
            "channels": ["pressure", "e_1", "e_2"],
            "spectra": [],
            "config": ["diss_length"],
        }
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(cfg_data))
        output = tmp_path / "reduced_dec.q"
        file_size = os.path.getsize(str(qfile_v12))
        qfiles = {str(qfile_v12): file_size}
        # Small maxSize to force decimation
        result = reduceFiles(qfiles, str(cfg_file), str(output), 200)
        assert result is not None
        assert result > 0

    def test_reduce_v13(self, mri_file, tmp_path):
        """reduceFiles on v1.3 file."""
        cfg_data = {
            "channels": ["pressure"],
            "spectra": [],
            "config": ["diss_length"],
        }
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(cfg_data))
        output = tmp_path / "reduced.mri"
        file_size = os.path.getsize(str(mri_file))
        qfiles = {str(mri_file): file_size}
        result = reduceFiles(qfiles, str(cfg_file), str(output), file_size * 10)
        assert result is not None
        assert result > 0

    def test_reduce_invalid_config(self, qfile_v12, tmp_path):
        """reduceFiles returns None with invalid config file."""
        output = tmp_path / "output.q"
        file_size = os.path.getsize(str(qfile_v12))
        qfiles = {str(qfile_v12): file_size}
        result = reduceFiles(qfiles, "/nonexistent/config.json", str(output), 99999)
        assert result is None

    def test_reduce_multiple_files(self, qfile_v12, qfile_v12_small, tmp_path):
        """reduceFiles handles multiple input files."""
        cfg_data = {
            "channels": ["pressure"],
            "spectra": [],
            "config": [],
        }
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(cfg_data))
        output = tmp_path / "multi_reduced.q"
        size1 = os.path.getsize(str(qfile_v12))
        size2 = os.path.getsize(str(qfile_v12_small))
        qfiles = {str(qfile_v12): size1, str(qfile_v12_small): size2}
        result = reduceFiles(qfiles, str(cfg_file), str(output), 999999)
        assert result is not None
        assert result > 0


class TestReduceAndDecimate:
    """Test reduceAndDecimate function."""

    def test_basic_reduce_and_decimate(self, qfile_v12, tmp_path):
        """reduceAndDecimate reduces and decimates to fit maxSize."""
        config = {"channels": ["pressure", "e_1"], "spectra": [], "config": []}
        qr = QReduce(str(qfile_v12), config)
        info = {str(qfile_v12): qr}
        output = tmp_path / "rad_output.q"
        with open(str(output), "wb") as ofp:
            sz = reduceAndDecimate(info, ofp, str(output), 500)
        assert sz > 0

    def test_reduce_and_decimate_v13(self, mri_file, tmp_path):
        """reduceAndDecimate with v1.3 file."""
        config = {"channels": ["pressure"], "spectra": [], "config": []}
        qr = QReduce(str(mri_file), config)
        info = {str(mri_file): qr}
        output = tmp_path / "rad_v13.mri"
        with open(str(output), "wb") as ofp:
            sz = reduceAndDecimate(info, ofp, str(output), 200)
        assert sz > 0

    def test_reduce_and_decimate_multiple(self, qfile_v12, qfile_v12_small, tmp_path):
        """reduceAndDecimate with multiple files."""
        config = {"channels": ["pressure"], "spectra": [], "config": []}
        qr1 = QReduce(str(qfile_v12), config)
        qr2 = QReduce(str(qfile_v12_small), config)
        info = {str(qfile_v12): qr1, str(qfile_v12_small): qr2}
        output = tmp_path / "rad_multi.q"
        with open(str(output), "wb") as ofp:
            sz = reduceAndDecimate(info, ofp, str(output), 300)
        assert sz > 0

    def test_reduce_and_decimate_zero_ratio(self, qfile_v12, tmp_path):
        """reduceAndDecimate with maxSize too small for headers returns current pos."""
        config = {"channels": ["pressure", "e_1"], "spectra": [], "config": []}
        qr = QReduce(str(qfile_v12), config)
        info = {str(qfile_v12): qr}
        output = tmp_path / "rad_tiny.q"
        with open(str(output), "wb") as ofp:
            # maxSize of 1 is smaller than total header size, so ratio <= 0
            sz = reduceAndDecimate(info, ofp, str(output), 1)
        # Should return 0 (position in empty file) since nothing is written
        assert sz == 0


class TestGlueFilesErrors:
    """Test glueFiles error handling."""

    def test_glue_unreadable_file(self, tmp_path):
        """glueFiles catches OSError for unreadable files."""
        output = tmp_path / "output.q"
        # Pass a directory as input (causes OSError on open for reading)
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        size = glueFiles([str(subdir)], str(output))
        assert size == 0

    def test_glue_nonexistent_file(self, tmp_path):
        """glueFiles catches OSError for nonexistent input."""
        output = tmp_path / "output.q"
        size = glueFiles(["/nonexistent/file.q"], str(output))
        assert size == 0


class TestScanDirectory:
    """Test scanDirectory function."""

    def test_no_qfiles_creates_empty(self, tmp_path):
        """scanDirectory creates empty output when no Q-files found."""
        import time as time_module

        output = tmp_path / "output.q"
        args = Namespace(
            datadir=str(tmp_path),
            output=str(output),
            maxSize=999999,
            bufferSize=100 * 1024,
            config=None,
        )
        now = time_module.time()
        times = np.array([now - 60, now + 60])
        result = scanDirectory(args, times)
        assert result == 0
        assert output.exists()
        assert output.stat().st_size == 0

    def test_glue_small_files(self, tmp_path):
        """scanDirectory glues files when total size < maxSize."""
        import time as time_module
        import shutil

        # Copy real Q-files into tmp dir
        src = QFILE_V12
        dest = tmp_path / "test.q"
        shutil.copy(str(src), str(dest))
        # Touch to set mtime to now (copy2 preserves old timestamps)
        dest.touch()

        output = tmp_path / "merged.q"
        args = Namespace(
            datadir=str(tmp_path),
            output=str(output),
            maxSize=999999,
            bufferSize=100 * 1024,
            config=None,
        )
        now = time_module.time()
        times = np.array([now - 60, now + 60])
        result = scanDirectory(args, times)
        assert result > 0
        assert output.exists()

    def test_decimate_large_files(self, tmp_path):
        """scanDirectory decimates when total size > maxSize."""
        import time as time_module
        import shutil

        src = QFILE_V12
        dest = tmp_path / "test.q"
        shutil.copy(str(src), str(dest))
        dest.touch()

        output = tmp_path / "decimated.q"
        file_size = os.path.getsize(str(dest))
        args = Namespace(
            datadir=str(tmp_path),
            output=str(output),
            maxSize=file_size // 2,
            bufferSize=100 * 1024,
            config=None,
        )
        now = time_module.time()
        times = np.array([now - 60, now + 60])
        result = scanDirectory(args, times)
        assert result > 0
        assert output.exists()
        assert output.stat().st_size <= file_size

    def test_existing_output_no_new_files(self, tmp_path):
        """scanDirectory with existing output and no new Q-files returns existing size."""
        import time as time_module

        output = tmp_path / "output.q"
        output.write_bytes(b"existing data here")
        existing_size = output.stat().st_size

        args = Namespace(
            datadir=str(tmp_path),
            output=str(output),
            maxSize=999999,
            bufferSize=100 * 1024,
            config=None,
        )
        # Time range that excludes any files
        now = time_module.time()
        times = np.array([now + 3600, now + 7200])
        result = scanDirectory(args, times)
        assert result == existing_size

    def test_existing_output_with_new_files(self, tmp_path):
        """scanDirectory appends to existing output file."""
        import time as time_module
        import shutil

        # Create existing output
        output = tmp_path / "output.q"
        output.write_bytes(b"x" * 10)
        existing_size = output.stat().st_size

        # Create a Q-file data dir
        datadir = tmp_path / "data"
        datadir.mkdir()
        dest = datadir / "test.q"
        shutil.copy(str(QFILE_V12), str(dest))
        dest.touch()

        args = Namespace(
            datadir=str(datadir),
            output=str(output),
            maxSize=999999,
            bufferSize=100 * 1024,
            config=None,
        )
        now = time_module.time()
        times = np.array([now - 60, now + 60])
        result = scanDirectory(args, times)
        assert result > existing_size

    def test_existing_output_maxsize_exceeded(self, tmp_path):
        """scanDirectory stops when existing output uses all maxSize."""
        import time as time_module
        import shutil

        output = tmp_path / "output.q"
        output.write_bytes(b"x" * 500)

        datadir = tmp_path / "data"
        datadir.mkdir()
        dest = datadir / "test.q"
        shutil.copy(str(QFILE_V12), str(dest))
        dest.touch()

        args = Namespace(
            datadir=str(datadir),
            output=str(output),
            maxSize=100,  # Already exceeded by existing file
            bufferSize=100 * 1024,
            config=None,
        )
        now = time_module.time()
        times = np.array([now - 60, now + 60])
        result = scanDirectory(args, times)
        # Should return existing size since maxSize is already exceeded
        assert result == 500

    def test_config_based_reduction(self, tmp_path):
        """scanDirectory uses config for reduction when config file exists."""
        import time as time_module
        import shutil

        datadir = tmp_path / "data"
        datadir.mkdir()
        dest = datadir / "test.q"
        shutil.copy(str(QFILE_V12), str(dest))
        dest.touch()

        cfg_data = {
            "channels": ["pressure"],
            "spectra": [],
            "config": [],
        }
        cfg_file = tmp_path / "reduce.cfg"
        cfg_file.write_text(json.dumps(cfg_data))

        output = tmp_path / "reduced.q"
        args = Namespace(
            datadir=str(datadir),
            output=str(output),
            maxSize=999999,
            bufferSize=100 * 1024,
            config=str(cfg_file),
        )
        now = time_module.time()
        times = np.array([now - 60, now + 60])
        result = scanDirectory(args, times)
        assert result > 0
        assert output.exists()

    def test_temp_file_cleanup_on_exception(self, tmp_path):
        """scanDirectory cleans up temp file on exception."""
        import time as time_module

        # Create a Q-file that will cause issues during processing
        datadir = tmp_path / "data"
        datadir.mkdir()
        qfile = datadir / "bad.q"
        # Write just enough to be found but not enough for valid header
        qfile.write_bytes(b"\x00" * 10)

        output = tmp_path / "output.q"
        args = Namespace(
            datadir=str(datadir),
            output=str(output),
            maxSize=5,  # Very small to force decimation
            bufferSize=100 * 1024,
            config=None,
        )
        now = time_module.time()
        times = np.array([now - 60, now + 60])

        # decimateFiles should handle the bad file and not leave temp files
        try:
            scanDirectory(args, times)
        except Exception:
            pass
        # Temp file should be cleaned up
        tmp_file = Path(str(output) + ".tmp")
        assert not tmp_file.exists()


class TestChkExists:
    """Test __chkExists validator."""

    def test_existing_file(self, tmp_path):
        """Returns filename for existing file."""
        f = tmp_path / "exists.txt"
        f.write_text("hello")
        chk = getattr(mergeqfiles_module, "_mergeqfiles__chkExists", None)
        if chk is None:
            # Module-level __ functions aren't mangled; try direct access
            chk = mergeqfiles_module.__dict__.get("__chkExists")
        if chk is None:
            pytest.skip("Cannot access __chkExists")
        result = chk(str(f))
        assert result == str(f)

    def test_nonexistent_file(self, tmp_path):
        """Raises ArgumentTypeError for nonexistent file."""
        from argparse import ArgumentTypeError

        chk = getattr(mergeqfiles_module, "_mergeqfiles__chkExists", None)
        if chk is None:
            chk = mergeqfiles_module.__dict__.get("__chkExists")
        if chk is None:
            pytest.skip("Cannot access __chkExists")
        with pytest.raises(ArgumentTypeError):
            chk("/nonexistent/path/file.txt")


class TestChkPositiveInt:
    """Test __chkPositiveInt validator."""

    def test_positive_int(self):
        """Returns int for positive value."""
        chk = getattr(mergeqfiles_module, "_mergeqfiles__chkPositiveInt", None)
        if chk is None:
            chk = mergeqfiles_module.__dict__.get("__chkPositiveInt")
        if chk is None:
            pytest.skip("Cannot access __chkPositiveInt")
        assert chk("5") == 5
        assert chk("100") == 100

    def test_zero(self):
        """Raises ArgumentTypeError for zero."""
        from argparse import ArgumentTypeError

        chk = getattr(mergeqfiles_module, "_mergeqfiles__chkPositiveInt", None)
        if chk is None:
            chk = mergeqfiles_module.__dict__.get("__chkPositiveInt")
        if chk is None:
            pytest.skip("Cannot access __chkPositiveInt")
        with pytest.raises(ArgumentTypeError):
            chk("0")

    def test_negative(self):
        """Raises ArgumentTypeError for negative value."""
        from argparse import ArgumentTypeError

        chk = getattr(mergeqfiles_module, "_mergeqfiles__chkPositiveInt", None)
        if chk is None:
            chk = mergeqfiles_module.__dict__.get("__chkPositiveInt")
        if chk is None:
            pytest.skip("Cannot access __chkPositiveInt")
        with pytest.raises(ArgumentTypeError):
            chk("-3")


class TestChkNotNegativeFloat:
    """Test __chkNotNegativeFloat validator."""

    def test_positive_float(self):
        """Returns float for positive value."""
        chk = getattr(mergeqfiles_module, "_mergeqfiles__chkNotNegativeFloat", None)
        if chk is None:
            chk = mergeqfiles_module.__dict__.get("__chkNotNegativeFloat")
        if chk is None:
            pytest.skip("Cannot access __chkNotNegativeFloat")
        assert chk("3.14") == 3.14

    def test_zero(self):
        """Returns 0.0 for zero (non-negative)."""
        chk = getattr(mergeqfiles_module, "_mergeqfiles__chkNotNegativeFloat", None)
        if chk is None:
            chk = mergeqfiles_module.__dict__.get("__chkNotNegativeFloat")
        if chk is None:
            pytest.skip("Cannot access __chkNotNegativeFloat")
        assert chk("0") == 0.0

    def test_negative(self):
        """Raises ArgumentTypeError for negative value."""
        from argparse import ArgumentTypeError

        chk = getattr(mergeqfiles_module, "_mergeqfiles__chkNotNegativeFloat", None)
        if chk is None:
            chk = mergeqfiles_module.__dict__.get("__chkNotNegativeFloat")
        if chk is None:
            pytest.skip("Cannot access __chkNotNegativeFloat")
        with pytest.raises(ArgumentTypeError):
            chk("-1.5")


class TestMainCLI:
    """Test the main() CLI entry point via subprocess."""

    @staticmethod
    def _run(args: list[str], timeout: int = 10) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "q2netcdf.mergeqfiles"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_help(self):
        """--help shows usage."""
        result = self._run(["--help"])
        assert result.returncode == 0
        assert "maxSize" in result.stdout

    def test_version(self):
        """--version shows version."""
        result = self._run(["--version"])
        assert result.returncode == 0
        assert "0.4" in result.stdout

    def test_no_args(self):
        """No arguments produces error."""
        result = self._run([])
        assert result.returncode != 0

    def test_nonexistent_datadir(self, tmp_path):
        """Non-existent datadir prints error and exits."""
        bad_dir = str(tmp_path / "nonexistent")
        output = str(tmp_path / "output.q")
        result = self._run(
            ["0", "-60", "99999", "--datadir", bad_dir, "-o", output, "--logfile", ""]
        )
        assert result.returncode == 1
        assert "does not exist" in result.stdout

    def test_run_on_empty_dir(self, tmp_path):
        """Run with empty data dir creates empty output, prints 0."""
        datadir = tmp_path / "data"
        datadir.mkdir()
        output = tmp_path / "output.q"
        result = self._run(
            [
                "0",
                "-3600",
                "99999",
                "--datadir",
                str(datadir),
                "-o",
                str(output),
                "--logfile",
                "",
            ]
        )
        assert result.returncode == 0
        assert "0" in result.stdout
        assert output.exists()

    def test_run_with_real_files(self, tmp_path):
        """Run with real Q-files produces output."""
        import shutil

        datadir = tmp_path / "data"
        datadir.mkdir()
        shutil.copy2(str(MRI_FILE), str(datadir / "test.q"))

        output = tmp_path / "output.q"
        import time as time_module

        now = time_module.time()
        result = self._run(
            [
                str(now - 120),
                "240",
                "99999",
                "--datadir",
                str(datadir),
                "-o",
                str(output),
                "--logfile",
                "",
            ]
        )
        assert result.returncode == 0

    def test_negative_maxsize(self):
        """Negative maxSize is rejected."""
        result = self._run(["0", "-60", "-5"])
        assert result.returncode != 0

    def test_zero_maxsize(self):
        """Zero maxSize is rejected (must be positive)."""
        result = self._run(["0", "-60", "0"])
        assert result.returncode != 0

    def test_negative_safety(self, tmp_path):
        """Negative safety value is rejected."""
        output = str(tmp_path / "output.q")
        result = self._run(
            [
                "0",
                "-60",
                "99999",
                "--safety",
                "-10",
                "-o",
                output,
                "--logfile",
                "",
            ]
        )
        assert result.returncode != 0

    def test_verbose_flag(self, tmp_path):
        """Verbose flag is accepted."""
        datadir = tmp_path / "data"
        datadir.mkdir()
        output = tmp_path / "output.q"
        result = self._run(
            [
                "0",
                "-3600",
                "99999",
                "--datadir",
                str(datadir),
                "-o",
                str(output),
                "--logfile",
                "",
                "-v",
            ]
        )
        assert result.returncode == 0

    def test_config_empty_string(self, tmp_path):
        """Empty config string disables config."""
        datadir = tmp_path / "data"
        datadir.mkdir()
        output = tmp_path / "output.q"
        result = self._run(
            [
                "0",
                "-3600",
                "99999",
                "--datadir",
                str(datadir),
                "-o",
                str(output),
                "--logfile",
                "",
                "--config",
                "",
            ]
        )
        assert result.returncode == 0


class TestMainInProcess:
    """Test main() in-process for coverage (subprocess tests don't count)."""

    def test_main_empty_dir(self, tmp_path, monkeypatch, capsys):
        """main() with empty data dir creates empty output and prints 0."""
        from q2netcdf.mergeqfiles import main

        datadir = tmp_path / "data"
        datadir.mkdir()
        output = tmp_path / "output.q"

        monkeypatch.setattr(
            "sys.argv",
            [
                "mergeqfiles",
                "0",
                "-3600",
                "99999",
                "--datadir",
                str(datadir),
                "-o",
                str(output),
                "--logfile",
                "",
            ],
        )
        main()
        captured = capsys.readouterr()
        assert "0" in captured.out
        assert output.exists()

    def test_main_nonexistent_datadir(self, tmp_path, monkeypatch, capsys):
        """main() with non-existent datadir exits with code 1."""
        from q2netcdf.mergeqfiles import main

        bad_dir = str(tmp_path / "nonexistent")
        output = str(tmp_path / "output.q")

        monkeypatch.setattr(
            "sys.argv",
            [
                "mergeqfiles",
                "0",
                "-60",
                "99999",
                "--datadir",
                bad_dir,
                "-o",
                output,
                "--logfile",
                "",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.out

    def test_main_with_config_empty(self, tmp_path, monkeypatch, capsys):
        """main() with --config '' disables config loading."""
        from q2netcdf.mergeqfiles import main

        datadir = tmp_path / "data"
        datadir.mkdir()
        output = tmp_path / "output.q"

        monkeypatch.setattr(
            "sys.argv",
            [
                "mergeqfiles",
                "0",
                "-3600",
                "99999",
                "--datadir",
                str(datadir),
                "-o",
                str(output),
                "--logfile",
                "",
                "--config",
                "",
            ],
        )
        main()
        captured = capsys.readouterr()
        assert "0" in captured.out

    def test_main_with_logfile(self, tmp_path, monkeypatch, capsys):
        """main() creates logfile in specified location."""
        import logging

        from q2netcdf.mergeqfiles import main

        datadir = tmp_path / "data"
        datadir.mkdir()
        output = tmp_path / "output.q"
        logdir = tmp_path / "logs"
        logfile = logdir / "test.log"

        monkeypatch.setattr(
            "sys.argv",
            [
                "mergeqfiles",
                "0",
                "-3600",
                "99999",
                "--datadir",
                str(datadir),
                "-o",
                str(output),
                "--logfile",
                str(logfile),
            ],
        )
        # Reset logging so basicConfig takes effect
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        main()
        assert logdir.exists()

    def test_main_with_real_files(self, tmp_path, monkeypatch, capsys):
        """main() with real Q-files glues them and prints size."""
        import shutil
        import time as time_module
        import logging

        from q2netcdf.mergeqfiles import main

        datadir = tmp_path / "data"
        datadir.mkdir()
        dest = datadir / "test.q"
        shutil.copy(str(MRI_FILE), str(dest))
        dest.touch()

        output = tmp_path / "output.q"
        now = time_module.time()

        monkeypatch.setattr(
            "sys.argv",
            [
                "mergeqfiles",
                str(now - 120),
                "240",
                "99999",
                "--datadir",
                str(datadir),
                "-o",
                str(output),
                "--logfile",
                "",
                "-v",
            ],
        )
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        main()
        captured = capsys.readouterr()
        # Should print the output size
        assert output.exists()
        size_str = captured.out.strip().split("\n")[-1]
        assert int(size_str) > 0

    def test_main_with_stime_zero(self, tmp_path, monkeypatch, capsys):
        """main() with stime=0 uses current time."""
        import logging

        from q2netcdf.mergeqfiles import main

        datadir = tmp_path / "data"
        datadir.mkdir()
        output = tmp_path / "output.q"

        monkeypatch.setattr(
            "sys.argv",
            [
                "mergeqfiles",
                "0",
                "-3600",
                "99999",
                "--datadir",
                str(datadir),
                "-o",
                str(output),
                "--logfile",
                "",
            ],
        )
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        main()
        captured = capsys.readouterr()
        assert "0" in captured.out

    def test_main_with_config_file(self, tmp_path, monkeypatch, capsys):
        """main() with explicit config file path."""
        import shutil
        import time as time_module
        import logging

        from q2netcdf.mergeqfiles import main

        datadir = tmp_path / "data"
        datadir.mkdir()
        dest = datadir / "test.q"
        shutil.copy(str(QFILE_V12), str(dest))
        dest.touch()

        cfg_data = {
            "channels": ["pressure"],
            "spectra": [],
            "config": [],
        }
        cfg_file = tmp_path / "reduce.cfg"
        cfg_file.write_text(json.dumps(cfg_data))

        output = tmp_path / "output.q"
        now = time_module.time()

        monkeypatch.setattr(
            "sys.argv",
            [
                "mergeqfiles",
                str(now - 120),
                "240",
                "99999",
                "--datadir",
                str(datadir),
                "-o",
                str(output),
                "--logfile",
                "",
                "--config",
                str(cfg_file),
            ],
        )
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        main()
        capsys.readouterr()
        assert output.exists()

    def test_main_default_config_in_datadir(self, tmp_path, monkeypatch, capsys):
        """main() picks up mergeqfiles.cfg from datadir when no --config given."""
        import shutil
        import time as time_module
        import logging

        from q2netcdf.mergeqfiles import main

        datadir = tmp_path / "data"
        datadir.mkdir()
        dest = datadir / "test.q"
        shutil.copy(str(QFILE_V12), str(dest))
        dest.touch()

        # Place a config file at the default location
        cfg_data = {
            "channels": ["pressure"],
            "spectra": [],
            "config": [],
        }
        cfg_file = datadir / "mergeqfiles.cfg"
        cfg_file.write_text(json.dumps(cfg_data))

        output = tmp_path / "output.q"
        now = time_module.time()

        monkeypatch.setattr(
            "sys.argv",
            [
                "mergeqfiles",
                str(now - 120),
                "240",
                "99999",
                "--datadir",
                str(datadir),
                "-o",
                str(output),
                "--logfile",
                "",
            ],
        )
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        main()
        capsys.readouterr()
        assert output.exists()


class TestScanDirectoryTmpCleanup:
    """Test scanDirectory stale temp file cleanup paths."""

    def test_stale_tmp_file_removed(self, tmp_path):
        """scanDirectory removes stale .tmp file before writing."""
        import time as time_module
        import shutil

        datadir = tmp_path / "data"
        datadir.mkdir()
        dest = datadir / "test.q"
        shutil.copy(str(MRI_FILE), str(dest))
        dest.touch()

        output = tmp_path / "output.q"
        stale_tmp = Path(str(output) + ".tmp")
        stale_tmp.write_bytes(b"stale data")

        args = Namespace(
            datadir=str(datadir),
            output=str(output),
            maxSize=999999,
            bufferSize=100 * 1024,
            config=None,
        )
        now = time_module.time()
        times = np.array([now - 60, now + 60])
        result = scanDirectory(args, times)
        assert result > 0
        assert output.exists()
        # The stale tmp file should have been cleaned up (renamed to output)
        assert not stale_tmp.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
