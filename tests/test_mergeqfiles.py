#
# Unit tests for mergeqfiles.py
#
# Mar-2025, Claude Code Assistant

import pytest
import os
import tempfile
import struct
import numpy as np
from pathlib import Path
import json

# Import from mergeqfiles (standalone module)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from q2netcdf.mergeqfiles import (
    RecordType,
    QVersion,
    QConfig,
    QHexCodes,
    QHeader,
    QReduce,
    glueFiles,
    fileCandidates,
    __chkExists,
)


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
        assert qc.size() == len(config_data)

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
        config_data = b'{}'
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue("3.14")
        assert result == 3.14
        assert isinstance(result, float)

    def test_parse_value_string(self):
        config_data = b'{}'
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue('"hello"')
        assert result == "hello"

    def test_parse_value_bool_true(self):
        config_data = b'{}'
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue("true")
        assert result is True

    def test_parse_value_bool_false(self):
        config_data = b'{}'
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue("false")
        assert result is False

    def test_parse_value_array_empty(self):
        config_data = b'{}'
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue("[]")
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_parse_value_array_numbers(self):
        config_data = b'{}'
        qc = QConfig(config_data, QVersion.v13)
        result = qc._QConfig__parseValue("[1, 2, 3]")
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert result[0] == 1
        assert result[2] == 3


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
            "spectra": []
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


class TestChkExists:
    """Test __chkExists function."""

    def test_exists(self, tmp_path):
        test_file = tmp_path / "test.q"
        test_file.write_text("test")
        result = __chkExists(str(test_file))
        assert result == str(test_file)

    def test_not_exists(self):
        from argparse import ArgumentTypeError
        with pytest.raises(ArgumentTypeError):
            __chkExists("/nonexistent/file.q")


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
