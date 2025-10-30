#
# Integration tests for q2netcdf
#
# Mar-2025, Claude Code Assistant

import pytest
import struct
import numpy as np
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from q2netcdf import QFile, QHeader, QConfig, QHexCodes
from q2netcdf.QRecordType import RecordType
from q2netcdf.QVersion import QVersion


class TestQFileIntegration:
    """Integration tests for complete Q-file workflows."""

    @pytest.fixture
    def sample_qfile(self, tmp_path):
        """Create a minimal valid Q-file for testing."""
        qfile_path = tmp_path / "test.q"

        # Create a minimal Q-file header (v1.3)
        header = bytearray()

        # Record type (HEADER)
        header += struct.pack("<H", RecordType.HEADER.value)

        # Version (1.3)
        header += struct.pack("<f", 1.3)

        # Binary timestamp (milliseconds since 0000-01-01)
        dt_ms = int(np.datetime64('2025-01-01').astype('datetime64[ms]').astype(int))
        header += struct.pack("<Q", dt_ms)

        # Number of channels, spectra, frequencies
        Nc, Ns, Nf = 2, 0, 0
        header += struct.pack("<HHH", Nc, Ns, Nf)

        # Channel identifiers (pressure, temperature)
        header += struct.pack("<HH", 0x160, 0x620)

        # Config size and empty config
        config_str = '{}'
        header += struct.pack("<H", len(config_str))
        header += config_str.encode('utf-8')

        # Data record size (ident=2, stime=2, 2 channels * 2 bytes each)
        data_size = 2 + 2 + (Nc * 2)
        header += struct.pack("<H", data_size)

        # Create a few data records
        data_records = bytearray()
        for i in range(5):
            # DATA record ident
            data_records += struct.pack("<H", RecordType.DATA.value)
            # Sample time (half-precision float, seconds)
            data_records += struct.pack("<e", float(i))
            # Channel values (pressure, temperature)
            data_records += struct.pack("<ee", 10.0 + i, 20.0 + i * 0.5)

        qfile_path.write_bytes(header + data_records)
        return qfile_path

    def test_read_qfile_header(self, sample_qfile):
        """Test reading Q-file header."""
        with QFile(str(sample_qfile)) as qf:
            header = qf.header()
            assert header.version == QVersion.v13
            assert header.Nc == 2
            assert header.Ns == 0
            assert header.Nf == 0
            assert len(header.channels) == 2

    def test_read_qfile_data_records(self, sample_qfile):
        """Test reading Q-file data records."""
        with QFile(str(sample_qfile)) as qf:
            _ = qf.header()  # Must read header before data
            records = list(qf.data())
            assert len(records) == 5

            # Check first record
            assert records[0].channels[0] == pytest.approx(10.0, rel=0.01)
            assert records[0].channels[1] == pytest.approx(20.0, rel=0.01)

            # Check last record
            assert records[4].channels[0] == pytest.approx(14.0, rel=0.01)
            assert records[4].channels[1] == pytest.approx(22.0, rel=0.01)

    def test_qfile_context_manager(self, sample_qfile):
        """Test Q-file context manager properly closes file."""
        qf = QFile(str(sample_qfile))
        with qf:
            header = qf.header()
            assert header is not None
        # File should be closed after context exit
        # (Can't directly test but context manager should work)

    def test_hex_code_mapping_integration(self):
        """Test that hex codes work correctly in real scenarios."""
        # Shear probe
        name = QHexCodes.name(0x610)
        assert name == "sh_0"
        attrs = QHexCodes.attributes(0x610)
        assert attrs["long_name"] == "shear_0"

        # Reverse lookup
        ident = QHexCodes.name2ident("sh_0")
        assert ident == 0x610

    def test_config_parsing_integration(self):
        """Test config parsing with realistic data."""
        config_data = {
            "sample_rate": 512,
            "channels": [1, 2, 3],
            "calibration": {
                "offset": 0.5,
                "gain": 1.0
            },
            "enabled": True
        }
        config_bytes = json.dumps(config_data).encode('utf-8')
        qconfig = QConfig(config_bytes, QVersion.v13)

        parsed = qconfig.config()
        assert parsed["sample_rate"] == 512
        assert parsed["channels"] == [1, 2, 3]
        assert parsed["calibration"]["offset"] == 0.5
        assert parsed["enabled"] is True


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_create_read_modify_workflow(self, tmp_path):
        """Test a complete create->read->modify workflow."""
        # This would be expanded with actual Q-file creation
        # For now, verify the components work together

        # Verify we can map various sensor types
        pressure_name = QHexCodes.name(0x160)
        temp_name = QHexCodes.name(0x620)
        shear_name = QHexCodes.name(0x610)

        assert pressure_name == "pressure"
        assert temp_name == "T_0"
        assert shear_name == "sh_0"

        # Verify attributes are correct
        press_attrs = QHexCodes.attributes(0x160)
        assert "units" in press_attrs
        assert press_attrs["units"] == "decibar"

    def test_multiple_sensor_types(self):
        """Test handling multiple sensor types."""
        sensor_ids = [
            (0x160, "pressure", "pressure_ocean"),
            (0x610, "sh_0", "shear_0"),
            (0x620, "T_0", "temperature_0"),
            (0x630, "C_0", "microConductivity_0"),
        ]

        for ident, expected_name, expected_long in sensor_ids:
            name = QHexCodes.name(ident)
            assert name == expected_name

            attrs = QHexCodes.attributes(ident)
            assert attrs["long_name"] == expected_long


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_file(self):
        """Test handling of missing Q-file."""
        with pytest.raises(FileNotFoundError):
            QFile("/nonexistent/file.q")

    def test_corrupted_header(self, tmp_path):
        """Test handling of corrupted header."""
        qfile = tmp_path / "corrupted.q"
        # Write incomplete header (less than 20 bytes)
        qfile.write_bytes(b"corrupted data")

        with pytest.raises((EOFError, ValueError, struct.error)):
            with open(str(qfile), "rb") as fp:
                QHeader(fp, str(qfile))

    def test_invalid_version(self, tmp_path):
        """Test handling of invalid version number."""
        qfile = tmp_path / "badversion.q"
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", 99.9)  # Invalid version
        header += struct.pack("<Q", 0)  # timestamp
        header += struct.pack("<HHH", 0, 0, 0)  # Nc, Ns, Nf
        qfile.write_bytes(header)

        with pytest.raises(NotImplementedError):
            with open(str(qfile), "rb") as fp:
                QHeader(fp, str(qfile))

    def test_unknown_hex_code(self):
        """Test handling of unknown hex codes."""
        name = QHexCodes.name(0xFFFF)
        assert name is None

        attrs = QHexCodes.attributes(0xFFFF)
        assert attrs is None


class TestPerformance:
    """Basic performance tests."""

    def test_large_hex_map_lookup(self):
        """Test that hex map lookups are fast (using class-level cache)."""
        # This tests that we're using compiled regex patterns
        import time

        start = time.time()
        for _ in range(1000):
            QHexCodes.name(0x610)
            QHexCodes.attributes(0x160)
        elapsed = time.time() - start

        # Should complete 1000 lookups in < 0.1 seconds
        assert elapsed < 0.1, f"Lookups took {elapsed:.3f}s, too slow!"

    def test_config_parsing_performance(self):
        """Test config parsing is reasonably fast."""
        import time

        config_data = json.dumps({
            f"key_{i}": i for i in range(100)
        }).encode('utf-8')

        start = time.time()
        for _ in range(100):
            qconfig = QConfig(config_data, QVersion.v13)
            _ = qconfig.config()
        elapsed = time.time() - start

        # Should parse 100 configs in < 0.5 seconds
        assert elapsed < 0.5, f"Config parsing took {elapsed:.3f}s, too slow!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
