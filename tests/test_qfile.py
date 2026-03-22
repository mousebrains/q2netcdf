"""Tests for QFile reader."""

import math
import struct
import numpy as np
import pytest
from pathlib import Path
from q2netcdf.QFile import QFile
from q2netcdf.QRecordType import RecordType
from q2netcdf.QVersion import QVersion


class TestQFile:
    """Test QFile functionality."""

    def test_nonexistent_file_raises(self):
        """Test that opening non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            QFile("/nonexistent/path/to/file.q")

    def test_context_manager(self, tmp_path):
        """Test QFile works as context manager."""
        # Create a dummy file
        test_file = tmp_path / "test.q"
        test_file.touch()

        # This should not raise even though file is empty/invalid
        # The actual error would come when trying to read header
        try:
            with QFile(str(test_file)) as qf:
                assert qf is not None
        except (EOFError, ValueError):
            # Expected if trying to read from empty file
            pass

    def test_data_before_header_raises(self, tmp_path):
        """Test that calling data() before header() raises EOFError."""
        test_file = tmp_path / "test.q"
        test_file.write_bytes(b"dummy content")

        with QFile(str(test_file)) as qf:
            with pytest.raises(RuntimeError, match="header must be read before"):
                # Generator doesn't raise until iteration starts
                next(qf.data())

    def test_read_real_qfile_header(self):
        """Test reading header from real Q-file sample."""
        sample_file = Path(__file__).parent / "sample.q"
        if not sample_file.exists():
            pytest.skip("sample.q not found")

        with QFile(str(sample_file)) as qf:
            hdr = qf.header()
            assert hdr is not None
            assert hdr.Nc > 0  # Should have channels
            assert hdr.time is not None  # Should have timestamp
            assert hdr.version is not None  # Should have version

    def test_read_real_qfile_data_records(self):
        """Test reading data records from real Q-file sample."""
        sample_file = Path(__file__).parent / "sample.q"
        if not sample_file.exists():
            pytest.skip("sample.q not found")

        with QFile(str(sample_file)) as qf:
            _ = qf.header()  # Must read header before data
            record_count = 0
            max_records = 5  # Read first 5 records

            for record in qf.data():
                assert record is not None
                assert record.t0 is not None  # Should have timestamp
                assert record.channels is not None  # Should have channel data
                record_count += 1
                if record_count >= max_records:
                    break

            assert record_count > 0, "Should have read at least one record"

    def test_qfile_channels_match_header(self):
        """Test that data record channels match header specification."""
        sample_file = Path(__file__).parent / "sample.q"
        if not sample_file.exists():
            pytest.skip("sample.q not found")

        with QFile(str(sample_file)) as qf:
            hdr = qf.header()
            record = next(qf.data())

            assert len(record.channels) == hdr.Nc, (
                f"Expected {hdr.Nc} channels, got {len(record.channels)}"
            )

            if hdr.Ns > 0 and hdr.Nf > 0:
                expected_shape = (hdr.Ns, hdr.Nf)
                assert record.spectra.shape == expected_shape, (
                    f"Expected spectra shape {expected_shape}, got {record.spectra.shape}"
                )

    def test_multiheader_reads_all_segments(self, synthetic_multiheader_qfile):
        """Test that data() reads across multiple header/data segments."""
        with QFile(str(synthetic_multiheader_qfile)) as qf:
            qf.header()
            records = list(qf.data())

            assert len(records) == 15  # 5 from segment 1 + 10 from segment 2

            # First 5 records: surface, epsilon is NaN
            for rec in records[:5]:
                assert math.isnan(rec.channels[0])  # e_2
                assert math.isnan(rec.channels[2])  # e_1

            # Last 10 records: profiling, epsilon is valid
            for rec in records[5:]:
                assert not math.isnan(rec.channels[0])  # e_2
                assert not math.isnan(rec.channels[2])  # e_1
                assert rec.channels[1] > 0  # pressure > 0


class TestQFileEdgeCases:
    """Tests for QFile uncovered lines."""

    def _build_v13_qfile_bytes(
        self,
        channel_idents: list[int],
        data_rows: list[list[float]],
        config_str: str = "{}",
    ) -> bytes:
        """Build complete v1.3 Q-file binary content."""
        Nc = len(channel_idents)
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v13.value)
        dt_ms = int(
            np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int)
        )
        header += struct.pack("<Q", dt_ms)
        header += struct.pack("<HHH", Nc, 0, 0)
        for ident in channel_idents:
            header += struct.pack("<H", ident)
        header += struct.pack("<H", len(config_str))
        header += config_str.encode("utf-8")
        data_size = 2 + 2 + (Nc * 2)
        header += struct.pack("<H", data_size)

        data = bytearray()
        for row in data_rows:
            data += struct.pack("<H", RecordType.DATA.value)
            data += struct.pack("<e", row[0])  # stime
            for val in row[1:]:
                data += struct.pack("<e", val)

        return bytes(header + data)

    def test_data_fp_none_raises(self, tmp_path):
        """Line 104: data() raises RuntimeError when fp is None."""
        # Create a valid Q-file, read header, then close and null the fp
        content = self._build_v13_qfile_bytes(
            [0x160], [[0.0, 100.0]]
        )
        f = tmp_path / "test_fp_none.q"
        f.write_bytes(content)

        with QFile(str(f)) as qf:
            qf.header()
            # Close and set fp to None manually
            qf._QFile__fp.close()
            qf._QFile__fp = None

            with pytest.raises(RuntimeError, match="File pointer is not open"):
                next(qf.data())

    def test_data_break_on_unknown_ident(self, tmp_path):
        """Line 112: data() breaks when QData.chkIdent returns False/None."""
        # Write valid header + one data record + unknown identifier bytes
        content = self._build_v13_qfile_bytes(
            [0x160], [[0.0, 100.0]]
        )
        # Append an unsupported identifier (not header, not data)
        extra = struct.pack("<H", 0xABCD)

        f = tmp_path / "test_unknown_ident_break.q"
        f.write_bytes(content + extra)

        with QFile(str(f)) as qf:
            qf.header()
            records = list(qf.data())

        # Should get only the one valid data record, then break
        assert len(records) == 1

    def test_validate_unknown_channel_identifiers(self, tmp_path):
        """Line 168: validate() adds unknown channel identifiers to set."""
        content = self._build_v13_qfile_bytes(
            [0x160, 0xFFFF], [[0.0, 100.0, 42.0]]
        )
        f = tmp_path / "test_validate_unknown.q"
        f.write_bytes(content)

        with QFile(str(f)) as qf:
            results = qf.validate()

        assert 0xFFFF in results["unknown_identifiers"]
        assert results["records_readable"] == 1

    def test_validate_eoferror(self, tmp_path):
        """Lines 175-176: validate() catches EOFError on truncated file."""
        # Write just a header identifier with not enough bytes
        content = struct.pack("<H", RecordType.HEADER.value) + b"\x00" * 8
        f = tmp_path / "test_validate_eof.q"
        f.write_bytes(content)

        with QFile(str(f)) as qf:
            results = qf.validate()

        assert results["valid"] is False
        assert any("EOF error" in e for e in results["errors"])

    def test_validate_generic_exception(self, tmp_path):
        """Lines 180-182: validate() catches generic Exception."""
        # Create a file with an invalid version to trigger NotImplementedError
        # which is a subclass of Exception but not EOFError/ValueError
        buf = bytearray()
        buf += struct.pack("<H", RecordType.HEADER.value)
        buf += struct.pack("<f", 9.9)  # Invalid version
        dt_ms = int(
            np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int)
        )
        buf += struct.pack("<Q", dt_ms)
        buf += struct.pack("<HHH", 1, 0, 0)
        buf += struct.pack("<H", 0x160)
        buf += struct.pack("<H", 2)
        buf += b"{}"
        buf += struct.pack("<H", 6)

        f = tmp_path / "test_validate_exception.q"
        f.write_bytes(bytes(buf))

        with QFile(str(f)) as qf:
            results = qf.validate()

        assert results["valid"] is False
        assert any("Unexpected error" in e for e in results["errors"])

    def test_data_load_returns_none_breaks(self, tmp_path):
        """Line 112: data() breaks when QData.load returns None (truncated)."""
        # Write a valid header + one valid data record + a truncated data record
        content = self._build_v13_qfile_bytes(
            [0x160], [[0.0, 100.0]]
        )
        # Add a truncated data record: data identifier + partial data
        truncated = struct.pack("<H", RecordType.DATA.value) + b"\x00"

        f = tmp_path / "test_load_none_break.q"
        f.write_bytes(content + truncated)

        with QFile(str(f)) as qf:
            qf.header()
            records = list(qf.data())

        # Should get only the one valid record, then break on truncated
        assert len(records) == 1

    def test_validate_unknown_spectra_identifiers(self, tmp_path):
        """Lines 166-168: validate() adds unknown spectra identifiers to set."""
        # Build a header with known channel and unknown spectra
        Nc, Ns, Nf = 1, 1, 2
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v13.value)
        dt_ms = int(
            np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int)
        )
        header += struct.pack("<Q", dt_ms)
        header += struct.pack("<HHH", Nc, Ns, Nf)
        header += struct.pack("<H", 0x160)  # pressure channel
        header += struct.pack("<H", 0xFFFF)  # unknown spectra
        header += struct.pack("<ee", 1.0, 2.0)  # frequencies
        config_str = "{}"
        header += struct.pack("<H", len(config_str))
        header += config_str.encode("utf-8")
        data_size = 2 + 2 + (Nc * 2) + (Ns * Nf * 2)
        header += struct.pack("<H", data_size)

        # One valid data record
        data = bytearray()
        data += struct.pack("<H", RecordType.DATA.value)
        data += struct.pack("<e", 0.0)  # stime
        data += struct.pack("<e", 100.0)  # pressure
        data += struct.pack("<ee", 0.5, 0.6)  # spectra values

        f = tmp_path / "test_validate_unknown_spectra.q"
        f.write_bytes(bytes(header) + bytes(data))

        with QFile(str(f)) as qf:
            results = qf.validate()

        assert 0xFFFF in results["unknown_identifiers"]
        assert results["records_readable"] == 1
