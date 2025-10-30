"""Tests for QFile reader."""
import pytest
from pathlib import Path
from q2netcdf.QFile import QFile


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
            with pytest.raises(EOFError, match="header must be read before"):
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

            assert len(record.channels) == hdr.Nc, \
                f"Expected {hdr.Nc} channels, got {len(record.channels)}"

            if hdr.Ns > 0 and hdr.Nf > 0:
                expected_shape = (hdr.Ns, hdr.Nf)
                assert record.spectra.shape == expected_shape, \
                    f"Expected spectra shape {expected_shape}, got {record.spectra.shape}"
