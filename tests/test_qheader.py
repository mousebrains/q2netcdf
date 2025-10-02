#! /usr/bin/env python3
"""Tests for QHeader parser."""

import pytest
from pathlib import Path
from q2netcdf.QHeader import QHeader
from q2netcdf.QVersion import QVersion


class TestQHeader:
    """Test suite for QHeader parsing."""

    def test_read_header_from_sample(self):
        """Test reading header from sample Q-file."""
        sample_file = Path(__file__).parent / "sample.q"
        if not sample_file.exists():
            pytest.skip("sample.q not found")

        with open(sample_file, "rb") as fp:
            hdr = QHeader(fp, str(sample_file))

            assert hdr.filename == str(sample_file)
            assert hdr.version in [QVersion.v12, QVersion.v13]
            assert hdr.time is not None
            assert hdr.Nc > 0, "Should have channels"
            assert hdr.channels is not None
            assert len(hdr.channels) == hdr.Nc

    def test_header_has_configuration(self):
        """Test that header contains configuration data."""
        sample_file = Path(__file__).parent / "sample.q"
        if not sample_file.exists():
            pytest.skip("sample.q not found")

        with open(sample_file, "rb") as fp:
            hdr = QHeader(fp, str(sample_file))

            assert hdr.config is not None
            config_dict = hdr.config.config()
            assert isinstance(config_dict, dict)
            assert len(config_dict) > 0

    def test_header_spectra_and_frequencies(self):
        """Test header spectra and frequency data."""
        sample_file = Path(__file__).parent / "sample.q"
        if not sample_file.exists():
            pytest.skip("sample.q not found")

        with open(sample_file, "rb") as fp:
            hdr = QHeader(fp, str(sample_file))

            if hdr.Ns > 0:
                assert hdr.spectra is not None
                assert len(hdr.spectra) == hdr.Ns

            if hdr.Nf > 0:
                assert hdr.frequencies is not None
                assert len(hdr.frequencies) == hdr.Nf

    def test_header_data_size(self):
        """Test that header reports correct data record size."""
        sample_file = Path(__file__).parent / "sample.q"
        if not sample_file.exists():
            pytest.skip("sample.q not found")

        with open(sample_file, "rb") as fp:
            hdr = QHeader(fp, str(sample_file))

            assert hdr.dataSize > 0
            assert hdr.hdrSize > 0

    def test_chkIdent_recognizes_header(self):
        """Test chkIdent correctly identifies header records."""
        sample_file = Path(__file__).parent / "sample.q"
        if not sample_file.exists():
            pytest.skip("sample.q not found")

        with open(sample_file, "rb") as fp:
            result = QHeader.chkIdent(fp)
            assert result is True, "Should recognize header record"

    def test_empty_file_raises_eof(self, tmp_path):
        """Test that empty file raises EOFError."""
        test_file = tmp_path / "empty.q"
        test_file.touch()

        with open(test_file, "rb") as fp:
            with pytest.raises(EOFError):
                QHeader(fp, str(test_file))

    def test_invalid_header_raises_valueerror(self, tmp_path):
        """Test that invalid header identifier raises ValueError."""
        test_file = tmp_path / "invalid.q"
        # Write 20 bytes with wrong identifier
        test_file.write_bytes(b"\x00\x00" + b"\x00" * 18)

        with open(test_file, "rb") as fp:
            with pytest.raises(ValueError, match="Invalid header identifer"):
                QHeader(fp, str(test_file))
