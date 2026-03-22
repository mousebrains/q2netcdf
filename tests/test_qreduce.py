"""Tests for QReduce module."""

import json

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from q2netcdf.QReduce import QReduce
from q2netcdf.QReduce import __chkExists as chkExists
from q2netcdf.QFile import QFile
from q2netcdf.QVersion import QVersion


class TestLoadConfig:
    """Tests for QReduce.loadConfig class method."""

    def test_nonexistent_file(self):
        assert QReduce.loadConfig("/nonexistent/file.json") is None

    def test_invalid_json(self, tmp_path):
        cfg = tmp_path / "bad.json"
        cfg.write_text("not json")
        assert QReduce.loadConfig(str(cfg)) is None

    def test_not_a_dict(self, tmp_path):
        cfg = tmp_path / "list.json"
        cfg.write_text("[1, 2, 3]")
        assert QReduce.loadConfig(str(cfg)) is None

    def test_missing_required_keys(self, tmp_path):
        cfg = tmp_path / "partial.json"
        cfg.write_text('{"channels": []}')  # missing spectra and config
        assert QReduce.loadConfig(str(cfg)) is None

    def test_valid_config(self, tmp_path):
        cfg = tmp_path / "valid.json"
        cfg.write_text(
            json.dumps(
                {
                    "channels": ["e_1", "pressure"],
                    "spectra": [],
                    "config": ["diss_length"],
                }
            )
        )
        result = QReduce.loadConfig(str(cfg))
        assert result is not None
        assert result["channels"] == ["e_1", "pressure"]


class TestQReduceWithRealFile:
    """Tests for QReduce with real MRI files."""

    def test_reduce_creates_smaller_output(self, mri_file, tmp_path):
        config = {
            "channels": ["e_2", "pressure"],
            "spectra": [],
            "config": ["diss_length"],
        }
        qr = QReduce(str(mri_file), config)

        # Reduced file should be smaller
        assert qr.fileSize < qr.fileSizeOrig
        assert qr.hdrSize <= qr.hdrSizeOrig
        assert qr.dataSize <= qr.dataSizeOrig

    def test_reduce_writes_valid_output(self, mri_file, tmp_path):
        config = {
            "channels": ["e_2", "pressure"],
            "spectra": [],
            "config": ["diss_length"],
        }
        qr = QReduce(str(mri_file), config)

        output = tmp_path / "reduced.q"
        with open(output, "wb") as fp:
            bytes_written = qr.reduceFile(fp)

        assert bytes_written > 0
        assert output.stat().st_size == bytes_written

        # The reduced file should be readable
        with QFile(str(output)) as qf:
            hdr = qf.header()
            assert hdr.version == QVersion.v13
            assert hdr.Nc == 2  # e_2 and pressure
            records = list(qf.data())
            assert len(records) > 0

    def test_reduce_preserves_selected_config(self, mri_file, tmp_path):
        config = {
            "channels": ["e_2", "pressure"],
            "spectra": [],
            "config": ["diss_length", "fft_length"],
        }
        qr = QReduce(str(mri_file), config)

        output = tmp_path / "reduced.q"
        with open(output, "wb") as fp:
            qr.reduceFile(fp)

        with QFile(str(output)) as qf:
            hdr = qf.header()
            cfg = hdr.config.config()
            assert "diss_length" in cfg

    def test_reduce_repr(self, mri_file):
        config = {
            "channels": ["pressure"],
            "spectra": [],
            "config": [],
        }
        qr = QReduce(str(mri_file), config)
        r = repr(qr)
        assert "fn" in r
        assert "hdr" in r
        assert "data" in r

    def test_decimate(self, mri_file, tmp_path):
        config = {
            "channels": ["e_2", "pressure"],
            "spectra": [],
            "config": ["diss_length"],
        }
        qr = QReduce(str(mri_file), config)

        # Decimate to every other record
        indices = np.arange(0, int(qr.nRecords), 2)
        output = tmp_path / "decimated.q"
        with open(output, "wb") as fp:
            bytes_written = qr.decimate(fp, indices)

        assert bytes_written > 0

        # Read back and verify
        with QFile(str(output)) as qf:
            qf.header()
            records = list(qf.data())
            assert len(records) == len(indices)

    def test_reduce_with_no_matching_channels(self, mri_file, tmp_path):
        config = {
            "channels": ["nonexistent_sensor"],
            "spectra": [],
            "config": [],
        }
        qr = QReduce(str(mri_file), config)
        # Should still work, just with 0 channels
        output = tmp_path / "empty_reduce.q"
        with open(output, "wb") as fp:
            qr.reduceFile(fp)


class TestQReduceWithV12File:
    """Tests for QReduce with v1.2 Q-files that have spectra."""

    def test_reduce_v12_with_spectra(self, qfile_v12, tmp_path):
        """Test reducing a v1.2 file that includes spectra channels."""
        config = {
            "channels": ["pressure"],
            "spectra": ["shear_gfd_1"],
            "config": ["diss_length"],
        }
        qr = QReduce(str(qfile_v12), config)

        # The reduced file should be smaller than original
        assert qr.fileSize < qr.fileSizeOrig

        output = tmp_path / "reduced_v12.q"
        with open(output, "wb") as fp:
            bytes_written = qr.reduceFile(fp)

        assert bytes_written > 0

        # Read back the reduced file; it should be v1.3 format
        with QFile(str(output)) as qf:
            hdr = qf.header()
            assert hdr.version == QVersion.v13
            assert hdr.Nc == 1  # pressure only
            assert hdr.Ns == 1  # shear_gfd_1 only
            assert hdr.Nf > 0  # frequencies preserved
            records = list(qf.data())
            assert len(records) > 0

    def test_reduce_v12_channels_only_no_spectra(self, qfile_v12, tmp_path):
        """Test reducing a v1.2 file keeping only channels, no spectra."""
        config = {
            "channels": ["pressure", "e_1"],
            "spectra": [],
            "config": [],
        }
        qr = QReduce(str(qfile_v12), config)

        output = tmp_path / "reduced_v12_no_spectra.q"
        with open(output, "wb") as fp:
            bytes_written = qr.reduceFile(fp)

        assert bytes_written > 0
        with QFile(str(output)) as qf:
            hdr = qf.header()
            assert hdr.Nc == 2
            assert hdr.Ns == 0
            assert hdr.Nf == 0

    def test_decimate_v12(self, qfile_v12, tmp_path):
        """Test decimating a v1.2 file."""
        config = {
            "channels": ["pressure"],
            "spectra": ["shear_gfd_1"],
            "config": [],
        }
        qr = QReduce(str(qfile_v12), config)

        indices = np.arange(0, int(qr.nRecords), 3)
        output = tmp_path / "decimated_v12.q"
        with open(output, "wb") as fp:
            bytes_written = qr.decimate(fp, indices)

        assert bytes_written > 0
        with QFile(str(output)) as qf:
            qf.header()
            records = list(qf.data())
            assert len(records) == len(indices)


class TestUpdateName2Ident:
    """Tests for QReduce.__updateName2Ident edge cases."""

    def test_non_dict_config_returns_none(self):
        """Test that non-dict config returns None."""
        # Access via the class method indirectly through QReduce constructor behavior
        # __updateName2Ident is called with config and key; test with non-dict
        result = QReduce._QReduce__updateName2Ident("not_a_dict", "channels")
        assert result is None

    def test_missing_key_returns_none(self):
        """Test that missing key returns None."""
        result = QReduce._QReduce__updateName2Ident({"other": []}, "channels")
        assert result is None

    def test_key_not_list_returns_none(self):
        """Test that non-list value returns None."""
        result = QReduce._QReduce__updateName2Ident(
            {"channels": "not_a_list"}, "channels"
        )
        assert result is None


class TestFindIndices:
    """Tests for QReduce.__findIndices edge cases."""

    def test_none_idents_returns_empty(self):
        """Test that None idents returns empty arrays."""
        known = np.array([0x160, 0x610], dtype="uint16")
        idents, indices = QReduce._QReduce__findIndices(None, known)
        assert len(idents) == 0
        assert len(indices) == 0


class TestChkExists:
    """Tests for the __chkExists argparse validator."""

    def test_existing_file(self, tmp_path):
        """Test that existing file path is returned."""
        f = tmp_path / "exists.q"
        f.write_text("data")
        assert chkExists(str(f)) == str(f)

    def test_nonexistent_file_raises(self):
        """Test that nonexistent file raises ArgumentTypeError."""
        from argparse import ArgumentTypeError

        with pytest.raises(ArgumentTypeError, match="does not exist"):
            chkExists("/nonexistent/path/file.q")


class TestQReduceVersionNone:
    """Test QReduce raises RuntimeError when header version is None."""

    def test_version_none_raises_runtime_error(self, mri_file):
        """Test that QReduce raises RuntimeError if QHeader.version is None.

        Covers line 48 in QReduce.py.
        """
        mock_hdr = MagicMock()
        mock_hdr.version = None

        with patch("q2netcdf.QReduce.QHeader", return_value=mock_hdr):
            with pytest.raises(RuntimeError, match="version must be set"):
                QReduce(str(mri_file), {"channels": [], "spectra": [], "config": []})


class TestReduceRecordWrongSize:
    """Test __reduceRecord returns None for wrong buffer size."""

    def test_wrong_buffer_size_returns_none(self, mri_file):
        """Test that __reduceRecord returns None when buffer size mismatches.

        Covers line 184 in QReduce.py.
        """
        config = {
            "channels": ["pressure"],
            "spectra": [],
            "config": [],
        }
        qr = QReduce(str(mri_file), config)

        # Call __reduceRecord with wrong-sized buffer
        result = qr._QReduce__reduceRecord(b"too short")
        assert result is None
