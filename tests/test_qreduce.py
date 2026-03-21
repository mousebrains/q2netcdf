"""Tests for QReduce module."""

import json

import numpy as np

from q2netcdf.QReduce import QReduce
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
        cfg.write_text(json.dumps({
            "channels": ["e_1", "pressure"],
            "spectra": [],
            "config": ["diss_length"],
        }))
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
