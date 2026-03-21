"""In-process tests for main() entry points to improve coverage."""

import pytest
from unittest.mock import patch

from q2netcdf.QHeader import main as qheader_main
from q2netcdf.QFile import main as qfile_main
from q2netcdf.QHexCodes import main as qhexcodes_main
from q2netcdf.QReduce import main as qreduce_main
from q2netcdf.q2netcdf import main as q2netcdf_main
from q2netcdf.mkISDPcfg import main as mkisdpcfg_main


class TestQHeaderMain:
    def test_main_with_real_file(self, mri_file, capsys):
        with patch("sys.argv", ["QHeader", str(mri_file)]):
            qheader_main()
        out = capsys.readouterr().out
        assert "File version" in out
        assert "Scalar" in out

    def test_main_nothing_flag(self, mri_file, capsys):
        with patch("sys.argv", ["QHeader", str(mri_file), "--nothing"]):
            qheader_main()
        out = capsys.readouterr().out
        assert "Channels: n=" in out

    def test_main_suppress_channels(self, mri_file, capsys):
        with patch("sys.argv", ["QHeader", str(mri_file), "--channels"]):
            qheader_main()
        out = capsys.readouterr().out
        assert "Channels: n=" in out

    def test_main_suppress_config(self, mri_file, capsys):
        with patch("sys.argv", ["QHeader", str(mri_file), "--config"]):
            qheader_main()
        out = capsys.readouterr().out
        assert "File version" in out


class TestQFileMain:
    def test_main_normal_mode(self, mri_file, capsys):
        with patch(
            "sys.argv", ["QFile", str(mri_file), "--n", "2", "--logLevel", "DEBUG"]
        ):
            qfile_main()

    def test_main_validate_mode(self, mri_file, capsys):
        with patch("sys.argv", ["QFile", str(mri_file), "--validate"]):
            qfile_main()
        out = capsys.readouterr().out
        assert "valid" in out.lower() or "Valid" in out

    def test_main_nonexistent_file(self, capsys):
        with patch("sys.argv", ["QFile", "/nonexistent/file.mri"]):
            qfile_main()  # Should handle gracefully (logs exception)

    def test_main_empty_file(self, empty_file, capsys):
        with patch("sys.argv", ["QFile", str(empty_file)]):
            qfile_main()  # Should handle EOFError gracefully

    def test_main_validate_corrupt(self, corrupt_file, capsys):
        with patch("sys.argv", ["QFile", str(corrupt_file), "--validate"]):
            qfile_main()
        out = capsys.readouterr().out
        assert "issues" in out.lower() or "Errors" in out

    def test_main_validate_unknown_idents(self, tmp_path, capsys):
        """Test validate display with unknown identifiers."""
        import struct
        import numpy as np
        from q2netcdf.QRecordType import RecordType
        from q2netcdf.QVersion import QVersion

        qfile = tmp_path / "unknown.q"
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v13.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        header += struct.pack("<Q", dt_ms)
        header += struct.pack("<HHH", 1, 0, 0)
        header += struct.pack("<H", 0xFFF0)  # unknown
        header += struct.pack("<H", 2) + b"{}"
        header += struct.pack("<H", 6)
        data = struct.pack("<Hee", RecordType.DATA.value, 1.0, 42.0)
        qfile.write_bytes(header + data)

        with patch("sys.argv", ["QFile", str(qfile), "--validate"]):
            qfile_main()
        out = capsys.readouterr().out
        assert "Unknown identifiers" in out


class TestQHexCodesMain:
    def test_main_no_args(self, capsys):
        with patch("sys.argv", ["QHexCodes", "--logLevel", "DEBUG"]):
            qhexcodes_main()

    def test_main_lookup_ident(self, capsys):
        with patch("sys.argv", ["QHexCodes", "610", "--logLevel", "DEBUG"]):
            qhexcodes_main()

    def test_main_lookup_name(self, capsys):
        with patch(
            "sys.argv", ["QHexCodes", "--name", "pressure", "--logLevel", "DEBUG"]
        ):
            qhexcodes_main()


class TestQReduceMain:
    def test_main_with_valid_config(self, mri_file, tmp_path, capsys):
        import json

        cfg = tmp_path / "reduce.cfg"
        cfg.write_text(
            json.dumps(
                {
                    "channels": ["e_2", "pressure"],
                    "spectra": [],
                    "config": ["diss_length"],
                }
            )
        )
        output = tmp_path / "reduced.q"
        with patch(
            "sys.argv",
            ["QReduce", str(mri_file), "--config", str(cfg), "--output", str(output)],
        ):
            qreduce_main()
        assert output.exists()

    def test_main_no_output(self, mri_file, tmp_path, capsys):
        import json

        cfg = tmp_path / "reduce.cfg"
        cfg.write_text(
            json.dumps(
                {
                    "channels": ["e_2", "pressure"],
                    "spectra": [],
                    "config": ["diss_length"],
                }
            )
        )
        with patch("sys.argv", ["QReduce", str(mri_file), "--config", str(cfg)]):
            qreduce_main()

    def test_main_bad_config(self, mri_file, tmp_path, capsys):
        cfg = tmp_path / "bad.cfg"
        cfg.write_text("not json")
        with patch("sys.argv", ["QReduce", str(mri_file), "--config", str(cfg)]):
            qreduce_main()


class TestQ2NetCDFMain:
    def test_main_convert(self, mri_file, tmp_path):
        nc = tmp_path / "out.nc"
        with patch("sys.argv", ["q2netcdf", str(mri_file), "--nc", str(nc)]):
            q2netcdf_main()
        assert nc.exists()

    def test_main_no_compression(self, mri_file, tmp_path):
        nc = tmp_path / "out.nc"
        with patch(
            "sys.argv",
            ["q2netcdf", str(mri_file), "--nc", str(nc), "--compressionLevel", "0"],
        ):
            q2netcdf_main()
        assert nc.exists()

    def test_main_empty_file(self, empty_file, tmp_path):
        nc = tmp_path / "out.nc"
        with patch("sys.argv", ["q2netcdf", str(empty_file), "--nc", str(nc)]):
            with pytest.raises(SystemExit) as exc_info:
                q2netcdf_main()
            assert exc_info.value.code == 0  # exits with 0 for "No data found"


class TestMkISDPcfgMain:
    def test_main_default(self, tmp_path, capsys):
        cfg = tmp_path / "isdp.cfg"
        with patch("sys.argv", ["mkISDPcfg", "--isdpConfig", str(cfg)]):
            mkisdpcfg_main()
        assert cfg.exists()
        content = cfg.read_text()
        assert "Generated" in content

    def test_main_with_options(self, tmp_path, capsys):
        cfg = tmp_path / "isdp.cfg"
        with patch(
            "sys.argv",
            [
                "mkISDPcfg",
                "--isdpConfig",
                str(cfg),
                "--instrument",
                "slocum_glider",
                "--fft_length",
                "4",
                "--hp_cut",
                "0.125",
                "--diss_length",
                "30",
                "--overlap",
                "0",
            ],
        ):
            mkisdpcfg_main()
        content = cfg.read_text()
        assert "fft_length = 4.0" in content
        assert "diss_length = 30.0" in content
        assert '"slocum_glider"' in content

    def test_main_fft_hp_mismatch_warning(self, tmp_path, capsys):
        cfg = tmp_path / "isdp.cfg"
        with patch(
            "sys.argv",
            [
                "mkISDPcfg",
                "--isdpConfig",
                str(cfg),
                "--fft_length",
                "4",
                "--hp_cut",
                "0.5",
            ],
        ):
            mkisdpcfg_main()
        out = capsys.readouterr().out
        assert "WARNING" in out

    def test_main_fft_without_hp_warning(self, tmp_path, capsys):
        cfg = tmp_path / "isdp.cfg"
        with patch(
            "sys.argv", ["mkISDPcfg", "--isdpConfig", str(cfg), "--fft_length", "4"]
        ):
            mkisdpcfg_main()
        out = capsys.readouterr().out
        assert "WARNING" in out

    def test_main_hp_without_fft_warning(self, tmp_path, capsys):
        cfg = tmp_path / "isdp.cfg"
        with patch(
            "sys.argv", ["mkISDPcfg", "--isdpConfig", str(cfg), "--hp_cut", "0.125"]
        ):
            mkisdpcfg_main()
        out = capsys.readouterr().out
        assert "WARNING" in out

    def test_main_despiking_params(self, tmp_path, capsys):
        cfg = tmp_path / "isdp.cfg"
        with patch(
            "sys.argv",
            ["mkISDPcfg", "--isdpConfig", str(cfg), "--shear_despiking", "3.0,0.5,10"],
        ):
            mkisdpcfg_main()
        content = cfg.read_text()
        assert "shear_despiking" in content

    def test_main_boolean_options(self, tmp_path, capsys):
        cfg = tmp_path / "isdp.cfg"
        with patch(
            "sys.argv",
            [
                "mkISDPcfg",
                "--isdpConfig",
                str(cfg),
                "--goodman_spectra",
                "true",
                "--band_averaging",
                "false",
            ],
        ):
            mkisdpcfg_main()
        content = cfg.read_text()
        assert "goodman_spectra = true" in content
        assert "band_averaging = false" in content

    def test_main_bad_directory(self, tmp_path):
        from argparse import ArgumentTypeError

        cfg = "/nonexistent/dir/isdp.cfg"
        with patch("sys.argv", ["mkISDPcfg", "--isdpConfig", cfg]):
            with pytest.raises(ArgumentTypeError):
                mkisdpcfg_main()
