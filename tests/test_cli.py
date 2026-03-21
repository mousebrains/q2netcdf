"""Tests for CLI entry points of all 6 commands."""

import subprocess
import sys



def run_cli(module: str, args: list[str], timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a CLI module and return the result."""
    return subprocess.run(
        [sys.executable, "-m", f"q2netcdf.{module}"] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestQHeaderCLI:
    def test_help(self):
        result = run_cli("QHeader", ["--help"])
        assert result.returncode == 0
        assert "filename" in result.stdout

    def test_no_args(self):
        result = run_cli("QHeader", [])
        assert result.returncode != 0

    def test_read_real_file(self, mri_file):
        result = run_cli("QHeader", [str(mri_file)])
        assert result.returncode == 0
        assert "File version" in result.stdout

    def test_nothing_flag(self, mri_file):
        result = run_cli("QHeader", [str(mri_file), "--nothing"])
        assert result.returncode == 0
        assert "Channels: n=" in result.stdout


class TestQFileCLI:
    def test_help(self):
        result = run_cli("QFile", ["--help"])
        assert result.returncode == 0
        assert "filename" in result.stdout

    def test_validate_real_file(self, mri_file):
        result = run_cli("QFile", [str(mri_file), "--validate"])
        assert result.returncode == 0
        assert "valid" in result.stdout.lower() or "Valid" in result.stdout

    def test_nonexistent_file(self):
        result = run_cli("QFile", ["/nonexistent/file.mri"])
        # Should handle gracefully (logs exception)
        assert result.returncode == 0 or result.returncode == 1


class TestQReduceCLI:
    def test_help(self):
        result = run_cli("QReduce", ["--help"])
        assert result.returncode == 0
        assert "filename" in result.stdout

    def test_no_args(self):
        result = run_cli("QReduce", [])
        assert result.returncode != 0


class TestQ2NetCDFCLI:
    def test_help(self):
        result = run_cli("q2netcdf", ["--help"])
        assert result.returncode == 0
        assert "qfile" in result.stdout
        assert "--nc" in result.stdout

    def test_no_args(self):
        result = run_cli("q2netcdf", [])
        assert result.returncode != 0

    def test_convert_real_file(self, mri_file, tmp_path):
        nc_out = tmp_path / "output.nc"
        result = run_cli("q2netcdf", [str(mri_file), "--nc", str(nc_out)])
        assert result.returncode == 0
        assert nc_out.exists()
        assert nc_out.stat().st_size > 0


class TestMkISDPcfgCLI:
    def test_help(self):
        result = run_cli("mkISDPcfg", ["--help"])
        assert result.returncode == 0
        assert "isdpConfig" in result.stdout

    def test_generate_config(self, tmp_path):
        cfg_out = tmp_path / "isdp.cfg"
        result = run_cli("mkISDPcfg", ["--isdpConfig", str(cfg_out)])
        assert result.returncode == 0
        assert cfg_out.exists()
        content = cfg_out.read_text()
        assert "Generated" in content

    def test_generate_with_options(self, tmp_path):
        cfg_out = tmp_path / "isdp.cfg"
        result = run_cli("mkISDPcfg", [
            "--isdpConfig", str(cfg_out),
            "--instrument", "slocum_glider",
            "--fft_length", "4",
            "--hp_cut", "0.125",
            "--diss_length", "30",
        ])
        assert result.returncode == 0
        content = cfg_out.read_text()
        assert "fft_length" in content
        assert "diss_length" in content


class TestQHexCodesCLI:
    def test_help(self):
        result = run_cli("QHexCodes", ["--help"])
        assert result.returncode == 0

    def test_lookup_ident(self):
        result = run_cli("QHexCodes", ["610"])
        assert result.returncode == 0

    def test_lookup_name(self):
        result = run_cli("QHexCodes", ["--name", "pressure"])
        assert result.returncode == 0
