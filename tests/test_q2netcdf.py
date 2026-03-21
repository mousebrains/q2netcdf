"""Tests for q2netcdf conversion module."""

import pytest
import xarray as xr

from q2netcdf.q2netcdf import loadQfile, cfCompliant, addEncoding


class TestLoadQfile:
    """Tests for loadQfile function."""

    def test_load_synthetic_v13(self, synthetic_v13_qfile):
        ds = loadQfile(str(synthetic_v13_qfile))
        assert ds is not None
        assert isinstance(ds, xr.Dataset)
        assert "time" in ds.coords
        assert len(ds.time) == 10

    def test_load_real_mri_file(self, mri_file):
        ds = loadQfile(str(mri_file))
        assert ds is not None
        assert isinstance(ds, xr.Dataset)
        assert "time" in ds.coords
        assert len(ds.time) > 0

    def test_load_real_mri_has_config_vars(self, mri_file):
        ds = loadQfile(str(mri_file))
        assert ds is not None
        assert "fileVersion" in ds
        assert "ftime" in ds.coords

    def test_load_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            loadQfile(str(tmp_path / "nonexistent.q"))

    def test_load_empty_file(self, empty_file):
        result = loadQfile(str(empty_file))
        assert result is None

    def test_channels_have_correct_names(self, mri_file):
        ds = loadQfile(str(mri_file))
        assert ds is not None
        # The file has channels 0xA12 (e_2), 0x160 (pressure), 0xA11 (e_1)
        assert "pressure" in ds
        assert "e_2" in ds
        assert "e_1" in ds


class TestCfCompliant:
    """Tests for cfCompliant function."""

    def test_adds_conventions(self, mri_file):
        ds = loadQfile(str(mri_file))
        assert ds is not None
        ds = cfCompliant(ds)
        assert "Conventions" in ds.attrs
        assert "CF-1" in ds.attrs["Conventions"]

    def test_adds_history(self, mri_file):
        ds = loadQfile(str(mri_file))
        assert ds is not None
        ds = cfCompliant(ds)
        assert "history" in ds.attrs
        assert "q2netcdf" in ds.attrs["history"]

    def test_adds_time_coverage(self, mri_file):
        ds = loadQfile(str(mri_file))
        assert ds is not None
        ds = cfCompliant(ds)
        assert "time_coverage_start" in ds.attrs
        assert "time_coverage_end" in ds.attrs

    def test_adds_known_attrs(self, mri_file):
        ds = loadQfile(str(mri_file))
        assert ds is not None
        ds = cfCompliant(ds)
        if "time" in ds:
            assert "standard_name" in ds["time"].attrs

    def test_preserves_existing_history(self, mri_file):
        ds = loadQfile(str(mri_file))
        assert ds is not None
        ds.attrs["history"] = "previous entry"
        ds = cfCompliant(ds)
        assert "previous entry" in ds.attrs["history"]
        assert "q2netcdf" in ds.attrs["history"]


class TestAddEncoding:
    """Tests for addEncoding function."""

    def test_adds_compression(self, mri_file):
        ds = loadQfile(str(mri_file))
        assert ds is not None
        ds = addEncoding(ds, level=5)
        for name in ds:
            if ds[name].dtype.kind != "U":
                assert ds[name].encoding.get("zlib") is True
                assert ds[name].encoding.get("complevel") == 5

    def test_zero_level_no_compression(self, mri_file):
        ds = loadQfile(str(mri_file))
        assert ds is not None
        ds = addEncoding(ds, level=0)
        for name in ds:
            assert ds[name].encoding.get("zlib") is not True

    def test_end_to_end_netcdf_write(self, mri_file, tmp_path):
        ds = loadQfile(str(mri_file))
        assert ds is not None
        ds = cfCompliant(ds)
        ds = addEncoding(ds)
        nc_path = tmp_path / "test.nc"
        ds.to_netcdf(str(nc_path))
        assert nc_path.exists()
        assert nc_path.stat().st_size > 0

        # Read back and verify
        ds2 = xr.open_dataset(str(nc_path))
        assert len(ds2.time) == len(ds.time)
        ds2.close()
