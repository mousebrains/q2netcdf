"""Tests for q2netcdf conversion module."""

import math
import pytest
import xarray as xr

from q2netcdf.q2netcdf import loadQfile, mergeDatasets, cfCompliant, addEncoding


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

    def test_merge_multiple_files(self, synthetic_multi_mri_files, tmp_path):
        """Test that merging multiple MRI files preserves all data."""
        frames = []
        for path in synthetic_multi_mri_files:
            ds = loadQfile(str(path))
            assert ds is not None
            frames.append(ds)

        ds = mergeDatasets(frames)

        # Total records: file1=3, file2=4, file3=2+3 (multiheader)=5 → 12
        assert len(ds.time) == 12

        # 3 files = 3 ftime entries (one per loadQfile call)
        assert len(ds.ftime) == 3

        # All pressure values should be valid (no NaN)
        assert ds.pressure.notnull().sum().values == 12

        # e_1 NaN: file1 has 3 NaN + file3 segment1 has 2 NaN = 5 NaN
        assert ds.e_1.isnull().sum().values == 5
        assert ds.e_1.notnull().sum().values == 7

        # e_2 NaN: same pattern
        assert ds.e_2.isnull().sum().values == 5
        assert ds.e_2.notnull().sum().values == 7

        # Verify actual values by sorting on time
        ds = ds.sortby("time")

        # File 1 (earliest): 3 records, pressure = -0.5, epsilon NaN
        for i in range(3):
            assert ds.pressure.values[i] == pytest.approx(-0.5)
            assert math.isnan(ds.e_1.values[i])
            assert math.isnan(ds.e_2.values[i])

        # File 2: 4 records with known values
        e2_f2 = [-8.0, -9.0, -10.0, -8.5]
        pr_f2 = [500.0, 490.0, 480.0, 470.0]
        e1_f2 = [-8.5, -9.5, -10.0, -9.0]
        for i in range(4):
            idx = 3 + i
            assert ds.e_2.values[idx] == pytest.approx(e2_f2[i], abs=0.01)
            assert ds.pressure.values[idx] == pytest.approx(pr_f2[i], abs=0.1)
            assert ds.e_1.values[idx] == pytest.approx(e1_f2[i], abs=0.01)

        # File 3 segment 1: 2 records, pressure valid, epsilon NaN
        assert ds.pressure.values[7] == pytest.approx(100.0, abs=0.1)
        assert ds.pressure.values[8] == pytest.approx(200.0, abs=0.1)
        assert math.isnan(ds.e_1.values[7])
        assert math.isnan(ds.e_1.values[8])

        # File 3 segment 2: 3 records, all valid
        e2_f3b = [-7.0, -8.0, -9.0]
        pr_f3b = [300.0, 400.0, 500.0]
        e1_f3b = [-7.5, -8.0, -9.0]
        for i in range(3):
            idx = 9 + i
            assert ds.e_2.values[idx] == pytest.approx(e2_f3b[i], abs=0.01)
            assert ds.pressure.values[idx] == pytest.approx(pr_f3b[i], abs=0.1)
            assert ds.e_1.values[idx] == pytest.approx(e1_f3b[i], abs=0.01)

        # Config variables: all files have same config
        assert (ds.diss_length.values == 16).all()
        assert (ds.fft_length.values == 4).all()
        assert ds.fileVersion.values == pytest.approx([1.3, 1.3, 1.3], abs=0.01)

        # Round-trip through NetCDF
        nc_path = tmp_path / "merged.nc"
        ds = cfCompliant(ds)
        ds = addEncoding(ds)
        ds.to_netcdf(str(nc_path))
        ds2 = xr.open_dataset(str(nc_path), decode_timedelta=False)
        assert len(ds2.time) == 12
        assert ds2.e_1.notnull().sum().values == 7
        assert ds2.pressure.notnull().sum().values == 12
        ds2.close()

    def test_merge_single_frame(self, synthetic_v13_qfile):
        """Test that merging a single frame returns it as-is."""
        ds = loadQfile(str(synthetic_v13_qfile))
        assert ds is not None
        result = mergeDatasets([ds])
        assert result is ds

    def test_merge_empty_raises(self):
        """Test that merging empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            mergeDatasets([])

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
