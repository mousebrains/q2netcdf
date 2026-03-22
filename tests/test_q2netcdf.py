"""Tests for q2netcdf conversion module."""

import io
import logging
import math
import struct
import numpy as np
import pytest
import xarray as xr
from unittest.mock import MagicMock

from q2netcdf.q2netcdf import loadQfile, mergeDatasets, cfCompliant, addEncoding
from q2netcdf.QData import QData, QRecord
from q2netcdf.QHeader import QHeader
from q2netcdf.QRecordType import RecordType
from q2netcdf.QVersion import QVersion


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


class TestMergeSchemas:
    """Test mergeDatasets with mismatched channel schemas."""

    def test_merge_mismatched_channels(self, synthetic_mismatched_channels_files):
        """Verify outer join produces correct NaN fill for mismatched schemas."""
        frames = []
        for path in synthetic_mismatched_channels_files:
            ds = loadQfile(str(path))
            assert ds is not None
            frames.append(ds)

        ds = mergeDatasets(frames)

        # File 1 has 5 records, file 2 has 3
        assert len(ds.time) == 8

        # pressure is in both files — all valid
        assert "pressure" in ds
        assert ds.pressure.notnull().sum().values == 8

        # e_2 and e_1 only in file 1 — NaN for file 2's 3 records
        assert "e_2" in ds
        assert "e_1" in ds
        assert ds.e_2.isnull().sum().values == 3
        assert ds.e_1.isnull().sum().values == 3

        # T_0 only in file 2 — NaN for file 1's 5 records
        assert "T_0" in ds
        assert ds.T_0.isnull().sum().values == 5
        assert ds.T_0.notnull().sum().values == 3


class TestRoundTrip:
    """Test full Q-file → NetCDF → read-back round trip."""

    def test_synthetic_roundtrip(self, synthetic_v13_qfile, tmp_path):
        """Create binary Q-file, load, convert to NetCDF, read back, verify."""
        ds = loadQfile(str(synthetic_v13_qfile))
        assert ds is not None

        # Write to NetCDF
        nc_path = tmp_path / "roundtrip.nc"
        ds = cfCompliant(ds)
        ds = addEncoding(ds)
        ds.to_netcdf(str(nc_path))

        # Read back
        ds2 = xr.open_dataset(str(nc_path), decode_timedelta=False)
        assert len(ds2.time) == len(ds.time)
        assert "pressure" in ds2
        assert "e_1" in ds2
        assert "e_2" in ds2

        # Verify pressure values are valid floats (not all NaN)
        assert ds2.pressure.notnull().sum().values > 0
        ds2.close()


class TestErrorHandling:
    """Test handling of truncated and corrupt files."""

    def test_truncated_header(self, tmp_path):
        """File with only 10 bytes (header needs 20) raises EOFError."""
        f = tmp_path / "truncated.q"
        f.write_bytes(b"\x29\x17" + b"\x00" * 8)  # Header ident + garbage
        with pytest.raises(EOFError):
            loadQfile(str(f))

    def test_truncated_data_record(self, tmp_path, caplog):
        """File with valid header but truncated data record logs warning."""
        import struct
        import numpy as np
        from q2netcdf.QRecordType import RecordType
        from q2netcdf.QVersion import QVersion

        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v13.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        header += struct.pack("<Q", dt_ms)
        header += struct.pack("<HHH", 1, 0, 0)  # Nc=1, Ns=0, Nf=0
        header += struct.pack("<H", 0x160)  # pressure channel
        header += struct.pack("<H", 2)  # config size = 2
        header += b"{}"
        header += struct.pack("<H", 6)  # data record size: 2+2+2=6

        # One valid record
        data = struct.pack("<H", RecordType.DATA.value)
        data += struct.pack("<e", 0.0)  # stime
        data += struct.pack("<e", 100.0)  # pressure

        # Truncated record (3 bytes of expected 6)
        truncated = b"\x57\x16\x00"

        f = tmp_path / "truncdata.q"
        f.write_bytes(header + data + truncated)

        import logging

        with caplog.at_level(logging.WARNING):
            ds = loadQfile(str(f))

        # Should still get the one valid record
        assert ds is not None
        assert len(ds.time) == 1
        # Should have logged a truncation warning
        assert any("Truncated" in r.message for r in caplog.records)

    def test_bad_json_config(self, tmp_path):
        """File with malformed JSON config should not crash."""
        import struct
        import numpy as np
        from q2netcdf.QRecordType import RecordType
        from q2netcdf.QVersion import QVersion

        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v13.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        header += struct.pack("<Q", dt_ms)
        header += struct.pack("<HHH", 1, 0, 0)
        header += struct.pack("<H", 0x160)
        bad_json = b"{invalid}"
        header += struct.pack("<H", len(bad_json))
        header += bad_json
        header += struct.pack("<H", 6)  # data record size

        data = struct.pack("<H", RecordType.DATA.value)
        data += struct.pack("<e", 0.0)
        data += struct.pack("<e", 100.0)

        f = tmp_path / "badjson.q"
        f.write_bytes(header + data)

        ds = loadQfile(str(f))
        assert ds is not None
        assert len(ds.time) == 1

    def test_bounds_check_rejects_huge_nc(self, tmp_path):
        """Header with Nc > 1024 should raise ValueError."""
        import struct
        import numpy as np
        from q2netcdf.QRecordType import RecordType
        from q2netcdf.QVersion import QVersion

        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v13.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        header += struct.pack("<Q", dt_ms)
        header += struct.pack("<HHH", 2000, 0, 0)  # Nc=2000 (over bounds)

        f = tmp_path / "hugeNc.q"
        f.write_bytes(header)

        with pytest.raises(ValueError, match="Suspect header counts"):
            loadQfile(str(f))


class TestRealFileRoundTrip:
    """Full roundtrip: Q-file -> Dataset -> NetCDF -> read back -> verify."""

    def test_v13_roundtrip(self, mri_file, tmp_path):
        ds = loadQfile(str(mri_file))
        assert ds is not None

        orig_times = ds.time.values.copy()
        orig_pressure = ds.pressure.values.copy()

        nc = tmp_path / "v13_roundtrip.nc"
        ds_out = cfCompliant(ds)
        ds_out = addEncoding(ds_out)
        ds_out.to_netcdf(str(nc))

        ds2 = xr.open_dataset(str(nc), decode_timedelta=False)
        assert len(ds2.time) == len(orig_times)
        assert "pressure" in ds2
        np.testing.assert_array_equal(ds2.time.values, orig_times)
        np.testing.assert_allclose(ds2.pressure.values, orig_pressure, rtol=1e-5)
        assert "CF-1.13" in ds2.attrs["Conventions"]
        assert "time_coverage_start" in ds2.attrs
        ds2.close()

    def test_v12_roundtrip(self, qfile_v12, tmp_path):
        ds = loadQfile(str(qfile_v12))
        assert ds is not None

        orig_times = ds.time.values.copy()
        orig_pressure = ds.pressure.values.copy()

        nc = tmp_path / "v12_roundtrip.nc"
        ds_out = cfCompliant(ds)
        ds_out = addEncoding(ds_out)
        ds_out.to_netcdf(str(nc))

        ds2 = xr.open_dataset(str(nc), decode_timedelta=False)
        assert len(ds2.time) == len(orig_times)
        assert "pressure" in ds2
        np.testing.assert_array_equal(ds2.time.values, orig_times)
        np.testing.assert_allclose(ds2.pressure.values, orig_pressure, rtol=1e-5)
        # v1.2 has spectra — verify freq coordinate survived
        assert "freq" in ds2.coords
        assert len(ds2.freq) == 18
        # Verify spectra variable survived
        assert "shear_gfd_1" in ds2
        assert "CF-1.13" in ds2.attrs["Conventions"]
        ds2.close()

    def test_multifile_roundtrip(self, mri_file, qfile_v12, tmp_path):
        frames = []
        for path in [mri_file, qfile_v12]:
            ds = loadQfile(str(path))
            assert ds is not None
            frames.append(ds)

        ds = mergeDatasets(frames)
        total_records = len(frames[0].time) + len(frames[1].time)
        assert len(ds.time) == total_records
        assert len(ds.ftime) == 2

        nc = tmp_path / "multi_roundtrip.nc"
        ds_out = cfCompliant(ds)
        ds_out = addEncoding(ds_out)
        ds_out.to_netcdf(str(nc))

        ds2 = xr.open_dataset(str(nc), decode_timedelta=False)
        assert len(ds2.time) == total_records
        assert len(ds2.ftime) == 2
        assert "pressure" in ds2
        ds2.close()


class TestLoadQfileEdgeCases:
    """Tests for loadQfile edge cases: unknown idents, data-before-header, etc."""

    def _build_v13_header(
        self, channel_idents: list[int], config_str: str = "{}"
    ) -> bytes:
        """Helper to build a minimal v1.3 header."""
        Nc = len(channel_idents)
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v13.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        header += struct.pack("<Q", dt_ms)
        header += struct.pack("<HHH", Nc, 0, 0)  # Nc, Ns=0, Nf=0
        for ident in channel_idents:
            header += struct.pack("<H", ident)
        header += struct.pack("<H", len(config_str))
        header += config_str.encode("utf-8")
        data_size = 2 + 2 + (Nc * 2)  # ident + stime + channels
        header += struct.pack("<H", data_size)
        return bytes(header)

    def _build_v13_data_record(self, channel_values: list[float]) -> bytes:
        """Helper to build a v1.3 data record."""
        rec = bytearray()
        rec += struct.pack("<H", RecordType.DATA.value)
        rec += struct.pack("<e", 0.0)  # stime
        for val in channel_values:
            rec += struct.pack("<e", val)
        return bytes(rec)

    def test_unknown_channel_identifier(self, tmp_path, caplog):
        """Line 70: Unknown channel identifier logs warning and skips it."""
        # Use 0xFFFF as an unknown channel identifier alongside a known one
        header = self._build_v13_header([0x160, 0xFFFF])
        data = self._build_v13_data_record([100.0, 42.0])

        f = tmp_path / "unknown_channel.q"
        f.write_bytes(header + data)

        with caplog.at_level(logging.WARNING):
            ds = loadQfile(str(f))

        assert ds is not None
        assert "pressure" in ds
        # The unknown channel should have been skipped
        assert any("Unknown channel identifier" in r.message for r in caplog.records)

    def test_unknown_spectra_identifier(self, tmp_path, caplog):
        """Line 90: Unknown spectra identifier logs warning and skips it."""
        # Build a header with known channels and an unknown spectra
        Nc, Ns, Nf = 1, 1, 2
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v13.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
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

        data = bytearray()
        data += struct.pack("<H", RecordType.DATA.value)
        data += struct.pack("<e", 0.0)  # stime
        data += struct.pack("<e", 100.0)  # pressure
        data += struct.pack("<ee", 0.5, 0.6)  # spectra values

        f = tmp_path / "unknown_spectra.q"
        f.write_bytes(bytes(header) + bytes(data))

        with caplog.at_level(logging.WARNING):
            ds = loadQfile(str(f))

        assert ds is not None
        assert "pressure" in ds
        assert any("Unknown spectra identifier" in r.message for r in caplog.records)

    def test_data_record_before_header(self, tmp_path):
        """Line 128: Data record identifier before any header raises ValueError."""
        # Write a data record identifier as the first thing in the file
        content = bytearray()
        content += struct.pack("<H", RecordType.DATA.value)
        content += b"\x00" * 20  # some padding

        f = tmp_path / "data_before_header.q"
        f.write_bytes(bytes(content))

        with pytest.raises(ValueError, match="Data record before header"):
            loadQfile(str(f))

    def test_unsupported_identifier(self, tmp_path, caplog):
        """Lines 138-142: Unsupported identifier warns and breaks."""
        # Write a valid header + data record, then an unknown identifier
        header = self._build_v13_header([0x160])
        data = self._build_v13_data_record([100.0])
        # Then an unsupported identifier
        unsupported = struct.pack("<H", 0xABCD)

        f = tmp_path / "unsupported_ident.q"
        f.write_bytes(header + data + unsupported)

        with caplog.at_level(logging.WARNING):
            ds = loadQfile(str(f))

        assert ds is not None
        assert len(ds.time) == 1
        assert any("Unsupported identifier" in r.message for r in caplog.records)

    def test_no_header_in_file(self, tmp_path, caplog):
        """Lines 148-149: File with no header record returns None with warning."""
        # Write bytes that are neither header nor data identifier
        # 0xABCD is not 0x1729 (header) or 0x1657 (data)
        content = struct.pack("<H", 0xABCD) + b"\x00" * 20

        f = tmp_path / "no_header.q"
        f.write_bytes(content)

        with caplog.at_level(logging.WARNING):
            result = loadQfile(str(f))

        assert result is None
        assert any("No header found" in r.message for r in caplog.records)


class TestQDataCoverage:
    """Tests for QData and QRecord uncovered lines."""

    def _make_header(
        self,
        channel_idents: list[int],
        spectra_idents: list[int] | None = None,
        nf: int = 0,
        version: QVersion = QVersion.v13,
    ) -> QHeader:
        """Build a real QHeader by constructing binary data and parsing it."""
        Nc = len(channel_idents)
        Ns = len(spectra_idents) if spectra_idents else 0
        Nf = nf

        buf = bytearray()
        buf += struct.pack("<H", RecordType.HEADER.value)
        buf += struct.pack("<f", version.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        buf += struct.pack("<Q", dt_ms)
        buf += struct.pack("<HHH", Nc, Ns, Nf)
        for ident in channel_idents:
            buf += struct.pack("<H", ident)
        if spectra_idents:
            for ident in spectra_idents:
                buf += struct.pack("<H", ident)
        if Nf > 0:
            for i in range(Nf):
                buf += struct.pack("<e", float(i + 1))

        config_str = "{}" if version == QVersion.v13 else ""
        if version == QVersion.v12:
            # v1.2 config has 4-byte header: 2-byte ident + 2-byte size
            config_bytes = config_str.encode("utf-8")
            buf += struct.pack("<HH", RecordType.CONFIG_V12.value, len(config_bytes))
            buf += config_bytes
        else:
            config_bytes = config_str.encode("utf-8")
            buf += struct.pack("<H", len(config_bytes))
            buf += config_bytes

        # data record size
        if version.isV12():
            data_size = 2 + 2 + 8 + 2 + 2 + (Nc * 2) + (Ns * Nf * 2)
        else:
            data_size = 2 + 2 + (Nc * 2) + (Ns * Nf * 2)
        buf += struct.pack("<H", data_size)

        fp = io.BytesIO(bytes(buf))
        hdr = QHeader(fp, "test.q")
        return hdr

    def test_qrecord_repr_with_t1(self):
        """Line 67: QRecord.__repr__ with t1 set (v1.2 records have t1)."""
        hdr = self._make_header([0x160], version=QVersion.v12)
        # v1.2 record: number, err, stime, etime
        record = QRecord(hdr, 1, 0, 0.0, 1.0, [100.0])
        repr_str = repr(record)
        assert "Record #:" in repr_str
        assert "to" in repr_str  # "Time: t0 to t1"

    def test_qrecord_split_unknown_channel(self, caplog):
        """Line 99: QRecord.split() with unknown channel logs warning."""
        hdr = self._make_header([0xFFFF])
        record = QRecord(hdr, None, None, 0.0, None, [100.0])

        with caplog.at_level(logging.WARNING):
            rec_dict, attrs = record.split(hdr)

        assert "time" in rec_dict
        # The unknown channel should not be in the dict
        assert any("Unknown channel identifier" in r.message for r in caplog.records)

    def test_qrecord_split_unknown_spectra(self, caplog):
        """Line 112: QRecord.split() with unknown spectra logs warning."""
        hdr = self._make_header([0x160], spectra_idents=[0xFFFF], nf=2)
        items = [100.0, 0.5, 0.6]  # 1 channel + 2 spectra values
        record = QRecord(hdr, None, None, 0.0, None, items)

        with caplog.at_level(logging.WARNING):
            rec_dict, attrs = record.split(hdr)

        assert "pressure" in rec_dict
        assert any("Unknown channel identifier" in r.message for r in caplog.records)

    def test_qrecord_prettyrecord_unknown_channel(self):
        """Line 139: prettyRecord with unknown channel falls back to hex name."""
        hdr = self._make_header([0xFFFF])
        record = QRecord(hdr, None, None, 0.0, None, [100.0])

        output = record.prettyRecord(hdr)
        assert "0xffff" in output

    def test_qrecord_prettyrecord_unknown_spectra(self):
        """Line 145: prettyRecord with unknown spectra falls back to hex name."""
        hdr = self._make_header([0x160], spectra_idents=[0xFFFF], nf=2)
        items = [100.0, 0.5, 0.6]
        record = QRecord(hdr, None, None, 0.0, None, items)

        output = record.prettyRecord(hdr)
        assert "0xffff" in output

    def test_qdata_version_none_raises(self):
        """Line 162: QData.__init__() with version=None raises RuntimeError."""
        hdr = MagicMock(spec=QHeader)
        hdr.version = None

        with pytest.raises(RuntimeError, match="version must be set"):
            QData(hdr)

    def test_qdata_load_struct_error(self):
        """Lines 193-199: QData.load() with malformed data logs warning."""
        hdr = self._make_header([0x160])
        qdata = QData(hdr)

        # The data size from header is the expected read size.
        # Put enough bytes so read succeeds but struct.unpack fails
        # by writing valid-length but wrong-format data.
        # Actually, the simplest approach: write correct length but
        # change the format string by mocking. Instead, let's create
        # data that's exactly the right length (dataSize) but
        # note that struct.unpack with 'e' format on 2 bytes should
        # always work. We need to cause struct.error.
        # Let's override the format to require more data than available.
        qdata._QData__format = "<HHH"  # expects 6 bytes

        # Create buffer with exactly hdr.dataSize bytes (should be 6 for 1 channel v1.3)
        # But now format expects 3 unsigned shorts = 6 bytes, dataSize is also 6
        # That won't cause struct.error. Let's make format expect more.
        qdata._QData__format = "<HHHH"  # expects 8 bytes, but buffer is 6

        fp = io.BytesIO(b"\x57\x16\x00\x00\x00\x00")  # 6 bytes

        with pytest.raises(struct.error):
            struct.unpack("<HHHH", fp.read(6))

        # Reset the fp
        fp.seek(0)

        result = qdata.load(fp)
        assert result is None  # Should return None after logging warning

    def test_qdata_load_ident_mismatch(self, caplog):
        """Line 213: QData.load() with data record identifier mismatch."""
        hdr = self._make_header([0x160])
        qdata = QData(hdr)

        # Build a data record with wrong identifier but correct size
        buf = bytearray()
        buf += struct.pack("<H", 0x0000)  # Wrong identifier (not 0x1657)
        buf += struct.pack("<e", 0.0)  # stime
        buf += struct.pack("<e", 100.0)  # pressure value

        fp = io.BytesIO(bytes(buf))

        with caplog.at_level(logging.WARNING):
            record = qdata.load(fp)

        # Should still return a record (it logs warning but doesn't fail)
        assert record is not None
        assert any(
            "Data record identifier mismatch" in r.message for r in caplog.records
        )
