"""Tests using real MRI files for QFile, QHeader, QData."""

import struct

import numpy as np
import pytest

from q2netcdf.QFile import QFile
from q2netcdf.QHeader import QHeader
from q2netcdf.QData import QData
from q2netcdf.QHexCodes import QHexCodes
from q2netcdf.QRecordType import RecordType
from q2netcdf.QVersion import QVersion


class TestQHeaderRealFile:
    """Test QHeader parsing with real MRI files."""

    def test_parse_header(self, mri_file):
        with open(str(mri_file), "rb") as fp:
            hdr = QHeader(fp, str(mri_file))
            assert hdr.version == QVersion.v13
            assert hdr.Nc == 3
            assert hdr.Ns == 0
            assert hdr.Nf == 0
            assert len(hdr.channels) == 3
            assert hdr.dataSize > 0
            assert hdr.hdrSize > 0

    def test_chkIdent_at_start(self, mri_file):
        with open(str(mri_file), "rb") as fp:
            result = QHeader.chkIdent(fp)
            assert result is True
            # File pointer should be restored
            assert fp.tell() == 0

    def test_chkIdent_at_data_record(self, mri_file):
        with open(str(mri_file), "rb") as fp:
            QHeader(fp, str(mri_file))
            # Now fp is at first data record
            result = QHeader.chkIdent(fp)
            assert result is False  # It's a data record, not a header

    def test_header_config(self, mri_file):
        with open(str(mri_file), "rb") as fp:
            hdr = QHeader(fp, str(mri_file))
            cfg = hdr.config.config()
            assert isinstance(cfg, dict)
            assert "diss_length" in cfg
            assert "fft_length" in cfg

    def test_header_repr(self, mri_file):
        with open(str(mri_file), "rb") as fp:
            hdr = QHeader(fp, str(mri_file))
            r = repr(hdr)
            assert "Version" in r
            assert "Channels" in r
            assert "Data Size" in r

    def test_read_exact_eof(self, tmp_path):
        f = tmp_path / "short.bin"
        f.write_bytes(b"\x00\x01")
        with open(str(f), "rb") as fp:
            with pytest.raises(EOFError, match="fixed header"):
                QHeader._read_exact(fp, 20, str(f), "fixed header")


class TestQDataRealFile:
    """Test QData parsing with real MRI files."""

    def test_load_records(self, mri_file):
        with open(str(mri_file), "rb") as fp:
            hdr = QHeader(fp, str(mri_file))
            data = QData(hdr)
            records = []
            while True:
                rec = data.load(fp)
                if rec is None:
                    break
                records.append(rec)
            assert len(records) > 0

    def test_record_has_channels(self, mri_file):
        with open(str(mri_file), "rb") as fp:
            hdr = QHeader(fp, str(mri_file))
            data = QData(hdr)
            rec = data.load(fp)
            assert rec is not None
            assert len(rec.channels) == hdr.Nc
            assert rec.t0 is not None

    def test_chkIdent_data_record(self, mri_file):
        with open(str(mri_file), "rb") as fp:
            hdr = QHeader(fp, str(mri_file))
            result = QData.chkIdent(fp)
            assert result is True
            # fp should be at same position
            assert fp.tell() == hdr.hdrSize

    def test_record_split(self, mri_file):
        with open(str(mri_file), "rb") as fp:
            hdr = QHeader(fp, str(mri_file))
            data = QData(hdr)
            rec = data.load(fp)
            assert rec is not None

            record_dict, attrs_dict = rec.split(hdr)
            assert "time" in record_dict
            # Should have named channels
            for ident in hdr.channels:
                name = QHexCodes.name(ident)
                if name:
                    assert name in record_dict

    def test_pretty_record(self, mri_file):
        with open(str(mri_file), "rb") as fp:
            hdr = QHeader(fp, str(mri_file))
            data = QData(hdr)
            rec = data.load(fp)
            assert rec is not None

            pretty = data.prettyRecord(rec)
            assert isinstance(pretty, str)
            assert "Time" in pretty
            assert "Channel" in pretty

    def test_record_repr(self, mri_file):
        with open(str(mri_file), "rb") as fp:
            hdr = QHeader(fp, str(mri_file))
            data = QData(hdr)
            rec = data.load(fp)
            assert rec is not None
            r = repr(rec)
            assert "Record" in r
            assert "Time" in r

    def test_v13_record_fields(self, mri_file):
        """V1.3 records should have no record number or end time."""
        with open(str(mri_file), "rb") as fp:
            hdr = QHeader(fp, str(mri_file))
            assert hdr.version == QVersion.v13
            data = QData(hdr)
            rec = data.load(fp)
            assert rec is not None
            assert rec.number is None  # v1.3 has no record number
            assert rec.error is None  # v1.3 has no error code
            assert rec.t1 is None  # v1.3 has no end time


class TestQFileRealFile:
    """Test QFile with real MRI files."""

    def test_read_header_and_data(self, mri_file):
        with QFile(str(mri_file)) as qf:
            hdr = qf.header()
            assert hdr is not None
            records = list(qf.data())
            assert len(records) > 100  # 02880000.mri has 198 records

    def test_context_manager_closes(self, mri_file):
        qf = QFile(str(mri_file))
        with qf:
            hdr = qf.header()
            assert hdr is not None
        # After exit, accessing internals should be clean
        # The __del__ should handle double-close gracefully

    def test_validate(self, mri_file):
        with QFile(str(mri_file)) as qf:
            results = qf.validate()
            assert results["valid"] is True
            assert results["version"] == QVersion.v13
            assert results["records_readable"] > 0
            assert results["records_failed"] == 0
            assert len(results["errors"]) == 0

    def test_pretty_record(self, mri_file):
        with QFile(str(mri_file)) as qf:
            qf.header()
            rec = next(qf.data())
            pretty = qf.prettyRecord(rec)
            assert pretty is not None
            assert "Time" in pretty

    def test_pretty_record_before_header(self, mri_file):
        with QFile(str(mri_file)) as qf:
            # Create a dummy record to test prettyRecord without header
            result = qf.prettyRecord(None)  # type: ignore
            assert result is None

    def test_multiple_files(self, mri_file, mri_file_small):
        """Test reading multiple different MRI files."""
        for fn in [mri_file, mri_file_small]:
            with QFile(str(fn)) as qf:
                hdr = qf.header()
                assert hdr.version == QVersion.v13
                records = list(qf.data())
                assert len(records) > 0

    def test_validate_corrupt_file(self, corrupt_file):
        with QFile(str(corrupt_file)) as qf:
            results = qf.validate()
            assert results["valid"] is False
            assert len(results["errors"]) > 0

    def test_validate_with_unknown_idents(self, tmp_path):
        """Test validate detects unknown channel identifiers."""
        qfile = tmp_path / "unknown_ident.q"
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v13.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        header += struct.pack("<Q", dt_ms)
        Nc, Ns, Nf = 1, 0, 0
        header += struct.pack("<HHH", Nc, Ns, Nf)
        header += struct.pack("<H", 0xFFF0)  # unknown ident
        config_str = "{}"
        header += struct.pack("<H", len(config_str))
        header += config_str.encode("utf-8")
        data_size = 2 + 2 + Nc * 2
        header += struct.pack("<H", data_size)
        # One data record
        data = struct.pack("<H", RecordType.DATA.value)
        data += struct.pack("<e", 1.0)
        data += struct.pack("<e", 42.0)
        qfile.write_bytes(header + data)

        with QFile(str(qfile)) as qf:
            results = qf.validate()
            assert 0xFFF0 in results["unknown_identifiers"]
            assert results["records_readable"] == 1


class TestQDataV12Synthetic:
    """Test QData with synthetic v1.2 records."""

    def test_v12_record_has_all_fields(self, tmp_path):
        """V1.2 records have record number, error, start time, end time."""
        qfile = tmp_path / "v12_data.q"
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v12.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        header += struct.pack("<Q", dt_ms)

        Nc, Ns, Nf = 2, 0, 0
        header += struct.pack("<HHH", Nc, Ns, Nf)
        header += struct.pack("<HH", 0x160, 0x620)  # pressure, T_0

        # v1.2 config
        config_bytes = b"{}"
        header += struct.pack("<HH", RecordType.CONFIG_V12.value, len(config_bytes))
        header += config_bytes

        # v1.2 data record: ident(H) + recno(H) + error(q) + stime(e) + etime(e) + channels
        data_size = 2 + 2 + 8 + 2 + 2 + (Nc * 2)
        header += struct.pack("<H", data_size)

        # Write one data record
        data = bytearray()
        data += struct.pack("<H", RecordType.DATA.value)  # ident
        data += struct.pack("<H", 1)  # record number
        data += struct.pack("<q", 0)  # error code
        data += struct.pack("<e", 1.0)  # start time
        data += struct.pack("<e", 2.0)  # end time
        data += struct.pack("<ee", 100.5, 15.0)  # pressure, temperature

        qfile.write_bytes(header + data)

        with QFile(str(qfile)) as qf:
            hdr = qf.header()
            assert hdr.version == QVersion.v12
            records = list(qf.data())
            assert len(records) == 1
            rec = records[0]
            assert rec.number == 1
            assert rec.error == 0
            assert rec.t1 is not None  # v1.2 has end time
            assert rec.channels[0] == pytest.approx(100.5, rel=0.01)

    def test_v12_record_split_includes_extras(self, tmp_path):
        """split() should include t1, record, error for v1.2."""
        qfile = tmp_path / "v12_split.q"
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v12.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        header += struct.pack("<Q", dt_ms)

        Nc, Ns, Nf = 1, 0, 0
        header += struct.pack("<HHH", Nc, Ns, Nf)
        header += struct.pack("<H", 0x160)

        config_bytes = b"{}"
        header += struct.pack("<HH", RecordType.CONFIG_V12.value, len(config_bytes))
        header += config_bytes

        data_size = 2 + 2 + 8 + 2 + 2 + (Nc * 2)
        header += struct.pack("<H", data_size)

        data = bytearray()
        data += struct.pack("<H", RecordType.DATA.value)
        data += struct.pack("<H", 42)
        data += struct.pack("<q", 5)
        data += struct.pack("<e", 1.0)
        data += struct.pack("<e", 2.0)
        data += struct.pack("<e", 100.0)

        qfile.write_bytes(header + data)

        with open(str(qfile), "rb") as fp:
            hdr = QHeader(fp, str(qfile))
            qdata = QData(hdr)
            rec = qdata.load(fp)
            assert rec is not None
            record_dict, attrs_dict = rec.split(hdr)
            assert "t1" in record_dict
            assert "record" in record_dict
            assert "error" in record_dict
            assert record_dict["record"] == 42

    def test_v12_pretty_record(self, tmp_path):
        """prettyRecord should show time range for v1.2."""
        qfile = tmp_path / "v12_pretty.q"
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v12.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        header += struct.pack("<Q", dt_ms)

        Nc, Ns, Nf = 1, 0, 0
        header += struct.pack("<HHH", Nc, Ns, Nf)
        header += struct.pack("<H", 0x160)

        config_bytes = b"{}"
        header += struct.pack("<HH", RecordType.CONFIG_V12.value, len(config_bytes))
        header += config_bytes

        data_size = 2 + 2 + 8 + 2 + 2 + (Nc * 2)
        header += struct.pack("<H", data_size)

        data = bytearray()
        data += struct.pack("<H", RecordType.DATA.value)
        data += struct.pack("<H", 1)
        data += struct.pack("<q", 0)
        data += struct.pack("<e", 1.0)
        data += struct.pack("<e", 2.0)
        data += struct.pack("<e", 100.0)

        qfile.write_bytes(header + data)

        with open(str(qfile), "rb") as fp:
            hdr = QHeader(fp, str(qfile))
            qdata = QData(hdr)
            rec = qdata.load(fp)
            assert rec is not None
            pretty = rec.prettyRecord(hdr)
            assert "Record #" in pretty
            assert "Error" in pretty
            assert " to " in pretty  # time range


class TestQDataWithSpectra:
    """Test QData with spectra data."""

    def test_spectra_record(self, tmp_path):
        qfile = tmp_path / "spectra_data.q"
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v13.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        header += struct.pack("<Q", dt_ms)

        Nc, Ns, Nf = 1, 2, 3
        header += struct.pack("<HHH", Nc, Ns, Nf)
        header += struct.pack("<H", 0x160)  # channel: pressure
        header += struct.pack("<HH", 0x610, 0x620)  # spectra: sh_0, T_0
        header += struct.pack("<eee", 1.0, 2.0, 4.0)  # frequencies

        config_str = "{}"
        header += struct.pack("<H", len(config_str))
        header += config_str.encode("utf-8")

        data_size = 2 + 2 + (Nc + Ns * Nf) * 2
        header += struct.pack("<H", data_size)

        data = bytearray()
        data += struct.pack("<H", RecordType.DATA.value)
        data += struct.pack("<e", 1.0)
        data += struct.pack("<e", 100.0)  # pressure
        # sh_0 spectra (3 freq bins)
        data += struct.pack("<eee", 0.1, 0.2, 0.3)
        # T_0 spectra (3 freq bins)
        data += struct.pack("<eee", 0.4, 0.5, 0.6)

        qfile.write_bytes(header + data)

        with open(str(qfile), "rb") as fp:
            hdr = QHeader(fp, str(qfile))
            qdata = QData(hdr)
            rec = qdata.load(fp)
            assert rec is not None
            assert rec.spectra.shape == (2, 3)
            assert rec.spectra[0, 0] == pytest.approx(0.1, abs=0.01)

            # Test split with spectra
            record_dict, attrs_dict = rec.split(hdr)
            assert "sh_0" in record_dict
            assert "T_0" in record_dict
            assert record_dict["sh_0"].shape == (3,)

            # Test prettyRecord with spectra
            pretty = rec.prettyRecord(hdr)
            assert "spectra[sh_0]" in pretty
            assert "spectra[T_0]" in pretty


class TestQHeaderSynthetic:
    """Test QHeader edge cases with synthetic data."""

    def test_v12_header_with_config(self, tmp_path):
        """Create and read a v1.2 header with config."""
        qfile = tmp_path / "v12.q"
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v12.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        header += struct.pack("<Q", dt_ms)

        Nc, Ns, Nf = 1, 0, 0
        header += struct.pack("<HHH", Nc, Ns, Nf)
        header += struct.pack("<H", 0x160)  # pressure

        # v1.2 config: cfgIdent + size
        config_bytes = b'"diss_length" => 16\n"fft_length" => 4'
        header += struct.pack("<HH", RecordType.CONFIG_V12.value, len(config_bytes))
        header += config_bytes

        # Data record size
        data_size = (
            2 + 2 + 8 + 2 + 2 + (Nc * 2)
        )  # v1.2: ident+recno+err+stime+etime+channels
        header += struct.pack("<H", data_size)

        qfile.write_bytes(header)

        with open(str(qfile), "rb") as fp:
            hdr = QHeader(fp, str(qfile))
            assert hdr.version == QVersion.v12
            assert hdr.Nc == 1
            cfg = hdr.config.config()
            assert cfg["diss_length"] == 16

    def test_header_with_spectra_and_frequencies(self, tmp_path):
        """Test header with spectra and frequency arrays."""
        qfile = tmp_path / "spectra.q"
        header = bytearray()
        header += struct.pack("<H", RecordType.HEADER.value)
        header += struct.pack("<f", QVersion.v13.value)
        dt_ms = int(np.datetime64("2025-01-01").astype("datetime64[ms]").astype(int))
        header += struct.pack("<Q", dt_ms)

        Nc, Ns, Nf = 1, 2, 4
        header += struct.pack("<HHH", Nc, Ns, Nf)
        header += struct.pack("<H", 0x160)  # channel: pressure
        header += struct.pack("<HH", 0x610, 0x620)  # spectra: sh_0, T_0
        header += struct.pack("<eeee", 1.0, 2.0, 4.0, 8.0)  # frequencies

        config_str = "{}"
        header += struct.pack("<H", len(config_str))
        header += config_str.encode("utf-8")

        data_size = 2 + 2 + (Nc + Ns * Nf) * 2
        header += struct.pack("<H", data_size)

        qfile.write_bytes(header)

        with open(str(qfile), "rb") as fp:
            hdr = QHeader(fp, str(qfile))
            assert hdr.Nc == 1
            assert hdr.Ns == 2
            assert hdr.Nf == 4
            assert len(hdr.spectra) == 2
            assert len(hdr.frequencies) == 4
            assert hdr.frequencies[0] == pytest.approx(1.0, rel=0.01)
