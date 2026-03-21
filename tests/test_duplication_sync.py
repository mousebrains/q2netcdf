"""Tests verifying mergeqfiles.py duplicated classes stay in sync with standalone modules."""

import numpy as np
import pytest

from q2netcdf.QRecordType import RecordType as StandaloneRecordType
from q2netcdf.QVersion import QVersion as StandaloneQVersion
from q2netcdf.QHexCodes import QHexCodes as StandaloneQHexCodes
from q2netcdf.QConfig import QConfig as StandaloneQConfig

from q2netcdf.mergeqfiles import (
    RecordType as MergeRecordType,
    QVersion as MergeQVersion,
    QHexCodes as MergeQHexCodes,
    QConfig as MergeQConfig,
)


class TestRecordTypeSync:
    """Verify mergeqfiles.RecordType matches standalone RecordType."""

    def test_header_value_matches(self):
        assert MergeRecordType.HEADER.value == StandaloneRecordType.HEADER.value

    def test_data_value_matches(self):
        assert MergeRecordType.DATA.value == StandaloneRecordType.DATA.value

    def test_config_v12_value_matches(self):
        assert MergeRecordType.CONFIG_V12.value == StandaloneRecordType.CONFIG_V12.value

    def test_all_members_present(self):
        standalone_names = {m.name for m in StandaloneRecordType}
        merge_names = {m.name for m in MergeRecordType}
        assert standalone_names == merge_names


class TestQVersionSync:
    """Verify mergeqfiles.QVersion matches standalone QVersion."""

    def test_v12_value_matches(self):
        assert MergeQVersion.v12.value == StandaloneQVersion.v12.value

    def test_v13_value_matches(self):
        assert MergeQVersion.v13.value == StandaloneQVersion.v13.value

    def test_isV12_matches(self):
        assert MergeQVersion.v12.isV12() == StandaloneQVersion.v12.isV12()
        assert MergeQVersion.v13.isV12() == StandaloneQVersion.v13.isV12()

    def test_isV13_matches(self):
        assert MergeQVersion.v12.isV13() == StandaloneQVersion.v12.isV13()
        assert MergeQVersion.v13.isV13() == StandaloneQVersion.v13.isV13()

    def test_all_members_present(self):
        standalone_names = {m.name for m in StandaloneQVersion}
        merge_names = {m.name for m in MergeQVersion}
        assert standalone_names == merge_names


class TestQHexCodesSync:
    """Verify mergeqfiles.QHexCodes produces identical results to standalone."""

    # Test a representative sample of hex codes across all categories
    SAMPLE_IDENTS = [
        0x010,
        0x110,
        0x160,
        0x240,
        0x320,
        0x410,
        0x610,
        0x611,
        0x620,
        0x630,
        0x810,
        0xA10,
        0xA20,
        0xD20,
    ]

    @pytest.mark.parametrize("ident", SAMPLE_IDENTS)
    def test_name_matches(self, ident):
        standalone_name = StandaloneQHexCodes.name(ident)
        merge_name = MergeQHexCodes.name(ident)
        assert standalone_name == merge_name, (
            f"Name mismatch for {ident:#06x}: standalone={standalone_name}, merge={merge_name}"
        )

    @pytest.mark.parametrize("ident", SAMPLE_IDENTS)
    def test_attributes_match(self, ident):
        standalone_attrs = StandaloneQHexCodes.attributes(ident)
        merge_attrs = MergeQHexCodes.attributes(ident)
        assert standalone_attrs == merge_attrs, (
            f"Attrs mismatch for {ident:#06x}: standalone={standalone_attrs}, merge={merge_attrs}"
        )

    def test_unknown_ident_both_return_none(self):
        assert StandaloneQHexCodes.name(0xFFFF) is None
        assert MergeQHexCodes.name(0xFFFF) is None

    def test_name2ident_matches(self):
        names = ["sh_0", "sh_1", "pressure", "T_0", "e_0"]
        for name in names:
            standalone = StandaloneQHexCodes.name2ident(name)
            merge = MergeQHexCodes.name2ident(name)
            assert standalone == merge, (
                f"name2ident mismatch for {name}: standalone={standalone}, merge={merge}"
            )

    def test_hex_maps_have_same_keys(self):
        """Verify both hex maps cover the same set of sensor types."""
        # Access the private hex maps via name-mangled attributes
        standalone_map = StandaloneQHexCodes._QHexCodes__hexMap
        merge_map = MergeQHexCodes._QHexCodes__hexMap
        assert set(standalone_map.keys()) == set(merge_map.keys()), (
            f"Missing in merge: {set(standalone_map.keys()) - set(merge_map.keys())}, "
            f"Extra in merge: {set(merge_map.keys()) - set(standalone_map.keys())}"
        )


class TestQConfigSync:
    """Verify mergeqfiles.QConfig matches standalone QConfig."""

    def test_v13_json_parsing_matches(self):
        data = b'{"key": "value", "number": 42, "flag": true}'
        standalone = StandaloneQConfig(data, StandaloneQVersion.v13).config()
        merge = MergeQConfig(data, MergeQVersion.v13).config()
        assert standalone == merge

    def test_v12_perl_parsing_matches(self):
        data = b'"sample_rate" => 512\n"enabled" => true\n"name" => "test"'
        standalone = StandaloneQConfig(data, StandaloneQVersion.v12).config()
        merge = MergeQConfig(data, MergeQVersion.v12).config()
        assert standalone == merge

    def test_empty_config_matches(self):
        data = b"{}"
        standalone = StandaloneQConfig(data, StandaloneQVersion.v13).config()
        merge = MergeQConfig(data, MergeQVersion.v13).config()
        assert standalone == merge

    def test_array_parsing_matches(self):
        data = b'"values" => [1, 2, 3]'
        standalone = StandaloneQConfig(data, StandaloneQVersion.v12).config()
        merge = MergeQConfig(data, MergeQVersion.v12).config()
        assert list(standalone.keys()) == list(merge.keys())
        np.testing.assert_array_equal(standalone["values"], merge["values"])

    def test_len_and_raw_match(self):
        data = b'{"test": 1}'
        s = StandaloneQConfig(data, StandaloneQVersion.v13)
        m = MergeQConfig(data, MergeQVersion.v13)
        assert len(s) == len(m)
        assert s.raw() == m.raw()


class TestQHeaderSync:
    """Verify mergeqfiles.QHeader produces identical results to standalone."""

    def test_parse_real_file_matches(self, mri_file):
        from q2netcdf.QHeader import QHeader as StandaloneQHeader
        from q2netcdf.mergeqfiles import QHeader as MergeQHeader

        with open(str(mri_file), "rb") as fp:
            s = StandaloneQHeader(fp, str(mri_file))

        with open(str(mri_file), "rb") as fp:
            m = MergeQHeader(fp, str(mri_file))

        assert s.version.value == m.version.value
        assert s.Nc == m.Nc
        assert s.Ns == m.Ns
        assert s.Nf == m.Nf
        assert s.channels == m.channels
        assert s.spectra == m.spectra
        assert s.frequencies == m.frequencies
        assert s.dataSize == m.dataSize
        assert s.hdrSize == m.hdrSize
        assert s.dtBinary == m.dtBinary
        assert s.config.config() == m.config.config()

    def test_repr_matches(self, mri_file):
        from q2netcdf.QHeader import QHeader as StandaloneQHeader
        from q2netcdf.mergeqfiles import QHeader as MergeQHeader

        with open(str(mri_file), "rb") as fp:
            s = StandaloneQHeader(fp, str(mri_file))

        with open(str(mri_file), "rb") as fp:
            m = MergeQHeader(fp, str(mri_file))

        # repr should produce identical output
        assert repr(s) == repr(m)

    def test_frequencies_default_when_nf_zero(self, mri_file):
        """Regression: mergeqfiles QHeader must init frequencies even when Nf=0."""
        from q2netcdf.mergeqfiles import QHeader as MergeQHeader

        with open(str(mri_file), "rb") as fp:
            m = MergeQHeader(fp, str(mri_file))

        # This file has Nf=0, so frequencies should be empty tuple, not missing
        assert m.Nf == 0
        assert m.frequencies == ()
        # repr should not raise AttributeError
        repr(m)
