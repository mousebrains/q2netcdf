#! /usr/bin/env python3
"""Tests for QRecordType enum."""

import pytest
from q2netcdf.QRecordType import RecordType


class TestQRecordType:
    """Test suite for QRecordType enum."""

    def test_record_type_values(self):
        """Test that record type enum has correct hex values."""
        assert RecordType.HEADER.value == 0x1729
        assert RecordType.CONFIG_V12.value == 0x0827
        assert RecordType.DATA.value == 0x1657

    def test_all_record_types_exist(self):
        """Test that expected record types are defined."""
        record_types = list(RecordType)
        assert len(record_types) == 3
        assert RecordType.HEADER in record_types
        assert RecordType.CONFIG_V12 in record_types
        assert RecordType.DATA in record_types

    def test_record_type_unique_values(self):
        """Test that all record type values are unique."""
        values = [rt.value for rt in RecordType]
        assert len(values) == len(set(values)), "Record type values should be unique"

    def test_record_type_string_representation(self):
        """Test string representation of record types."""
        assert "HEADER" in str(RecordType.HEADER)
        assert "CONFIG_V12" in str(RecordType.CONFIG_V12)
        assert "DATA" in str(RecordType.DATA)

    def test_record_type_by_value(self):
        """Test accessing record types by value."""
        assert RecordType(0x1729) == RecordType.HEADER
        assert RecordType(0x0827) == RecordType.CONFIG_V12
        assert RecordType(0x1657) == RecordType.DATA

    def test_invalid_value_raises(self):
        """Test that invalid value raises ValueError."""
        with pytest.raises(ValueError):
            RecordType(0xFFFF)
