"""Tests for QVersion enum."""
import pytest
from q2netcdf.QVersion import QVersion


def test_version_enum_values():
    """Test that QVersion enum has expected values."""
    assert hasattr(QVersion, 'v12')
    assert hasattr(QVersion, 'v13')


def test_version_isV12():
    """Test isV12 method for different versions."""
    assert QVersion.v12.isV12() is True
    assert QVersion.v13.isV12() is False


def test_version_float_values():
    """Test that version values are correct floats."""
    assert abs(QVersion.v12.value - 1.2) < 0.01
    assert abs(QVersion.v13.value - 1.3) < 0.01
