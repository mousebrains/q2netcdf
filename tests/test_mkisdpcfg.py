"""Tests for mkISDPcfg module."""

import pytest
from argparse import ArgumentTypeError

from q2netcdf.mkISDPcfg import chkNotNegative, chkPositive, chkDespiking


class TestChkNotNegative:
    def test_zero(self):
        assert chkNotNegative("0") == 0.0

    def test_positive(self):
        assert chkNotNegative("3.14") == 3.14

    def test_negative_raises(self):
        with pytest.raises(ArgumentTypeError, match="< 0"):
            chkNotNegative("-1")

    def test_not_numeric_raises(self):
        with pytest.raises(ArgumentTypeError, match="not numeric"):
            chkNotNegative("abc")


class TestChkPositive:
    def test_positive(self):
        assert chkPositive("5.0") == 5.0

    def test_zero_raises(self):
        with pytest.raises(ArgumentTypeError, match="<= 0"):
            chkPositive("0")

    def test_negative_raises(self):
        with pytest.raises(ArgumentTypeError, match="<= 0"):
            chkPositive("-1")

    def test_not_numeric_raises(self):
        with pytest.raises(ArgumentTypeError, match="not numeric"):
            chkPositive("xyz")


class TestChkDespiking:
    def test_valid_input(self):
        result = chkDespiking("3.0,0.5,10")
        assert result == (3.0, 0.5, 10)

    def test_wrong_field_count(self):
        with pytest.raises(ArgumentTypeError, match="three fields"):
            chkDespiking("1,2")

    def test_bad_threshold(self):
        with pytest.raises(ArgumentTypeError, match="threshold"):
            chkDespiking("abc,0.5,10")

    def test_bad_smoothing(self):
        with pytest.raises(ArgumentTypeError, match="smoothing"):
            chkDespiking("3.0,abc,10")

    def test_bad_npoints(self):
        with pytest.raises(ArgumentTypeError, match="integer"):
            chkDespiking("3.0,0.5,abc")

    def test_float_npoints_raises(self):
        with pytest.raises(ArgumentTypeError, match="integer"):
            chkDespiking("3.0,0.5,1.5")
