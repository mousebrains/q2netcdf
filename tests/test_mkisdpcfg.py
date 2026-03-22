"""Tests for mkISDPcfg module."""

import sys

import pytest
from argparse import ArgumentTypeError
from unittest.mock import patch

from q2netcdf.mkISDPcfg import chkNotNegative, chkPositive, chkDespiking, main


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


class TestMainQuoting:
    """Tests for string quoting logic in main()."""

    def test_main_writes_config_with_instrument(self, tmp_path):
        """Test that main() writes config file with instrument string double-quoted."""
        output = tmp_path / "isdp.cfg"
        test_args = [
            "mkISDPcfg",
            "--isdpConfig",
            str(output),
            "--instrument",
            "vmp",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        content = output.read_text()
        # String values (not true/false) should be wrapped in double quotes
        assert 'instrument = "vmp"' in content

    def test_main_single_quote_fallback(self, tmp_path):
        """Test that strings with double quotes get wrapped in single quotes.

        Covers lines 284-285 in mkISDPcfg.py.
        """
        output = tmp_path / "isdp.cfg"
        from argparse import Namespace

        mock_args = Namespace(
            isdpConfig=str(output),
            instrument='has"double"quotes',
        )
        with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
            main()

        content = output.read_text()
        assert "instrument = 'has\"double\"quotes'" in content

    def test_main_both_quotes_raises_value_error(self, tmp_path):
        """Test that strings with both quote types raise ValueError.

        Covers lines 286-290 in mkISDPcfg.py.
        """
        output = tmp_path / "isdp.cfg"
        from argparse import Namespace

        mock_args = Namespace(
            isdpConfig=str(output),
            instrument="""has"double"and'single'quotes""",
        )
        with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
            with pytest.raises(ValueError):
                main()
