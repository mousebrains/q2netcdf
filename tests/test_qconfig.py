"""Tests for QConfig parser."""
import pytest
import numpy as np
from q2netcdf.QConfig import QConfig
from q2netcdf.QVersion import QVersion


class TestQConfig:
    """Test QConfig configuration parsing."""

    def test_parse_v13_json_config(self):
        """Test parsing v1.3 JSON configuration."""
        config_str = b'{"foo": 42, "bar": 3.14, "baz": "test"}'
        qconfig = QConfig(config_str, QVersion.v13)
        parsed = qconfig.config()

        assert parsed["foo"] == 42
        assert parsed["bar"] == 3.14
        assert parsed["baz"] == "test"

    def test_parse_v12_perl_config(self):
        """Test parsing v1.2 Perl-style configuration."""
        config_str = b'"int_val" => 42\n"float_val" => 3.14\n"str_val" => "hello"'
        qconfig = QConfig(config_str, QVersion.v12)
        parsed = qconfig.config()

        assert parsed["int_val"] == 42
        assert parsed["float_val"] == 3.14
        assert parsed["str_val"] == "hello"

    def test_parse_v12_array(self):
        """Test parsing arrays in v1.2 config."""
        config_str = b'"array" => [1, 2, 3]'
        qconfig = QConfig(config_str, QVersion.v12)
        parsed = qconfig.config()

        assert isinstance(parsed["array"], np.ndarray)
        assert len(parsed["array"]) == 3
        assert parsed["array"][0] == 1

    def test_config_size(self):
        """Test config size reporting."""
        config_str = b'{"test": 123}'
        qconfig = QConfig(config_str, QVersion.v13)
        assert len(qconfig) == len(config_str)
        assert qconfig.size() == len(config_str)

    def test_raw_config(self):
        """Test raw config retrieval."""
        config_str = b'{"test": 123}'
        qconfig = QConfig(config_str, QVersion.v13)
        assert qconfig.raw() == config_str

    def test_parse_v12_boolean(self):
        """Test parsing boolean values in v1.2 config."""
        config_str = b'"enabled" => true\n"disabled" => false'
        qconfig = QConfig(config_str, QVersion.v12)
        parsed = qconfig.config()

        assert parsed["enabled"] is True
        assert parsed["disabled"] is False

    def test_parse_invalid_v13_json(self):
        """Test handling of invalid JSON in v1.3 config."""
        config_str = b'{invalid json}'
        qconfig = QConfig(config_str, QVersion.v13)

        with pytest.raises(Exception):
            qconfig.config()

    def test_empty_config(self):
        """Test handling of empty configuration."""
        config_str = b''
        qconfig = QConfig(config_str, QVersion.v13)
        assert len(qconfig) == 0

    def test_parse_v12_nested_array(self):
        """Test parsing nested arrays in v1.2 config."""
        config_str = b'"matrix" => [[1, 2], [3, 4]]'
        qconfig = QConfig(config_str, QVersion.v12)
        parsed = qconfig.config()

        assert isinstance(parsed["matrix"], np.ndarray)
        # Should be flattened or nested depending on implementation
        assert len(parsed["matrix"]) >= 2

    def test_parse_v12_float_with_exponent(self):
        """Test parsing scientific notation floats."""
        config_str = b'"scientific" => 1.5E-5'
        qconfig = QConfig(config_str, QVersion.v12)
        parsed = qconfig.config()

        assert isinstance(parsed["scientific"], float)
        assert abs(parsed["scientific"] - 1.5e-5) < 1e-10

    def test_parse_v12_negative_numbers(self):
        """Test parsing negative integers and floats."""
        config_str = b'"neg_int" => -42\n"neg_float" => -3.14'
        qconfig = QConfig(config_str, QVersion.v12)
        parsed = qconfig.config()

        assert parsed["neg_int"] == -42
        assert parsed["neg_float"] == -3.14

    def test_parse_v12_quoted_string(self):
        """Test parsing quoted strings with special characters."""
        config_str = b'"message" => "Hello, World!"'
        qconfig = QConfig(config_str, QVersion.v12)
        parsed = qconfig.config()

        assert parsed["message"] == "Hello, World!"

    def test_parse_v12_empty_array(self):
        """Test parsing empty array."""
        config_str = b'"empty" => []'
        qconfig = QConfig(config_str, QVersion.v12)
        parsed = qconfig.config()

        assert isinstance(parsed["empty"], np.ndarray)
        assert len(parsed["empty"]) == 0

    def test_parse_v13_complex_structure(self):
        """Test parsing complex nested JSON structure."""
        config_str = b'{"outer": {"inner": 123}, "list": [1, 2, 3]}'
        qconfig = QConfig(config_str, QVersion.v13)
        parsed = qconfig.config()

        assert "outer" in parsed
        assert "list" in parsed
        assert isinstance(parsed["list"], list)

    def test_parse_v12_multiple_fields(self):
        """Test parsing multiple fields in v1.2 config."""
        config_str = (
            b'"fft_length" => 4\n'
            b'"diss_length" => 32\n'
            b'"f_aa" => 98\n'
            b'"hp_cut" => 0.125\n'
            b'"algorithm" => "glide"'
        )
        qconfig = QConfig(config_str, QVersion.v12)
        parsed = qconfig.config()

        assert parsed["fft_length"] == 4
        assert parsed["diss_length"] == 32
        assert parsed["f_aa"] == 98
        assert parsed["hp_cut"] == 0.125
        assert parsed["algorithm"] == "glide"

    def test_parse_v12_array_with_floats(self):
        """Test parsing array containing floats."""
        config_str = b'"despiking" => [8.0, 0.25, 0.04]'
        qconfig = QConfig(config_str, QVersion.v12)
        parsed = qconfig.config()

        assert isinstance(parsed["despiking"], np.ndarray)
        assert len(parsed["despiking"]) == 3
        assert abs(parsed["despiking"][1] - 0.25) < 1e-10

    def test_repr_output(self):
        """Test string representation of QConfig."""
        config_str = b'{"test": 123, "foo": "bar"}'
        qconfig = QConfig(config_str, QVersion.v13)
        repr_str = repr(qconfig)

        assert "test" in repr_str
        assert "123" in repr_str
        assert "foo" in repr_str

    def test_parse_v12_whitespace_handling(self):
        """Test handling of extra whitespace."""
        config_str = b'"key"  =>  42  '
        qconfig = QConfig(config_str, QVersion.v12)
        parsed = qconfig.config()

        assert parsed["key"] == 42

    def test_parse_v13_unicode_string(self):
        """Test parsing unicode strings in v1.3."""
        config_str = '{"name": "测试"}'.encode('utf-8')
        qconfig = QConfig(config_str, QVersion.v13)
        parsed = qconfig.config()

        assert parsed["name"] == "测试"

    def test_parse_v12_mixed_types_array(self):
        """Test parsing array with mixed types."""
        config_str = b'"mixed" => [1, 2.5, 3]'
        qconfig = QConfig(config_str, QVersion.v12)
        parsed = qconfig.config()

        assert isinstance(parsed["mixed"], np.ndarray)
        # All should be converted to same type
        assert len(parsed["mixed"]) == 3
