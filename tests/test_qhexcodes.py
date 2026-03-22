"""Tests for QHexCodes identifier mapping."""

import pytest

from q2netcdf.QHexCodes import QHexCodes


class TestQHexCodes:
    """Test QHexCodes identifier ↔ name mapping."""

    def test_name_lookup_basic(self):
        """Test basic identifier to name lookup."""
        # 0x610 should map to shear probe
        name = QHexCodes.name(0x610)
        assert name == "sh_0"

    def test_name_lookup_with_instance(self):
        """Test identifier with instance number."""
        # Lower 4 bits encode instance (0-15)
        name1 = QHexCodes.name(0x610)  # Instance 0
        name2 = QHexCodes.name(0x611)  # Instance 1
        name3 = QHexCodes.name(0x612)  # Instance 2

        assert name1 == "sh_0"
        assert name2 == "sh_1"
        assert name3 == "sh_2"

    def test_name_lookup_unknown_ident(self):
        """Test that unknown identifier returns None."""
        name = QHexCodes.name(0xFFFF)
        assert name is None

    def test_attributes_lookup(self):
        """Test getting attributes for identifier."""
        attrs = QHexCodes.attributes(0x610)
        assert attrs is not None
        assert "long_name" in attrs
        assert attrs["long_name"] == "shear_0"

    def test_attributes_unknown_ident(self):
        """Test that unknown identifier returns None for attributes."""
        attrs = QHexCodes.attributes(0xFFFF)
        assert attrs is None

    def test_name2ident_basic(self):
        """Test reverse lookup: name to identifier."""
        ident = QHexCodes.name2ident("sh_1")
        assert ident == 0x611

    def test_name2ident_no_instance(self):
        """Test reverse lookup for name without instance number."""
        ident = QHexCodes.name2ident("pressure")
        assert ident == 0x160

    def test_name2ident_unknown_name(self):
        """Test that unknown name returns None."""
        ident = QHexCodes.name2ident("nonexistent_sensor")
        assert ident is None

    def test_bidirectional_mapping(self):
        """Test that name ↔ ident mapping is consistent."""
        # Forward: ident → name
        name = QHexCodes.name(0x620)
        assert name is not None

        # Reverse: name → ident
        ident = QHexCodes.name2ident(name)
        assert ident == 0x620

    def test_temperature_identifier(self):
        """Test temperature sensor identifier."""
        name = QHexCodes.name(0x620)
        assert name == "T_0"

        attrs = QHexCodes.attributes(0x620)
        assert "units" in attrs
        assert attrs["units"] == "Celsius"

    def test_velocity_identifier(self):
        """Test velocity identifier mapping."""
        # 0x320 with cnt=0 returns first element
        attrs = QHexCodes.attributes(0x320)
        assert "long_name" in attrs
        assert attrs["long_name"] == "velocity_eastward"

        # Different instances return different values from the list
        attrs1 = QHexCodes.attributes(0x321)
        assert attrs1["long_name"] == "velocity_northward"

    def test_dissolved_oxygen_spelling(self):
        """Test that dissolved oxygen is spelled correctly (not disolved)."""
        attrs = QHexCodes.attributes(0x530)
        assert attrs["long_name"] == "dissolved_oxygen"

    def test_list_name_cnt_too_large_raises_value_error(self):
        """Test ValueError when cnt exceeds list-type name length.

        0x110 maps to ["A0", "Ax", "Ay", "Az"] (4 items).
        0x114 has cnt=4 which is >= len(list), so it should raise ValueError.
        """
        with pytest.raises(ValueError, match=r"cnt\(4\) >= \(4\)"):
            QHexCodes.name(0x114)

    def test_list_name_attributes_cnt_too_large_raises_value_error(self):
        """Test ValueError in attributes() when cnt exceeds list-type attr values."""
        with pytest.raises(ValueError, match=r"cnt\(4\) >= \(4\)"):
            QHexCodes.attributes(0x114)

    def test_repr(self):
        """Test __repr__ returns formatted hex map string."""
        hm = QHexCodes()
        result = repr(hm)
        assert isinstance(result, str)
        # Should contain hex codes from the map
        assert "0x0110" in result or "0x110" in result or "0x0610" in result
        # Should have multiple lines (one per hex map entry)
        lines = result.strip().split("\n")
        assert len(lines) > 10

    def test_list_name_valid_instances(self):
        """Test that valid instances of list-type names work correctly."""
        # 0x110 -> ["A0", "Ax", "Ay", "Az"]
        assert QHexCodes.name(0x110) == "A0"
        assert QHexCodes.name(0x111) == "Ax"
        assert QHexCodes.name(0x112) == "Ay"
        assert QHexCodes.name(0x113) == "Az"
