#
# Q-File record type identifiers
#
# Mar-2025, Pat Welch, pat@mousebrains.com

from enum import Enum


class RecordType(Enum):
    """
    Q-file binary record type identifiers.

    Q-files contain different types of records, each identified by a
    16-bit hexadecimal value at the start of the record.

    Attributes:
        HEADER: Header record (0x1729) - Contains file metadata, sensor list
        CONFIG_V12: Configuration record for v1.2 (0x0827) - Beta version only
        DATA: Data record (0x1657) - Contains measurements and spectra
    """

    HEADER = 0x1729  # Header record with channels, spectra, frequencies
    CONFIG_V12 = 0x0827  # v1.2 configuration record identifier (beta)
    DATA = 0x1657  # Data record with measurements
