#
# Q-File versions
#
# Mar-2025, Pat Welch, pat@mousebrains.com

from enum import Enum

class QVersion(Enum):
    """
    Q-File format versions.

    Rockland Scientific's ISDP data logger produces Q-files in different
    format versions. This enum identifies the version for proper parsing.

    Attributes:
        v12: Version 1.2 - Documented in Rockland's TN-054
        v13: Version 1.3 - Reduced redundancy version of v1.2
    """
    v12 = 1.2 # Documented in Rockland's TN054
    v13 = 1.3 # My reduced redundancy version of v1.2

    def isV12(self) -> bool:
        """Check if this is version 1.2."""
        return self == QVersion.v12

    def isV13(self) -> bool:
        """Check if this is version 1.3."""
        return self == QVersion.v13
