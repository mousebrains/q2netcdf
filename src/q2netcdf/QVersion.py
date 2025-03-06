#
# Q-File versions
#
# Mar-2025, Pat Welch, pat@mousebrains.com

from enum import Enum

class QVersion(Enum):
    v12 = 1.2 # Documented in Rockland's TN054
    v13 = 1.3 # My reduced redundancy version of v1.2

    def isV12(self):
        return self == QVersion.v12

    def isV13(self):
        return self == QVersion.v13
