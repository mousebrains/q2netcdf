#
# Decode QFile config record into a dictionary
#
# Feb-2025, Pat Welch, pat@mousebrains.com

import re
import numpy as np
import json
from typing import Union, Optional
from .QVersion import QVersion

class QConfig:
    """
    Parser for Q-file configuration records.

    Configuration records contain key-value pairs with various data types
    including integers, floats, strings, booleans, and arrays.
    """

    # Cached compiled regex patterns for performance (class-level constants)
    _PATTERN_ARRAY = re.compile(r"^\[(.*)\]$")
    _PATTERN_INT = re.compile(r"^[+-]?\d+$")
    _PATTERN_FLOAT = re.compile(r"^[+-]?\d+[.]\d*(|[Ee][+-]?\d+)$")
    _PATTERN_STRING = re.compile(r'^"(.*)"$')
    _PATTERN_TRUE = re.compile(r"^true$")
    _PATTERN_FALSE = re.compile(r"^false$")
    _PATTERN_V12_LINE = re.compile(r"^\"(.*)\"\s*=>\s*(.*)$")

    def __init__(self, config: bytes, version: QVersion) -> None:
        self.__config = config
        self.__version = version
        self.__dict: Optional[dict] = None

    def __repr__(self) -> str:
        config = self.config()
        msg = []
        for key in sorted(config):
            msg.append(f"{key} -> {config[key]}")
        return "\n".join(msg)

    def __parseValue(self, val: str) -> Union[int, float, str, bool, np.ndarray]:
        # Use cached compiled patterns for better performance
        matches = self._PATTERN_ARRAY.match(val)
        if matches:
            # Handle empty arrays
            content = matches[1].strip()
            if not content:
                return np.array([])
            fields = []
            for field in content.split(","):
                fields.append(self.__parseValue(field.strip()))
            return np.array(fields)

        matches = self._PATTERN_INT.match(val)
        if matches:
            return int(val)

        matches = self._PATTERN_FLOAT.match(val)
        if matches:
            return float(val)

        matches = self._PATTERN_STRING.match(val)
        if matches:
            return matches[1]

        matches = self._PATTERN_TRUE.match(val)
        if matches:
            return True

        matches = self._PATTERN_FALSE.match(val)
        if matches:
            return False

        return val

    def __splitConfigV12(self) -> None:
        self.__dict = dict()
        for line in self.__config.split(b"\n"):
            try:
                line_str = line.decode("utf-8").strip()
                matches = self._PATTERN_V12_LINE.match(line_str)
                if matches:
                    self.__dict[matches[1]] = self.__parseValue(matches[2])
            except (UnicodeDecodeError, ValueError):
                pass

    def __splitConfigv13(self) -> None:
        self.__dict = json.loads(self.__config)

    def __len__(self) -> int:
        return len(self.__config)

    def size(self) -> int:
        return len(self)

    def raw(self) -> bytes:
        return self.__config

    def config(self) -> dict:
        if self.__dict is None:
            if self.__version.isV12():
                self.__splitConfigV12()
            else:
                self.__splitConfigv13()
        assert self.__dict is not None  # After split methods, dict is populated
        return self.__dict
