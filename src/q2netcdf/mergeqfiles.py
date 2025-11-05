#! /usr/bin/env python3
#
# This is a rewrite of Rockland's mergeqfiles for use by TWR's Slocum uRider proglet.
#
# The stock script will return a zero length file if the maximum size allowed is smaller
# than the size of a q-file.
#
# This script has several modes:
#   if no Q-file reduction is requested it finds all Q-files
#            modified in the specified interval plus a safety margin
#      if their total size is less than the maximum allowed size, they are merged together
#      else they are decimated to reach the maximum allowed size
#
#   if Q-file reduction is requested, the Q-files reduced sizes are estimated and then
#            we follow the above prescription in terms of decimation
#
# The internal Q-file structure is based on Rockland's TN 054
# The reduced Q-file format is a modified version of TN 054
#
# Oct-2024, Pat Welch, pat@mousebrains.com
# Feb-2025, Pat Welch, pat@mousebrains.com, update module usage
# Mar-2025, Pat Welch, pat@mousebrains.com, Reduce Q-file contents

from argparse import ArgumentParser, Namespace
import os
import sys
import time
import logging
import math
import json
import struct
import re
from enum import Enum
from typing import Any, Optional, Union, Tuple, Dict, List, IO
import numpy as np


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


class QVersion(Enum):
    """
    Q-File format versions.

    Rockland Scientific's ISDP data logger produces Q-files in different
    format versions. This enum identifies the version for proper parsing.

    Attributes:
        v12: Version 1.2 - Documented in Rockland's TN-054
        v13: Version 1.3 - Reduced redundancy version of v1.2
    """

    v12 = 1.2  # Documented in Rockland's TN054
    v13 = 1.3  # My reduced redundancy version of v1.2

    def isV12(self) -> bool:
        """Check if this is version 1.2."""
        return self == QVersion.v12

    def isV13(self) -> bool:
        """Check if this is version 1.3."""
        return self == QVersion.v13


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
        self.__dict: Optional[Dict[str, Any]] = None

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

    def config(self) -> Dict[str, Any]:
        if self.__dict is None:
            if self.__version.isV12():
                self.__splitConfigV12()
            else:
                self.__splitConfigv13()
        assert self.__dict is not None  # After split methods, dict is populated
        return self.__dict


class QHexCodes:
    """
    Mapping between Q-file hex identifiers and sensor/spectra names.

    Q-files use hexadecimal identifiers to label channels (scalar measurements)
    and spectra (frequency-domain data). This class provides bidirectional
    mapping between identifiers and human-readable names with attributes.

    The identifier scheme uses:
    - Upper 12 bits (0xFFF0): Sensor/spectra type
    - Lower 4 bits (0x000F): Instance number (0-15)

    Example:
        0x610 -> "sh_1" (shear probe #1)
        0x611 -> "sh_2" (shear probe #2)
    """

    __hexMap = {
        0x010: [
            "dT_",
            {
                "long_name": "preThermal_",
            },
        ],
        0x020: [
            "dC_",
            {
                "long_name": "preUConductivity_",
            },
        ],
        0x030: [
            "P_dP",
            {
                "long_name": "prePressure",
            },
        ],
        0x110: [
            ["A0", "Ax", "Ay", "Az"],
            {
                "long_name": [
                    "acceleration_0",
                    "acceleration_X",
                    "acceleration_Y",
                    "acceleration_Z",
                ],
            },
        ],
        0x120: [
            ["A0", "Ax", "Ay"],
            {
                "long_name": [
                    "piezo_0",
                    "piezo_X",
                    "piezo_Y",
                ],
            },
        ],
        0x130: [
            ["Incl_0", "Incl_X", "Incl_Y", "Incl_T"],
            {
                "long_name": [
                    "Inclinometer_0",
                    "Inclinometer_X",
                    "Inclinometer_Y",
                    "Inclinometer_T",
                ],
                "units": ["degrees", "degrees", "Celsius"],
            },
        ],
        0x140: [
            ["theta_0", "thetaX", "thetaY"],
            {
                "long_name": ["Theta_0", "Theta_X", "Theta_Y"],
                "units": "degrees",
            },
        ],
        0x150: [
            ["M_0", "Mx", "My", "Mz"],
            {
                "long_name": [
                    "magnetic_0",
                    "magnetic_X",
                    "magnetic_Y",
                    "magnetic_Z",
                ],
            },
        ],
        0x160: [
            "pressure",
            {
                "long_name": "pressure_ocean",
                "units": "decibar",
            },
        ],
        0x170: [
            "AOA",
            {
                "long_name": "angle_of_attack",
                "units": "degrees",
            },
        ],
        0x210: [
            "VBat",
            {
                "long_name": "battery",
                "units": "Volts",
            },
        ],
        0x220: [
            "PV",
            {
                "long_name": "pressure_transducer",
                "units": "Volts",
            },
        ],
        0x230: [
            "EMCur",
            {
                "long_name": "EM_current",
                "units": "Amps",
            },
        ],
        0x240: [
            ["latitude", "longitude"],
            {
                "long_name": ["Latitude", "Longitude"],
                "units": ["degrees North", "degrees East"],
            },
        ],
        0x250: [
            "noise",
            {
                "long_name": "glider_noise",
            },
        ],
        0x310: [
            "EM",
            {
                "long_name": "speed",
                "units": "meters/second",
            },
        ],
        0x320: [
            ["U", "V", "W", "speed_squared"],
            {
                "long_name": [
                    "velocity_eastward",
                    "velocity_northward",
                    "velocity_upwards",
                    "velocity_squared",
                ],
                "units": [
                    "meters/second",
                    "meters/second",
                    "meters/second",
                    "meters^2/second^2",
                ],
            },
        ],
        0x330: [
            "dzdt",
            {
                "long_name": "fallRate",
                "units": "meters/second",
            },
        ],
        0x340: [
            "dzdt_adj",
            {
                "long_name": "fallRate_adjusted_for_AOA",
                "units": "meters/second",
            },
        ],
        0x350: [
            "speed_hotel",
            {
                "long_name": "speed_hotel",
                "units": "meters/second",
            },
        ],
        0x360: [
            "speed",
            {
                "long_name": "speed_computation",
                "units": "meters/second",
            },
        ],
        0x410: [
            [
                "temperature_JAC",
                "temperature_SB",
                "temperature_RBR",
                "temperature_Hotel",
                "temperature_Contant",
            ],
            {
                "long_name": "temperature",
                "units": "Celsius",
            },
        ],
        0x420: [
            [
                "conductivity_JAC",
                "conductivity_SB",
                "conductivity_RBR",
                "conductivity_Hotel",
                "conductivity_Constant",
            ],
            {
                "long_name": "conductivity",
            },
        ],
        0x430: [
            [
                "salinity_JAC",
                "salinity_SB",
                "salinity_RBR",
                "salinity_Hotel",
                "salinity_Constant",
            ],
            {
                "long_name": "salinity",
                "units": "PSU",
            },
        ],
        0x440: [
            "sigma0",
            {
                "long_name": "sigma_0",
                "units": "kilogram/meter^3",
            },
        ],
        0x450: [
            "visc",
            {
                "long_name": "viscosity",
                "units": "meter^2/second",
            },
        ],
        0x510: [
            "chlor",
            {
                "long_name": "chlorophyll",
            },
        ],
        0x520: [
            "turb",
            {
                "long_name": "turbidity",
            },
        ],
        0x530: [
            "DO",
            {
                "long_name": "dissolved_oxygen",
            },
        ],
        0x610: [
            "sh_",
            {
                "long_name": "shear_",
            },
        ],
        0x620: [
            "T_",
            {
                "long_name": "temperature_",
                "units": "Celsius",
            },
        ],
        0x630: [
            "C_",
            {
                "long_name": "microConductivity_",
            },
        ],
        0x640: [
            "dT_",
            {
                "long_name": "gradient_temperature_",
                "units": "Celsius/meter",
            },
        ],
        0x650: [
            "dC_",
            {
                "long_name": "gradient_conductivity_",
            },
        ],
        0x710: [
            "sh_GTD_",
            {
                "long_name": "shear_goodman_",
            },
        ],
        0x720: [
            "sh_DSP_",
            {
                "long_name": "shear_despiked_",
            },
        ],
        0x730: [
            "uCond_DSP_",
            {
                "long_name": "microConductivity_despiked_",
            },
        ],
        0x740: [
            "sh_fraction_",
            {
                "long_name": "shear_fraction_",
            },
        ],
        0x750: [
            "sh_passes_",
            {
                "long_name": "shear_passes_",
            },
        ],
        0x760: [
            "uCond_fraction_",
            {
                "long_name": "microConductivity_fraction_",
            },
        ],
        0x770: [
            "uCond_passes_",
            {
                "long_name": "microConductivity_passes_",
            },
        ],
        0x810: [
            "K_max_",
            {
                "long_name": "integration_limit_",
            },
        ],
        0x820: [
            "var_res_",
            {
                "long_name": "variance_resolved_",
            },
        ],
        0x830: [
            "MAD_",
            {
                "long_name": "mean_averaged_deviation_",
            },
        ],
        0x840: [
            "FM_",
            {
                "long_name": "figure_of_merit_",
            },
        ],
        0x850: [
            "CI_",
            {
                "long_name": "confidence_interval_",
            },
        ],
        0x860: [
            "MAD_T_",
            {
                "long_name": "mean_average_deviation_temperature_",
            },
        ],
        0x870: [
            "QC_",
            {
                "long_name": "quality_control_flags_",
            },
        ],
        0x910: [
            "freq",
            {
                "long_name": "frequency",
            },
        ],
        0x920: [
            "shear_raw",
            {
                "long_name": "shear_raw",
            },
        ],
        0x930: [
            "shear_gfd_",
            {
                "long_name": "shear_goodman_",
            },
        ],
        0x940: [
            "gradT_raw",
            {
                "long_name": "thermistor_raw",
            },
        ],
        0x950: [
            "gradT_gfd_",
            {
                "long_name": "thermistor_goodman_",
            },
        ],
        0x960: [
            "uCond_raw",
            {
                "long_name": "microConductivity_raw",
            },
        ],
        0x970: [
            "uCond_gfd_",
            {
                "long_name": "microConductivity_goodman_",
            },
        ],
        0x980: [
            "piezo",
            {
                "long_name": "vibration",
            },
        ],
        0x990: [
            "accel",
            {
                "long_name": "accelerometer",
            },
        ],
        0x9A0: [
            "T_ref",
            {
                "long_name": "temperature_reference",
            },
        ],
        0x9B0: [
            "T_noise",
            {
                "long_name": "temperature_noise",
            },
        ],
        0xA10: [
            "e_",
            {
                "long_name": "epsilon_",
            },
        ],
        0xA20: [
            "N2",
            {
                "long_name": "buoyancy_frequency",
            },
        ],
        0xA30: [
            "eddy_diff",
            {
                "long_name": "eddy_diffusivity",
            },
        ],
        0xA40: [
            "chi_",
            {
                "long_name": "chi_",
            },
        ],
        0xA50: [
            "e_T_",
            {
                "long_name": "thermal_dissipation_",
            },
        ],
        0xD20: [
            "diagnostic_",
            {},
        ],  # Value that shouldn't be here
    }

    def __init__(self) -> None:
        pass

    @classmethod
    def __repr__(cls) -> str:
        msg = []
        for key in sorted(cls.__hexMap):
            msg.append(f"{key:#05x} {cls.__hexMap[key]}")
        return "\n".join(msg)

    @staticmethod
    def __fixName(name: Union[str, List, Tuple], cnt: int) -> str:
        if isinstance(name, str):
            if not name.endswith("_"):
                return name
            cnt = cnt  # 0-15 -> 1-16
            return f"{name}{cnt}"

        if isinstance(name, (list, tuple)):
            if len(name) > cnt:
                return name[cnt]
            raise ValueError(f"cnt({cnt}) >= ({len(name)}) names <-  {name}")

        raise NotImplementedError(f"Unsupported name type, {type(name)} <- {name}")

    @classmethod
    def __findIdent(cls, ident: int) -> Tuple[Optional[List], Optional[int]]:
        key = ident & 0xFFF0
        cnt = ident & 0x0F
        if key in cls.__hexMap:
            return (cls.__hexMap[key], cnt)

        logging.warning(f"{key:#06x} not in map, ident {ident:#06x}")
        return (None, None)

    @classmethod
    def name(cls, ident: int) -> Optional[str]:
        """
        Get the name for a given identifier.

        Args:
            ident: Hexadecimal identifier (e.g., 0x610)

        Returns:
            Human-readable name (e.g., "sh_0") or None if not found
        """
        (item, cnt) = cls.__findIdent(ident)
        if item is None:
            return None

        assert cnt is not None  # __findIdent returns both None or both non-None
        name = item[0]
        return cls.__fixName(name, cnt)

    @classmethod
    def attributes(cls, ident: int) -> Optional[Dict[str, Any]]:
        """
        Get the metadata attributes for a given identifier.

        Args:
            ident: Hexadecimal identifier

        Returns:
            Dictionary with long_name, units, etc., or None if not found
        """
        (item, cnt) = cls.__findIdent(ident)
        if item is None:
            return None

        assert cnt is not None  # __findIdent returns both None or both non-None
        attrs = item[1].copy()  # In case I modify it

        for attr in attrs:
            attrs[attr] = cls.__fixName(attrs[attr], cnt)

        return attrs

    @classmethod
    def name2ident(cls, name: str) -> Optional[int]:
        """
        Convert a name to its hexadecimal identifier (reverse lookup).

        Args:
            name: Human-readable name (e.g., "sh_1")

        Returns:
            Hexadecimal identifier (e.g., 0x611) or None if not found
        """
        matches = re.match(r"^(.*_)(\d+)$", name)
        if matches:
            prefix = matches[1]
            cnt = int(matches[2])
        else:
            prefix = name
            cnt = 0

        for ident in cls.__hexMap:
            if cls.__hexMap[ident][0] == prefix:
                return ident + cnt
        logging.warning(f"{name} not found in hexMap")
        return None


class QHeader:
    """
    Parser for Rockland Scientific Q-file header records.

    A header record contains:
    - File version
    - Start time
    - Channel identifiers (scalar measurements)
    - Spectra identifiers (frequency-domain data)
    - Frequency array
    - Configuration dictionary

    See Rockland Technical Note TN-054 for format specification.
    """

    @classmethod
    def chkIdent(cls, fp: IO[bytes]) -> Optional[bool]:
        n = 2
        buffer = fp.read(n)
        if len(buffer) != n:
            return None
        (ident,) = struct.unpack("<H", buffer)
        fp.seek(-n, 1)  # Back up n bytes
        return ident == RecordType.HEADER.value

    def __init__(self, fp: IO[bytes], fn: str) -> None:
        self.filename = fn
        hdrSize = 0

        buffer = fp.read(20)  # Read fixed header
        n = len(buffer)
        if n != 20:
            raise EOFError(f"EOF in fixed header, {n} != 20, in {fn}")
        hdrSize += n

        (ident, version, dt, self.Nc, self.Ns, self.Nf) = struct.unpack(
            "<HfQHHH", buffer
        )

        if ident != RecordType.HEADER.value:
            raise ValueError(
                f"Invalid header identifer, {ident:#05x} != {RecordType.HEADER.value:#05x}, in {fn}"
            )

        self.version = None
        for v in QVersion:
            if abs(version - v.value) < 0.0001:
                self.version = v
                break
        if self.version is None:
            raise NotImplementedError(f"Invalid version, {version}, in {fn}")

        self.dtBinary = dt
        self.time = np.datetime64("0000-01-01") + np.timedelta64(dt, "ms")

        self.channels: Tuple[int, ...] = ()
        self.spectra: Tuple[int, ...] = ()

        if self.Nc:  # Some channel identifiers to read
            sz = self.Nc * 2
            buffer = fp.read(sz)  # Get channel identifiers
            n = len(buffer)
            if n != sz:
                raise EOFError(
                    f"EOF while reading channel identifiers, {n} != {sz}, in {fn}"
                )
            hdrSize += n
            self.channels = struct.unpack("<" + ("H" * self.Nc), buffer)

        if self.Ns:  # Some spectra identifiers to read
            sz = self.Ns * 2
            buffer = fp.read(sz)  # Get spectra identifiers
            n = len(buffer)
            if n != sz:
                raise EOFError(
                    f"EOF while reading spectra identifiers, {n} != {sz}, in {fn}"
                )
            hdrSize += n
            self.spectra = struct.unpack("<" + ("H" * self.Ns), buffer)

        if self.Nf:  # Some frequencies to read
            sz = self.Nf * 2
            buffer = fp.read(sz)  # Get spectra identifiers
            n = len(buffer)
            if n != sz:
                raise EOFError(f"EOF while reading frequencies, {n} != {sz}, in {fn}")
            hdrSize += n
            self.frequencies = struct.unpack("<" + ("e" * self.Nf), buffer)

        cfgHdrSz = 4 if self.version == QVersion.v12 else 2
        buffer = fp.read(cfgHdrSz)  # Grab configuration record's fixed fields
        n = len(buffer)
        if n != cfgHdrSz:
            raise EOFError(
                f"EOF while reading fixed configuration record, {len(buffer)} != {cfgHdrSz} in {fn}"
            )
        hdrSize += n

        if self.version == QVersion.v12:
            # cfgIdent is bad in the beta version of 1.2, should be RecordType.CONFIG_V12
            (cfgIdent, sz) = struct.unpack("<HH", buffer)
        else:
            (sz,) = struct.unpack("<H", buffer)  # No config ident

        if sz:  # Some configuration record to read
            buffer = fp.read(sz)
            n = len(buffer)
            if n != sz:
                raise EOFError(
                    f"EOF while reading configuration record, {n} != {sz}, in {fn}"
                )
            hdrSize += n
            self.config = QConfig(buffer, self.version)

        buffer = fp.read(2)  # Get data record size
        n = len(buffer)
        if n != 2:
            raise EOFError(f"EOF while reading data record size, {n} != 2, in {fn}")
        hdrSize += n

        self.dataSize = struct.unpack("<H", buffer)[0]
        self.hdrSize = hdrSize

    def __repr__(self) -> str:
        msg = []
        msg.append(f"filename:    {self.filename}")
        msg.append(f"Version:     {self.version}")
        msg.append(f"Time:        {self.time}")
        msg.append(f"Channels:    {self.channels}")
        msg.append(f"Spectra:     {self.spectra}")
        msg.append(f"Frequencies: {self.frequencies}")
        msg.append(f"Data Size:   {self.dataSize}")
        msg.append(f"Header Size: {self.hdrSize}")
        msg.append(f"Config:      {self.config}")
        return "\n".join(msg)


class QReduce:
    """
    Reduce Q-file size by pruning channels, spectra, and config fields.

    This class creates a reduced version of a Q-file containing only
    specified channels, spectra, and configuration parameters. Useful
    for reducing file size when only certain measurements are needed.

    The reduced file uses v1.3 format regardless of input version.
    """

    __name2ident: Dict[str, int] = {}

    def __init__(self, filename: str, config: dict) -> None:
        self.filename = filename

        channelIdents = self.__updateName2Ident(
            config, "channels"
        )  # Get intersecting idents
        spectraIdents = self.__updateName2Ident(config, "spectra")

        with open(filename, "rb") as fp:
            hdr = QHeader(fp, filename)
            self.fileSizeOrig = os.fstat(fp.fileno()).st_size

        # Convert tuples to ndarrays for processing
        channelArray = np.array(hdr.channels, dtype="uint16")
        spectraArray = np.array(hdr.spectra, dtype="uint16")

        (channelIdents, channelIndices) = self.__findIndices(
            channelIdents, channelArray
        )
        (spectraIdents, spectraIndices) = self.__findIndices(
            spectraIdents, spectraArray
        )

        qFreq = spectraIndices.size != 0
        if qFreq:  # Some spectra, so build full indices
            spectraIndices = self.__spectraIndices(hdr, spectraIndices)
            allIndices = np.concatenate((channelIndices, spectraIndices))
        else:
            allIndices = channelIndices

        body = struct.pack("<Hf", RecordType.HEADER.value, QVersion.v13.value)
        body += np.array(hdr.dtBinary, dtype="uint64").tobytes()
        body += struct.pack(
            "<HHH", len(channelIdents), len(spectraIdents), hdr.Nf if qFreq else 0
        )
        body += channelIdents.astype("<u2").tobytes()
        if qFreq:
            body += spectraIdents.astype("<u2").tobytes()
            body += np.array(hdr.frequencies).astype("<f2").tobytes()

        myConfig: Dict[str, Any] = {}
        if isinstance(config, dict) and "config" in config:
            hdrConfig = hdr.config.config()
            for name in config["config"]:
                if name in hdrConfig:
                    myConfig[name] = hdrConfig[name]

        myConfigStr: str
        if myConfig:
            myConfigStr = json.dumps(myConfig, separators=(",", ":"))
        else:
            myConfigStr = ""

        body += struct.pack("<H", len(myConfigStr))
        body += myConfigStr.encode("utf-8")

        self.dataSize = 4 + 2 * allIndices.size
        body += struct.pack("<H", self.dataSize)

        self.__header = body
        self.__indices = allIndices
        self.hdrSize = len(self.__header)
        self.hdrSizeOrig = hdr.hdrSize
        self.dataSizeOrig = hdr.dataSize

        self.nRecords = np.floor((self.fileSizeOrig - hdr.hdrSize) / hdr.dataSize)
        self.fileSize = self.hdrSize + self.nRecords * self.dataSize

    def __repr__(self) -> str:
        msgs = [
            f"fn {self.filename}",
            f"hdr {self.hdrSizeOrig} -> {self.hdrSize}",
            f"data {self.dataSizeOrig} -> {self.dataSize}",
            f"file {self.fileSizeOrig} -> {self.fileSize}",
        ]
        return ", ".join(msgs)

    @classmethod
    def loadConfig(cls, filename: str) -> Optional[Dict[str, Any]]:
        if os.path.isfile(filename):
            try:
                with open(filename, "r") as fp:
                    config = json.load(fp)
                    if not isinstance(config, dict):
                        logging.error(f"{filename} config {config} is not a dictionary")
                        return None
                    for key in ["config", "channels", "spectra"]:
                        if key not in config:
                            logging.error(f"{key} is not in {filename}, {config}")
                            return None
                    cls.__updateName2Ident(config, "channels")
                    cls.__updateName2Ident(config, "spectra")
                    return config
            except Exception:
                logging.exception(f"Loading {filename}")
                return None
        else:
            return None

    @staticmethod
    def __spectraIndices(hdr: QHeader, indices: np.ndarray) -> np.ndarray:
        Nf = hdr.Nf  # Number of frequencies
        indices = indices.reshape(-1, 1)
        freq = np.arange(Nf, dtype="uint16").reshape(1, -1)
        indices = hdr.Nc + (indices * Nf + freq)
        return indices.flatten()

    @staticmethod
    def __findIndices(
        idents: Optional[np.ndarray], known: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if idents is None:
            return (np.array([], dtype=int), np.array([], dtype=int))

        (idents, iLHS, iRHS) = np.intersect1d(idents, known, return_indices=True)
        ix = iRHS.argsort()
        return (idents[ix], iRHS[ix])

    @classmethod
    def __updateName2Ident(
        cls, config: Dict[str, Any], key: str
    ) -> Optional[np.ndarray]:
        if not isinstance(config, dict):
            return None
        if key not in config or not isinstance(config[key], list):
            return None

        idents = []
        for name in config[key]:
            ident: Optional[int]
            if name in cls.__name2ident:
                ident = cls.__name2ident[name]
            else:
                ident = QHexCodes.name2ident(name)
                if ident is None:
                    logging.warning(f"Unknown name({key}) to ident({name})")
                else:
                    cls.__name2ident[name] = ident
            if ident is not None:
                idents.append(ident)

        return np.array(idents, dtype="uint16")

    def __reduceRecord(self, buffer: bytes) -> Optional[bytes]:
        if len(buffer) != self.dataSizeOrig:
            return None

        record = buffer[:2] + buffer[12:14]  # Ident + stime
        data = np.frombuffer(buffer, dtype="<f2", offset=16)
        data = data[self.__indices]
        record += data.tobytes()
        return record

    def reduceFile(self, ofp: IO[bytes]) -> int:
        """
        Write reduced Q-file to output file pointer.

        Args:
            ofp: Output file pointer opened in binary write mode

        Returns:
            Total bytes written
        """
        with open(self.filename, "rb") as ifp:
            ifp.seek(self.hdrSizeOrig)  # Skip the header
            totSize = ofp.write(self.__header)
            while True:
                data = ifp.read(self.dataSizeOrig)
                if not data:
                    break
                record = self.__reduceRecord(data)
                if record is not None:
                    totSize += ofp.write(record)
            return totSize

    def decimate(self, ofp: IO[bytes], indices: np.ndarray) -> int:
        """
        Write decimated Q-file records to output file pointer.

        Args:
            ofp: Output file pointer opened in binary write mode
            indices: Array of record indices to include

        Returns:
            Total bytes written
        """
        with open(self.filename, "rb") as ifp:
            ifp.seek(self.hdrSizeOrig)  # Skip the header
            totSize = ofp.write(self.__header)
            for index in indices:
                ifp.seek(self.hdrSizeOrig + index * self.dataSizeOrig)
                data = ifp.read(self.dataSizeOrig)
                record = self.__reduceRecord(data)
                if record is not None:
                    totSize += ofp.write(record)
            return totSize


def __chkExists(filename: str) -> str:
    """Validate that file exists for argparse."""
    from argparse import ArgumentTypeError

    if os.path.isfile(filename):
        return filename
    raise ArgumentTypeError(f"{filename} does not exist")


def reduceAndDecimate(
    info: Dict[str, QReduce], ofp: IO[bytes], ofn: str, maxSize: int
) -> int:
    """
    Reduce and decimate Q-files to fit within maximum size.

    Args:
        info: Dictionary mapping filenames to QReduce objects
        ofp: Output file pointer
        ofn: Output filename
        maxSize: Maximum output file size in bytes

    Returns:
        Total bytes written
    """
    totHdrSize = 0
    totDataSize = 0
    for fn in info:
        qr = info[fn]
        totHdrSize += qr.hdrSize
        totDataSize += qr.fileSize - qr.hdrSize

    availSize = maxSize - totHdrSize
    ratio = availSize / totDataSize
    if ratio <= 0:
        logging.warning("Not adding to %s since ratio is %s <= 0", ofn, ratio)
        if ofp.seekable():
            return ofp.tell()
        st = os.fstat(ofp.fileno())
        return st.st_size

    logging.info("Sizes max %s avail %s ratio %s", maxSize, availSize, ratio)

    for fn in sorted(info):
        qr = info[fn]
        indices = np.unique(
            np.floor(
                np.linspace(
                    0,
                    qr.nRecords,
                    np.floor(qr.nRecords * ratio).astype(int),
                    endpoint=False,
                )
            ).astype(int)
        )
        sz = qr.decimate(ofp, indices)
        logging.info(
            "Decimated %s to %s -> %s -> %s n=%s/%s",
            qr.filename,
            ofn,
            qr.fileSizeOrig,
            sz,
            indices.size,
            qr.nRecords.astype(int),
        )
    if ofp.seekable():
        return ofp.tell()
    st = os.fstat(ofp.fileno())
    return st.st_size


def reduceFiles(
    qFiles: Dict[str, int], fnConfig: str, ofn: str, maxSize: int
) -> Optional[int]:
    """
    Reduce Q-files using configuration and write to output.

    Args:
        qFiles: Dictionary of Q-files to reduce
        fnConfig: Path to JSON configuration file
        ofn: Output filename
        maxSize: Maximum output file size in bytes

    Returns:
        Total bytes written or None if config loading failed
    """
    qrConfig = QReduce.loadConfig(fnConfig)
    logging.info("Config %s -> %s", fnConfig, qrConfig)
    if qrConfig is None:
        return None

    info = {}
    totSize = 0

    for fn in qFiles:
        qr = QReduce(fn, qrConfig)
        totSize += qr.fileSize
        info[fn] = qr

    with open(ofn, "ab") as ofp:
        if totSize <= maxSize:  # no need to decimate, so append glued reduced files
            for fn in sorted(info):
                qr = info[fn]
                sz = qr.reduceFile(ofp)
                logging.info(
                    "Appending %s to %s, %s -> %s", fn, ofn, qr.fileSizeOrig, sz
                )
        else:
            reduceAndDecimate(info, ofp, ofn, maxSize)
        if ofp.seekable():
            return ofp.tell()  # Actual file size
        st = os.fstat(ofp.fileno())
        return st.st_size # for non-seekable files


def decimateFiles(qFiles: Dict[str, int], ofn: str, totSize: int, maxSize: int) -> int:
    """
    Decimate Q-files to fit within maximum size.

    Args:
        qFiles: Dictionary of Q-files to decimate
        ofn: Output filename
        totSize: Total size of input files
        maxSize: Maximum output file size in bytes

    Returns:
        Total bytes written
    """
    try:
        filenames = sorted(qFiles, reverse=False)  # sorted filenames to work on
        totHdrSize = 0
        totDataSize = 0
        info = {}
        for ifn in filenames:
            try:
                with open(ifn, "rb") as fp:
                    hdr = QHeader(fp, ifn)
                    st = os.fstat(fp.fileno())
                    item = {}
                    item["hdrSize"] = hdr.hdrSize
                    item["dataSize"] = hdr.dataSize
                    item["nRecords"] = np.floor(
                        int(st.st_size - item["hdrSize"]) / item["dataSize"]
                    )
                    logging.info(
                        "%s hdr %s data %s n %s",
                        ifn,
                        item["hdrSize"],
                        item["dataSize"],
                        item["nRecords"],
                    )
                    info[ifn] = item
                    totHdrSize += item["hdrSize"]
                    totDataSize += item["dataSize"] * item["nRecords"]
            except EOFError:
                pass
            except (OSError, ValueError):
                logging.exception("filename %s", ifn)

        logging.info("Total header size %s data size %s", totHdrSize, totDataSize)

        availSize = maxSize - totHdrSize
        ratio = availSize / totDataSize
        logging.info("availSize %s ratio %s", availSize, ratio)

        with open(ofn, "ab") as ofp:
            if ratio <= 0:
                logging.warning("Not adding to %s since ratio is %s <= 0", ofn, ratio)
                st = os.fstat(ofp.fileno())
                return st.st_size

            for ifn in filenames:
                item = info[ifn]
                n = item["nRecords"]
                indices = np.unique(
                    np.floor(
                        np.linspace(0, n, math.floor(n * ratio), endpoint=False)
                    ).astype(int)
                )
                hdrSize = item["hdrSize"]
                dataSize = item["dataSize"]
                offsets = hdrSize + indices * dataSize
                logging.info(
                    "Decimating file %s hdr sz %s data sz %s n %s of %s",
                    ifn,
                    hdrSize,
                    dataSize,
                    len(offsets),
                    item["nRecords"],
                )
                with open(ifn, "rb") as ifp:
                    buffer = ifp.read(hdrSize)
                    if len(buffer) != hdrSize:
                        continue
                    ofp.write(buffer)
                    for offset in offsets:
                        ifp.seek(offset)
                        buffer = ifp.read(dataSize)
                        if len(buffer) != dataSize:
                            break
                        ofp.write(buffer)
            if ofp.seekable():
                return ofp.tell()
            st = os.fstat(ofp.fileno())
            return st.st_size

    except (OSError, ValueError):
        logging.exception("Unable to decimate %s to %s", filenames, ofn)
        return 0


def glueFiles(filenames: List[str], ofn: str, bufferSize: int = 1024 * 1024) -> int:
    """
    Concatenate Q-files into single output file.

    Args:
        filenames: List of input filenames to concatenate
        ofn: Output filename
        bufferSize: Size of read buffer in bytes

    Returns:
        Total bytes written
    """
    try:
        totSize = 0
        with open(ofn, "ab") as ofp:
            for ifn in filenames:
                with open(ifn, "rb") as ifp:
                    while True:
                        buffer = ifp.read(bufferSize)
                        if len(buffer) <= 0:
                            break  # EOF
                        ofp.write(buffer)
                        logging.info(
                            "Appended %s to %s with %s bytes", ifn, ofn, len(buffer)
                        )
                        totSize += len(buffer)
            if ofp.seekable():
                fSize = ofp.tell()
            else:
                st = os.fstat(ofp.fileno())
                fSize = st.st_size
            logging.info("Glued %s to %s fSize %s", totSize, ofn, fSize)
            return fSize
    except OSError:
        logging.exception("Unable to glue %s to %s", filenames, ofn)
        return 0


def fileCandidates(args: Namespace, times: np.ndarray) -> Tuple[Dict[str, int], int]:
    """
    Find Q-file candidates within time range.

    Args:
        args: Parsed command-line arguments
        times: Array with [start_time, end_time] in seconds since epoch

    Returns:
        Tuple of (qFiles dict mapping paths to sizes, total size in bytes)
    """
    with os.scandir(args.datadir) as it:
        qFiles = {}
        totSize = 0
        for entry in it:
            if not entry.name.endswith(".q") or not entry.is_file():
                continue
            # N.B. on the MR, c_time and m_time are identical
            # This is from mounting an exFAT filesystem with FUSE
            st = entry.stat()
            qKeep = st.st_mtime >= times[0] and st.st_mtime <= times[1]
            logger = logging.info if qKeep else logging.debug
            logger(
                "%s sz %s mtime %s %s",
                entry.name,
                st.st_size,
                np.datetime64(round(st.st_mtime * 1000), "ms"),
                qKeep,
            )
            if qKeep:
                qFiles[entry.path] = st.st_size
                totSize += st.st_size
        return (qFiles, totSize)


def scanDirectory(args: Namespace, times: np.ndarray) -> int:
    """
    Scan directory for Q-files and merge/reduce them.

    Args:
        args: Parsed command-line arguments
        times: Array with [start_time, end_time] in seconds since epoch

    Returns:
        Total bytes written to output file
    """
    (qFiles, totSize) = fileCandidates(args, times)

    if os.path.isfile(args.output):  # File already exist, so reduce maxsize
        fSize = os.path.getsize(args.output)

        if not qFiles:  # No Q-files found, nothing to do
            logging.info("No new files to append to %s", args.output)
            return fSize

        args.maxSize -= fSize
        if args.maxSize <= 0:
            logging.info("Can't append more to file, %s >= %s", fSize, args.maxSize)
            return fSize
        logging.info("Reduced maxsize by %s since file already exists", fSize)
    elif not qFiles:  # No q-files, so create empty file and return 0
        with open(args.output, "wb"):
            pass
        logging.info("No new files, so created an empty file %s", args.output)
        return 0

    if args.config and os.path.isfile(
        args.config
    ):  # We're going to reduce the size of the Q-files,
        value = reduceFiles(qFiles, args.config, args.output, args.maxSize)
        # Fall through if config is empty
        if value is not None:
            return value

    logging.info("Total size %s max %s", totSize, args.maxSize)

    if totSize <= args.maxSize:
        # Glue the files together since their total size is small enough
        return glueFiles(sorted(qFiles, reverse=False), args.output, args.bufferSize)

    # Parse the Q-files and pull out roughly equally spaced in time records
    return decimateFiles(qFiles, args.output, totSize, args.maxSize)


def main() -> None:
    """Command-line interface for mergeqfiles."""
    parser = ArgumentParser()
    parser.add_argument(
        "stime", type=float, help="Unix seconds for earliest sample, or 0 for now"
    )
    parser.add_argument(
        "dt", type=float, help="Seconds added to stime for other end of samples"
    )
    parser.add_argument("maxSize", type=int, help="Maximum output filesize in bytes")
    parser.add_argument(
        "--output", "-o", type=str, default="/dev/stdout", help="Output filename"
    )
    parser.add_argument(
        "--bufferSize",
        type=int,
        default=100 * 1024,
        help="Maximum buffer size to read at a time in bytes",
    )
    parser.add_argument(
        "--datadir", type=str, default="~/data", help="Where Q-files are stored"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable logging.debug messages"
    )
    parser.add_argument("--logfile", type=str, help="Output of logfile messages")
    parser.add_argument(
        "--safety",
        type=float,
        default=30,
        help="Extra seconds to add to end time for race condition issue",
    )
    parser.add_argument("--config", type=str, help="JSON config file")
    args = parser.parse_args()

    args.datadir = os.path.abspath(os.path.expanduser(args.datadir))

    if not os.path.isdir(args.datadir):
        print(f'ERROR: Data directory "{args.datadir}" does not exist')
        sys.exit(1)

    try:
        if args.logfile is None:
            args.logfile = os.path.join(args.datadir, "mergeqfiles.log")
        elif args.logfile == "":
            args.logfile = None  # Spew out to the console
        else:
            args.logfile = os.path.abspath(os.path.expanduser(args.logfile))
            dirname = os.path.dirname(args.logfile)
            if not os.path.isdir(dirname):
                os.makedirs(dirname, 0o755, exist_ok=True)

        logging.basicConfig(
            format="%(asctime)s %(levelname)s: %(message)s",
            level=logging.DEBUG if args.verbose else logging.INFO,
            filename=args.logfile,
        )

        if args.config is None:
            fn = os.path.abspath(
                os.path.expanduser(os.path.join(args.datadir, "mergeqfiles.cfg"))
            )
            if os.path.isfile(fn):
                args.config = fn
        elif args.config == "":
            args.config = None
        else:
            args.config = os.path.abspath(os.path.expanduser(args.config))

        if args.stime <= 0:
            args.stime = time.time()  # Current time

        logging.info("Args: %s", args)

        times = np.sort([args.stime, args.stime + args.dt + args.safety])

        logging.info("Time limits %s", times.astype("datetime64[s]"))

        outSize = scanDirectory(args, times)
        logging.info("printing outSize %s to console", outSize)
        print(outSize)
    except Exception:
        logging.exception("Unexpected exception executing %s", args)
        sys.exit(1)


if __name__ == "__main__":
    main()
