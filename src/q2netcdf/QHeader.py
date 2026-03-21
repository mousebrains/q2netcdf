#! /usr/bin/env python3
#
# Decode QFiles header and config
#
# Feb-2025, Pat Welch, pat@mousebrains.com

import struct
import logging
import numpy as np
from .QConfig import QConfig
from .QVersion import QVersion
from .QRecordType import RecordType


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
    def chkIdent(cls, fp) -> bool | None:
        n = 2
        buffer = fp.read(n)
        if len(buffer) != n:
            return None
        (ident,) = struct.unpack("<H", buffer)
        fp.seek(-n, 1)  # Back up n bytes
        return ident == RecordType.HEADER.value

    @staticmethod
    def _read_exact(fp, sz: int, fn: str, what: str) -> bytes:
        """Read exactly sz bytes from fp, raising EOFError on short read."""
        buffer = fp.read(sz)
        if len(buffer) != sz:
            raise EOFError(f"EOF while reading {what}, {len(buffer)} != {sz}, in {fn}")
        return buffer

    def __init__(self, fp, fn: str) -> None:
        self.filename = fn
        hdrSize = 0

        buffer = self._read_exact(fp, 20, fn, "fixed header")
        hdrSize += 20

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

        hdrSize += self._read_identifiers(fp, fn)
        hdrSize += self._read_config(fp, fn)

        buffer = self._read_exact(fp, 2, fn, "data record size")
        hdrSize += 2

        self.dataSize = struct.unpack("<H", buffer)[0]
        self.hdrSize = hdrSize

    def _read_identifiers(self, fp, fn: str) -> int:
        """Read channel identifiers, spectra identifiers, and frequencies."""
        bytesRead = 0

        self.channels: tuple[int, ...] = ()
        self.spectra: tuple[int, ...] = ()
        self.frequencies: tuple[float, ...] = ()

        if self.Nc:
            sz = self.Nc * 2
            buffer = self._read_exact(fp, sz, fn, "channel identifiers")
            bytesRead += sz
            self.channels = struct.unpack("<" + ("H" * self.Nc), buffer)

        if self.Ns:
            sz = self.Ns * 2
            buffer = self._read_exact(fp, sz, fn, "spectra identifiers")
            bytesRead += sz
            self.spectra = struct.unpack("<" + ("H" * self.Ns), buffer)

        if self.Nf:
            sz = self.Nf * 2
            buffer = self._read_exact(fp, sz, fn, "frequencies")
            bytesRead += sz
            self.frequencies = struct.unpack("<" + ("e" * self.Nf), buffer)

        return bytesRead

    def _read_config(self, fp, fn: str) -> int:
        """Read the configuration record (v1.2 or v1.3 format)."""
        assert self.version is not None  # Validated in __init__ before this call
        bytesRead = 0
        self.config = QConfig(b"{}", self.version)

        cfgHdrSz = 4 if self.version == QVersion.v12 else 2
        buffer = self._read_exact(fp, cfgHdrSz, fn, "fixed configuration record")
        bytesRead += cfgHdrSz

        if self.version == QVersion.v12:
            # cfgIdent is bad in the beta version of 1.2, should be RecordType.CONFIG_V12
            (cfgIdent, sz) = struct.unpack("<HH", buffer)
        else:
            (sz,) = struct.unpack("<H", buffer)

        if sz:
            buffer = self._read_exact(fp, sz, fn, "configuration record")
            bytesRead += sz
            self.config = QConfig(buffer, self.version)

        return bytesRead

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


def main() -> None:
    """Command-line interface for QHeader."""
    from argparse import ArgumentParser
    import os.path
    from .QHexCodes import QHexCodes

    parser = ArgumentParser()
    parser.add_argument("filename", type=str, nargs="+", help="Input filename(s)")
    parser.add_argument("--config", action="store_false", help="Don't display config")
    parser.add_argument(
        "--channels", action="store_false", help="Don't display channel names"
    )
    parser.add_argument(
        "--spectra", action="store_false", help="Don't display spectra names"
    )
    parser.add_argument(
        "--frequencies", action="store_false", help="Don't display frequencies"
    )
    parser.add_argument("--nothing", action="store_true", help="Don't display extra")
    parser.add_argument(
        "--logLevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logLevel))

    hex = QHexCodes()

    for fn in args.filename:
        fn = os.path.abspath(os.path.expanduser(fn))
        print("filename:", fn)
        with open(fn, "rb") as fp:
            hdr = QHeader(fp, fn)
            print(f"File version: {hdr.version}")
            print("Time:", hdr.time)

            if args.channels and not args.nothing:
                for ident in hdr.channels:
                    name = hex.name(ident)
                    attrs = hex.attributes(ident)
                    print(f"Scalar[{ident:#06x}] ->", name, "->", attrs)
            else:
                print(f"Channels: n={len(hdr.channels)}")

            if args.spectra and not args.nothing:
                for ident in hdr.spectra:
                    name = hex.name(ident)
                    attrs = hex.attributes(ident)
                    print(f"Spectra[{ident:#06x}] ->", name, "->", attrs)
            else:
                print(f"Spectra: n={len(hdr.spectra)}")

            n = len(hdr.frequencies)
            if args.frequencies and not args.nothing:
                print(f"Frequencies n={n}", hdr.frequencies, "Hz")
            else:
                print(f"Frequencies n={n}")

            if args.config and not args.nothing:
                config = hdr.config.config()
                for key in sorted(config):
                    print(f"Config[{key:18s}] ->", config[key])

            print(f"Data   Record Size: {hdr.dataSize}")
            print(f"Header Record Size: {hdr.hdrSize} config size: {len(hdr.config)}")


if __name__ == "__main__":
    main()
