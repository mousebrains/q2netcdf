#! /usr/bin/env python3
#
# Decode QFiles header and config
#
# Feb-2025, Pat Welch, pat@mousebrains.com

import struct
import logging
import numpy as np
from typing import Optional, Tuple
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
    def chkIdent(cls, fp) -> Optional[bool]:
        n = 2
        buffer = fp.read(n)
        if len(buffer) != n:
            return None
        (ident,) = struct.unpack("<H", buffer)
        fp.seek(-n, 1)  # Back up n bytes
        return ident == RecordType.HEADER.value

    def __init__(self, fp, fn: str) -> None:
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
