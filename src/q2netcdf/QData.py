#! /usr/bin/env python3
#
# Read and parse Q-file data recrods
#
# Feb-2025, Pat Welch, pat@mousebrains.com

import struct
from QHeader import QHeader
from QHexCodes import QHexCodes
import numpy as np
import logging

class QRecord:
    def __init__(self, hdr:QHeader, number:int, err:int, stime:float, etime:float, 
                 items:list) -> None:
        self.number = number
        self.error = err
        self.t0 = (hdr.time + 
                   np.array(stime * 1000).astype("timedelta64[ms]")).astype("datetime64[ns]")
        if etime is not None:
            self.t1 = (hdr.time + 
                       np.array(etime * 1000).astype("timedelta64[ms]")).astype("datetime64[ns]")
        else:
            self.t1 = None
        self.channels = np.array(items[:hdr.Nc]).astype("f4")
        self.spectra = np.array(items[hdr.Nc:]).astype("f4")
        self.spectra = self.spectra.reshape((hdr.Ns, hdr.Nf))

    def __repr__(self) -> str:
        msg = []
        msg.append(f"Record #: {self.number}")
        msg.append(f"Error: {self.error}")
        msg.append(f"Time: {self.t0} to {self.t1}")
        msg.append(f"Channels: {self.channels}")
        msg.append(f"Spectra: {self.spectra}")
        return "\n".join(msg)

    def split(self, hdr:QHeader) -> tuple:
        hexMap = QHexCodes()
        record = {}
        attrs = {}

        record["time"] = self.t0
        attrs["time"] = dict(long_name="time")

        if self.t1 is not None:
            record["t1"] = self.t1
            attrs["t1"] = dict(long_name="timeStop")
        if self.number is not None:
            record["record"] = self.number
            attrs["record"] = dict(long_name="recordNumber")
        if self.error is not None:
            record["error"] = self.error
            attrs["error"] = dict(long_name="errorCode")

        for index in range(hdr.Nc):
            ident = hdr.channels[index]
            name = hexMap.name(ident)
            if name: 
                record[name] = self.channels[index]
                attrs[name] = hexMap.attributes(ident)

        for index in range(hdr.Ns):
            ident = hdr.spectra[index]
            name = hexMap.name(ident)
            if name: 
                record[name] = self.spectra[index]
                attrs[name] = hexMap.attributes(ident)

        return (record, attrs)

    def prettyRecord(self, hdr:QHeader) -> str:
        hexMap = QHexCodes()
        msg = []

        if self.number is not None: 
            msg.append(f"Record #: {self.number}")

        if self.error is not None: 
            msg.append(f"Error: {self.error}")

        if self.t1 is not None:
            msg.append(f"Time: {self.t0} to {self.t1}")
        else:
            msg.append(f"Time: {self.t0}")

        for index in range(hdr.Nc):
            ident = hdr.channels[index]
            name = hexMap.name(ident)
            if not name: name = f"{ident:#06x}"
            msg.append(f"Channel[{name}] = {self.channels[index]}")
        for index in range(hdr.Ns):
            ident = hdr.spectra[index]
            name = hexMap.name(ident)
            if not name: name = f"{ident:#06x}"
            msg.append(f"spectra[{name}] = {self.spectra[index,:]}")

        return "\n".join(msg)

class QData:
    dataIdent = 0x1657

    def __init__(self, hdr:QHeader) -> None:
        self.__hdr = hdr
        if hdr.version.isV12():
            self.__format = "<HHqee" + ("e" * hdr.Nc) + ("e" * hdr.Ns * hdr.Nf)
        else: # >v12
            self.__format = "<He" + ("e" * hdr.Nc) + ("e" * hdr.Ns * hdr.Nf)

    @classmethod
    def chkIdent(cls, fp) -> bool:
        n = 2
        buffer = fp.read(n)
        if len(buffer) != n: return None
        (ident,) = struct.unpack("<H", buffer)
        fp.seek(-n, 1) # Backup n bytes
        return ident == cls.dataIdent # A data ident

    def load(self, fp) -> QRecord:
        hdr = self.__hdr
        buffer = fp.read(hdr.dataSize)
        if len(buffer) != hdr.dataSize: return None # EOF while reading

        items = struct.unpack(self.__format, buffer)
        if hdr.version.isV12():
            offset = 5
            (ident, number, err, stime, etime) = items[:offset]
        else:
            offset = 2
            (ident, stime) = items[:offset]
            number = None
            err = None
            etime = None

        if ident != self.dataIdent:
            logging.warning(f"Data record identifier mismatch, {ident:#06x} != {self.dataIdent:#06x} in at byte %s in %s",
                            fp.tell() - len(buffer), self.__hdr.filename)

        record = QRecord(hdr, number, err, stime, etime, items[offset:])
        return record

    def prettyRecord(self, record:QRecord) -> str:
        return record.prettyRecord(self.__hdr)
