#! /usr/bin/env python3
#
# Reduce the size of a Q-file
#  drop redundant fields
#  prune config records
#  prune channels
#  prune spectra
#
# Mar-2025, Pat Welch, pat@mousebrains.com

import json
import logging
import struct
import os
import numpy as np
from typing import Any, Dict, Optional, Tuple, IO
from .QHeader import QHeader
from .QHexCodes import QHexCodes
from .QVersion import QVersion
from .QRecordType import RecordType

class QReduce:
    """
    Reduce Q-file size by pruning channels, spectra, and config fields.

    This class creates a reduced version of a Q-file containing only
    specified channels, spectra, and configuration parameters. Useful
    for reducing file size when only certain measurements are needed.

    The reduced file uses v1.3 format regardless of input version.
    """
    __name2ident: Dict[str, int] = {}

    def __init__(self, filename: str, config: Dict[str, Any]) -> None:
        self.filename = filename

        channelIdents = self.__updateName2Ident(config, "channels") # Get intersecting idents
        spectraIdents = self.__updateName2Ident(config, "spectra")

        with open(filename, "rb") as fp:
            hdr = QHeader(fp, filename)
            self.fileSizeOrig = os.fstat(fp.fileno()).st_size

        # Convert tuples to ndarrays for processing
        channelArray = np.array(hdr.channels, dtype="uint16")
        spectraArray = np.array(hdr.spectra, dtype="uint16")

        (channelIdents, channelIndices) = self.__findIndices(channelIdents, channelArray)
        (spectraIdents, spectraIndices) = self.__findIndices(spectraIdents, spectraArray)

        qFreq = spectraIndices.size != 0
        if qFreq: # Some spectra, so build full indices
            spectraIndices = self.__spectraIndices(hdr, spectraIndices)
            allIndices = np.concatenate((channelIndices, spectraIndices))
        else:
            allIndices = channelIndices

        body = struct.pack("<Hf", RecordType.HEADER.value, QVersion.v13.value)
        body += np.array(hdr.dtBinary, dtype="uint64").tobytes()
        body += struct.pack("<HHH", 
                            len(channelIdents), 
                            len(spectraIdents),
                            hdr.Nf if qFreq else 0)
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
        Nf = hdr.Nf # Number of frequencies
        indices = indices.reshape(-1, 1)
        freq = np.arange(Nf, dtype="uint16").reshape(1, -1)
        indices = hdr.Nc + (indices * Nf + freq)
        return indices.flatten()

    @staticmethod
    def __findIndices(idents: Optional[np.ndarray], known: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if idents is None:
            return (np.array([], dtype=int), np.array([], dtype=int))

        (idents, iLHS, iRHS) = np.intersect1d(idents, known, return_indices=True)
        ix = iRHS.argsort()
        return (idents[ix], iRHS[ix])

    @classmethod
    def __updateName2Ident(cls, config: Dict[str, Any], key: str) -> Optional[np.ndarray]:
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

        record = buffer[:2] + buffer[12:14] # Ident + stime
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
            ifp.seek(self.hdrSizeOrig) # Skip the header
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
            ifp.seek(self.hdrSizeOrig) # Skip the header
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

def main() -> None:
    """Command-line interface for QReduce."""
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Reduce the size of a Q-file")
    parser.add_argument("filename", type=__chkExists, help="Q-file to reduce")
    parser.add_argument("--output", type=str, help="Output file name")
    parser.add_argument("--config", type=__chkExists, default="mergeqfiles.cfg",
                        help="JSON configuration file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.filename = os.path.abspath(os.path.expanduser(args.filename))

    qrConfig = QReduce.loadConfig(args.config) # Do this once per file
    if qrConfig is None:
        logging.error(f"Failed to load config from {args.config}")
        return
    qr = QReduce(args.filename, qrConfig)
    logging.info(f"QR {qr}")

    if args.output:
        args.output = os.path.abspath(os.path.expanduser(args.output))
        with open(args.output, "ab") as fp:
            qr.reduceFile(fp)

if __name__ == "__main__":
    main()
