#! /usr/bin/env python3
#
# This is a rewrite of Rockland's mergeqfiles for use by TWR's Slocum uRider proglet.
#
# The stock script will return a zero length file if the maximum size allowed is smaller
# than the size of a q-file.
#
# This script takes equally spaced in time samples from the
# q-files to reach the maximum allowed size.
#
# The internal q-file structure is based on Rockland's TN 054
#
# Oct-2024, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import os
import time
import struct
import numpy as np
import logging
import re
import math
import sys

def splitHeader(buffer:bytes) -> dict:
    info = {}
    for line in buffer.split(b"\n"):
        try:
            line = str(line, "utf-8") # It should be UTF-8
            matches = re.fullmatch(r'[#"\s]*([\w_]+)"?\s*=>\s*"?(["\s].*)"?', line.strip())
            if matches:
                key = matches[1]
                val = matches[2]
                try:
                    key = str(key, "utf-8")
                    val = str(val, "utf-8")
                except:
                    pass
                try:
                    val = float(val)
                except:
                    pass
                info[key] = val
        except:
            pass # Skip non-UTF8 lines
    return info

def loadQHeader(fn:str) -> tuple:
    badReturn = (None, None, None, None, None, None)
    with open(fn, "rb") as fp:
        buffer = fp.read(20) # Load the first 20 bytes of the header
        if len(buffer) != 20:
            logging.error("Reading initial header, %s!=20, in %s", len(buffer), fn)
            return badReturn

        (ident, fVer, stime, Nc, Ns, Nf) = struct.unpack("<HfQHHH", buffer)

        if ident != 0x1729:
            logging.error(f"%s ident incorrect, {ident:#04x}!=0x1729", fn)
            return badReturn

        if abs(fVer - 1.2) > 0.0001:
            logging.error("Invalid file version, %s, in %s", fVer, fn)
            return badReturn

        stime = np.datetime64("0000-01-01") + np.timedelta64(stime, "ms")

        sz = Nc * 2 + Ns * 2 + Nf * 2 # Number of bytes in ident information
        body = buffer
        buffer = fp.read(sz)
        if len(buffer) != sz:
            logging.error("Reading Channel/Spectra/Frequency information, %s != %s in %s",
                          len(buffer), sz, fn)
            return badReturn

        body += buffer

        buffer = fp.read(4) # Read configuration ident and size
        if len(buffer) != 4:
            logging.error("Reading configuration header, %s != 4, in %s", len(buffer), fn)
            return badReturn
        body += buffer

        (ident, sz) = struct.unpack("<HH", buffer)

        buffer = fp.read(sz)
        if len(buffer) != sz:
            logging.error("Reading configuration record, %s != %s, in %s", len(buffer), sz, fn)
            return badReturn
        body += buffer
        hdr = splitHeader(buffer)

        buffer = fp.read(2)
        if len(buffer) != 2:
            logging.error("Reading data record size, %s != 2, in %s", len(buffer), fn)
            return badReturn
        body += buffer
        (dataSize,) = struct.unpack("<H", buffer)

        hdrSize = fp.tell()

        st = os.fstat(fp.fileno())
        nRecords = math.floor((st.st_size - hdrSize) / dataSize)
        dissLength = hdr["diss_length"] if "diss_length" in hdr else 1
        dt = np.timedelta64(int(hdr["diss_length"] * 1000), "ms")
        etime = stime + dt * nRecords

        return (hdrSize, dataSize, stime, etime, nRecords, dt)

def loadQData(f, fn:str, hdr:dict):
    if hdr is None:
        logging.error("No header record before first data record in %s", fn)
        return None

    Nc = hdr["Nc"]
    Ns = hdr["Ns"]
    Nf = hdr["Nf"]

    sz = hdr["recordSize"] - 2 # Ident already read

    buffer = f.read(sz)
    if len(buffer) != sz:
        logging.error("Unable to read data record, %s!=%s, in %s", len(buffer), sz, fn)
        return None

    items = struct.unpack("<Hqee" + ("e" * Nc) + ("e" * Ns * Nf), buffer)

    stime = (hdr["time"] + np.timedelta64(int(items[2] * 1000), "ms")).astype("datetime64[ns]")
    etime = (hdr["time"] + np.timedelta64(int(items[3] * 1000), "ms")).astype("datetime64[ns]")

    return ds

def writePartialFile(ifn:str, ofp, szHeader:int, szData:int, nRecords:int, indices:np.array):
    logging.info("Partial file %s sz %s n %s of %s", ifn, szHeader, len(indices), nRecords)
    with open(ifn, "rb") as ifp:
        buffer = ifp.read(szHeader)
        if len(buffer) != szHeader: return
        ofp.write(buffer)
        for index in indices:
            offset = szHeader + index * szData
            ifp.seek(offset)
            buffer = ifp.read(szData)
            if len(buffer) != szData: return
            ofp.write(buffer)


def decimateFiles(qfiles:dict, ofn:str, totSize:int, maxSize:int) -> int:
    try:
        filenames = sorted(qfiles, reverse=False) # sorted filenames to work on
        info = {}
        totHdrSize = 0
        totDataSize = 0
        totTime = np.timedelta64(0, "s")
        for ifn in filenames:
            (hdrSize, dataSize, stime, etime, nRecords, dt) = loadQHeader(ifn)
            totHdrSize += hdrSize
            totDataSize += dataSize * nRecords
            totTime += etime - stime
            info[ifn] = dict(
                    hdrSize = hdrSize,
                    dataSize = dataSize,
                    stime = stime,
                    etime = etime,
                    nRecords = nRecords,
                    dt = dt,
                    )
            logging.info("%s n %s", ifn, nRecords)

        logging.info("total header size %s data size %s totalTime %s", totHdrSize, totDataSize, totTime)
        availSize = maxSize - totHdrSize
        ratio = availSize / totDataSize
        logging.info("availSize %s ratio %s", availSize, ratio)

        with open(ofn, "ab") as ofp:
            if ratio <= 0:
                logging.warning("Creating an empty file since ratio, %s, <=0", ratio)
                st = os.fstat(ofp.fileno())
                return st.st_size

            for ifn in filenames:
                item = info[ifn]
                n = item["nRecords"]
                indices = np.unique(
                        np.floor(
                            np.linspace(0, n, math.floor(n * ratio), endpoint=False)).astype(int)
                        )
                writePartialFile(ifn,
                                 ofp,
                                 item["hdrSize"],
                                 item["dataSize"],
                                 item["nRecords"],
                                 indices)
            fSize = ofp.tell()
            return fSize
    except:
        logging.exception("Unable to decimate %s to %s", filenames, ofn)

def glueFiles(filenames:list, ofn:str, bufferSize:int=1024*1024) -> int:
    try:
        totSize = 0
        with open(ofn, "ab") as ofp:
            for ifn in filenames:
                with open(ifn, "rb") as ifp:
                    while True:
                        buffer = ifp.read(bufferSize)
                        if len(buffer) <= 0: break # EOF
                        ofp.write(buffer)
                        logging.info("Appended %s to %s with %s bytes", ifn, ofn, len(buffer))
                        totSize += len(buffer)
            fSize = ofp.tell()
            logging.info("Glued %s to %s fSize %s", totSize, ofn, fSize)
            return fSize
    except:
        logging.exception("Unable to glue %s to %s", filenames, ofn)

def scanDirectory(args:ArgumentParser, times:np.array) -> int:
    with os.scandir(args.datadir) as it:
        qfiles = {}
        totSize = 0

        t0 = times[0].astype("datetime64[s]")
        t1 = times[1].astype("datetime64[s]")

        for entry in it:
            if not entry.name.endswith(".q") or not entry.is_file():
                continue
            # N.B. on the MR, c_time and m_time are identical
            st = entry.stat()
            qKeep = st.st_mtime >= times[0] and st.st_mtime <= times[1]
            logging.info("%s sz %s mtime %s times %s %s qKeep %s",
                         entry.name,
                         st.st_size,
                         np.datetime64(round(st.st_mtime * 1000), "ms"),
                         t0,
                         t1,
                         qKeep
                         )
            if qKeep:
                qfiles[entry.path] = st.st_size
                totSize += st.st_size

    if os.path.isfile(args.output): # File already exist, so reduce maxsize
        fSize = os.path.getsize(args.output)
        args.maxSize -= fSize
        if args.maxSize <= 0:
            logging.info("Can't append more to file, %s >= %s", fSize, args.maxSize)
            return fSize
        logging.info("Reducing maxsize by %s since file already exists", fSize)

    logging.info("Total size %s max %s", totSize, args.maxSize)
    if totSize <= args.maxSize:
        # Glue the files together since their total size is small enough
        # This handles the no-files case and will generate an empty .mri file
        return glueFiles(sorted(qfiles, reverse=False), args.output, args.bufferSize)

    # Parse the qfiles and pull out roughly equally spaced in time records
    return decimateFiles(qfiles, args.output, totSize, args.maxSize)

def main():
    parser = ArgumentParser()
    parser.add_argument("stime", type=float, help="Unix seconds for earliest sample, or 0 for now")
    parser.add_argument("dt", type=float, help="Seconds added to stime for other end of samples")
    parser.add_argument("maxSize", type=int, help="Maximum output filesize in bytes")
    parser.add_argument("--output", "-o", type=str, default="/dev/stdout", help="Output filename")
    parser.add_argument("--bufferSize", type=int, default=100*1024,
                        help="Maximum buffer size to read at a time in bytes")
    parser.add_argument("--datadir", type=str, default="~/data", help="Where Q-files are stored")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable logging.debug messages")
    parser.add_argument("--logfile", type=str, help="Output of logfile messages")
    parser.add_argument("--safety", type=float, default=30,
                        help="Extra seconds to add to end time for race condition issue")
    args = parser.parse_args()

    args.datadir = os.path.abspath(os.path.expanduser(args.datadir))

    if not os.path.isdir(args.datadir):
        print(f"ERROR: Data directory '{args.datadir}' does not exist")
        sys.exit(1)

    try:
        if args.logfile is None:
            args.logfile = os.path.join(args.datadir, "mergeqfiles.log")
        elif args.logfile == "":
            args.logfile = None # Spew out to the console
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

        logging.info("Args: %s", args)

        if args.stime <= 0:
            args.stime = time.time() # Current time

        times = np.sort([args.stime, args.stime + args.dt + args.safety])

        logging.info("Time limits %s", times.astype("datetime64[s]"))

        outSize = scanDirectory(args, times)
        logging.info("printing outSize %s to console", outSize)
        print(outSize)
    except:
        logging.exception("Unexpected exception executing %s", args)


if __name__ == "__main__":
    main()
