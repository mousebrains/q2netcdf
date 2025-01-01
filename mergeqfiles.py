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

def splitHeader(buffer:bytes) -> dict:
    info = {}
    for line in buffer.split(b"\n"):
        matches = re.fullmatch(b'[#"\s]*([\w_]+)"?\s*=>\s*"?(["\s].*)"?', line.strip())
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

        st = os.fstat(fp.fileno());
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

def loadQfile(fn:str):
    logging.info("Processing %s", fn)
    records = []
    with open(fn, "rb") as f:
        body = bytes()
        while True:
            n = f.tell()
            buffer = f.read(2)
            if len(buffer) != 2:
                if buffer is None or len(buffer) == 0: break # EOF
                logging.error("Unable to read identification field, %s!=2, in %s", len(buffer), fn)
                break
            body += buffer
            (ident,) = struct.unpack("<H", buffer)
            match ident:
                case 0x1729: # Header+Config record
                    logging.info(f"HDR at {n:#05x}")
                    (hdrSize, dataSize, stime, etime, nRecords, dt) = loadQHeader(ifn)
                    if hdrSize is None: 
                        body = None
                        break

                case 0x1657: # Data record
                    logging.info(f"Data at {n:#05x}")
                    rec = loadQData(f, fn, hdr)
                    if rec is None: break
                    records.append(rec)
                case 0x1a1a: # EOF ??
                    logging.info(f"0x1a1a at {n:#05x}")
                    break
                case _:
                    logging.error(f"Unrecognized record identifier, {ident:#04x}, in %s", fn)
                    break

    if not records:
        logging.warning("No records found in %s", fn)
        return None

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


def decimateFiles(qfiles:dict, ofn:str, totSize:int, maxSize:int):
    try:
        filenames = sorted(qfiles) # sorted filenames to work on
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

        with open(ofn, "wb") as ofp:
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
    except:
        logging.exception("Unable to decimate %s to %s", filenames, ofn)

def glueFiles(filenames:list[str], ofn:str, bufferSize:int=1024*1024):
    try:
        totSize = 0
        with open(ofn, "wb") as ofp:
            for ifn in filenames:
                with open(ifn, "rb") as ifp:
                    while True:
                        buffer = ifp.read(bufferSize)
                        if len(buffer) <= 0: break # EOF
                        ofp.write(buffer)
                        logging.info("Appended %s to %s with %s bytes", ifn, ofn, len(buffer))
                        totSize += len(buffer)
        logging.info("Created %s with %s bytes", ofn, totSize)
    except:
        logging.exception("Unable to glue %s to %s", filenames, ofn)

parser = ArgumentParser()
parser.add_argument("stime", type=float, help="Unix seconds for earliest sample, or 0 for now")
parser.add_argument("dt", type=float, help="Seconds added to stime for other end of samples")
parser.add_argument("maxSize", type=int, help="Maximum output filesize in bytes")
parser.add_argument("mri", type=str, help="Output filename")
parser.add_argument("--bufferSize", type=int, default=100*1024,
                    help="Maximum buffer size to read at a time in bytes")
parser.add_argument("--datadir", type=str, default="~/data", help="Where Q-files are stored")
parser.add_argument("--verbose", action="store_true", help="Enable logging.debug messages")
parser.add_argument("--logfile", type=str, default="~/data/mergeqfiles.log",
                    help="Output of logfile messages")
# parser.add_argument("--config", type=str, default="~/data/mergeqfiles.config",
                    # help="YAML config file")
args = parser.parse_args()

if args.logfile: args.logfile = os.path.abspath(os.path.expanduser(args.logfile))
if args.datadir: args.datadir = os.path.abspath(os.path.expanduser(args.datadir))

logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        filename=args.logfile,
        )

logging.info("Command line, %s", args)

if args.stime <= 0:
    args.stime = time.time() # Current time

times = np.sort([args.stime, args.stime + args.dt])

logging.info("Time limits %s", times.astype("datetime64[s]"))

with os.scandir(args.datadir) as it:
    qfiles = {}
    totSize = 0
    for entry in it:
        if not entry.name.endswith(".q") or not entry.is_file(): continue
        st = entry.stat()
        logging.info("%s sz %s ctime %s mtime %s times %s %s", 
                     entry.name, st.st_size, st.st_ctime, st.st_mtime, 
                     times[0], times[1])
        if st.st_ctime > times[1] or st.st_mtime < times[0]: 
            logging.info("Ignoring %s due to times out of range, %s>%s %s or %s<%s %s",
                         entry.name,
                         np.datetime64(round(st.st_ctime*1e9), "ns"),
                         np.datetime64(round(times[1] * 1e9), "ns"), 
                         st.st_ctime > times[1],
                         np.datetime64(round(st.st_mtime*1e9), "ns"),
                         np.datetime64(round(times[0]*1e9), "ns"),
                         st.st_mtime < times[0],
                         )
            continue
        qfiles[entry.path] = st.st_size
        totSize += st.st_size

logging.info("Total size %s max %s", totSize, args.maxSize)
if totSize <= args.maxSize:
    # Glue the files together since their total size is small enough
    # This handles the no-files case and will generate an empty .mri file
    glueFiles(sorted(qfiles), args.mri, args.bufferSize) 
else:
    # Parse the qfiles and pull out equally spaced in time records
    decimateFiles(qfiles, args.mri, totSize, args.maxSize)
