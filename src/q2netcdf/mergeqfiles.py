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

from argparse import ArgumentParser
import os
import time
import numpy as np
import logging
import math
import sys
import yaml
import json
import struct
try:
    from QFile import QFile
    from QHeader import QHeader
    from QConfig import QConfig
    from QReduce import QReduce
except:
    from q2netcdf.QFile import QFile
    from q2netcdf.QHeader import QHeader
    from q2netcdf.QConfig import QConfig
    from q2netcdf.QReduce import QReduce

def reduceAndDecimate(info:dict, ofp, ofn:str, maxSize:int) -> int:
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
        return ofp.tell()

    logging.info("Sizes max %s avail %s ratio %s", maxSize, availSize, ratio)

    for fn in sorted(info):
        qr = info[fn]
        indices = np.unique(
                np.floor(
                    np.linspace(
                        0, 
                        qr.nRecords, 
                        np.floor(qr.nRecords * ratio).astype(int), 
                        endpoint=False)
                    ).astype(int)
                )
        sz = qr.decimate(ofp, indices)
        logging.info("Decimated %s to %s -> %s -> %s n=%s/%s", 
                     qr.filename, ofn, qr.fileSizeOrig, sz, indices.size, qr.nRecords.astype(int))
    return ofp.tell()

def reduceFiles(qFiles:dict, fnConfig:str, ofn:str, maxSize:int) -> int:
    qrConfig = QReduce.loadConfig(fnConfig)
    logging.info("Config %s -> %s", fnConfig, qrConfig)

    info = {}
    totSize = 0

    for fn in qFiles:
        qr = QReduce(fn, qrConfig)
        totSize += qr.fileSize
        info[fn] = qr

    with open(ofn, "ab") as ofp:
        if totSize <= maxSize: # no need to decimate, so append glued reduced files
            for fn in sorted(info):
                qr = info[fn]
                sz = qr.reduceFile(ofp)
                logging.info("Appending %s to %s, %s -> %s", fn, ofn, qr.fileSizeOrig, sz)
        else:
            reduceAndDecimate(info, ofp, ofn, maxSize)
        return ofp.tell() # Actual file size


def decimateFiles(qFiles:dict, ofn:str, totSize:int, maxSize:int) -> int:
    try:
        filenames = sorted(qFiles, reverse=False) # sorted filenames to work on
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
                            (st.st_size - item["hdrSize"]).astype(int) / item["dataSize"]
                            )
                    logging.info("%s hdr %s data %s n %s",
                                 ifn, item["hdrSize"], item["dataSize"], item["nRecords"])
                    info[ifn] = item
                    totHdrSize += item["hdrSize"]
                    totDataSize += item["dataSize"] * item["nRecords"]
            except EOFError:
                pass
            except:
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
                            np.linspace(0, n, math.floor(n * ratio), endpoint=False))
                        .astype(int)
                        )
                hdrSize = item["hdrSize"]
                dataSize = item["dataSize"]
                offsets = hdrSize + indices * dataSize
                logging.info("Decimating file %s hdr sz %s data sz %s n %s of %s", 
                             ifn, hdrSize, dataSize, len(offsets), item["nRecords"])
                with open(ifn, "rb") as ifp:
                    buffer = ifp.read(hdrSize)
                    if len(buffer) != hdrSize: continue
                    ofp.write(buffer)
                    for offset in offsets:
                        ifp.seek(offset)
                        buffer = ifp.read(dataSize)
                        if len(buffer) != dataSize: break
                        ofp.write(buffer)
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

def fileCandidates(args:ArgumentParser, times:np.array) -> tuple:
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
            logger("%s sz %s mtime %s %s",
                   entry.name,
                   st.st_size,
                   np.datetime64(round(st.st_mtime * 1000), "ms"),
                   qKeep
                   )
            if qKeep:
                qFiles[entry.path] = st.st_size
                totSize += st.st_size
        return (qFiles, totSize)

def scanDirectory(args:ArgumentParser, times:np.array) -> int:
    (qFiles, totSize) = fileCandidates(args, times)

    if os.path.isfile(args.output): # File already exist, so reduce maxsize
        fSize = os.path.getsize(args.output)

        if not qFiles: # No Q-files found, nothing to do
            logging.info("No new files to append to %s", args.output)
            return fSize 

        args.maxSize -= fSize
        if args.maxSize <= 0:
            logging.info("Can't append more to file, %s >= %s", fSize, args.maxSize)
            return fSize
        logging.info("Reduced maxsize by %s since file already exists", fSize)
    elif not qFiles: # No q-files, so create empty file and return 0
        with open(args.output, "wb") as fp:
            pass
        logging.info("No new files, so created an empty file %s", args.output)
        return 0

    if args.config and os.path.isfile(args.config): # We're going to reduce the size of the Q-files,
        return reduceFiles(qFiles, args.config, args.output, args.maxSize)

    logging.info("Total size %s max %s", totSize, args.maxSize)


    if totSize <= args.maxSize:
        # Glue the files together since their total size is small enough
        return glueFiles(sorted(qFiles, reverse=False), args.output, args.bufferSize)

    # Parse the Q-files and pull out roughly equally spaced in time records
    return decimateFiles(qFiles, args.output, totSize, args.maxSize)

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
    parser.add_argument("--config", type=str, help="YAML config file")
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

        if args.config is None:
            fn = os.path.abspath(os.path.expanduser(os.path.join(args.datadir, "mergeqfiles.yaml")))
            if os.path.isfile(fn): args.config = fn
        elif args.config == "": 
            args.config = None
        else:
            args.config = os.path.abspath(os.path.expanduser(args.config))

        if args.stime <= 0:
            args.stime = time.time() # Current time

        logging.info("Args: %s", args)

        times = np.sort([args.stime, args.stime + args.dt + args.safety])

        logging.info("Time limits %s", times.astype("datetime64[s]"))

        outSize = scanDirectory(args, times)
        logging.info("printing outSize %s to console", outSize)
        print(outSize)
    except:
        logging.exception("Unexpected exception executing %s", args)


if __name__ == "__main__":
    main()
