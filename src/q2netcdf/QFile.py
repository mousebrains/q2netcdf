#! /usr/bin/env python3
#
# Decode QFiles exposing header, config, and data records
#
# Feb-2025, Pat Welch, pat@mousebrains.com

import os.path
import logging
from QHeader import QHeader
from QData import QData, QRecord

class QFile:
    def __init__(self, fn:str) -> None:
        self.__fn = os.path.abspath(os.path.expanduser(fn))
        self.__fp = None
        self.__data = None

        if not os.path.isfile(fn):
            raise FileNotFoundError(f"%s does not exist {fn}")

    def __enter__(self):
        self.__maybeOpen__()
        return self

    def __exit__(self, *args):
        if self.__fp:
            if not self.__fp.closed: self.__fp.close()
            self.__fp = None

    def __del__(self) -> None:
        self.__exit__()

    def __maybeOpen__(self):
        if not self.__fp or self.__fp.closed: 
            self.__fp = open(self.__fn, "rb")
        return self.__fp

    def header(self) -> QHeader:
        fp = self.__maybeOpen__()
        fp.seek(0) # Rewind to beginning
        hdr = QHeader(fp, self.__fn)
        self.__data = QData(hdr)
        return hdr

    def data(self) -> QRecord:
        if not self.__data: 
            raise EOFError(f"A header must be before any data records in {self.__fn}")
        
        return self.__data.load(self.__fp)

    def prettyRecord(self, record:QRecord) -> str:
        return self.__data.prettyRecord(record) if self.__data else None

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("filename", type=str, nargs="+", help="Input filename(s)")
    parser.add_argument("--n", type=int, default=10, help="Number of data records to print out")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    for fn in args.filename:
        try:
            with QFile(fn) as qf:
                logging.info("fn %s", fn)
                logging.info("QFile %s", qf)
                hdr = qf.header() # First record should be a header+config record
                logging.info("hdr %s", hdr)
                for cnt in range(args.n):
                    record = qf.data()
                    if not record: break
                    logging.info("Record %s", qf.prettyRecord(record))
        except EOFError:
            logging.info("EOF while reading %s", fn)
        except:
            logging.exception("While reading %s", fn)

if __name__ == "__main__":
    main()
