#! /usr/bin/env python3
#
# Display the Q-file's header record
#
# Nov-2024, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import struct
import numpy as np

def main():
    parser = ArgumentParser()
    parser.add_argument("fn", type=str, help="Input Q-Filename")
    args = parser.parse_args()

    with open(args.fn, "rb") as fp:
        buffer = fp.read(20) # Read the fixed initial block
        if len(buffer) != 20:
            print("Initial buffer read failed,", len(buffer))
            raise EOFError

        (ident, version, dt, Nc, Ns, Nf) = struct.unpack("<HfQHHH", buffer)

        t = np.datetime64("0000-01-01") + np.timedelta64(dt, "ms")

        print(f"{ident:#04x} identifier, should be 0x1729")
        print(f"{version} file version")
        print(f"{dt:#07x} dt {t} time")
        print(Nc, "Nc")
        print(Ns, "Ns")
        print(Nf, "Nf")

        buffer = fp.read(Nc * 2)
        if len(buffer) != (Nc * 2):
            print("Problem reading spectra identifiers,", len(buffer), "!=", Nc*2)
            raise EOFError

        idents = struct.unpack("<" + ("H" * Nc), buffer)
        idents = list(map(lambda x: f"{x:#04x}", idents))
        print("Channel identifiers")
        print(idents)

        buffer = fp.read(Ns * 2)
        if len(buffer) != (Ns * 2):
            print("Problem reading spectra identifiers,", len(buffer), "!=", Ns*2)
            raise EOFError

        idents = struct.unpack("<" + ("H" * Ns), buffer)
        idents = list(map(lambda x: f"{x:#04x}", idents))
        print("Spectra identifiers")
        print(idents)

        buffer = fp.read(Nf * 2)
        if len(buffer) != (Nf * 2):
            print("Problem reading frequencies,", len(buffer), "!=", Nf*2)
            raise EOFError

        idents = struct.unpack("<" + ("e" * Nf), buffer)
        print("Frequencies")
        print(idents)

        buffer = fp.read(4) # Grab configuration record
        if len(buffer) != 4:
            print("Problem reading configuratio record,", len(buffer), "!= 4")
            raise EOFError
        (ident, sz) = struct.unpack("<HH", buffer)
        print(f"Configuration ident {ident:#04x} size {sz}")

        buffer= fp.read(sz)
        try:
            print(str(buffer, "utf-8"))
        except:
            print(buffer)

        buffer = fp.read(2)
        if len(buffer) != 2:
            print("Problem reading data record size,", len(buffer), "!= 2")
            raise EOFError
        sz = struct.unpack("<H", buffer)
        print("Data record size", sz)

if __name__ == "__main__":
    main()
