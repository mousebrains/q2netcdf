#!/usr/bin/env python3
"""
Example: Reading and inspecting Q-file contents.

This script demonstrates how to:
- Open a Q-file using the QFile context manager
- Read the header to get metadata
- Iterate through data records
- Access channel and spectra data

Usage:
    python read_qfile.py <qfile.q>
    python read_qfile.py --max-records 5 <qfile.q>
"""

from argparse import ArgumentParser
import sys

try:
    from q2netcdf.QFile import QFile
    from q2netcdf.QHexCodes import QHexCodes
except ImportError:
    print("Error: q2netcdf not installed. Install with: pip install -e .")
    sys.exit(1)


def read_qfile(filename: str, max_records: int = 10) -> None:
    """
    Read and display Q-file contents.

    Args:
        filename: Path to Q-file
        max_records: Maximum number of data records to display
    """
    print(f"Reading Q-file: {filename}\n")

    hexmap = QHexCodes()

    # Open Q-file using context manager
    with QFile(filename) as qf:
        # Read header
        header = qf.header()

        print("=" * 60)
        print("HEADER INFORMATION")
        print("=" * 60)
        print(f"Version:        {header.version}")
        print(f"Start Time:     {header.time}")
        print(f"Channels:       {header.Nc}")
        print(f"Spectra:        {header.Ns}")
        print(f"Frequencies:    {header.Nf}")
        print(f"Data Size:      {header.dataSize} bytes")
        print(f"Header Size:    {header.hdrSize} bytes")

        # Display channel names
        if header.Nc > 0:
            print(f"\nChannel Identifiers:")
            for i, ident in enumerate(header.channels):
                name = hexmap.name(ident)
                attrs = hexmap.attributes(ident)
                long_name = attrs.get('long_name', 'unknown') if attrs else 'unknown'
                print(f"  [{i}] {ident:#06x} -> {name:15s} ({long_name})")

        # Display spectra names
        if header.Ns > 0:
            print(f"\nSpectra Identifiers:")
            for i, ident in enumerate(header.spectra):
                name = hexmap.name(ident)
                attrs = hexmap.attributes(ident)
                long_name = attrs.get('long_name', 'unknown') if attrs else 'unknown'
                print(f"  [{i}] {ident:#06x} -> {name:15s} ({long_name})")

        # Display configuration
        if header.config:
            config = header.config.config()
            print(f"\nConfiguration ({len(config)} parameters):")
            for key in sorted(config):
                print(f"  {key:20s} = {config[key]}")

        # Read and display data records
        print("\n" + "=" * 60)
        print(f"DATA RECORDS (showing first {max_records})")
        print("=" * 60)

        record_count = 0
        for record in qf.data():
            record_count += 1

            print(f"\nRecord #{record_count}:")
            print(f"  Time:     {record.t0}")
            if record.t1 is not None:
                print(f"  End Time: {record.t1}")
            if record.number is not None:
                print(f"  Number:   {record.number}")
            if record.error is not None:
                print(f"  Error:    {record.error}")

            # Display channel values
            print(f"  Channels ({len(record.channels)}):")
            for i in range(min(5, len(record.channels))):  # Show first 5
                ident = header.channels[i]
                name = hexmap.name(ident) or f"{ident:#06x}"
                print(f"    {name:15s} = {record.channels[i]:10.4f}")
            if len(record.channels) > 5:
                print(f"    ... and {len(record.channels) - 5} more")

            # Display spectra info (not all values, just shape)
            if record.spectra.size > 0:
                print(f"  Spectra shape: {record.spectra.shape}")

            if record_count >= max_records:
                break

        print(f"\nTotal records read: {record_count}")


def main() -> None:
    """Parse arguments and read Q-file."""
    parser = ArgumentParser(description="Read and inspect Q-file contents")
    parser.add_argument("qfile", help="Path to Q-file")
    parser.add_argument(
        "--max-records",
        type=int,
        default=10,
        help="Maximum number of records to display (default: 10)"
    )
    args = parser.parse_args()

    try:
        read_qfile(args.qfile, args.max_records)
    except FileNotFoundError:
        print(f"Error: File not found: {args.qfile}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading Q-file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
