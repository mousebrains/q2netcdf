#!/usr/bin/env python3
"""
Example: Detailed Q-file header inspection.

This script demonstrates how to:
- Parse Q-file headers
- Display all metadata fields
- Show sensor identifier mappings
- Examine configuration parameters

Usage:
    python inspect_header.py <qfile.q>
    python inspect_header.py --show-all <qfile.q>
"""

from argparse import ArgumentParser
import sys

try:
    from q2netcdf.QHeader import QHeader
    from q2netcdf.QHexCodes import QHexCodes
    from q2netcdf.QVersion import QVersion
except ImportError:
    print("Error: q2netcdf not installed. Install with: pip install -e .")
    sys.exit(1)


def inspect_header(filename: str, show_all: bool = False) -> None:
    """
    Inspect Q-file header in detail.

    Args:
        filename: Path to Q-file
        show_all: Show all details including frequencies
    """
    print(f"Inspecting header: {filename}\n")

    hexmap = QHexCodes()

    with open(filename, "rb") as fp:
        # Parse header
        header = QHeader(fp, filename)

        # Basic information
        print("=" * 70)
        print("FILE INFORMATION")
        print("=" * 70)
        print(f"Filename:           {header.filename}")
        print(f"Q-File Version:     {header.version} ({header.version.value})")
        print(f"Start Time:         {header.time}")
        print(f"Binary Timestamp:   {header.dtBinary} ms since year 0")
        print()

        # Size information
        print("=" * 70)
        print("SIZE INFORMATION")
        print("=" * 70)
        print(f"Header Size:        {header.hdrSize:,} bytes")
        print(f"Data Record Size:   {header.dataSize:,} bytes")
        print(f"Number of Channels: {header.Nc}")
        print(f"Number of Spectra:  {header.Ns}")
        print(f"Number of Freqs:    {header.Nf}")
        print()

        # Channel identifiers
        if header.Nc > 0:
            print("=" * 70)
            print(f"CHANNEL IDENTIFIERS ({header.Nc} channels)")
            print("=" * 70)
            print(f"{'Idx':<5} {'Hex ID':<8} {'Name':<20} {'Long Name':<30} {'Units':<10}")
            print("-" * 70)

            for i, ident in enumerate(header.channels):
                name = hexmap.name(ident) or "unknown"
                attrs = hexmap.attributes(ident)
                if attrs:
                    long_name = attrs.get('long_name', '-')
                    units = attrs.get('units', '-')
                else:
                    long_name = '-'
                    units = '-'

                print(f"{i:<5} {ident:#06x}   {name:<20} {long_name:<30} {units:<10}")
            print()

        # Spectra identifiers
        if header.Ns > 0:
            print("=" * 70)
            print(f"SPECTRA IDENTIFIERS ({header.Ns} spectra)")
            print("=" * 70)
            print(f"{'Idx':<5} {'Hex ID':<8} {'Name':<20} {'Long Name':<30} {'Units':<10}")
            print("-" * 70)

            for i, ident in enumerate(header.spectra):
                name = hexmap.name(ident) or "unknown"
                attrs = hexmap.attributes(ident)
                if attrs:
                    long_name = attrs.get('long_name', '-')
                    units = attrs.get('units', '-')
                else:
                    long_name = '-'
                    units = '-'

                print(f"{i:<5} {ident:#06x}   {name:<20} {long_name:<30} {units:<10}")
            print()

        # Frequency array
        if header.Nf > 0 and show_all:
            print("=" * 70)
            print(f"FREQUENCY ARRAY ({header.Nf} frequencies)")
            print("=" * 70)
            freqs = header.frequencies
            print(f"Min:  {min(freqs):.4f} Hz")
            print(f"Max:  {max(freqs):.4f} Hz")
            print(f"Step: {(max(freqs) - min(freqs)) / (len(freqs) - 1):.4f} Hz")
            if len(freqs) <= 20:
                print("\nAll frequencies:")
                for i, f in enumerate(freqs):
                    print(f"  [{i:2d}] {f:8.4f} Hz")
            else:
                print("\nFirst 10 frequencies:")
                for i in range(10):
                    print(f"  [{i:2d}] {freqs[i]:8.4f} Hz")
                print(f"  ... and {len(freqs) - 10} more")
            print()

        # Configuration
        if header.config:
            config = header.config.config()
            print("=" * 70)
            print(f"CONFIGURATION ({len(config)} parameters)")
            print("=" * 70)
            print(f"{'Parameter':<25} {'Value':<45}")
            print("-" * 70)

            for key in sorted(config):
                value = config[key]
                if isinstance(value, (list, tuple)):
                    value_str = f"[{', '.join(str(v) for v in value)}]"
                else:
                    value_str = str(value)

                # Truncate long values
                if len(value_str) > 45:
                    value_str = value_str[:42] + "..."

                print(f"{key:<25} {value_str:<45}")
            print()

        # Version-specific information
        print("=" * 70)
        print("VERSION DETAILS")
        print("=" * 70)
        if header.version == QVersion.v12:
            print("Version 1.2: Original format with full metadata in each record")
            print("  - Record number included")
            print("  - Error code included")
            print("  - End time (t1) included")
        elif header.version == QVersion.v13:
            print("Version 1.3: Optimized format with reduced redundancy")
            print("  - Record number removed")
            print("  - Error code removed")
            print("  - End time removed")
        print()


def main() -> None:
    """Parse arguments and inspect header."""
    parser = ArgumentParser(description="Inspect Q-file header in detail")
    parser.add_argument("qfile", help="Path to Q-file")
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all details including frequency arrays"
    )
    args = parser.parse_args()

    try:
        inspect_header(args.qfile, args.show_all)
    except FileNotFoundError:
        print(f"Error: File not found: {args.qfile}")
        sys.exit(1)
    except Exception as e:
        print(f"Error inspecting header: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
