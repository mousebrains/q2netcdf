#!/usr/bin/env python3
"""
Example: Reduce Q-file size by selecting channels/spectra.

This script demonstrates how to:
- Create a reduced Q-file with only selected channels
- Use JSON configuration for reduction
- Reduce file size while preserving important data

Usage:
    python reduce_qfile.py input.q -o output.q --config reduction.json
    python reduce_qfile.py input.q -o output.q --channels sh_0 sh_1
"""

from argparse import ArgumentParser
import sys
import json
from pathlib import Path

try:
    from q2netcdf.QReduce import QReduce
    from q2netcdf.QHexCodes import QHexCodes
except ImportError:
    print("Error: q2netcdf not installed. Install with: pip install -e .")
    sys.exit(1)


def create_config(
    channels: list[str] | None = None,
    spectra: list[str] | None = None,
    config_params: list[str] | None = None
) -> dict:
    """
    Create reduction configuration dictionary.

    Args:
        channels: List of channel names to keep
        spectra: List of spectra names to keep
        config_params: List of config parameter names to keep

    Returns:
        Configuration dictionary
    """
    return {
        "channels": channels or [],
        "spectra": spectra or [],
        "config": config_params or []
    }


def reduce_qfile(
    input_file: str,
    output_file: str,
    config: dict
) -> None:
    """
    Reduce Q-file using configuration.

    Args:
        input_file: Input Q-file path
        output_file: Output Q-file path
        config: Reduction configuration
    """
    print(f"Reducing {input_file}...")
    print(f"Configuration: {json.dumps(config, indent=2)}\n")

    # Create QReduce object
    qr = QReduce(input_file, config)

    # Display reduction information
    print("Reduction Information:")
    print(f"  Original header size:  {qr.hdrSizeOrig:,} bytes")
    print(f"  Reduced header size:   {qr.hdrSize:,} bytes")
    print(f"  Original data size:    {qr.dataSizeOrig:,} bytes/record")
    print(f"  Reduced data size:     {qr.dataSize:,} bytes/record")
    print(f"  Original file size:    {qr.fileSizeOrig:,} bytes")
    print(f"  Reduced file size:     {qr.fileSize:,} bytes")
    print(f"  Size reduction:        {100 * (1 - qr.fileSize / qr.fileSizeOrig):.1f}%")
    print()

    # Write reduced file
    print(f"Writing to {output_file}...")
    with open(output_file, "wb") as ofp:
        size = qr.reduceFile(ofp)

    print(f"Done! Wrote {size:,} bytes")

    # Verify output
    if Path(output_file).exists():
        actual_size = Path(output_file).stat().st_size
        print(f"Output file size: {actual_size:,} bytes")
    else:
        print("Warning: Output file not created")


def list_available_channels(input_file: str) -> None:
    """
    List all available channels in Q-file.

    Args:
        input_file: Input Q-file path
    """
    from q2netcdf.QHeader import QHeader

    print(f"Available channels in {input_file}:\n")

    hexmap = QHexCodes()

    with open(input_file, "rb") as fp:
        header = QHeader(fp, input_file)

        if header.Nc > 0:
            print("Channels:")
            for ident in header.channels:
                name = hexmap.name(ident)
                attrs = hexmap.attributes(ident)
                long_name = attrs.get('long_name', '') if attrs else ''
                print(f"  {name:15s} - {long_name}")

        if header.Ns > 0:
            print("\nSpectra:")
            for ident in header.spectra:
                name = hexmap.name(ident)
                attrs = hexmap.attributes(ident)
                long_name = attrs.get('long_name', '') if attrs else ''
                print(f"  {name:15s} - {long_name}")


def main() -> None:
    """Parse arguments and reduce Q-file."""
    parser = ArgumentParser(description="Reduce Q-file size by selecting channels/spectra")
    parser.add_argument("input", help="Input Q-file")
    parser.add_argument(
        "-o", "--output",
        help="Output Q-file (default: input_reduced.q)"
    )
    parser.add_argument(
        "--config",
        help="JSON configuration file"
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Channel names to keep (e.g., sh_0 sh_1)"
    )
    parser.add_argument(
        "--spectra",
        nargs="+",
        help="Spectra names to keep"
    )
    parser.add_argument(
        "--config-params",
        nargs="+",
        help="Configuration parameters to keep"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available channels and exit"
    )
    args = parser.parse_args()

    # Check input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # List channels if requested
    if args.list:
        try:
            list_available_channels(args.input)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    # Determine output file
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_reduced{input_path.suffix}")

    # Load or create configuration
    if args.config:
        config = QReduce.loadConfig(args.config)
        if config is None:
            print(f"Error: Could not load configuration from {args.config}")
            sys.exit(1)
    elif args.channels or args.spectra or args.config_params:
        config = create_config(
            channels=args.channels,
            spectra=args.spectra,
            config_params=args.config_params
        )
    else:
        print("Error: Must specify either --config or at least one of --channels/--spectra/--config-params")
        print("Use --list to see available channels")
        sys.exit(1)

    try:
        reduce_qfile(args.input, args.output, config)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
