#!/usr/bin/env python3
"""
Example: Convert Q-file to NetCDF format.

This script demonstrates how to:
- Load Q-files as xarray Datasets
- Add CF-1.8 compliant metadata
- Write to NetCDF format with compression
- Handle multiple Q-files

Usage:
    python convert_to_netcdf.py input.q -o output.nc
    python convert_to_netcdf.py file1.q file2.q file3.q -o merged.nc
    python convert_to_netcdf.py input.q -o output.nc --compression 9
"""

from argparse import ArgumentParser
import sys
from pathlib import Path

try:
    import xarray as xr
    import numpy as np
except ImportError:
    print("Error: xarray and numpy required. Install with: pip install xarray numpy")
    sys.exit(1)

try:
    from q2netcdf.QFile import QFile
    from q2netcdf.QHeader import QHeader
    from q2netcdf.QData import QData
except ImportError:
    print("Error: q2netcdf not installed. Install with: pip install -e .")
    sys.exit(1)


def load_qfile(filename: str) -> xr.Dataset | None:
    """
    Load a Q-file and convert to xarray Dataset.

    Args:
        filename: Path to Q-file

    Returns:
        xarray Dataset or None if file is invalid
    """
    print(f"Loading {filename}...")

    records = []
    hdr = None

    with open(filename, "rb") as fp:
        data = None
        while True:
            if QHeader.chkIdent(fp):
                hdr = QHeader(fp, filename)
                if hdr is None:
                    break
                data = QData(hdr)
            elif QData.chkIdent(fp):
                record = data.load(fp)
                if record is None:
                    break
                record_data, attrs = record.split(hdr)

                # Split into scalar channels and spectra
                t0 = record_data["time"]
                del record_data["time"]

                values = {}
                has_freq = False

                for key in record_data:
                    val = record_data[key]
                    if np.isscalar(val):
                        values[key] = (
                            "time",
                            [val],
                            attrs.get(key),
                        )
                    else:
                        has_freq = True
                        values[key] = (
                            ("time", "freq"),
                            val.reshape(1, -1),
                            attrs.get(key),
                        )

                coords = {"time": [t0]}
                if has_freq:
                    coords["freq"] = np.array(hdr.frequencies)

                ds = xr.Dataset(data_vars=values, coords=coords)
                records.append(ds)
            else:
                buffer = fp.read(2)
                if len(buffer) != 2:
                    break
                print(f"Warning: Unknown record type in {filename}")

    if hdr is None:
        print(f"Warning: No header found in {filename}")
        return None

    if not records:
        print(f"Warning: No data records found in {filename}")
        return None

    # Concatenate all records along time dimension
    ds = xr.concat(records, "time")

    # Add file-level metadata
    ftime = ds.time.data.min()
    ds = ds.assign_coords(ftime=[ftime])

    # Add file version
    ds = ds.assign({"fileversion": ("ftime", [hdr.version.value])})

    # Add configuration parameters
    config = hdr.config.config()
    for key in config:
        val = config[key]
        if np.isscalar(val):
            ds = ds.assign({key: ("ftime", [val])})

    return ds


def add_cf_metadata(ds: xr.Dataset) -> xr.Dataset:
    """
    Add CF-1.8 compliant metadata to Dataset.

    Args:
        ds: Input Dataset

    Returns:
        Dataset with CF metadata
    """
    # Add standard attributes
    ds.attrs.update({
        "Conventions": "CF-1.8",
        "title": "NetCDF translation of Rockland's Q-File(s)",
        "keywords": "turbulence, ocean, microstructure",
        "summary": "See Rockland's TN-054 for description of Q-Files",
        "time_coverage_start": str(ds.time.data.min()),
        "time_coverage_end": str(ds.time.data.max()),
        "time_coverage_duration": str(ds.time.data.max() - ds.time.data.min()),
        "date_created": str(np.datetime64("now")),
    })

    return ds


def add_compression(ds: xr.Dataset, level: int = 5) -> xr.Dataset:
    """
    Add compression encoding to all variables.

    Args:
        ds: Input Dataset
        level: Compression level (0-9)

    Returns:
        Dataset with compression encoding
    """
    if level <= 0:
        return ds

    for name in ds:
        if ds[name].dtype.kind != "U":  # Skip string variables
            ds[name].encoding = {
                "compression": "zlib",
                "compression_level": level
            }

    return ds


def convert_qfiles(
    qfiles: list[str],
    output: str,
    compression: int = 5
) -> None:
    """
    Convert Q-files to NetCDF.

    Args:
        qfiles: List of Q-file paths
        output: Output NetCDF path
        compression: Compression level (0-9)
    """
    # Load all Q-files
    datasets = []
    for qfile in qfiles:
        if not Path(qfile).exists():
            print(f"Error: File not found: {qfile}")
            continue

        ds = load_qfile(qfile)
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        print("Error: No valid Q-files loaded")
        sys.exit(1)

    # Merge if multiple files
    if len(datasets) > 1:
        print(f"\nMerging {len(datasets)} datasets...")
        ds = xr.concat(datasets, "time")
    else:
        ds = datasets[0]

    # Add CF metadata
    print("Adding CF-1.8 metadata...")
    ds = add_cf_metadata(ds)

    # Add compression
    if compression > 0:
        print(f"Adding compression (level {compression})...")
        ds = add_compression(ds, compression)

    # Write to NetCDF
    print(f"\nWriting to {output}...")
    ds.to_netcdf(output)

    # Report file sizes
    total_input = sum(Path(f).stat().st_size for f in qfiles if Path(f).exists())
    output_size = Path(output).stat().st_size

    print("\nConversion complete!")
    print(f"Input size:  {total_input:,} bytes")
    print(f"Output size: {output_size:,} bytes")
    print(f"Compression: {100 * (1 - output_size / total_input):.1f}%")
    print(f"Records:     {len(ds.time)}")


def main() -> None:
    """Parse arguments and convert Q-files."""
    parser = ArgumentParser(description="Convert Q-files to NetCDF format")
    parser.add_argument("qfiles", nargs="+", help="Input Q-file(s)")
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output NetCDF filename"
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=5,
        choices=range(10),
        help="Compression level 0-9 (default: 5, 0=none)"
    )
    args = parser.parse_args()

    try:
        convert_qfiles(args.qfiles, args.output, args.compression)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
