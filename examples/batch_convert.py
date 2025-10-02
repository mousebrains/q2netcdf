#!/usr/bin/env python3
"""
Example: Batch convert multiple Q-files to NetCDF.

This script demonstrates how to:
- Process entire directories of Q-files
- Handle errors gracefully
- Show progress for large batches
- Generate summary reports

Usage:
    python batch_convert.py /path/to/qfiles/ /path/to/output/
    python batch_convert.py /path/to/qfiles/ /path/to/output/ --pattern "*.q"
    python batch_convert.py /path/to/qfiles/ /path/to/output/ --one-file
"""

from argparse import ArgumentParser
import sys
from pathlib import Path
from typing import List
import time

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
    Load single Q-file to Dataset.

    Args:
        filename: Path to Q-file

    Returns:
        Dataset or None on error
    """
    try:
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

                    t0 = record_data["time"]
                    del record_data["time"]

                    values = {}
                    has_freq = False

                    for key in record_data:
                        val = record_data[key]
                        if np.isscalar(val):
                            values[key] = ("time", [val], attrs.get(key))
                        else:
                            has_freq = True
                            values[key] = (("time", "freq"), val.reshape(1, -1), attrs.get(key))

                    coords = {"time": [t0]}
                    if has_freq:
                        coords["freq"] = np.array(hdr.frequencies)

                    ds = xr.Dataset(data_vars=values, coords=coords)
                    records.append(ds)
                else:
                    buffer = fp.read(2)
                    if len(buffer) != 2:
                        break

        if hdr is None or not records:
            return None

        ds = xr.concat(records, "time")
        ftime = ds.time.data.min()
        ds = ds.assign_coords(ftime=[ftime])
        ds = ds.assign({"fileversion": ("ftime", [hdr.version.value])})

        return ds

    except Exception as e:
        print(f"  Error loading {filename}: {e}")
        return None


def find_qfiles(directory: str, pattern: str = "*.q") -> List[Path]:
    """
    Find all Q-files in directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern for Q-files

    Returns:
        List of Path objects
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    files = sorted(dir_path.glob(pattern))
    return files


def batch_convert_separate(
    qfiles: List[Path],
    output_dir: Path,
    compression: int = 5
) -> dict:
    """
    Convert each Q-file to separate NetCDF.

    Args:
        qfiles: List of Q-file paths
        output_dir: Output directory
        compression: Compression level

    Returns:
        Summary statistics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": len(qfiles),
        "success": 0,
        "failed": 0,
        "total_input_size": 0,
        "total_output_size": 0,
        "start_time": time.time()
    }

    for i, qfile in enumerate(qfiles, 1):
        print(f"[{i}/{len(qfiles)}] Processing {qfile.name}...", end=" ")

        # Load Q-file
        ds = load_qfile(str(qfile))
        if ds is None:
            print("FAILED")
            stats["failed"] += 1
            continue

        # Generate output filename
        output_file = output_dir / f"{qfile.stem}.nc"

        # Add compression
        if compression > 0:
            for name in ds:
                if ds[name].dtype.kind != "U":
                    ds[name].encoding = {
                        "compression": "zlib",
                        "compression_level": compression
                    }

        # Write NetCDF
        try:
            ds.to_netcdf(output_file)
            print(f"OK -> {output_file.name}")
            stats["success"] += 1
            stats["total_input_size"] += qfile.stat().st_size
            stats["total_output_size"] += output_file.stat().st_size
        except Exception as e:
            print(f"FAILED: {e}")
            stats["failed"] += 1

    stats["elapsed_time"] = time.time() - stats["start_time"]
    return stats


def batch_convert_merged(
    qfiles: List[Path],
    output_file: Path,
    compression: int = 5
) -> dict:
    """
    Convert all Q-files to single merged NetCDF.

    Args:
        qfiles: List of Q-file paths
        output_file: Output NetCDF path
        compression: Compression level

    Returns:
        Summary statistics dictionary
    """
    stats = {
        "total": len(qfiles),
        "success": 0,
        "failed": 0,
        "total_input_size": 0,
        "total_output_size": 0,
        "start_time": time.time()
    }

    datasets = []

    for i, qfile in enumerate(qfiles, 1):
        print(f"[{i}/{len(qfiles)}] Loading {qfile.name}...", end=" ")

        ds = load_qfile(str(qfile))
        if ds is None:
            print("FAILED")
            stats["failed"] += 1
            continue

        print("OK")
        datasets.append(ds)
        stats["success"] += 1
        stats["total_input_size"] += qfile.stat().st_size

    if not datasets:
        raise ValueError("No Q-files loaded successfully")

    # Merge all datasets
    print(f"\nMerging {len(datasets)} datasets...")
    merged = xr.concat(datasets, "time")

    # Add compression
    if compression > 0:
        for name in merged:
            if merged[name].dtype.kind != "U":
                merged[name].encoding = {
                    "compression": "zlib",
                    "compression_level": compression
                }

    # Write merged file
    print(f"Writing to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(output_file)

    stats["total_output_size"] = output_file.stat().st_size
    stats["elapsed_time"] = time.time() - stats["start_time"]
    return stats


def print_summary(stats: dict) -> None:
    """Print conversion summary."""
    print("\n" + "=" * 60)
    print("BATCH CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total files:      {stats['total']}")
    print(f"Successful:       {stats['success']}")
    print(f"Failed:           {stats['failed']}")
    print(f"Success rate:     {100 * stats['success'] / stats['total']:.1f}%")
    print(f"Input size:       {stats['total_input_size']:,} bytes")
    print(f"Output size:      {stats['total_output_size']:,} bytes")
    if stats['total_input_size'] > 0:
        ratio = 100 * (1 - stats['total_output_size'] / stats['total_input_size'])
        print(f"Size reduction:   {ratio:.1f}%")
    print(f"Elapsed time:     {stats['elapsed_time']:.2f} seconds")
    print("=" * 60)


def main() -> None:
    """Parse arguments and batch convert."""
    parser = ArgumentParser(description="Batch convert Q-files to NetCDF")
    parser.add_argument("input_dir", help="Input directory containing Q-files")
    parser.add_argument("output_dir", help="Output directory for NetCDF files")
    parser.add_argument(
        "--pattern",
        default="*.q",
        help="Glob pattern for Q-files (default: *.q)"
    )
    parser.add_argument(
        "--one-file",
        action="store_true",
        help="Merge all Q-files into single NetCDF"
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=5,
        choices=range(10),
        help="Compression level 0-9 (default: 5)"
    )
    args = parser.parse_args()

    try:
        # Find Q-files
        print(f"Searching for Q-files in {args.input_dir}...")
        qfiles = find_qfiles(args.input_dir, args.pattern)

        if not qfiles:
            print(f"No Q-files found matching pattern: {args.pattern}")
            sys.exit(1)

        print(f"Found {len(qfiles)} Q-file(s)\n")

        # Convert
        output_path = Path(args.output_dir)

        if args.one_file:
            output_file = output_path / "merged.nc"
            stats = batch_convert_merged(qfiles, output_file, args.compression)
        else:
            stats = batch_convert_separate(qfiles, output_path, args.compression)

        # Print summary
        print_summary(stats)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
