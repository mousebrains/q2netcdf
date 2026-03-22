#! /usr/bin/env python3
#
# Translate Rockland ISDP Q file(s) to a NetCDF file
#
# Based on Rockland TN 054
#
# Oct-2024, Pat Welch, pat@mousebrains.com
# Feb-2025, Pat Welch, pat@mousebrains.com update to using QFile...

from argparse import ArgumentParser
import numpy as np
import xarray as xr
import logging
import os
import sys
import struct
from typing import Any
from .QHeader import QHeader
from .QData import QData, QRecord
from .QHexCodes import QHexCodes


def _buildSegmentDataset(
    hdr: QHeader,
    qrecords: list[QRecord],
    hexMap: QHexCodes,
) -> xr.Dataset:
    """Build a single xr.Dataset from a header and its associated records.

    Instead of creating one Dataset per record and concatenating, this
    builds numpy arrays across all records first, then constructs a
    single Dataset.
    """
    # Times
    times = np.array([r.t0 for r in qrecords])

    data_vars: dict[str, Any] = {}

    # v1.2 extra fields (record number, error code, end time)
    if qrecords[0].t1 is not None:
        data_vars["t1"] = (
            "time",
            np.array([r.t1 for r in qrecords]),
            {"long_name": "timeStop"},
        )
    if qrecords[0].number is not None:
        data_vars["record"] = (
            "time",
            np.array([r.number for r in qrecords]),
            {"long_name": "recordNumber"},
        )
    if qrecords[0].error is not None:
        data_vars["error"] = (
            "time",
            np.array([r.error for r in qrecords]),
            {"long_name": "errorCode"},
        )

    # Stack all channel arrays: (n, Nc)
    if hdr.Nc > 0:
        all_channels = np.stack([r.channels for r in qrecords])
        for idx, ident in enumerate(hdr.channels):
            name = hexMap.name(ident)
            if name:
                data_vars[name] = (
                    "time",
                    all_channels[:, idx],
                    hexMap.attributes(ident),
                )
            else:
                logging.warning(
                    "Unknown channel identifier %#06x at index %d, skipping",
                    ident,
                    idx,
                )

    # Stack all spectra arrays: (n, Ns, Nf)
    coords: dict[str, Any] = {"time": times}
    if hdr.Ns > 0 and hdr.Nf > 0:
        all_spectra = np.stack([r.spectra for r in qrecords])
        coords["freq"] = list(hdr.frequencies)
        for idx, ident in enumerate(hdr.spectra):
            name = hexMap.name(ident)
            if name:
                data_vars[name] = (
                    ("time", "freq"),
                    all_spectra[:, idx, :],
                    hexMap.attributes(ident),
                )
            else:
                logging.warning(
                    "Unknown spectra identifier %#06x at index %d, skipping",
                    ident,
                    idx,
                )

    return xr.Dataset(data_vars=data_vars, coords=coords)


def loadQfile(fn: str) -> xr.Dataset | None:
    """
    Load a Q-file and convert it to an xarray Dataset.

    Args:
        fn: Path to Q-file

    Returns:
        xarray.Dataset with time-indexed data variables for all channels
        and spectra, or None if file is invalid/empty
    """
    # Parse binary file into segments: each segment = (header, [records])
    segments: list[tuple[QHeader, list[QRecord]]] = []
    hdr: QHeader | None = None
    data: QData | None = None
    seg_records: list[QRecord] = []

    with open(fn, "rb") as fp:
        while True:
            if QHeader.chkIdent(fp):
                if hdr is not None and seg_records:
                    segments.append((hdr, seg_records))
                hdr = QHeader(fp, fn)
                if hdr is None:
                    break  # EOF
                data = QData(hdr)
                seg_records = []
            elif QData.chkIdent(fp):
                if data is None or hdr is None:
                    raise ValueError(f"Data record before header in {fn}")

                qrecord = data.load(fp)
                if qrecord is None:
                    break  # EOF
                seg_records.append(qrecord)
            else:
                buffer = fp.read(2)
                if len(buffer) != 2:
                    break  # EOF
                (ident,) = struct.unpack("<H", buffer)
                logging.warning(
                    f"Unsupported identifier, {ident:#06x}, at {fp.tell() - 2} in {fn}"
                )
                break

    # Flush final segment
    if hdr is not None and seg_records:
        segments.append((hdr, seg_records))

    if not segments:
        if os.path.getsize(fn) > 0:
            logging.warning(f"No header found in {fn}")
        return None

    # Build one Dataset per segment using batch numpy construction
    hexMap = QHexCodes()
    datasets = [_buildSegmentDataset(h, recs, hexMap) for h, recs in segments]

    if len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = xr.concat(datasets, "time")

    # Add file-level metadata
    last_hdr = segments[-1][0]
    ftime = ds.time.data.min()
    ds = ds.assign_coords(ftime=[ftime], despike=np.arange(3))

    if last_hdr.version is None:
        raise RuntimeError("QHeader.version must be set after reading header")
    toAdd: dict[str, Any] = dict(fileVersion=("ftime", [last_hdr.version.value]))

    config = last_hdr.config.config()
    for key in config:
        val = config[key]
        if np.isscalar(val):
            toAdd[key] = ("ftime", [val])
        elif len(val) == 1:
            toAdd[key] = ("ftime", val)
        elif len(val) == 3:  # Despiking
            toAdd[key] = (("ftime", "despike"), val.reshape(1, -1))

    ds = ds.assign(toAdd)

    return ds


def mergeDatasets(frames: list[xr.Dataset]) -> xr.Dataset:
    """
    Merge multiple Q-file Datasets into a single Dataset.

    Time-indexed variables are concatenated along 'time', while per-file
    metadata variables (on the 'ftime' dimension) are concatenated along
    'ftime', then the two groups are merged.

    Args:
        frames: List of Datasets from loadQfile()

    Returns:
        Single merged Dataset

    Raises:
        ValueError: If frames is empty
    """
    if not frames:
        raise ValueError("Cannot merge empty list of Datasets")

    if len(frames) == 1:
        return frames[0]

    time_vars: set[str] = set()
    ftime_vars: set[str] = set()
    for f in frames:
        for n in f.data_vars:
            name = str(n)
            if "time" in f[name].dims:
                time_vars.add(name)
            if "ftime" in f[name].dims:
                ftime_vars.add(name)

    time_frames = [
        f[[v for v in time_vars if v in f]].drop_vars(
            ["ftime", "despike"], errors="ignore"
        )
        for f in frames
    ]
    ftime_frames = [
        f[[v for v in ftime_vars if v in f]].assign_coords(ftime=f.ftime)
        for f in frames
    ]

    ds_time = xr.concat(time_frames, "time", join="outer")
    ds_ftime = xr.concat(ftime_frames, "ftime")
    ds_ftime = ds_ftime.assign_coords(despike=np.arange(3))
    return xr.merge([ds_time, ds_ftime])


def cfCompliant(ds: xr.Dataset) -> xr.Dataset:
    """
    Add CF-1.13 compliant metadata to Dataset.

    Args:
        ds: Input Dataset

    Returns:
        Dataset with added attributes for CF compliance
    """
    known = {
        "aoa": {"long_name": "angle_of_attack", "units": "degrees"},
        "band_averaging": {"long_name": "band_averaging", "units": "1"},
        "channel": {
            "long_name": "scalar_all",
        },
        "channelIdent": {
            "long_name": "Channel_identifier",
        },
        "despike": {"long_name": "despike_index", "units": "1"},
        "diss_length": {"long_name": "dissipation_length", "units": "seconds"},
        "f_aa": {
            "units": "Hz",
        },
        "fft_length": {"long_name": "fourier_transform_length", "units": "seconds"},
        "fileVersion": {
            "long_name": "Q_file_version",
        },
        "fit_order": {"long_name": "fit_order", "units": "1"},
        "freq": {
            "long_name": "frequency",
            "standard_name": "frequency",
            "units": "Hz",
        },
        "frequency": {
            "long_name": "frequency_spectra",
            "standard_name": "frequency",
            "units": "Hz",
        },
        "ftime": {
            "long_name": "time_file_start",
            "standard_name": "time",
            "units_metadata": "leap_seconds: none",
        },
        "error": {"long_name": "error_code", "units": "1"},
        "goodman_length": {
            "units": "seconds",
        },
        "hp_cut": {"long_name": "high_pass_cut", "units": "Hz"},
        "inertial_sr": {"long_name": "inertial_subrange"},
        "overlap": {"long_name": "overlap_fft", "units": "1"},
        "record": {"long_name": "record_number", "units": "1"},
        "spectra": {
            "long_name": "spectra_all",
        },
        "spectraIdent": {
            "long_name": "Spectra_identifier",
        },
        "t1": {
            "long_name": "time_end_of_interval",
            "standard_name": "time",
            "units_metadata": "leap_seconds: none",
        },
        "time": {
            "long_name": "time_start_of_interval",
            "standard_name": "time",
            "units_metadata": "leap_seconds: none",
        },
        "ucond_despiking": {"long_name": "microconductivity_despiking", "units": "1"},
    }

    for name in known:
        if name in ds:
            ds[name] = ds[name].assign_attrs(known[name])

    now = np.datetime64("now")
    now_str = str(now)
    history_entry = f"{now_str} created by q2netcdf"
    history = ds.attrs.get("history")
    if history:
        history = f"{history}\n{history_entry}"
    else:
        history = history_entry

    ds = ds.assign_attrs(
        dict(
            Conventions="CF-1.13, CF-1.11",
            title="NetCDF translation of Rockland's Q-File(s)",
            keywords=["turbulence", "ocean"],
            summary="See Rockland's TN-054 for description of Q-Files",
            history=history,
            time_coverage_start=str(ds.time.data.min()),
            time_coverage_end=str(ds.time.data.max()),
            time_coverage_duration=str(ds.time.data.max() - ds.time.data.min()),
            date_created=now_str,
        )
    )

    return ds


def addEncoding(ds: xr.Dataset, level: int = 5) -> xr.Dataset:
    """
    Add zlib compression encoding to Dataset variables.

    Args:
        ds: Input Dataset
        level: Compression level (0-9), 0 disables compression

    Returns:
        Dataset with compression encoding added
    """
    if level <= 0:
        return ds

    for name in ds:
        if ds[name].dtype.kind == "U":
            continue
        ds[name].encoding = {"zlib": True, "complevel": level}

    return ds


def main() -> None:
    """Command-line interface for q2netcdf converter."""
    from . import __version__

    parser = ArgumentParser()
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + __version__
    )
    parser.add_argument("qfile", nargs="+", type=str, help="Q filename(s)")
    parser.add_argument("--nc", type=str, required=True, help="Output NetCDF filename")
    parser.add_argument(
        "--compressionLevel",
        type=int,
        default=5,
        help="Compression level in NetCDF file",
    )
    parser.add_argument(
        "--logLevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logLevel))

    frames = []
    for fn in args.qfile:
        try:
            ds = loadQfile(fn)
        except Exception:
            logging.exception("Failed to load %s", fn)
            continue
        if ds is not None:
            frames.append(ds)

    if not frames:  # Empty
        logging.warning("No data found in any input file")
        sys.exit(1)

    ds = mergeDatasets(frames)

    ds = cfCompliant(ds)
    ds = addEncoding(ds, args.compressionLevel)
    ds.to_netcdf(args.nc)


if __name__ == "__main__":
    main()
