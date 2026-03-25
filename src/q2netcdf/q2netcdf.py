#! /usr/bin/env python3
#
# Translate Rockland ISDP Q file(s) to a NetCDF file
#
# Based on Rockland TN 054
#
# Oct-2024, Pat Welch, pat@mousebrains.com
# Feb-2025, Pat Welch, pat@mousebrains.com update to using QFile...

from argparse import ArgumentParser
import io
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
from .QRecordType import RecordType


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

    ds_time = xr.concat(time_frames, "time", join="outer", coords="minimal")
    ds_ftime = xr.concat(ftime_frames, "ftime")
    ds_ftime = ds_ftime.assign_coords(despike=np.arange(3))
    return xr.merge([ds_time, ds_ftime])


def _parse_segments(
    fn: str,
    file_bytes: bytes,
    hexMap: QHexCodes,
) -> list[
    tuple[
        QHeader,
        np.ndarray,
        tuple[tuple[str | None, ...], np.ndarray | None],
        tuple[tuple[str | None, ...], np.ndarray | None, tuple[float, ...] | None],
        dict[str, np.ndarray],
    ]
]:
    """Parse a Q-file's bytes into segment tuples using bulk numpy reads.

    Returns a list of (hdr, times, (ch_names, ch_data), (sp_names, sp_data,
    frequencies), extras) per segment. extras is a dict with 't1', 'number',
    'error' for v1.2, or empty for v1.3.
    """
    segments: list[
        tuple[
            QHeader,
            np.ndarray,
            tuple[tuple[str | None, ...], np.ndarray | None],
            tuple[tuple[str | None, ...], np.ndarray | None, tuple[float, ...] | None],
            dict[str, np.ndarray],
        ]
    ] = []
    bio = io.BytesIO(file_bytes)

    while bio.tell() < len(file_bytes):
        pos = bio.tell()
        if pos + 2 > len(file_bytes):
            break
        ident_val = struct.unpack("<H", file_bytes[pos : pos + 2])[0]
        if ident_val != RecordType.HEADER.value:
            break

        hdr = QHeader(bio, fn)
        data_start = bio.tell()

        # Find next header or end of file
        data_end = len(file_bytes)
        scan = data_start
        while scan + hdr.dataSize <= len(file_bytes):
            if scan > data_start and scan + 2 <= len(file_bytes):
                check = struct.unpack("<H", file_bytes[scan : scan + 2])[0]
                if check == RecordType.HEADER.value:
                    data_end = scan
                    break
            scan += hdr.dataSize

        n_recs = (data_end - data_start) // hdr.dataSize
        bio.seek(data_end)
        if n_recs == 0:
            continue

        seg_bytes = file_bytes[data_start : data_start + n_recs * hdr.dataSize]

        # Build a structured numpy dtype matching the binary record layout
        assert hdr.version is not None
        if hdr.version.isV12():
            fields: list[tuple[str, str]] = [
                ("ident", "<u2"),
                ("number", "<u2"),
                ("error", "<i8"),
                ("stime", "<f2"),
                ("etime", "<f2"),
            ]
        else:
            fields = [("ident", "<u2"), ("stime", "<f2")]
        for i in range(hdr.Nc):
            fields.append((f"ch{i}", "<f2"))
        for i in range(hdr.Ns * hdr.Nf):
            fields.append((f"sp{i}", "<f2"))

        dt = np.dtype(fields)
        if dt.itemsize != hdr.dataSize:
            logging.warning(
                "Dtype size %d != dataSize %d for %s, falling back",
                dt.itemsize,
                hdr.dataSize,
                fn,
            )
            continue

        records = np.frombuffer(seg_bytes, dtype=dt)

        # Compute timestamps in bulk
        stimes = records["stime"].astype(np.float64)
        times = hdr.time + (stimes * 1000).astype("timedelta64[ms]")
        times = times.astype("datetime64[ns]")

        # Extract channels
        ch_names: list[str | None] = []
        ch_data: np.ndarray | None = None
        if hdr.Nc > 0:
            ch_data = np.column_stack(
                [records[f"ch{i}"].astype(np.float32) for i in range(hdr.Nc)]
            )
            for c_ident in hdr.channels:
                ch_names.append(hexMap.name(c_ident))

        # Extract spectra
        sp_names: list[str | None] = []
        sp_data: np.ndarray | None = None
        freqs: tuple[float, ...] | None = None
        if hdr.Ns > 0 and hdr.Nf > 0:
            sp_cols = [
                records[f"sp{i}"].astype(np.float32) for i in range(hdr.Ns * hdr.Nf)
            ]
            sp_data = np.column_stack(sp_cols).reshape(n_recs, hdr.Ns, hdr.Nf)
            freqs = hdr.frequencies
            for s_ident in hdr.spectra:
                sp_names.append(hexMap.name(s_ident))

        # v1.2 extra fields
        extras: dict[str, np.ndarray] = {}
        if hdr.version.isV12():
            etimes = records["etime"].astype(np.float64)
            extras["t1"] = (
                hdr.time + (etimes * 1000).astype("timedelta64[ms]")
            ).astype("datetime64[ns]")
            extras["number"] = records["number"].copy()
            extras["error"] = records["error"].copy()

        segments.append(
            (
                hdr,
                times,
                (tuple(ch_names), ch_data),
                (tuple(sp_names), sp_data, freqs),
                extras,
            )
        )

    return segments


def loadQfiles(filenames: list[str]) -> xr.Dataset | None:
    """
    Load multiple Q-files into a single xarray Dataset using bulk parsing.

    Uses numpy structured dtypes to parse all records per file in a single
    C-level call, then accumulates raw arrays across all files and builds
    one Dataset at the end.  This avoids per-file and per-record xarray
    overhead, giving ~10-20x speedup over loadQfile() + mergeDatasets().

    Handles mixed v1.2/v1.3 files, heterogeneous channels/spectra/config
    keys, varying frequency dimensions, multi-segment files, and empty
    files.

    Args:
        filenames: List of paths to Q-files

    Returns:
        Single merged xarray.Dataset, or None if no data found
    """
    hexMap = QHexCodes()

    # Accumulators for time-indexed data (one entry per segment)
    all_times: list[np.ndarray] = []
    seg_time_counts: list[int] = []
    seg_channels: list[tuple[tuple[str | None, ...], np.ndarray | None]] = []
    seg_spectra: list[
        tuple[tuple[str | None, ...], np.ndarray | None, tuple[float, ...] | None]
    ] = []
    seg_has_extras: list[bool] = []
    seg_t1: list[np.ndarray | None] = []
    seg_number: list[np.ndarray | None] = []
    seg_error: list[np.ndarray | None] = []
    channel_attrs: dict[str, dict[str, Any]] = {}
    spectra_attrs: dict[str, dict[str, Any]] = {}

    # Accumulators for per-file metadata
    ftime_list: list[np.datetime64] = []
    file_versions: list[float] = []
    file_configs: list[dict[str, Any]] = []
    all_freq_sets: list[tuple[float, ...]] = []

    for fn in filenames:
        try:
            with open(fn, "rb") as fp:
                file_bytes = fp.read()
        except OSError:
            logging.warning("Cannot read %s, skipping", fn)
            continue

        if len(file_bytes) < 20:
            if len(file_bytes) > 0:
                logging.warning("File too small: %s", fn)
            continue

        file_seg_start = len(all_times)

        try:
            segments = _parse_segments(fn, file_bytes, hexMap)
        except (EOFError, ValueError, NotImplementedError):
            logging.exception("Failed to parse %s", fn)
            continue

        last_hdr: QHeader | None = None
        for hdr, times, (ch_names, ch_data), (
            sp_names,
            sp_data,
            freqs,
        ), extras in segments:
            last_hdr = hdr
            all_times.append(times)
            seg_time_counts.append(len(times))

            seg_channels.append((ch_names, ch_data))

            seg_spectra.append((sp_names, sp_data, freqs))
            if freqs is not None:
                all_freq_sets.append(freqs)

            seg_has_extras.append(bool(extras))
            seg_t1.append(extras.get("t1"))
            seg_number.append(extras.get("number"))
            seg_error.append(extras.get("error"))

            # Collect attributes from first occurrence
            for name in ch_names:
                if name and name not in channel_attrs:
                    c_ident = hdr.channels[ch_names.index(name)]
                    attrs = hexMap.attributes(c_ident)
                    if attrs is not None:
                        channel_attrs[name] = attrs
            for name in sp_names:
                if name and name not in spectra_attrs:
                    s_ident = hdr.spectra[sp_names.index(name)]
                    attrs = hexMap.attributes(s_ident)
                    if attrs is not None:
                        spectra_attrs[name] = attrs

        # No data segments found in this file
        if len(all_times) == file_seg_start:
            if os.path.getsize(fn) > 0:
                logging.warning("No data found in %s", fn)
            continue

        assert last_hdr is not None
        assert last_hdr.version is not None
        file_times = [all_times[i] for i in range(file_seg_start, len(all_times))]
        ftime_list.append(min(t.min() for t in file_times))
        file_versions.append(last_hdr.version.value)
        file_configs.append(last_hdr.config.config())

    if not all_times:
        logging.warning("No data found in any input file")
        return None

    # --- Build a single Dataset from accumulated raw arrays ---

    total_time = sum(seg_time_counts)
    merged_times = np.concatenate(all_times)

    data_vars: dict[str, Any] = {}
    coords: dict[str, Any] = {"time": merged_times}

    # v1.2 extras with NaN/NaT fill for v1.3 segments
    if any(seg_has_extras):
        t1_arr = np.full(total_time, np.datetime64("NaT"), dtype="datetime64[ns]")
        num_arr = np.full(total_time, np.nan, dtype=np.float64)
        err_arr = np.full(total_time, np.nan, dtype=np.float64)
        offset = 0
        for i, count in enumerate(seg_time_counts):
            if seg_has_extras[i]:
                assert seg_t1[i] is not None
                assert seg_number[i] is not None
                assert seg_error[i] is not None
                t1_arr[offset : offset + count] = seg_t1[i]
                num_arr[offset : offset + count] = seg_number[i]
                err_arr[offset : offset + count] = seg_error[i]
            offset += count
        data_vars["t1"] = ("time", t1_arr, {"long_name": "timeStop"})
        data_vars["record"] = ("time", num_arr, {"long_name": "recordNumber"})
        data_vars["error"] = ("time", err_arr, {"long_name": "errorCode"})

    # Channels — union of all names, NaN where absent
    all_ch_names: set[str] = set()
    for names, _ in seg_channels:
        all_ch_names.update(n for n in names if n)
    for var_name in sorted(all_ch_names):
        arr = np.full(total_time, np.nan, dtype=np.float32)
        offset = 0
        for (names, ch_data), count in zip(seg_channels, seg_time_counts):
            if var_name in names:
                assert ch_data is not None
                arr[offset : offset + count] = ch_data[:, names.index(var_name)]
            offset += count
        data_vars[var_name] = ("time", arr, channel_attrs.get(var_name, {}))

    # Spectra — unified frequency coordinate, NaN where absent
    all_sp_names: set[str] = set()
    for names, _, _ in seg_spectra:
        all_sp_names.update(n for n in names if n)
    if all_freq_sets:
        unified_freqs = sorted(set(f for freqs in all_freq_sets for f in freqs))
        n_unified = len(unified_freqs)
        coords["freq"] = unified_freqs

        # Pre-compute frequency index maps
        freq_map: dict[tuple[float, ...], list[int]] = {}
        for freqs in all_freq_sets:
            if freqs not in freq_map:
                freq_map[freqs] = [unified_freqs.index(f) for f in freqs]

        for var_name in sorted(all_sp_names):
            sp_arr = np.full((total_time, n_unified), np.nan, dtype=np.float32)
            offset = 0
            for (names, sp_data, freqs), count in zip(seg_spectra, seg_time_counts):
                if var_name in names and freqs is not None:
                    assert sp_data is not None
                    col_indices = freq_map[freqs]
                    sp_arr[offset : offset + count][:, col_indices] = sp_data[
                        :, names.index(var_name), :
                    ]
                offset += count
            data_vars[var_name] = (
                ("time", "freq"),
                sp_arr,
                spectra_attrs.get(var_name, {}),
            )

    # Per-file metadata
    ftimes = np.array(ftime_list)
    coords["ftime"] = ftimes
    coords["despike"] = np.arange(3)
    data_vars["fileVersion"] = ("ftime", file_versions)

    # Config vars — union of all keys, NaN/NaT where absent
    all_config_keys: set[str] = set()
    for cfg in file_configs:
        all_config_keys.update(cfg.keys())

    for key in sorted(all_config_keys):
        # Find the first file that has this key to determine shape
        sample: Any = None
        for cfg in file_configs:
            if key in cfg:
                sample = cfg[key]
                break
        if sample is None:
            continue

        if np.isscalar(sample):
            vals = [cfg.get(key, np.nan) for cfg in file_configs]
            data_vars[key] = ("ftime", vals)
        elif len(sample) == 1:
            vals = [cfg[key][0] if key in cfg else np.nan for cfg in file_configs]
            data_vars[key] = ("ftime", vals)
        elif len(sample) == 3:  # Despiking
            rows = [
                cfg[key] if key in cfg else np.full(3, np.nan) for cfg in file_configs
            ]
            data_vars[key] = (("ftime", "despike"), np.stack(rows))

    return xr.Dataset(data_vars=data_vars, coords=coords)


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
        default=1,
        help="Compression level in NetCDF file (0-9, default 1)",
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

    ds = loadQfiles(args.qfile)
    if ds is None:
        sys.exit(1)

    ds = cfCompliant(ds)
    ds = addEncoding(ds, args.compressionLevel)
    ds.to_netcdf(args.nc)


if __name__ == "__main__":
    main()
