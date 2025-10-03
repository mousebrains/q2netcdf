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
import sys
import struct
try: # First try parent directory
    from .QHeader import QHeader
    from .QData import QData
except ImportError:
    try: # Then try local directory
        from QHeader import QHeader
        from QData import QData
    except ImportError:
        raise

def loadQfile(fn:str) -> xr.Dataset | None:
    """
    Load a Q-file and convert it to an xarray Dataset.

    Args:
        fn: Path to Q-file

    Returns:
        xarray.Dataset with time-indexed data variables for all channels
        and spectra, or None if file is invalid/empty
    """
    records = []
    hdr = None
    with open(fn, "rb") as fp:
        data = None
        while True:
            if QHeader.chkIdent(fp):
                hdr = QHeader(fp, fn)
                if hdr is None:
                    break # EOF
                data = QData(hdr)
            elif QData.chkIdent(fp):
                record = data.load(fp)
                if record is None:
                    break # EOF
                [record, attrs] = record.split(hdr)
                qFreq = False
                t0 = record["time"]
                del record["time"]
                values = {}
                qFreq = False
                for key in record:
                    val = record[key]
                    if np.isscalar(val):
                        values[key] = ("time", 
                                       [val], 
                                       attrs[key] if key in attrs else None,
                                       )
                    else:
                        qFreq = True
                        values[key] = (("time", "freq"), 
                                       val.reshape(1,-1),
                                       attrs[key] if key in attrs else None,
                                       )
                coords = dict(time=[t0])
                if qFreq:
                    coords["freq"] = np.array(hdr.frequencies)
                ds = xr.Dataset(data_vars=values, coords=coords)

                records.append(ds)
            else:
                buffer = fp.read(2)
                if len(buffer) != 2:
                    break # EOF
                (ident,) = struct.unpack("<H", buffer)
                logging.warning(f"Unsupported identifier, {ident:#06x}, at {fp.tell()-2} in {fn}")
                break

    if not hdr:
        logging.warning(f"No header found in {fn}")
        return None

    if not records:
        logging.warning(f"No records found in {fn}")
        return None

    ds = xr.concat(records, "time")

    ftime = ds.time.data.min()
    ds = ds.assign_coords(ftime=[ftime], despike=np.arange(3))

    toAdd = dict(fileversion=("ftime", [hdr.version.value]))

    config = hdr.config.config()
    for key in config: 
        val = config[key]
        if np.isscalar(val):
            toAdd[key] = ("ftime", [val])
        elif len(val) == 1:
            toAdd[key] = ("ftime", val)
        elif len(val) == 3: # Despiking
            toAdd[key] = (["ftime", "despike"], val.reshape(1,-1))

    ds = ds.assign(toAdd)

    return ds

def cfCompliant(ds:xr.Dataset) -> xr.Dataset:
    """
    Add CF-1.8 compliant metadata to Dataset.

    Args:
        ds: Input Dataset

    Returns:
        Dataset with added attributes for CF compliance
    """
    known = {
            "aoa": {"long_name": "angle_of_attack", "units": "degrees"},
            "band_averaging": {"long_name": "band_averaging", "units": "1"},
            "channel": {"long_name": "scalar_all",},
            "channelIdent": {"long_name": "Channel_identifier",},
            "despike": {"long_name": "despike_index", "units": "1"},
            "diss_length": {"long_name": "dissipation_length", "units": "seconds"},
            "f_aa": {"units": "Hz",},
            "fft_length": {"long_name": "fourier_transform_length", "units": "seconds"},
            "fileVersion": {"long_name": "Q_file_version",},
            "fit_order": {"long_name": "fit_order", "units": "1"},
            "freq": {"long_name": "frequency", "units": "Hz",},
            "frequency": {"units": "Hz", "long_name": "frequency_spectra",},
            "ftime": {"long_name": "time_file_start",},
            "error": {"long_name": "error_code", "units": "1"},
            "goodman_length": {"units": "seconds",},
            "hp_cut": {"long_name": "high_pass_cut", "units": "Hz"},
            "inertial_sr": {"long_name": "inertial_subrange"},
            "overlap": {"long_name": "overlap_fft", "units": "1"},
            "record": {"long_name": "record_number", "units": "1"},
            "spectra": {"long_name": "spectra_all",},
            "spectraIdent": {"long_name": "Spectra_identifier",},
            "t1": {"long_name": "time_end_of_interval",},
            "time": {"long_name": "time_start_of_interval",},
            "ucond_despiking": {"long_name": "microconductivity_despiking", "units": "1"},
            }

    for name in known:
        if name in ds:
            ds[name] = ds[name].assign_attrs(known[name])

    ds = ds.assign_attrs(dict(
        Conventions = "CF-1.8",
        title = "NetCDF translation of Rockland's Q-File(s)",
        keywords = ["turbulence", "ocean"],
        summary = "See Rockland's TN-054 for description of Q-Files",
        time_coverage_start = str(ds.time.data.min()),
        time_coverage_end = str(ds.time.data.max()),
        time_coverage_duration = str(ds.time.data.max() - ds.time.data.min()),
        date_created = str(np.datetime64("now"))
        ))

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
        ds[name].encoding = {"compression": "zlib", "compression_level": level}

    return ds

def main() -> None:
    """Command-line interface for q2netcdf converter."""
    parser = ArgumentParser()
    parser.add_argument("qfile", nargs="+", type=str, help="Q filename(s)")
    parser.add_argument("--nc", type=str, required=True, help="Output NetCDF filename")
    parser.add_argument("--compressionLevel", type=int, default=5,
                        help="Compression level in NetCDF file")
    parser.add_argument("--logLevel", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logLevel))

    frames = []
    for fn in args.qfile:
        ds = loadQfile(fn)
        if ds is not None:
            frames.append(ds)

    if not frames: # Empty
        logging.info("No data found")
        sys.exit(0)

    if len(frames) == 1:
        ds = frames[0]
    else:
        ds = xr.merge(frames)

    ds = cfCompliant(ds)
    ds = addEncoding(ds, args.compressionLevel)
    ds.to_netcdf(args.nc)

if __name__ == "__main__":
    main()

