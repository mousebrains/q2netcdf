#! /usr/bin/env python3
#
# Translatea Rockland ISDP Q file(s) to a NetCDF file
#
# Based on Rockland TN 054
#
# Oct-2024, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import struct
import numpy as np
import xarray as xr
import logging
import re
import copy

def parseQconfig(config:str) -> dict:
    items = {}
    for line in config.split("\n"):
        matches = re.match(r'^\s*"(\w+)"\s+=>\s+(.*)$', line)
        if not matches: continue
        name = matches[1]
        val = matches[2]
        if re.match(r"^\d+$", val):
            val = int(val)
        elif re.match(r"^\d*[.]\d+$", val):
            val = float(val)
        elif re.match(r"^\[\d*[.]\d+,\s*\d*[.]\d+,\s*\d*[.]\d+\]$", val):
            val = np.array(val)
        elif re.match(r"^true$", val):
            val = True
        elif re.match(r"^false$", val):
            val = False
        items[name] = val
    return items

def loadQHeader(f, fn:str) -> dict:
    buffer = f.read(18) # Load the first 18 bytes of the header
    if len(buffer) != 18:
        logging.error("Reading initial header, %s!=18, in %s", len(buffer), fn)
        return None

    (fVer, stime, Nc, Ns, Nf) = struct.unpack("<fQHHH", buffer)

    if abs(fVer - 1.2) > 0.0001:
        logging.error("Invalid file version, %s, in %s", fVer, fn)
        return None

    hdr = dict(
            fVer=fVer,
            time=np.datetime64("0000-01-01") + np.timedelta64(stime, "ms"),
            Nc=Nc,
            Ns=Ns,
            Nf=Nf,
            )

    buffer = f.read(Nc * 2)
    if len(buffer) != (Nc * 2):
        logging.error("Reading channel identifiers, %s != %s, in %s",
                      len(buffer), Nc * 2, fn)
        return None
    hdr["channelIdent"] = struct.unpack("<" + ("H" * Nc), buffer)

    buffer = f.read(Ns * 2)
    if len(buffer) != (Ns * 2):
        logging.error("Reading spectra identifiers, %s != %s, in %s",
                      len(buffer), Ns * 2, fn)
        return None
    hdr["spectraIdent"] = struct.unpack("<" + ("H" * Ns), buffer)

    buffer = f.read(Nf * 2)
    if len(buffer) != (Nf * 2):
        logging.error("Reading frequency bins, %s != %s, in %s",
                      len(buffer), Nf * 2, fn)
        return None
    hdr["frequencyBins"] = struct.unpack("<" + ("e" * Nf), buffer)

    buffer = f.read(4)
    if len(buffer) != 4:
        logging.error("Reading configuration record, %s != 4, in %s", len(buffer), fn)
        return None
    (ident, sz) = struct.unpack("<HH", buffer)

    hdr["config"] = str(f.read(sz), "utf-8")

    buffer = f.read(2)
    if len(buffer) != 2:
        logging.error("Reading data record size, %s != 2, in %s", len(buffer), fn)
        return None
    (hdr["recordSize"], ) = struct.unpack("<H", buffer)

    logging.info("hdr %s", hdr["time"])
    return hdr

def loadQData(f, fn:str, hdr:dict) -> xr.Dataset:
    if hdr is None:
        logging.error("No header record before first data record in %s", fn)
        return None

    Nc = hdr["Nc"]
    Ns = hdr["Ns"]
    Nf = hdr["Nf"]

    sz = hdr["recordSize"] - 2 # Ident already read

    buffer = f.read(sz)
    if len(buffer) != sz:
        logging.error("Unable to read data record, %s!=%s, in %s", len(buffer), sz, fn)
        return None

    items = struct.unpack("<Hqee" + ("e" * Nc) + ("e" * Ns * Nf), buffer)

    stime = (hdr["time"] + np.timedelta64(int(items[2] * 1000), "ms")).astype("datetime64[ns]")
    etime = (hdr["time"] + np.timedelta64(int(items[3] * 1000), "ms")).astype("datetime64[ns]")
 
    ds = xr.Dataset(
            coords=dict(
                time=[stime + (etime - stime) / 2],
                channelIdent=np.array(hdr["channelIdent"]).astype("uint16"),
                spectraIdent=np.array(hdr["spectraIdent"]).astype("uint16"),
                frequency=np.array(hdr["frequencyBins"]).astype("f4"),
                ),
            data_vars=dict(
                stime=("time", [stime]),
                etime=("time", [etime]),
                error=("time", np.array([items[1]]).astype("uint16")),
                channel=(("time", "channelIdent"), 
                         np.reshape(np.array(items[4:(Nc+4)]).astype("f4"), (1,Nc))),
                spectra=(("time", "spectraIdent", "frequency"), 
                         np.reshape(np.array(items[(Nc+4):]).astype("f4"), (1, Ns, Nf))),
                ),
            )
    return ds

def loadQfile(fn:str) -> xr.Dataset:
    logging.info("Processing %s", fn)
    records = []
    with open(fn, "rb") as f:
        hdr = None
        while True:
            n = f.tell()
            buffer = f.read(2)
            if len(buffer) != 2:
                if buffer is None or len(buffer) == 0: break; # EOF
                logging.error("Unable to read identification field, %s!=2, in %s", len(buffer), fn)
                break
            (ident,) = struct.unpack("<H", buffer)
            match ident:
                case 0x1729: # Header+Config record
                    hdr = loadQHeader(f, fn)
                    if hdr is None: break
                case 0x1657: # Data record
                    rec = loadQData(f, fn, hdr)
                    if rec is None: break
                    records.append(rec)
                case 0x1a1a: # EOF
                    logging.info(f"0x1a1a at {n:#05x}")
                    break
                case _:
                    logging.error(f"Unrecognized record identifier, {ident:#04x}, in %s", fn)
                    break

    if not records:
        logging.warning("No records found in %s", fn)
        return None

    ds = xr.concat(records, "time")
    ds = ds.assign(fileVersion=hdr["fVer"],)
    ds = ds.assign(parseQconfig(hdr["config"]))

    for name in ds:
        ds.encoding = dict(zlib=True, compleve=9)

    return ds

def fixName(name, cnt:int) -> str:
    if isinstance(name, str): return name + str(cnt)
    if cnt <= len(name): return name[cnt-1]
    print(f"Name array too short {len(name)} <= {cnt}")
    return None

def decryptIdent(ident:int) -> tuple:
    known = {
            0x010: ["dT_", {"long_name": "preThermal_"}],
            0x020: ["dC_", {"long_name": "preUConductivity_"}],
            0x030: ["P_dP", {"long_name": "prePressure"}],
            0x110: [["Ax", "Ay", "Az"], {"long_name": "acceleration_"}],
            0x120: [["Ax", "Ay"], {"long_name": "piezo_"}],
            0x130: [["Incl_X", "Incl_Y", "Incl_T"],
                    {"long_name": "Incl_", "units": ["degrees", "degrees", "Celsius"]}],
            0x140: [["thetaX", "thetaY"],
                    {"long_name": "Theta_", "units": "degrees"}],
            0x150: [["Mx", "My", "Mz"],
                    {"long_name": "magnetic_"}],
            0x160: ["pressure", {"long_name": "pressure_ocean", "units": "decibar"}],
            0x170: ["AOA", {"long_name": "angle_of_attack", "units": "degrees"}],
            0x210: ["VBat", {"long_name": "battery", "units": "Volts"}],
            0x220: ["PV", {"long_name": "pressure_transducer", "units": "Volts"}],
            0x230: ["EMCur", ["EMCur"], {"long_name": "EM_current", "units": "Amps"}],
            0x240: [["latitude", "longitude"],
                    {"long_name": None, "units": ["degrees North", "degrees East"]}],
            0x250: ["noise", {"long_name": "glider_noise"}],
            0x310: ["EM", {"long_name": "speed", "units": "meters/second"}],
            0x320: [["U", "V", "W", "speed_squared"],
                    {"long_name": ["velocity_eastwared", "velocity_northwrd", "velocity_upwards",
                                   "velocity_squard"],
                     "units": ["meters/second", "meters/second", "meters/second", 
                               "meters^2/second%2"],
                     },
                    ],
            0x330: ["dzdt", {"long_name": "fallRate", "units": "meters/second"},],
            0x340: ["dzdt_adj", {"long_name": "fallRate_adjusted_for_AOA", 
                                 "units": "meters/second"},],
            0x350: ["speed_hotel", {"long_name": "speed_hotel", "units": "meters/second"},],
            0x360: ["speed", {"long_name": "speed_computation", "units": "meters/second"},],
            0x410: ["temperature", {"long_name": "temperature", "units": "Celsius"},],
            0x420: ["conductivity", {"long_name": "conductivity",},],
            0x430: ["salinity", {"long_name": "salinity", "units": "PSU"},],
            0x440: ["density", {"long_name": "density_0", "units": "kilogram/meter^3"},],
            0x450: ["visc", {"long_name": "viscosity", "units": "meter^2/second"},],
            0x510: ["chlor", {"long_name": "chlorophyll",},],
            0x520: ["turb", {"long_name": "turbidity",},],
            0x530: ["DO", {"long_name": "disolved_oxygen",},],
            0x610: ["sh_", {"long_name": "shear_",},],
            0x620: ["T_", {"long_name": "temperature_", "units": "Celsius",},],
            0x630: ["C_", {"long_name": "microConductivity_",},],
            0x640: ["dT_", {"long_name": "gradient_temperature_", "units": "Celsius/meter",},],
            0x640: ["dC_", {"long_name": "gradient_conductivity_",},],
            0x710: ["sh_GTD_", {"long_name": "shear_goodman_",},],
            0x720: ["sh_DSP_", {"long_name": "shear_despiked_",},],
            0x730: ["uCond_DSP_", {"long_name": "microConductivity_despiked_",},],
            0x740: ["sh_fraction_", {"long_name": "shear_fraction_",},],
            0x750: ["sh_passes_", {"long_name": "shear_passes_",},],
            0x760: ["uCond_fraction_", {"long_name": "microConductivity_fraction_",},],
            0x770: ["uCond_passes_", {"long_name": "microConductivity_passes_",},],
            0x810: ["K_max_", {"long_name": "integration_limit_"},],
            0x820: ["var_res_", {"long_name": "variance_resolved_"},],
            0x830: ["MAD_", {"long_name": "mean_averaged_deviation_"},],
            0x840: ["FM_", {"long_name": "figure_of_merit_"},],
            0x850: ["CI_", {"long_name": "confidence_interval_"},],
            0x860: ["MAD_T_", {"long_name": "mean_average_deviation_temperature_"},],
            0x870: ["QC_", {"long_name": "quality_control_flags_"},],
            0x910: ["freq", {"long_name": "frequency"},],
            0x920: ["shear_raw", {"long_name": "shear_raw"},],
            0x930: ["shear_gfd", {"long_name": "shear_goodman"},],
            0x940: ["gradT_raw", {"long_name": "thermistor_raw"},],
            0x950: ["gradT_gfd", {"long_name": "thermistor_goodman"},],
            0x960: ["uCond_raw", {"long_name": "microConductivity_raw"},],
            0x970: ["uCond_gfd", {"long_name": "microConductivity_goodman"},],
            0x980: ["piezo", {"long_name": "vibration"},],
            0x990: ["accel", {"long_name": "accelerometer"},],
            0x9A0: ["T_ref", {"long_name": "temperature_reference"},],
            0x9B0: ["T_noise", {"long_name": "temperature_noise"},],
            0xA10: ["e_", {"long_name": "epsilon_"},],
            0xA20: ["N2", {"long_name": "buoyancy_frequency"},],
            0xA30: ["eddy_diff", {"long_name": "eddy_diffusivity"},],
            0xA40: ["chi_", {"long_name": "chi_"},],
            0xA50: ["e_T_", {"long_name": "thermal_dissipation_"},],
            }

    key = ident & 0xfff0
    if key not in known:
        print(f"Ident {ident:#04x} not known")
        return (None, None)

    cnt = ident & 0x0f
    (name, attrs) = known[key]

    if cnt == 0: return (name, attrs)

    name= fixName(name, cnt)

    attrs = copy.copy(attrs) # In case we modify something
    for attr in attrs:
        attrs[attr] = fixName(attrs[attr], cnt)

    return (name, attrs)

def splitIdenties(ds:xr.Dataset) -> xr.Dataset:
    used = []
    for ident in ds.channelIdent.data:
        (name, attrs) = decryptIdent(ident)
        if name is None: continue
        ds = ds.assign({name: ds.channel.sel(channelIdent=ident)})
        ds[name] = ds[name].assign_attrs(attrs)
        used.append(ident)

    if len(used) == ds.channelIdent.size: ds = ds.drop_vars(("channel", "channelIdent"))

    used = []
    for ident in ds.spectraIdent.data:
        (name, attrs) = decryptIdent(ident)
        if name is None: continue
        ds = ds.assign({name: ds.spectra.sel(spectraIdent=ident)})
        ds[name] = ds[name].assign_attrs(attrs)
        used.append(ident)

    if len(used) == ds.spectraIdent.size: ds = ds.drop_vars(("spectra", "spectraIdent"))

    return ds

def cfCompliant(ds:xr.Dataset) -> xr.Dataset:
    known = {
            "time": {"long_name": "time_end_of_interval",},
            "diss_length": {"units": "seconds",},
            "f_aa": {"units": "Hz",},
            "fft_length": {"units": "seconds",},
            "fileVersion": {"long_name": "Q_file_version",},
            "frequency": {"units": "Hz", "long_name": "frequency_spectra",},
            "goodman_length": {"units": "seconds",},
            "error": {"long_name": "error_bit_mask",},
            "channel": {"long_name": "scalar_all",},
            "spectra": {"long_name": "spectra_all",},
            "channelIdent": {"long_name": "Channel_identifier",},
            "spectraIdent": {"long_name": "Spectra_identifier",},
            }

    for name in known:
        if name in ds:
            ds[name] = ds[name].assign_attrs(known[name])

    ds = ds.assign_attrs(dict(
        conventions = "CF 1.8",
        title = "NetCDF translation of Rockland's Q-File(s)",
        keywords = ["turbulence", "ocean"],
        summary = "See Rockland's TN-054 for description of Q-Files",
        time_coverage_start = str(ds.time.data.min()),
        time_coverage_end = str(ds.time.data.max()),
        time_coverage_duration = str(ds.time.data.max() - ds.time.data.min()),
        date_created = str(np.datetime64("now"))
        ))

    return ds

def addEncoding(ds:xr.Dataset, level:int=5) -> xr.Dataset:
    if level <= 0: return ds

    for name in ds:
        if ds[name].dtype == "object": continue
        ds[name].encoding = {'compression': 'zlib', 'compression_level': level}

    return ds

def addFileIndex(ds:xr.Dataset) -> xr.Dataset:
    t0 = ds.stime.data.min()
    ds = ds.assign_coords(ftime = [t0])
    for name in ds:
        if ds[name].dims: continue # Not a scalar
        ds = ds.assign({name: ("ftime", [ds[name].data])})
    return ds

parser = ArgumentParser()
parser.add_argument("qfile", nargs="+", type=str, help="Q filename(s)")
parser.add_argument("--nc", type=str, required=True, help="Output NetCDF filename")
parser.add_argument("--compressionLevel", type=int, default=5, 
                    help="Compression level in NetCDF file")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)

frames = []
for fn in args.qfile:
    ds = loadQfile(fn)
    if ds is not None:
        frames.append(ds)

if len(frames) == 1:
    ds = frames[0]
else:
    for index in range(len(frames)):
        frames[index] = addFileIndex(frames[index])
    ds = xr.merge(frames)

ds = splitIdenties(ds)
ds = cfCompliant(ds)
ds = addEncoding(ds, args.compressionLevel)
ds.to_netcdf(args.nc)
