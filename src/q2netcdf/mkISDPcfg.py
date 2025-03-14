#! /usr/bin/env python3
#
# Write an isdp.cfg file based on command line arguments
#
# Jan-2025, Pat Welch, pat@mousebrains.com

from argparse import ArgumentTypeError

def chkNotNegative(val:float) -> float:
    try:
        val = float(val)
        if val >= 0: return val
        msg = ArgumentTypeError(f"{val} is < 0")
    except:
        msg = ArgumentTypeError(f"{val} is not numeric")
    raise msg

def chkPositive(val:float) -> float:
    try:
        val = float(val)
        if val > 0: return val
        msg = ArgumentTypeError(f"{val} is <= 0")
    except:
        msg = ArgumentTypeError(f"{val} is not numeric")
    return msg

def chkDespiking(val:str) -> tuple:
    fields = val.split(",")
    if len(fields) != 3:
        raise ArgumentTypeError(f"{val} is does not have three fields")
    val = []
    msg = None
    try:
        val.append(float(fields[0])) # Threshold
    except:
        msg = f"{fields[0]} threshold is not numeric in {fields}"
    if msg is not None: raise ArgumentTypeError(msg)

    try:
        val.append(float(fields[1])) # smoothing
    except:
        msg = f"{fields[1]} smoothing is not numeric in {fields}"
    if msg is not None: raise ArgumentTypeError(msg)

    try:
        val.append(int(fields[2])) # number of points to remove
    except:
        msg = f"{fields[2]} number of points is not an integer in {fields}"
    if msg is not None: raise ArgumentTypeError(msg)

    return val

def main ():
    from argparse import ArgumentParser
    from datetime import datetime, timezone
    import os.path

    parser = ArgumentParser(description="Generate isdp.cfg for Rockland's MicroRider")
    parser.add_argument("--isdpConfig", type=str, default="/home/debian/data/isdp.cfg",
                        help="Filename of output config file")

    grp = parser.add_argument_group(description="Platform speed related options")
    grp.add_argument("--instrument", type=str, 
                     choices=("vmp", "sea_explorer", "slocum_glider"),
                     default="slocum_glider",
                     help="Instrument platform, vmp, sea_explorer, slocum_glider")
    grp.add_argument("--tau", type=chkPositive, 
                     help="Smoothing value for pressure record to compute speed in seconds,"
                     + " default is 3 seconds")
    grp.add_argument("--aoa", type=chkPositive,
                     help="Angle-of-attack of glider in degrees, default is 3 degrees")
    grp.add_argument("--algorithm", type=str, choices=("glide",),
                     help="Algorithm for calculating instrument speed")

    grp = parser.add_argument_group(description="Spectra computation parameters")
    parser.add_argument("--fft_length", type=chkPositive,
                        help="Length of data segments used to compute individual FFTs in seconds,"
                        + " default is 4")
    grp.add_argument("--diss_length", type=chkPositive, 
                     help="Length of to calculate dissipation over in seconds, default is 30")
    grp.add_argument("--overlap", type=chkNotNegative,
                     help="Overlatp between dissipation estimates in seconds, default is 0")
    grp.add_argument("--hp_cut", type=chkPositive,
                     help="High-pass cut-off frequency in Hz, default is 0.125."
                     + "This should be 1/(2*fft_length")
    grp.add_argument("--shear_despiking", type=chkDespiking,
                     help="Shear despiking paramters,"
                     + " threshold, smoothing, number of points to remove")
    grp.add_argument("--ucond_despiking", type=chkDespiking,
                     help="Micro-conductivity despiking paramters,"
                     + " threshold, smoothing, number of points to remove")
    grp.add_argument("--order", type=int, choices=(0,1,2,3),
                     help="Polynomial order for detrending data, default is 1")
    grp.add_argument("--f_aa", type=chkPositive,
                     help="Anti-aliasing filter of the instrument in Hz, default is 98")
    grp.add_argument("--goodman_spectra", type=str, choices=("true", "false"),
                     help="Use Goodman filter or not")
    grp.add_argument("--goodman_length", type=chkNotNegative,
                     help="Length of FFT in seconds to perform Goodman coherent-noise removal,"
                     + " default is 0."
                     + " 0 means time-domain routine is not applied.")

    grp = parser.add_argument_group(description="Dissipation estimate parameters")
    grp.add_argument("--inertial_sr", type=chkPositive,
                     help="Threshold [epsilon | W m^3] to use the inertial subrange routine"
                     + " to compute dissipation estimates")
    grp.add_argument("--fit_order", type=int, choices=(0,1,2,3),
                     help="Order of polynomial fit used to identify minimas of the spectra")

    grp = parser.add_argument_group(description="Output data parameters")
    grp.add_argument("--num_frequency", type=chkNotNegative,
                     help="Number of frequency bins to write to the q-file, default is 28")
    grp.add_argument("--band_averaging", type=str, choices=("true", "false"),
                     help="Band average spectra")
    grp.add_argument("--scalar_processing", type=str, choices=("true", "false"),
                     help="Process scalar data")
    grp.add_argument("--q", type=chkPositive,
                     help="turbulent parameter typically 5.26 for the Kraichnan spectra")
    grp.add_argument("--scalar_spectra_ref", type=str, choices=("k", "b"),
                     help="Spectral model to use for scalar processing k = Kraichnan; b = Batchelor")
    grp.add_argument("--FP07_response", type=str, choices=("RSI",),
                     help="type of frequency correction to apply to the scalar spectra")
    args = parser.parse_args()

    args.isdpConfig = os.path.abspath(os.path.expanduser(args.isdpConfig))
    if not os.path.isdir(os.path.dirname(args.isdpConfig)):
        dirname = os.path.dirname(args.isdpConfig)
        raise ArgumentTypeError(f"WARNING, {dirname}, is not a directory");

    config = {}
    values = vars(args)
    for key in values:
        if key == "isdpConfig": continue
        val = values[key]
        if val is not None: 
            config[key] = val

    if "fft_length" in config:
        if "hp_cut" not in config:
            print("WARNING: You specified fft_length, but not hp_cut")
        else:
            fft = config["fft_length"]
            hp = config["hp_cut"]
            f = 1/(2 * fft)
            if abs(f - hp) > 1e-5:
                print(f"WARNING: Your fft_length({fft}) is not consistent with hp_cut {hp}")
    elif "hp_cut" in config:
        print("WARNING: You specified hp_cut, but not fft_length")

    with open(args.isdpConfig, "w") as fp:
        now = datetime.now().replace(tzinfo=timezone.utc)
        fp.write(f"# Generated {now}\n")
        for key in sorted(config):
            val = config[key]
            if isinstance(val, str) and val not in ("true", "false"):
                if '"' not in val:
                    val = '"' + val + '"'
                elif "'" not in val:
                    val = "'" + val + "'"
                else:
                    print(f"Both single and double quote in a string are not supported. ({val})")
                    raise ValueError

            fp.write(f"{key} = {val}\n")

if __name__ == "__main__":
    main()
