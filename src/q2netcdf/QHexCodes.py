#! /usr/bin/env python3
#
# Qfile Hex codes to sensor/spectra map
#
# Feb-2025, Pat Welch, pat@mousebrains.com

import logging
import re

class QHexCodes:
    """
    Mapping between Q-file hex identifiers and sensor/spectra names.

    Q-files use hexadecimal identifiers to label channels (scalar measurements)
    and spectra (frequency-domain data). This class provides bidirectional
    mapping between identifiers and human-readable names with attributes.

    The identifier scheme uses:
    - Upper 12 bits (0xFFF0): Sensor/spectra type
    - Lower 4 bits (0x000F): Instance number (0-15)

    Example:
        0x610 -> "sh_1" (shear probe #1)
        0x611 -> "sh_2" (shear probe #2)
    """
    __hexMap = {
            0x010: ["dT_",
                    {
                        "long_name": "preThermal_",
                        },
                    ],
            0x020: ["dC_",
                    {
                        "long_name": "preUConductivity_",
                        },
                    ],
            0x030: ["P_dP",
                    {
                        "long_name": "prePressure",
                        },
                    ],
            0x110: [["A0", "Ax", "Ay", "Az"],
                    {
                        "long_name": [
                            "acceleration_0",
                            "acceleration_X",
                            "acceleration_Y",
                            "acceleration_Z",
                            ],
                        },
                    ],
            0x120: [["A0", "Ax", "Ay"],
                    {
                        "long_name": ["piezo_0", "piezo_X", "piezo_Y",],
                        },
                    ],
            0x130: [["Incl_0", "Incl_X", "Incl_Y", "Incl_T"],
                    {
                        "long_name": [
                            "Inclinometer_0",
                            "Inclinometer_X",
                            "Inclinometer_Y",
                            "Inclinometer_T",
                            ],
                        "units": ["degrees", "degrees", "Celsius"],
                        },
                    ],
            0x140: [["theta_0", "thetaX", "thetaY"],
                    {
                        "long_name": ["Theta_0", "Theta_X", "Theta_Y"],
                        "units": "degrees",
                        },
                    ],
            0x150: [["M_0", "Mx", "My", "Mz"],
                    {
                        "long_name": ["magnetic_0", "magnetic_X", "magnetic_Y", "magnetic_Z",],
                        },
                    ],
            0x160: ["pressure",
                    {
                        "long_name": "pressure_ocean",
                        "units": "decibar",
                        },
                    ],
            0x170: ["AOA",
                    {
                        "long_name": "angle_of_attack",
                        "units": "degrees",
                        },
                    ],
            0x210: ["VBat",
                    {
                        "long_name": "battery",
                        "units": "Volts",
                        },
                    ],
            0x220: ["PV",
                    {
                        "long_name": "pressure_transducer",
                        "units": "Volts",
                        },
                    ],
            0x230: ["EMCur",
                    {
                        "long_name": "EM_current",
                        "units": "Amps",
                        },
                    ],
            0x240: [["latitude", "longitude"],
                    {
                        "long_name": ["Latitude", "Longitude"],
                        "units": ["degrees North", "degrees East"],
                        },
                    ],
            0x250: ["noise",
                    {
                        "long_name": "glider_noise",
                        },
                    ],
            0x310: ["EM",
                    {
                        "long_name": "speed",
                        "units": "meters/second",
                        },
                    ],
            0x320: [["U", "V", "W", "speed_squared"],
                    {
                        "long_name": [
                            "velocity_eastward",
                            "velocity_northward",
                            "velocity_upwards",
                            "velocity_squared",
                            ],
                         "units": [
                             "meters/second",
                             "meters/second",
                             "meters/second",
                             "meters^2/second^2",
                             ],
                         },
                    ],
            0x330: ["dzdt",
                    {
                        "long_name": "fallRate",
                        "units": "meters/second",
                        },
                    ],
            0x340: ["dzdt_adj",
                    {
                        "long_name": "fallRate_adjusted_for_AOA",
                        "units": "meters/second",
                        },
                    ],
            0x350: ["speed_hotel",
                    {
                        "long_name": "speed_hotel",
                        "units": "meters/second",
                        },
                    ],
            0x360: ["speed",
                    {
                        "long_name": "speed_computation",
                        "units": "meters/second",
                        },
                    ],
            0x410: [[
                "temperature_JAC",
                "temperature_SB",
                "temperature_RBR",
                "temperature_Hotel",
                "temperature_Contant",
                ],
                    {
                        "long_name": "temperature",
                        "units": "Celsius",
                        },
                    ],
            0x420: [[
                "conductivity_JAC",
                "conductivity_SB",
                "conductivity_RBR",
                "conductivity_Hotel",
                "conductivity_Constant",
                ],
                    {
                        "long_name": "conductivity",
                        },
                    ],
            0x430: [[
                "salinity_JAC",
                "salinity_SB",
                "salinity_RBR",
                "salinity_Hotel",
                "salinity_Constant",
                ],
                    {
                        "long_name": "salinity",
                        "units": "PSU",
                        },
                    ],
            0x440: ["sigma0",
                    {
                        "long_name": "sigma_0",
                        "units": "kilogram/meter^3",
                        },
                    ],
            0x450: ["visc",
                    {
                        "long_name": "viscosity",
                        "units": "meter^2/second",
                        },
                    ],
            0x510: ["chlor",
                    {
                        "long_name": "chlorophyll",
                        },
                    ],
            0x520: ["turb",
                    {
                        "long_name": "turbidity",
                        },
                    ],
            0x530: ["DO",
                    {
                        "long_name": "dissolved_oxygen",
                        },
                    ],
            0x610: ["sh_",
                    {
                        "long_name": "shear_",
                        },
                    ],
            0x620: ["T_",
                    {
                        "long_name": "temperature_",
                        "units": "Celsius",
                        },
                    ],
            0x630: ["C_",
                    {
                        "long_name": "microConductivity_",
                        },
                    ],
            0x640: ["dT_",
                    {
                        "long_name": "gradient_temperature_",
                        "units": "Celsius/meter",
                        },
                    ],
            0x650: ["dC_",
                    {
                        "long_name": "gradient_conductivity_",
                        },
                    ],
            0x710: ["sh_GTD_",
                    {"long_name": "shear_goodman_",
                     },
                    ],
            0x720: ["sh_DSP_",
                     {
                         "long_name": "shear_despiked_",
                         },
                    ],
            0x730: ["uCond_DSP_",
                    {
                        "long_name": "microConductivity_despiked_",
                        },
                    ],
            0x740: ["sh_fraction_",
                    {
                        "long_name": "shear_fraction_",
                        },
                    ],
            0x750: ["sh_passes_",
                    {
                        "long_name": "shear_passes_",
                        },
                    ],
            0x760: ["uCond_fraction_",
                    {
                        "long_name": "microConductivity_fraction_",
                        },
                    ],
            0x770: ["uCond_passes_",
                    {
                        "long_name": "microConductivity_passes_",
                        },
                    ],
            0x810: ["K_max_",
                    {
                        "long_name": "integration_limit_",
                        },
                    ],
            0x820: ["var_res_",
                    {
                        "long_name": "variance_resolved_",
                        },
                    ],
            0x830: ["MAD_",
                    {
                        "long_name": "mean_averaged_deviation_",
                        },
                    ],
            0x840: ["FM_",
                    {
                        "long_name": "figure_of_merit_",
                        },
                    ],
            0x850: ["CI_",
                    {
                        "long_name": "confidence_interval_",
                        },
                    ],
            0x860: ["MAD_T_",
                    {
                        "long_name": "mean_average_deviation_temperature_",
                        },
                    ],
            0x870: ["QC_",
                    {
                        "long_name": "quality_control_flags_",
                        },
                    ],
            0x910: ["freq",
                    {
                        "long_name": "frequency",
                        },
                    ],
            0x920: ["shear_raw",
                    {
                        "long_name": "shear_raw",
                        },
                    ],
            0x930: ["shear_gfd_",
                    {"long_name": "shear_goodman_",
                     },
                    ],
            0x940: ["gradT_raw",
                    {
                        "long_name": "thermistor_raw",
                        },
                    ],
            0x950: ["gradT_gfd_",
                    {
                        "long_name": "thermistor_goodman_",
                        },
                    ],
            0x960: ["uCond_raw",
                    {
                        "long_name": "microConductivity_raw",
                        },
                    ],
            0x970: ["uCond_gfd_",
                    {
                        "long_name": "microConductivity_goodman_",
                        },
                    ],
            0x980: ["piezo",
                    {
                        "long_name": "vibration",
                        },
                    ],
            0x990: ["accel",
                    {
                        "long_name": "accelerometer",
                        },
                    ],
            0x9A0: ["T_ref",
                    {
                        "long_name": "temperature_reference",
                        },
                    ],
            0x9B0: ["T_noise",
                    {
                        "long_name": "temperature_noise",
                        },
                    ],
            0xA10: ["e_",
                    {
                        "long_name": "epsilon_",
                        },
                    ],
            0xA20: ["N2",
                    {
                        "long_name": "buoyancy_frequency",
                        },
                    ],
            0xA30: ["eddy_diff",
                    {
                        "long_name": "eddy_diffusivity",
                        },
                    ],
            0xA40: ["chi_",
                    {
                        "long_name": "chi_",
                        },
                    ],
            0xA50: ["e_T_",
                    {
                        "long_name": "thermal_dissipation_",
                        },
                    ],
            0xD20: ["diagnostic_", { }, ], # Value that shouldn't be here
            }

    def __init__(self):
        pass

    @classmethod
    def __repr__(cls) -> str:
        msg = []
        for key in sorted(cls.__hexMap):
            msg.append(f"{key:#05x} {cls.__hexMap[key]}")
        return "\n".join(msg)

    @staticmethod
    def __fixName(name: str | list | tuple, cnt: int) -> str:
        if isinstance(name, str):
            if not name.endswith("_"):
                return name
            cnt = cnt # 0-15 -> 1-16
            return f"{name}{cnt}"

        if isinstance(name, (list, tuple)):
            if len(name) > cnt:
                return name[cnt]
            raise ValueError(f"cnt({cnt}) >= ({len(name)}) names <-  {name}")

        raise NotImplementedError(f"Unsupported name type, {type(name)} <- {name}")

    @classmethod
    def __findIdent(cls, ident: int) -> tuple[list | None, int | None]:
        key = ident & 0xfff0
        cnt = ident & 0x0f
        if key in cls.__hexMap:
            return (cls.__hexMap[key], cnt)

        logging.warning(f"{key:#06x} not in map, ident {ident:#06x}")
        return (None, None)

    @classmethod
    def name(cls, ident: int) -> str | None:
        """
        Get the name for a given identifier.

        Args:
            ident: Hexadecimal identifier (e.g., 0x610)

        Returns:
            Human-readable name (e.g., "sh_0") or None if not found
        """
        (item, cnt) = cls.__findIdent(ident)
        if item is None:
            return None

        assert cnt is not None  # __findIdent returns both None or both non-None
        name = item[0]
        return cls.__fixName(name, cnt)

    @classmethod
    def attributes(cls, ident: int) -> dict | None:
        """
        Get the metadata attributes for a given identifier.

        Args:
            ident: Hexadecimal identifier

        Returns:
            Dictionary with long_name, units, etc., or None if not found
        """
        (item, cnt) = cls.__findIdent(ident)
        if item is None:
            return None

        assert cnt is not None  # __findIdent returns both None or both non-None
        attrs = item[1].copy() # In case I modify it

        for attr in attrs:
            attrs[attr] = cls.__fixName(attrs[attr], cnt)

        return attrs

    @classmethod
    def name2ident(cls, name: str) -> int | None:
        """
        Convert a name to its hexadecimal identifier (reverse lookup).

        Args:
            name: Human-readable name (e.g., "sh_1")

        Returns:
            Hexadecimal identifier (e.g., 0x611) or None if not found
        """
        matches = re.match(r"^(.*_)(\d+)$", name)
        if matches:
            prefix = matches[1]
            cnt = int(matches[2])
        else:
            prefix = name
            cnt = 0

        for ident in cls.__hexMap:
            if cls.__hexMap[ident][0] == prefix:
                return ident + cnt
        logging.warning(f"{name} not found in hexMap")
        return None


def main() -> None:
    """Command-line interface for QHexCodes."""
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("ident", type=str, nargs="*", help="hex ident(s) to look up")
    parser.add_argument("--name", type=str, action="append", help="Name to translate to ident")
    parser.add_argument(
        "--logLevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logLevel))

    if args.name:
        for name in args.name:
            ident = QHexCodes.name2ident(name)
            logging.info(f"{name} -> {ident:#06x}")

    hexMap = QHexCodes()

    if args.ident:
        for ident in args.ident:
            ident = int(ident, 16)
            logging.info(f"ident {ident:#04x} {hexMap.name(ident)} -> {hexMap.attributes(ident)}")
    elif not args.name:
        logging.info(f"Hex Map\n{hexMap}")

if __name__ == "__main__":
    main()
