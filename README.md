# q2netcdf manipulates [Rockland Scientific's](https://rocklandscientific.com) Q-files as generated by [ISDP](https://rocklandscientific.com/news/rockland-data-logger/)

- `q2netcdf.py` translates Q-files into a mostly CF compliant NetCDF file
- `dumpQHeader.py` and `QHeader.py` dump the initial header record of a Q-file
- `QFile.py` dumps the header and data records of a Q-file
- `QReduce.py` reduces the size of an individual Q-file
- `QHexCodes.py` displays the hex identifier and names used by Rockland
- `mergeqfiles.py` merges Q-files together. This is designed to be run on the MicroRider as `/usr/local/bin/mergeqfiles` in support of [TWR's Slocum Glider uRider proglet](https://www.teledynemarine.com/brands/webb-research/slocum-glider) 
- `mkISDPcfg.py` generates syntatically correct `isdp.cfg` files. It is designed to be run on the MR

## Installation

The command line scripts in this package may be installed using 
[pipx](https://pipx.pypa.io/stable/installation/) after cloning this repository.

```bash
git clone https://github.com/OSUGliders/q2netcdf
cd q2netcdf
python3 -m pipx install .
```

`pipx` will install in a pipxy fashion:
- `q2netcdf`
- `dumpQHeader`
- `mergeqfiles`
- `mkISDPcfg`

If you don't trust `pipx` you can also directly run the scripts from the src/q2netcdf directory.

