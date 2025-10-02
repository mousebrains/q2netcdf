# q2netcdf Examples

This directory contains example scripts demonstrating how to use q2netcdf for various tasks.

## Prerequisites

Install q2netcdf:
```bash
pip install -e .
```

Or if installed via pipx:
```bash
pipx install git+https://github.com/mousebrains/q2netcdf
```

## Examples

### Basic Usage

1. **[read_qfile.py](read_qfile.py)** - Read and inspect Q-file contents
   - Open Q-files and read headers
   - Iterate through data records
   - Display channel and spectra information

2. **[convert_to_netcdf.py](convert_to_netcdf.py)** - Convert Q-file to NetCDF
   - Load Q-files as xarray Datasets
   - Add CF-1.8 compliant metadata
   - Write to NetCDF format with compression

3. **[reduce_qfile.py](reduce_qfile.py)** - Reduce Q-file size
   - Select specific channels and spectra
   - Create reduced Q-files with only needed data
   - Configure reduction via JSON

4. **[merge_qfiles.py](merge_qfiles.py)** - Merge multiple Q-files
   - Combine multiple Q-files into one
   - Handle size constraints with decimation
   - Time-based file selection

### Advanced Usage

5. **[inspect_header.py](inspect_header.py)** - Detailed header inspection
   - Parse Q-file headers
   - Display all metadata
   - Show sensor identifiers and mappings

6. **[batch_convert.py](batch_convert.py)** - Batch convert multiple files
   - Process entire directories
   - Error handling for corrupted files
   - Progress reporting

## Running Examples

Each example is self-contained and includes usage instructions:

```bash
cd examples
python read_qfile.py path/to/file.q
```

Most examples accept a Q-file path as the first argument. Run with `--help` for detailed options:

```bash
python read_qfile.py --help
```

## Sample Data

If you have sample Q-files, place them in this directory for testing. The examples will work with any valid Rockland Scientific Q-file.
