# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This package provides Python tools for reading, parsing, and converting Rockland Scientific Q-files (oceanographic binary data files from the ISDP data logger) to NetCDF format. It supports Q-file versions 1.2 and 1.3, handles 200+ sensor identifier codes, and includes utilities for merging, reducing, and inspecting Q-files.

## Development Commands

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_qfile.py

# Run specific test class or method
pytest tests/test_qfile.py::TestQFile::test_context_manager

# Run with verbose output
pytest -v
```

### Linting and Type Checking
```bash
# Lint all code (must pass - no errors allowed)
ruff check src/ tests/

# Type check (must pass - no errors allowed)
mypy src/

# Both tools are strictly enforced - all issues must be fixed
```

### Installation for Development
```bash
# Install with development dependencies
pip install -e ".[dev]"
```

## Code Architecture

### Q-file Binary Format (Rockland TN-054)

Q-files contain two types of binary records, each prefixed with a 2-byte identifier:

- **Header Record (0x1729)**: Contains file version, timestamp, channel/spectra identifiers, frequencies, and configuration. Appears at start of file.
- **Data Records (0x1657)**: Contains timestamped measurements. Multiple records follow the header.

### Core Data Flow

The package follows a layered parsing architecture:

1. **Low-level parsing** (`QHeader`, `QData`): Read binary structs from file pointers, handle version differences
2. **High-level interface** (`QFile`): Context manager that coordinates header/data reading
3. **Conversion layer** (`q2netcdf`, `mergeqfiles`): Transform parsed data to NetCDF/merged formats

### Version Handling

Q-file format has two versions with different binary layouts:

- **v1.2**: Includes record number, error code, start/end times in each data record
  - Format: `<HHqee` (ident, record#, error, start_time, end_time) + channels + spectra
- **v1.3**: Optimized with only start time in data records
  - Format: `<He` (ident, start_time) + channels + spectra

The `QVersion` enum (values 1.2, 1.3) drives conditional logic throughout parsing code. Always check `version.isV12()` to branch between format variants.

### Sensor Identifier Mapping

`QHexCodes` provides bidirectional mapping between:
- **Hex codes** (0x0001 - 0x00FF): Binary identifiers in Q-files
- **Names** (e.g., "W1", "T1", "Sh1"): Human-readable sensor names
- **Attributes**: Units, long names, CF-compliant metadata

The mapping includes instance numbering (e.g., W1, W2 for multiple shear probes).

### Type Safety Requirements

**CRITICAL**: Both `ruff` and `mypy` must pass with zero errors. Key patterns:

- Use `Type | None` union syntax (Python 3.10+)
- Add assertions after None checks to help mypy narrow types
- Annotate dicts with `dict[str, Any]` when values have mixed types
- Convert `tuple` to `ndarray` or `list` when needed for operations
- For argparse: validators take `str` input, functions take `Namespace` for parsed args

### Configuration Parsing

`QConfig` handles two different configuration formats based on version:

- **v1.2**: Perl-style key-value pairs, arrays as `[1,2,3]`, nested structures
- **v1.3**: JSON strings embedded in header

The class automatically detects format and exposes unified `config()` dict interface.

### File Operations Best Practices

**Reading Q-files**:
- Always use `QFile` context manager (handles file closing)
- Call `header()` before `data()` (enforced with assertion)
- Files are read with 64KB buffer for performance

**Writing/Merging Q-files**:
- `mergeqfiles`: Concatenates multiple Q-files, can decimate if size exceeds limit
- `QReduce`: Selectively removes channels/spectra/records based on JSON config
- Both preserve binary format (don't parse/rewrite, just copy bytes)

### NetCDF Conversion Pipeline

`q2netcdf.loadQfile()` converts Q-file → xarray Dataset:

1. Read header to get channel/spectra identifiers and frequencies
2. For each data record:
   - Split into scalar channels and spectra arrays
   - Create mini Dataset with proper dimensions (time, freq)
3. Concatenate all records along time dimension
4. Add file-level config as coordinate variables
5. Apply CF-1.8 compliant metadata via `cfCompliant()`

**Critical**: Scalars use `("time",)` dimension, spectra use `("time", "freq")`.

## Testing Patterns

- Use `io.BytesIO` to create in-memory binary files for testing
- Pack binary data with `struct.pack()` using correct format for version
- Test both v1.2 and v1.3 paths separately
- Verify round-trip: binary → parsed → attributes match expected
