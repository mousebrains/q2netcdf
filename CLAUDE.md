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
- **Hex codes** (0x0010 - 0x0D20): Binary identifiers in Q-files (upper 12 bits = sensor type, lower 4 bits = instance)
- **Names** (e.g., "sh_0", "T_0", "pressure"): Human-readable sensor names with zero-based instance numbering
- **Attributes**: Units, long names, CF-compliant metadata

The mapping includes instance numbering (e.g., sh_0, sh_1 for multiple shear probes).

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
2. Parse all data records into `QRecord` objects, grouped by segment (header)
3. Per segment: stack channels/spectra into numpy arrays via `np.stack`, build one `xr.Dataset`
4. Concat segments (if multi-header file), add file-level config as coordinate variables
5. Apply CF-1.13 compliant metadata via `cfCompliant()`

**Critical**: Scalars use `("time",)` dimension, spectra use `("time", "freq")`.

**Performance note (v0.4.3)**: Batch numpy construction replaced per-record `xr.Dataset` creation + `xr.concat`, yielding ~22x speedup. The bottleneck was xarray Python overhead (96% of time), not I/O (4%).

## Planned Performance Work: mergeqfiles.py

The following optimizations are documented for implementation when hardware is
available for testing. All are single-threaded and low-energy. Benchmark on
the actual Slocum/MR hardware before and after each change.

### 1. glueFiles — replace Python copy loop with zero-copy I/O

**Where**: `glueFiles()` lines 1353-1359
**Current**: `while True: buffer = ifp.read(bufferSize); ofp.write(buffer)`
**Proposed**: Replace with `shutil.copyfileobj(ifp, ofp, bufferSize)` (C-level
loop) or `os.sendfile(ofp.fileno(), ifp.fileno(), None, filesize)` on Linux
(zero-copy kernel transfer, no userspace memory).
**Expected gain**: 2-5x on the glue path. Also bump CLI default `--bufferSize`
from 100KB to 1MB (line 1522).
**Risk**: Low. `shutil.copyfileobj` is a drop-in. `os.sendfile` is
Linux-only — need platform guard.

### 2. QReduce.reduceFile — vectorize record reduction

**Where**: `QReduce.reduceFile()` lines 1078-1098, `__reduceRecord()` lines
1061-1076
**Current**: Python `while True` loop reads one record at a time, calls
`np.frombuffer` + fancy index + `tobytes()` per record.
**Proposed**: Read all data bytes after the header in one `ifp.read()`. Reshape
to `(nRecords, fieldsPerRecord)` as a 2D numpy array. Select columns with
`data[:, self.__indices]` in one operation. Prepend ident+stime columns. Write
result with a single `ofp.write(result.tobytes())`.
**Expected gain**: 10-20x for the reduce path (same pattern as the loadQfile
v0.4.3 optimization — eliminates N Python iterations in favor of one bulk
numpy operation).
**Risk**: Medium. Must handle partial final records (file size not exact
multiple of dataSize). Verify byte-identical output.

### 3. QReduce.decimate — same vectorization

**Where**: `QReduce.decimate()` lines 1100-1120
**Current**: Per-record seek+read+reduce loop.
**Proposed**: Read entire file data section into memory once, reshape to 2D
array, select rows by `indices` and columns by `self.__indices` in one numpy
operation, write result.
**Expected gain**: 5-10x for the decimate+reduce path.
**Risk**: Same as #2. Memory usage bounded by file size (already bounded by
`maxSize`).

### 4. decimateFiles — batch reads for consecutive indices

**Where**: `decimateFiles()` lines 1313-1323
**Current**: Seeks to each record offset individually, reads one `dataSize`
chunk per iteration.
**Proposed**: Read entire file into memory (or at least coalesce adjacent index
runs into single reads), then slice the records as a numpy array and write
selected records in bulk.
**Expected gain**: 2-3x for the decimate-without-reduce path. Larger gain on
storage with high seek latency (SD cards, network mounts).
**Risk**: Low. Files are small (bounded by maxSize).

### Verification protocol

For each change:
1. Generate reference output from current code on the target hardware
2. Apply optimization
3. Verify byte-identical output (or document acceptable differences)
4. Benchmark wall-clock time and energy (if measurable) on representative
   file sets from the Slocum/MR deployment

### What not to optimize

- **fileCandidates()**: `os.scandir` is already optimal; directory scan is
  I/O-latency-bound.
- **QHeader parsing**: ~0.4ms per file, done once. Not a bottleneck.
- **scanDirectory() control flow**: Pure logic, no hot loops.
- **Threading/multiprocessing**: Adds complexity, not suitable for the
  low-energy embedded deployment target. The CPU-bound work (numpy operations)
  is better served by vectorization than parallelism for this workload.

## Testing Patterns

- Use `io.BytesIO` to create in-memory binary files for testing
- Pack binary data with `struct.pack()` using correct format for version
- Test both v1.2 and v1.3 paths separately
- Verify round-trip: binary → parsed → attributes match expected
