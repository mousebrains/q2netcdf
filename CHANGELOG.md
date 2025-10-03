# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-10-02

### Added
- **Generator pattern for `QFile.data()`** - Now yields QRecord objects for Pythonic iteration
- **Version management** - Single source of truth using `importlib.metadata.version()`
- **Comprehensive test suite** - 54 unit tests with pytest (all passing)
- **Examples** - 5 complete working examples with documentation
- **Documentation** - CHANGELOG, CONTRIBUTING, format documentation, examples
- **CI/CD** - GitHub Actions workflow with pytest, ruff, and mypy
- **Development tools** - Pre-commit hooks, editorconfig, gitattributes
- Type hints throughout codebase using Python 3.11+ union syntax
- Comprehensive docstrings for all public APIs and modules
- `RecordType` enum for type-safe binary record identifiers
- Configurable logging with `--logLevel` argument in all CLI tools
- Package metadata and clean API exports in `__init__.py`
- Robust import patterns with try/except fallback for better compatibility
- Named constants for magic numbers (e.g., `BUFFER_SIZE = 64 * 1024`)

### Fixed
- **Critical**: Duplicate dictionary key 0x640 in QHexCodes (second occurrence changed to 0x650)
- **Generator pattern**: `QFile.data()` now properly yields records instead of returning single/None
- **Empty array parsing**: QConfig now correctly handles empty arrays `[]`
- **Whitespace handling**: v1.2 config parser now tolerates extra whitespace around `=>`
- **FileNotFoundError**: Fixed mixed format string bug
- Typos in NetCDF attributes: "velocity_squard" → "velocity_squared", "disolved_oxygen" → "dissolved_oxygen"
- Typos in help text: "Overlatp", "paramters", "minimas", "dsplay"
- Bare except clauses replaced with specific exception types
- Magic numbers replaced with RecordType enum constants
- Long lines reformatted for readability

### Changed
- **Breaking**: `QFile.data()` returns Generator instead of `QRecord | None`
  - Old: `while record := qf.data(): ...`
  - New: `for record in qf.data(): ...`
- **Code style**: All quotes standardized to double quotes
- **Regex patterns**: Renamed `_REGEX_*` to `_PATTERN_*` for clarity
- Converted all string formatting to f-strings for consistency
- Standardized to explicit relative imports throughout package
- Improved error messages with file context and byte positions
- Enhanced logging messages with f-string formatting

### Testing
- 54 unit tests covering core functionality
- Test coverage: QRecordType 100%, QConfig 90%, QVersion 88%
- Real Q-file samples for integration testing
- All edge cases tested (empty arrays, whitespace, unicode, errors)

## [0.2.0] - 2025-02-XX

Initial public release.

### Added
- Support for Rockland Scientific Q-file format v1.2 and v1.3
- Conversion to CF-1.8 compliant NetCDF files
- Command-line tools: `QFile`, `QHeader`, `QHexCodes`, `mkISDPcfg`
- Binary parsing for header, configuration, and data records
- Hexadecimal identifier mapping for 200+ sensor types
- Support for scalar channels and frequency spectra
