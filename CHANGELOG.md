# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Type hints throughout codebase using Python 3.11+ union syntax
- Comprehensive docstrings for all public APIs and modules
- `RecordType` enum for type-safe binary record identifiers
- Test suite with pytest configuration and 20+ tests
- GitHub Actions CI/CD workflow with pytest, ruff, and mypy
- Configurable logging with `--logLevel` argument in all CLI tools
- Package metadata and clean API exports in `__init__.py`
- `.editorconfig` for consistent code formatting across editors
- `.gitattributes` for proper line ending and binary file handling

### Fixed
- Critical: Duplicate dictionary key 0x640 in QHexCodes (second occurrence changed to 0x650)
- Typos in NetCDF attributes: "velocity_squard" → "velocity_squared", "disolved_oxygen" → "dissolved_oxygen"
- Typos in help text: "Overlatp", "paramters", "minimas", "dsplay"
- Bare except clauses replaced with specific exception types
- Magic numbers replaced with RecordType enum constants
- Long lines in mkISDPcfg.py reformatted for readability

### Changed
- Converted all string formatting to f-strings for consistency
- Standardized to explicit relative imports throughout package
- Improved error messages with file context and byte positions
- Enhanced logging messages with f-string formatting

## [0.2.0] - 2025-02-XX

Initial public release.

### Added
- Support for Rockland Scientific Q-file format v1.2 and v1.3
- Conversion to CF-1.8 compliant NetCDF files
- Command-line tools: `QFile`, `QHeader`, `QHexCodes`, `mkISDPcfg`
- Binary parsing for header, configuration, and data records
- Hexadecimal identifier mapping for 200+ sensor types
- Support for scalar channels and frequency spectra
