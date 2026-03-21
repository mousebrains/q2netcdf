# Changelog

All notable changes to the q2netcdf project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2024-11-05

### Changed
- Removed CI testing for Python 3.7, 3.8, and 3.9 (mergeqfiles.py remains compatible with Python 3.7+, but these versions are no longer actively tested in CI)
- Updated pre-commit hook versions to latest stable releases (ruff 0.8.4, mypy 1.13.0, pre-commit-hooks 5.0.0, bandit 1.8.0)
- Optimized QHexCodes.name2ident() with reverse lookup cache for O(1) performance
- Reorganized documentation into `documents/` directory for better project structure
- Updated minimum Python version in pyproject.toml to 3.10 for main package
- Improved mergeqfiles.py code quality:
  - Removed duplicate class definitions (QConfig, RecordType, QVersion)
  - Consolidated duplicate imports
  - Reduced file size from 1,478 to 1,344 lines

### Added
- Coverage badge to README
- GitHub issue templates for bug reports and feature requests
- SECURITY.md file with vulnerability reporting instructions and best practices
- Consolidated development documentation in docs/development/ directory
- documents/README.md to organize and index all documentation
- **CI/CD Pipeline**: GitHub Actions workflow testing Python 3.10-3.13 across Linux, macOS, and Windows
  - Automated pytest with coverage reporting
  - Ruff linting and formatting checks
  - mypy type checking for all Python versions
  - Coverage artifact uploads to Codecov
- Comprehensive type hints to mergeqfiles.py for better IDE support and type checking
- Unit tests for mergeqfiles.py module (test_mergeqfiles.py) with 50+ tests
- Integration tests for end-to-end workflows (test_integration.py)
- Performance tests for hex code lookups and config parsing
- Error handling tests for corrupted files and invalid inputs

### Fixed
- Type hint compatibility issues with Python 3.7 and 3.8
- Duplicate class definitions in mergeqfiles.py
- Duplicate import statements
- Missing type annotations in QReduce.py, QFile.py, QHexCodes.py, and q2netcdf.py
- Untyped function definitions flagged by mypy --strict mode
- Missing typing imports (Union, Optional, Dict, Tuple, IO) in multiple modules
- Ruff formatting issue in mergeqfiles.py (inline comment spacing)
- Typo in README.md: "restablished" → "reestablished"

## [0.3.0] - 2025-03-15

### Added
- Initial project structure with src/ layout
- Q-file to NetCDF conversion (q2netcdf)
- Q-file header parsing (QHeader)
- Q-file data record parsing (QData)
- Q-file configuration parser (QConfig)
- Hex code to sensor/spectra name mapping (QHexCodes with 200+ codes)
- Q-file merging functionality (mergeqfiles)
- Q-file size reduction (QReduce)
- ISDP configuration generator (mkISDPcfg)
- Support for Q-file versions 1.2 and 1.3
- Context manager for safe file handling
- Command-line tools for all major operations
- Example scripts demonstrating usage

### Documentation
- Comprehensive README with installation and usage instructions
- Docstrings on all public classes and methods
- Inline comments explaining complex logic
- Multiple working examples

### Testing
- pytest-based test suite
- Coverage tracking configuration
- Tests for core modules (QFile, QHeader, QRecordType, QHexCodes, QConfig, QVersion)
- Test fixtures and conftest.py setup

---

## Migration Guides

### Migrating from 0.3.0 to 0.4.0

**Python Version:**
- Minimum Python version for main package is now 3.10 (was 3.11)
- mergeqfiles.py standalone tool remains compatible with Python 3.7+
- If you were using Python 3.11+ features in custom code, you may need to update:
  - Use `Union[X, Y]` instead of `X | Y` for type hints
  - Use `Dict`, `List`, `Tuple` from `typing` module instead of built-in generics

**Dependencies:**
- If you have strict version requirements, update your dependency specifications
- All dependencies have been tested with their new minimum versions

**Code Changes:**
- No breaking changes to public APIs
- All existing code should work without modification
- Type hints are now more comprehensive (beneficial for type checkers)

---

## Contributors

### Lead Developer
- Pat Welch (pat@mousebrains.com)

### Contributors
- Claude Code Assistant (Type hints, testing, documentation improvements)

### Acknowledgments
- Rockland Scientific for Q-file format specification (TN-054)
- TWR for Slocum Glider uRider proglet integration requirements

---

## Links

- **Repository**: https://github.com/mousebrains/q2netcdf
- **Issues**: https://github.com/mousebrains/q2netcdf/issues
- **Documentation**: https://github.com/mousebrains/q2netcdf/blob/main/README.md

---

[Unreleased]: https://github.com/mousebrains/q2netcdf/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/mousebrains/q2netcdf/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mousebrains/q2netcdf/releases/tag/v0.3.0
