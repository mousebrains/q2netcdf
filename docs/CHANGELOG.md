# Changelog

All notable changes to the q2netcdf project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.2] - 2026-08-12

### Fixed
- **Record timestamps no longer overflow under numpy 2.5**: record times are now
  held as `datetime64[ms]` instead of `datetime64[ns]`. `QHeader` builds its
  timestamp from a year-zero epoch, but the `datetime64[ns]` range is only
  1677-09-21 to 2262-04-11, so any header timestamp outside that window
  overflowed on conversion. numpy <= 2.4 wrapped silently and returned a wrong
  time (year 0 became 1753-08-29T22:43:41); numpy 2.5 raises `OverflowError`,
  which surfaced as 28 test failures. Milliseconds match the resolution the
  Q-file actually stores, so the change is lossless, and the `datetime64[ms]`
  range removes the overflow entirely -- a Q-file with a corrupt or zero header
  timestamp now converts instead of aborting. NetCDF output is unchanged:
  written time values are identical and xarray still reads them back as `ns`.
  Note that `Dataset.time.dtype` from `loadQfile`/`loadQfiles` is now `ms`.

### Changed
- **Lint rules are now selected explicitly** in `pyproject.toml`, and the ruff
  version is pinned in CI and pre-commit. Ruff 0.16.0 expanded its implicit
  default rule set, reporting 218 findings across 26 rules on a tree that
  ruff 0.15.0 reported 0 on, with no source change. The deferred rule groups
  and their current counts are recorded in `pyproject.toml` so the cost of
  opting into each is known.

## [0.6.1] - 2026-08-12

### Added
- `CITATION.cff` so GitHub and reference managers can generate a citation for the package
- `.zenodo.json` supplying archive metadata (title, author with ORCID, affiliation, license, keywords) to the Zenodo GitHub integration

### Notes
- This release exists to trigger the first Zenodo archive and mint a DOI. No code changes.

## [0.6.0] - 2026-03-25

### Added
- **`loadQfiles()` bulk parser for batch conversion**: parses every record of a
  file in a single numpy structured-dtype call and builds one `xr.Dataset` at
  the end, removing the per-file xarray overhead and the per-record
  `struct.unpack` calls. Handles mixed v1.2/v1.3 inputs, heterogeneous
  channel/spectra/config keys, varying frequency dimensions, multi-segment
  files, and empty files.

  Benchmarks reported with the release (773 Q-files, 101K records; author's
  measurement, single run, not independently reproduced):

  |              | v0.5.1 | v0.6.0 | Speedup |
  |--------------|--------|--------|---------|
  | Load + merge | 3.5s   | 0.27s  | 13x     |
  | Write        | 5.8s   | 1.6s   | 3.6x    |
  | Total        | 9.3s   | 2.1s   | 4.4x    |

- `loadQfile`, `loadQfiles`, and `mergeDatasets` are now exported from the
  `q2netcdf` package root, so `from q2netcdf import loadQfiles` works

### Changed
- Default `--compressionLevel` lowered from 5 to 1. Produces a larger NetCDF
  file in exchange for roughly twice the write speed.
- Applied ruff formatting to `q2netcdf.py`

### Fixed
- `mergeDatasets()` no longer fails when the `freq` coordinate is absent from
  some of the datasets being merged (`coords="minimal"` on the time concat)
- mypy could not infer the spectra array shape under Python 3.10

## [0.5.1] - 2026-03-22

### Changed
- Version bump only. The tag contains no source, test, or documentation
  changes -- the sole diff against v0.5.0 is the version string in
  `pyproject.toml`. (The v0.5.1 GitHub release notes restate the v0.5.0
  changes; they do not describe anything introduced in this tag.)

## [0.5.0] - 2026-03-22

### Added
- PyPI publishing on GitHub release via trusted publishing (OIDC)
- TestPyPI publish workflow, triggered manually with `workflow_dispatch`
- PyPI version badge, plus `Changelog` and `Issues` project URLs
- Python API examples in the README (streaming records with `QFile`,
  converting with `loadQfile`)

### Changed
- README states the GPL-3.0-or-later license explicitly
- Updated GitHub Actions versions to clear the Node.js 20 deprecation warnings
- Documented the planned `mergeqfiles.py` performance optimizations in
  `CLAUDE.md` for implementation once hardware is available for testing

### Fixed
- Suppressed the spurious "No header found" warning for zero-length input
  files; the warning now fires only when the file is non-empty

## [0.4.3] - 2025-03-22

### Changed
- **22x speedup for `loadQfile()`**: Replaced per-record `xr.Dataset` construction + `xr.concat` with batch numpy array stacking (`np.stack`) and single `xr.Dataset` build. The bottleneck was xarray Python overhead (96% of per-file time), not I/O (4%).
- Handles multi-header Q-files by processing segments independently, then concatenating segments (typically 1-2) instead of individual records (100+)

### Added
- 120 new tests across all modules, increasing test coverage from 78% to 96%
  - Error paths: unknown identifiers, data-before-header, truncated files, malformed binary data
  - `mergeqfiles.py`: QReduce class, `decimateFiles()`, `reduceFiles()`, `reduceAndDecimate()`, `scanDirectory()`, argument validators, CLI `main()`
  - `QReduce.py`: v1.2 spectra reduction, `decimate()`, `loadConfig()` validation, `__chkExists()`
  - `QHexCodes.py`: list-type name overflow, `__repr__()`
  - `mkISDPcfg.py`: single-quote fallback, both-quotes error
  - `QFile.py`: `validate()` with unknown identifiers, EOF errors, closed file pointer
  - `QConfig.py`: invalid UTF-8 in v1.2 config parsing
- Documented planned `mergeqfiles.py` performance optimizations in CLAUDE.md for future hardware testing

## [0.4.2] - 2025-03-18

### Changed
- Added `--version` flag to all 7 CLI entry points
- CF-1.13 compliant metadata in NetCDF output
- QHexCodes fixes for edge cases in sensor mapping
- Committed test data files (v1.2 and v1.3 Q-files) to repository

### Added
- Roundtrip validation tests (Q-file → NetCDF → verify)
- Tests for error handling, mismatched schemas across multi-file merge
- Hardened error handling across main package with proper exception types

### Fixed
- Critical mergeqfiles bugs: version handling, `fsync` on output, crash recovery with temp files
- Windows CI: replaced Unicode checkmark with ASCII in test output
- Bumped GitHub Actions artifact actions to Node.js 24 compatible versions
- Codecov action v5: renamed `file` input to `files`
- Cancel in-progress CI runs on new pushes to same branch
- Removed `.coverage` from repo, added to `.gitignore`

## [0.4.1] - 2025-03-16

### Changed
- Decomposed QHeader into smaller, testable methods
- Consolidated documentation into `docs/` directory
- Applied ruff formatter across all source and test files

### Added
- PyPI trusted publishing workflow
- Expanded test coverage to 77%
- Multi-file merge and multi-header read support in QFile

### Fixed
- Multi-file merge correctness for mismatched channel sets
- Multi-header file reads (files with multiple header/data segments)
- Code quality improvements from Codex 5.2 review

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

[Unreleased]: https://github.com/mousebrains/q2netcdf/compare/v0.4.3...HEAD
[0.4.3]: https://github.com/mousebrains/q2netcdf/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/mousebrains/q2netcdf/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/mousebrains/q2netcdf/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/mousebrains/q2netcdf/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mousebrains/q2netcdf/releases/tag/v0.3.0
