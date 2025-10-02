# Import Pattern Solution for q2netcdf

## Problem

The original codebase used try/except blocks for imports to handle both direct script execution and package imports:

```python
try:
    from QHeader import QHeader
    from QData import QData
except:
    from q2netcdf.QHeader import QHeader
    from q2netcdf.QData import QData
```

This pattern has several issues:
1. **Bare except**: Catches all exceptions, including keyboard interrupts and system errors
2. **Complexity**: Every module needs this boilerplate
3. **Maintenance**: Changes to imports require updating both branches
4. **Ambiguity**: It's unclear which import path should be used

## Solution

Use **explicit relative imports** within the package:

```python
from .QHeader import QHeader
from .QData import QData
```

## Why This Works

When installed as a package (via pip/pipx), Python treats `src/q2netcdf/` as a package, and relative imports work correctly. The entry points defined in `pyproject.toml` handle script execution:

```toml
[project.scripts]
q2netcdf = "q2netcdf.q2netcdf:main"
dumpQHeader = "q2netcdf.dumpQHeader:main"
QFile = "q2netcdf.QFile:main"
```

## Files Updated

All internal imports were changed to use explicit relative imports:

- `QData.py`: `from .QHeader import QHeader`
- `QFile.py`: `from .QHeader import QHeader`
- `QHeader.py`: `from .QConfig import QConfig`
- `QConfig.py`: `from .QVersion import QVersion`
- `q2netcdf.py`: `from .QHeader import QHeader`
- `QReduce.py`: `from .QHeader import QHeader`
- `dumpQHeader.py`: `from . import QHeader`

## Benefits

1. **Cleaner code**: No try/except boilerplate
2. **Better error messages**: Import errors are explicit
3. **Explicit is better than implicit**: Clear that these are relative imports
4. **Standard practice**: Follows Python packaging best practices
5. **Type checkers happy**: Tools like mypy understand relative imports better

## Running Scripts

### As installed commands (recommended):
```bash
pip install -e .
q2netcdf input.q --nc output.nc
dumpQHeader data.q
```

### As modules (for development):
```bash
python -m q2netcdf.q2netcdf input.q --nc output.nc
python -m q2netcdf.QHeader data.q
```

### Note on Direct Execution

Direct execution like `python src/q2netcdf/q2netcdf.py` won't work with relative imports, which is intentional. This enforces proper package installation and usage patterns.

## Additional Improvements

Along with fixing imports, we also:

1. Removed circular dependency in `pyproject.toml`
2. Fixed typo "recrods" → "records"
3. Replaced bare `except:` with specific exception types
4. Added comprehensive docstrings
5. Improved type hints (using `| None` syntax)
6. Standardized to f-strings for logging
7. Made logging level configurable via `--logLevel` argument
