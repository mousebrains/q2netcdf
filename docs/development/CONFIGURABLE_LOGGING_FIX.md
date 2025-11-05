# Fix for Issue #4: Configurable Logging Levels

## Problem

Three CLI scripts still have hardcoded logging levels:
- `QFile.py:112` - hardcoded `logging.DEBUG`
- `QHeader.py:154` - hardcoded `logging.DEBUG`
- `QHexCodes.py:469` - hardcoded `logging.DEBUG`

## Solution

Add `--logLevel` argument to each script's `main()` function, following the pattern already implemented in `q2netcdf.py:175-180`.

---

## Fix for QFile.py

**Current code (lines 107-112):**
```python
parser = ArgumentParser()
parser.add_argument("filename", type=str, nargs="+", help="Input filename(s)")
parser.add_argument("--n", type=int, default=10, help="Number of data records to print out")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
```

**Fixed code:**
```python
parser = ArgumentParser()
parser.add_argument("filename", type=str, nargs="+", help="Input filename(s)")
parser.add_argument("--n", type=int, default=10, help="Number of data records to print out")
parser.add_argument(
    "--logLevel",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Logging level",
)
args = parser.parse_args()

logging.basicConfig(level=getattr(logging, args.logLevel))
```

---

## Fix for QHeader.py

**Current code (lines 145-154):**
```python
parser = ArgumentParser()
parser.add_argument("filename", type=str, nargs="+", help="Input filename(s)")
parser.add_argument("--config", action="store_false", help="Don't display config")
parser.add_argument("--channels", action="store_false", help="Don't display channel names")
parser.add_argument("--spectra", action="store_false", help="Don't display spectra names")
parser.add_argument("--frequencies", action="store_false", help="Don't display frequencies")
parser.add_argument("--nothing", action="store_true", help="Don't display extra")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
```

**Fixed code:**
```python
parser = ArgumentParser()
parser.add_argument("filename", type=str, nargs="+", help="Input filename(s)")
parser.add_argument("--config", action="store_false", help="Don't display config")
parser.add_argument("--channels", action="store_false", help="Don't display channel names")
parser.add_argument("--spectra", action="store_false", help="Don't display spectra names")
parser.add_argument("--frequencies", action="store_false", help="Don't display frequencies")
parser.add_argument("--nothing", action="store_true", help="Don't display extra")
parser.add_argument(
    "--logLevel",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Logging level",
)
args = parser.parse_args()

logging.basicConfig(level=getattr(logging, args.logLevel))
```

---

## Fix for QHexCodes.py

**Current code (lines 464-469):**
```python
parser = ArgumentParser()
parser.add_argument("ident", type=str, nargs="*", help="hex ident(s) to look up")
parser.add_argument("--name", type=str, action="append", help="Name to translate to ident")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
```

**Fixed code:**
```python
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
```

---

## Benefits

1. **User control**: Users can adjust verbosity without code changes
2. **Consistency**: All CLI tools now use the same logging interface
3. **Production-friendly**: Default to INFO instead of DEBUG (less noise)
4. **Debugging**: Users can still enable DEBUG when needed

## Usage Examples

```bash
# Use default INFO level
QFile data.q

# Enable debug logging
QFile data.q --logLevel DEBUG

# Suppress most output
QHeader data.q --logLevel ERROR
```

## Implementation Notes

- Uses `getattr(logging, args.logLevel)` to convert string to logging constant
- Default changed from DEBUG to INFO (more appropriate for normal use)
- Choices restricted to valid logging levels
- Pattern matches existing implementation in `q2netcdf.py`
