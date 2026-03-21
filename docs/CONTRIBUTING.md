# Contributing to q2netcdf

Thank you for considering contributing to q2netcdf! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites
- Python 3.11 or later
- Git

### Setting Up Your Development Environment

1. Clone the repository:
   ```bash
   git clone git@github.com:mousebrains/q2netcdf.git
   cd q2netcdf
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Coding Standards

### Python Style
- **Python Version**: Use Python 3.11+ features
- **Type Hints**: All functions should have type hints using modern syntax (`str | None` instead of `Optional[str]`)
- **String Formatting**: Use f-strings exclusively
- **Imports**: Use explicit relative imports within the package (`.QHeader`, `.QData`)
- **Docstrings**: All public classes, methods, and functions must have docstrings
- **Line Length**: Maximum 100 characters (configurable in `.editorconfig`)

### Code Examples

**Good:**
```python
def process_data(self, value: float) -> dict | None:
    """Process sensor data and return results.

    Args:
        value: Raw sensor reading

    Returns:
        Processed data dictionary or None if invalid
    """
    if value < 0:
        return None
    return {"processed": value * 2.0}
```

**Bad:**
```python
def process_data(self, value):  # Missing type hints and docstring
    if value < 0:
        return None
    return {'processed': value * 2.0}  # Use f-strings when interpolating
```

### Error Handling
- Avoid bare `except:` clauses
- Use specific exception types: `except (ValueError, TypeError):`
- Provide context in error messages with f-strings

## Running Tests

### Run All Tests
```bash
pytest
```

### Run with Coverage Report
```bash
pytest --cov=q2netcdf --cov-report=term-missing
```

### Run Specific Test File
```bash
pytest tests/test_qhexcodes.py
```

### Run Specific Test
```bash
pytest tests/test_qhexcodes.py::test_name_lookup
```

## Code Quality Tools

### Linting with Ruff
```bash
ruff check src/ tests/
```

### Type Checking with mypy
```bash
mypy src/
```

### Auto-formatting
```bash
ruff format src/ tests/
```

## Testing Guidelines

- Write tests for all new functionality
- Maintain or improve code coverage
- Use pytest fixtures for common test data (see `tests/conftest.py`)
- Test both success and error cases
- Include edge cases (empty inputs, None values, boundary conditions)

## Submitting Changes

### Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the coding standards above
   - Add tests for new functionality
   - Update docstrings and comments
   - Run tests and linters locally

3. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

   Commit message format:
   - Use present tense ("Add feature" not "Added feature")
   - Use imperative mood ("Move cursor to..." not "Moves cursor to...")
   - First line should be 50 characters or less
   - Reference issues: "Fix #123: Add support for..."

4. **Push to GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Provide clear description of changes
   - Reference related issues
   - Ensure CI tests pass
   - Request review

### Before Submitting

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Type hints added for new code
- [ ] Docstrings added/updated
- [ ] CHANGELOG.md updated (under `[Unreleased]`)
- [ ] No ruff or mypy warnings

## Project Structure

```
q2netcdf/
├── src/q2netcdf/          # Main package source
│   ├── __init__.py        # Package exports
│   ├── QFile.py           # Main Q-file parser
│   ├── QHeader.py         # Header record parser
│   ├── QData.py           # Data record parser
│   ├── QConfig.py         # Configuration parser
│   ├── QHexCodes.py       # Sensor identifier mapping
│   ├── QVersion.py        # Version enum
│   ├── QRecordType.py     # Record type enum
│   └── ...
├── tests/                 # Test suite
│   ├── conftest.py        # Shared fixtures
│   ├── test_*.py          # Test modules
│   └── ...
├── pyproject.toml         # Package configuration
├── CHANGELOG.md           # Version history
└── CONTRIBUTING.md        # This file
```

## Questions or Problems?

- Open an issue on GitHub for bugs or feature requests
- For questions about Q-file format, refer to Rockland Technical Note TN-054
- Email: pat@mousebrains.com

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing!
