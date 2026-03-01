# AGENTS.md - Cubes OLAP Framework

## Project Context

This is a **Greenfield Rebuild (Rip and Replace)** of a legacy Python 2/3 OLAP framework into modern Python.

- **Legacy code**: `cubes/` - Old architecture, legacy internal APIs, outdated patterns
- **New rebuild**: `src/cubes/` - Modern Python with zero backward compatibility constraints
- **Preserved**: External specifications, business value, domain invariants

## Environment Setup

This project uses **Nix + direnv** for reproducible development with **two separate Python environments**:

### 1. Legacy Environment (root directory)
For working with the legacy `cubes/` codebase:

```bash
# The root directory uses the legacy venv
# Install legacy dependencies
pip install -r requirements.txt
pip install -r requirements-optional.txt
pip install -e ".[dev]"
```

### 2. Modern Environment (src/ directory)
For working with the new `src/cubes/` rebuild:

```bash
# The direnv automatically activates src/.venv when you enter the project
# Or manually activate the modern venv:
source src/.venv/bin/activate

# Install modern dependencies (when pyproject.toml is added)
cd src && uv pip install -e ".[dev]"
```

### Nix Shell

The project uses Nix for reproducible system dependencies:

```bash
# Enter development shell (handled automatically by direnv)
nix develop

# Or manually:
nix-shell
```

Python 3.11 is used for both environments.

## Build/Test/Lint Commands

### Legacy Codebase (`cubes/`)

```bash
# Run all tests
python -m pytest tests/

# Run single test file
python -m pytest tests/test_model.py

# Run single test class
python -m pytest tests/test_model.py::AttributeTestCase

# Run single test method
python -m pytest tests/test_model.py::AttributeTestCase::test_basics

# Run with coverage
python -m pytest --cov=cubes --cov-report=term-missing tests/

# Install in dev mode
pip install -e ".[dev]"

# Run legacy test suite
python setup.py test
```

### New Rebuild (`src/cubes/`)

For the new codebase (when pyproject.toml is added):

```bash
# Install with modern tooling (uv recommended)
cd src && uv pip install -e ".[dev]"

# Run tests
pytest tests/metadata/

# Type checking
mypy src/cubes/

# Linting
ruff check src/cubes/
ruff format src/cubes/

# Run specific test
pytest src/tests/metadata/test_attributes.py -v
```

### Docstrings (Google Style)

```python
def rollup(self, path: List[str]) -> List[str]:
    """Get rollup levels for a given path.

    Args:
        path: List of level names representing the current drill-down path.

    Returns:
        List of level names that can be rolled up to.

    Raises:
        HierarchyError: If the path is invalid or empty.
    """
```


## File Locations

- **Legacy source**: `cubes/` (Python 2/3 compatible)
- **New source**: `src/cubes/` (modern Python)
- **Legacy tests**: `tests/` (uses `unittest`)
- **New tests**: `src/tests/metadata/` (use `pytest`)
- **Test data**: `src/tests/data/`
- **Test models**: `src/tests/models/`
- **JSON schemas**: `cubes/schemas/`
- **Documentation**: `doc/`

## Codebase Patterns

### Metadata Models Architecture (New Rebuild)

The new metadata module uses a dual-layer architecture:

1. **Pydantic Schemas** (`src/cubes/metadata/schemas/`): Input validation and parsing
   - Handle raw JSON/dict input validation
   - Normalize and coerce values (e.g., "ASC" -> "asc")
   - Schema compliance with legacy JSON schemas

2. **Dataclass Domain Models** (`src/cubes/metadata/models/`): Business logic and relationships
   - Clean, type-safe domain objects
   - Implement business logic (dependencies, lookups, etc.)
   - Use `to_dict()` for serialization

### ModelObject Base Class

All model objects extend `ModelObject` which provides:

- Standard attributes: `name`, `label`, `description`, `info`
- `to_dict(create_label=False)`: Serialization with optional label generation

### Testing Patterns

- Use pytest for new tests in `src/tests/metadata/`
- Run from `src/` directory to avoid importing legacy code
- Test both base class and subclass behavior
- Use mypy --strict for type checking
- Use ruff for linting
