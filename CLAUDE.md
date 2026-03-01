# CLAUDE.md

## Project Overview

Cubes is a Python framework for Online Analytical Processing (OLAP) and multidimensional analysis. It provides tools for browsing aggregated data with a focus on logical modeling that abstracts physical data implementation.

## Development Commands

### Installation
```bash
# Install development dependencies
pip install -e .[dev]
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_model.py
python -m pytest tests/sql/test_browser.py
```

### Documentation
```bash
cd doc
make html        # Build HTML documentation
make help        # See all available targets
```

### Slicer Command Line Tool
```bash
slicer --help           # Show all commands
slicer serve            # Start OLAP server
slicer model validate   # Validate model files
```

## Architecture Overview

### Core Components

**Metadata Layer** (`cubes/metadata/`)
- `model.py` - Logical model definitions (cubes, dimensions, measures)
- `cube.py` - Cube metadata and structure
- `dimension.py` - Dimension hierarchies and attributes
- `providers.py` - Model providers for loading metadata

**Query Engine** (`cubes/query/`)
- `browser.py` - Main interface for data browsing and aggregation
- `cells.py` - Cell definitions for slicing and dicing
- `computation.py` - Aggregation computation logic

**SQL Backend** (`cubes/sql/`)
- `browser.py` - SQL-specific browser implementation
- `mapper.py` - Logical-to-physical mapping
- `query.py` - SQL query generation
- `store.py` - Data store abstraction

**Server Components** (`cubes/server/`)
- `app.py` - Flask application setup
- `blueprint.py` - HTTP API endpoints
- `auth.py` - Authentication mechanisms

### Key Design Patterns

**Logical vs Physical Separation**: The framework uses logical models that abstract away physical database implementation details. The mapper component handles translation between logical and physical schemas.

**Provider Pattern**: Metadata providers allow loading model definitions from various sources (files, databases, etc.).

**Browser Pattern**: The browser provides a consistent interface for data exploration regardless of the backend storage system.

## Model Files

Models are typically defined in JSON format and located in:
- `models/` - Example model definitions
- `examples/` - Complete working examples with data
- `tests/models/` - Test model definitions

Common model file patterns:
- `model.json` - Main model definition
- `cube_*.json` - Individual cube definitions
- `dim_*.json` - Dimension definitions

## Configuration

Configuration files typically named `slicer.ini` contain:
- Database connection settings
- Model file locations
- Server configuration options
- Authentication settings

## Testing Structure

Tests are organized by component:
- `tests/test_*.py` - Core framework tests
- `tests/sql/` - SQL backend specific tests
- `tests/models/` - Test model definitions
- `tests/data/` - Test data and fixtures