# Greenfield Rebuild Implementation Plan: `cubes.metadata`

This document details the file-by-file, class-by-class migration plan for rebuilding the `cubes.metadata` module into modern Python using Pydantic v2 for parsing and standard dataclasses for domain logic.

## Goal Description
The purpose of this migration is to decouple input parsing from the deep business logic inside the metadata module. All legacy Python 2/3 compatibility baggage and deprecated design mechanisms will be dropped. The new architecture strictly parses JSON/dictionary schemas using Pydantic v2 schemas and outputs robust domain `dataclasses`, guaranteeing 100% equivalence in external invariants. 

## Proposed Changes

We will iterate file-by-file. For each component, we will:
1. Write the Pydantic schema in `src/cubes/metadata/schemas/`. **Note: The Pydantic models MUST fully support the existing JSON schemas located in `cubes/schemas/` (`cube.json`, `dimension.json`, `model.json`).**
2. Write the Domain dataclass in `src/cubes/metadata/models/`.
3. Port the respective legacy tests from `tests/test_model.py` into a new test structure (`tests/metadata/`).
4. Ensure the ported tests pass fully before moving to the next component.

---
### 1. The Core Base and Common Models
We start by defining the root objects and validation base classes that subsequent classes will build upon.

#### [NEW] `src/cubes/metadata/models/base.py`
- `ModelObject`: Convert to a standard `@dataclass`.
- `object_dict`: Convert into a generic `dict` generation utility.
- *Test Porting:* Ensure `ModelObject` initialization dict tests pass.

---
### 2. Attributes and Measures
The lowest-level building blocks.

#### [NEW] `src/cubes/metadata/schemas/attributes.py`
#### [NEW] `src/cubes/metadata/models/attributes.py`
- `AttributeBase`: Domain dataclass holding format, expression dependencies. Pydantic schema to validate order strings (`asc`/`desc`).
- `Attribute`: Add `dimension` back-reference modeling natively. 
- `Measure`: Ensure `nonadditive` validation logic is captured purely in the Pydantic schema validator (`@field_validator`).
- `MeasureAggregate`: Implement exclusive measure OR expression validations in Pydantic.
- *Test Porting:* Port `AttributeTestCase`, `MeasuresTestsCase` from `tests/test_model.py` directly to `/tests/metadata/test_attributes.py`.

---
### 3. Hierarchies, Levels, and Dimensions
The core structural grouping.

#### [NEW] `src/cubes/metadata/schemas/dimension.py`
#### [NEW] `src/cubes/metadata/models/dimension.py`
- `Level`: Dataclass. Move the default initialization logic (`key`, `label_attribute`, `order_attribute`) to the Pydantic parser logic, leaving the dataclass pure.
- `Hierarchy`: Dataclass containing ordered levels. Port the rollup and traversal array mathematics verbatim.
- `Dimension`: Dataclass. The complex "clone" and "template override" logic must be heavily tested.
- *Test Porting:* Port `LevelTestCase`, `HierarchyTestCase`, `DimensionTestCase` into `/tests/metadata/test_dimension.py`. Ensure template overriding works beautifully.

---
### 4. The Cube Object
The aggregate root combining dimensions, attributes, schemas, and measures.

#### [NEW] `src/cubes/metadata/schemas/cube.py`
#### [NEW] `src/cubes/metadata/models/cube.py`
- Pydantic schema will handle parsing `dimension_links` and triggering implicit generation of the default `fact_count` measure.
- Domain `Cube` class will cleanly index properties like `all_fact_attributes()`, `base_attributes()`, etc., using `@cached_property` where applicable to avoid expensive re-evaluation.
- *Test Porting:* Port `CubeTestCase` to `/tests/metadata/test_cube.py`. Validate `distilled_hierarchies` and dimension collection.

---
### 5. Providers and Repositories (Final Linkage)
Replace `defaults.py` and `providers.py` with the new architecture.

#### [NEW] `src/cubes/metadata/provider.py`
- Drop `defaults.py` as jsonschema draft 4 is replaced natively by Python `Model.model_validate(json)`.
- Reconstruct `StaticModelProvider` into an explicit metadata repository that resolves JSON references to final initialized Pydantic payloads, avoiding the dirty dictionary merging from the legacy framework. 
- *Test Porting:* Port relevant tests over to `/tests/metadata/test_providers.py`.

## Verification Plan

### Automated Tests
- After every single class is written, `pytest tests/metadata/test_[component].py` will be run locally. 
- Validation strategy: The inputs tested will be exact JSON dictionaries matching the old API specifications to ensure parsing via Pydantic matches perfectly.
- The Domain model methods (`rollup`, `levels_for_path`, `collect_dependencies`) will be tested against complex outputs to ensure mathematical and logical equality to the legacy models.
