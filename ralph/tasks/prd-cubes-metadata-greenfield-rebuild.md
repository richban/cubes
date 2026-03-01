# PRD: Cubes Metadata Greenfield Rebuild

## Introduction

Rebuild the `cubes.metadata` module from scratch using modern Python (3.11+) with Pydantic v2 for input parsing and standard dataclasses for domain logic. This is a complete architectural migration (Rip and Replace) with zero backward compatibility constraints, while preserving all domain invariants and external specifications.

The legacy codebase contains Python 2/3 compatibility baggage, deprecated design patterns, and tightly coupled parsing/business logic. The new architecture strictly separates input parsing (Pydantic schemas) from domain logic (dataclasses), providing type safety, validation, and maintainability.

## Goals

- Decouple input parsing from business logic using Pydantic v2 schemas and Python dataclasses
- Eliminate all Python 2/3 compatibility code and deprecated patterns
- Preserve 100% domain invariant equivalence with legacy implementation
- Provide strict type safety throughout the metadata module
- Support existing JSON schemas (`cube.json`, `dimension.json`, `model.json`) for backward data compatibility
- Enable template inheritance and dimension cloning with clean, testable code
- Maintain mathematical equivalence for hierarchy traversal, drill-downs, and aggregations
- Achieve comprehensive test coverage for all ported components

## User Stories

### Phase 1: Core Base and Common Models

#### US-001: Create ModelObject Domain Dataclass
**Description:** As a developer, I want a clean base dataclass for all model objects so that common properties (name, label, description, info) are standardized and type-safe.

**Acceptance Criteria:**
- [ ] Create `src/cubes/metadata/models/base.py` with `@dataclass` ModelObject
- [ ] Properties: `name: str`, `label: Optional[str]`, `description: Optional[str]`, `info: dict` (default `{}`)
- [ ] Implement `to_dict()` method with optional `create_label` parameter
- [ ] Port all ModelObject initialization tests from legacy
- [ ] Typecheck passes with `mypy --strict`

#### US-002: Create Metadata I/O Utilities
**Description:** As a developer, I want utility functions for reading/writing model metadata bundles so that I can load multi-file models and directory structures.

**Acceptance Criteria:**
- [ ] Implement `read_model_metadata()` supporting file paths, URLs, dicts, and directories
- [ ] Implement `read_model_metadata_bundle()` for `model.cubesmodel/` directories
- [ ] Implement `write_model_metadata_bundle()` for exporting models
- [ ] Support `dim_*.json` and `cube_*.json` file patterns
- [ ] Use `pathlib` instead of `os.path`
- [ ] Port legacy I/O tests with exact JSON fixtures
- [ ] Handle Windows drive letter edge cases (legacy compatibility)

#### US-003: Create Object Dictionary Utility
**Description:** As a developer, I want a utility to convert model object lists to keyed dictionaries so that I can enforce uniqueness and enable fast lookups.

**Acceptance Criteria:**
- [ ] Implement `object_dict()` function with `by_ref` option
- [ ] Raise `ModelError` on duplicate keys with descriptive message
- [ ] Return `dict` (Python 3.7+ ordered) instead of legacy `OrderedDict`
- [ ] Support custom error messages via `error_message` and `error_dict` parameters
- [ ] Port legacy `object_dict` tests
- [ ] Typecheck passes

---

### Phase 2: Attributes and Measures

#### US-004: Create Attribute Pydantic Schema
**Description:** As a developer, I want a Pydantic schema for parsing attribute definitions so that raw JSON/dict inputs are validated and coerced correctly.

**Acceptance Criteria:**
- [ ] Create `src/cubes/metadata/schemas/attributes.py`
- [ ] Schema fields: `name: str`, `label: Optional[str]`, `description: Optional[str]`, `order: Optional[Literal["asc", "desc"]]`, `info: dict`, `format: Optional[str]`, `missing_value: Any`, `expression: Optional[str]`
- [ ] Implement `@field_validator("order")` to normalize case (ASC/asc -> "asc")
- [ ] Implement `BeforeValidator` to coerce string inputs to `{"name": string}` dicts
- [ ] All legacy `expand_attribute_metadata` edge cases covered
- [ ] JSON schema compliance with `cubes/schemas/dimension.json`
- [ ] Comprehensive validation tests

#### US-005: Create AttributeBase Domain Dataclass
**Description:** As a developer, I want a domain dataclass for attribute base properties so that format, expression dependencies, and ordering are cleanly modeled.

**Acceptance Criteria:**
- [ ] Create `src/cubes/metadata/models/attributes.py` with `AttributeBase` dataclass
- [ ] Properties: `name`, `label`, `description`, `order` (normalized to "asc"/"desc"), `info`, `format`, `missing_value`, `expression`, `ref` (computed from name)
- [ ] Implement `__eq__`, `__hash__` based on `ref`
- [ ] Implement `to_dict()` using `dataclasses.asdict`
- [ ] Implement `dependencies` property that parses expression via `inspect_variables`
- [ ] Implement `is_base` property (True if no expression)
- [ ] Port legacy AttributeBase tests
- [ ] Typecheck passes

#### US-006: Create Attribute Domain Dataclass
**Description:** As a developer, I want a domain dataclass for dimension attributes.

**Acceptance Criteria:**
- [ ] Extend `AttributeBase` with `Attribute` dataclass
- [ ] Add `dimension: Optional[Dimension]` property
- [ ] Implement `dimension` property setter that updates `ref` to `dimension.name + "." + name`
- [ ] Ensure `__deepcopy__` disowns dimension (sets to None)
- [ ] Port legacy Attribute tests

#### US-007: Create Measure Pydantic Schema
**Description:** As a developer, I want a Pydantic schema for measure definitions so that nonadditive types and aggregates are validated correctly.

**Acceptance Criteria:**
- [ ] Extend attribute schema with measure-specific fields
- [ ] Fields: `formula: Optional[str]`, `aggregates: list[str]`, `nonadditive: Optional[Literal["none", "all", "time"]]`, `window_size: Optional[int]`
- [ ] Implement `@field_validator("nonadditive")` to normalize values ("none"->None, "all"/"any"->"all")
- [ ] Implement `@model_validator(mode="after")` for aggregate function validation
- [ ] Support string coercion for simple measure names
- [ ] Port legacy measure validation tests

#### US-008: Create Measure Domain Dataclass
**Description:** As a developer, I want a domain dataclass for measures so that formulas, aggregates, and nonadditive behavior are cleanly modeled.

**Acceptance Criteria:**
- [ ] Extend `AttributeBase` with `Measure` dataclass
- [ ] Properties: `formula`, `aggregates`, `nonadditive`, `window_size`
- [ ] Implement `default_aggregates()` method that generates `MeasureAggregate` objects
- [ ] Handle "identity" aggregate magic string
- [ ] Normalize nonadditive in `__post_init__` ("none"->None, etc.)
- [ ] Port legacy measure tests including aggregate generation

#### US-009: Create MeasureAggregate Pydantic Schema
**Description:** As a developer, I want a Pydantic schema for measure aggregates so that mutual exclusivity of measure vs expression is enforced.

**Acceptance Criteria:**
- [ ] Schema fields: `name`, `label`, `function`, `formula`, `measure`, `expression`, `nonadditive`, `window_size`
- [ ] Implement `@model_validator(mode="after")` to enforce: cannot have both `measure` AND `expression`
- [ ] Validate function names against supported aggregation functions
- [ ] Support implicit aggregate generation patterns
- [ ] Port legacy validation tests

#### US-010: Create MeasureAggregate Domain Dataclass
**Description:** As a developer, I want a domain dataclass for measure aggregates so that aggregation logic and dependencies are cleanly modeled.

**Acceptance Criteria:**
- [ ] Extend `AttributeBase` with `MeasureAggregate` dataclass
- [ ] Properties: `function`, `formula`, `measure`, `nonadditive`, `window_size`
- [ ] Implement `is_base` property (True if no expression AND no function)
- [ ] Implement `dependencies` property: returns `{measure}` if measure set, else expression variables
- [ ] Raise `ModelError` if both measure and expression set (defensive programming)
- [ ] Port legacy MeasureAggregate tests

#### US-011: Create Attribute Dependency Utilities
**Description:** As a developer, I want utilities for collecting and sorting attribute dependencies so that topological ordering is maintained for derived attributes.

**Acceptance Criteria:**
- [ ] Implement `collect_attributes()` utility for flattening attribute lists
- [ ] Implement `collect_dependencies()` for topological sorting
- [ ] Implement `depsort_attributes()` with cycle detection
- [ ] Raise `ExpressionError` on circular dependencies with descriptive message
- [ ] Raise `ExpressionError` on unknown attribute references
- [ ] Port legacy dependency sorting tests
- [ ] Ensure mathematical equivalence with legacy topological sort algorithm

---

### Phase 3: Hierarchies, Levels, and Dimensions

#### US-012: Create Level Pydantic Schema
**Description:** As a developer, I want a Pydantic schema for hierarchy levels so that key, label, and order attribute defaults are handled during parsing.

**Acceptance Criteria:**
- [ ] Create `src/cubes/metadata/schemas/dimension.py`
- [ ] Schema fields: `name`, `attributes` (list of Attribute schemas), `key`, `label_attribute`, `order_attribute`, `order`, `cardinality`, `role`, `nonadditive`, `info`, `description`
- [ ] Implement `@model_validator(mode="after")` to set defaults: `key` = first attribute, `label_attribute` = second attribute (or first if only one), `order_attribute` = first attribute
- [ ] Support string coercion for level names (creates single attribute)
- [ ] Integrate with Attribute schema for nested validation
- [ ] Port legacy level expansion tests

#### US-013: Create Level Domain Dataclass
**Description:** As a developer, I want a domain dataclass for levels so that attribute ownership and defaults are cleanly modeled.

**Acceptance Criteria:**
- [ ] Create `src/cubes/metadata/models/dimension.py` with `Level` dataclass
- [ ] Properties: `name`, `attributes` (list), `key` (Attribute), `label_attribute` (Attribute), `order_attribute` (Attribute), `order`, `cardinality`, `role`, `nonadditive`, `info`, `label`, `description`
- [ ] Implement `__post_init__` to resolve attribute references to actual Attribute objects
- [ ] Implement `attribute(name)` lookup method
- [ ] Implement `has_details` property (True if len(attributes) > 1)
- [ ] Implement `to_dict()` with `full_attribute_names` option
- [ ] Port legacy Level tests including default attribute resolution

#### US-014: Create Hierarchy Pydantic Schema
**Description:** As a developer, I want a Pydantic schema for dimension hierarchies so that level references are validated.

**Acceptance Criteria:**
- [ ] Schema fields: `name`, `levels` (list of strings or Level schemas), `label`, `info`, `description`
- [ ] Validate that levels list is not empty
- [ ] Support level name references (strings) that resolve later
- [ ] Validate hierarchy name uniqueness within dimension
- [ ] Port legacy hierarchy expansion tests

#### US-015: Create Hierarchy Domain Dataclass
**Description:** As a developer, I want a domain dataclass for hierarchies so that level ordering and traversal algorithms are cleanly implemented.

**Acceptance Criteria:**
- [ ] Create `Hierarchy` dataclass with properties: `name`, `levels` (ordered list of Level objects), `label`, `info`, `description`
- [ ] Implement `__post_init__` to convert level names to Level objects if needed
- [ ] Implement `keys(depth)` method returning level keys up to depth
- [ ] Implement `levels_for_path(path, drilldown=False)` returning levels for a path
- [ ] Implement `levels_for_depth(depth, drilldown=False)` with bounds checking
- [ ] Implement `next_level(level)` returning next level or None
- [ ] Implement `previous_level(level)` returning previous level or None
- [ ] Implement `level_index(level)` returning positional index
- [ ] Implement `is_last(level)` checking if level is terminal
- [ ] Implement `rollup(path, level=None)` rolling up path to specified level
- [ ] Implement `path_is_base(path)` checking if path reaches hierarchy bottom
- [ ] Implement `all_attributes` property flattening all level attributes
- [ ] Implement collection methods: `__len__`, `__getitem__`, `__contains__`
- [ ] Port legacy Hierarchy traversal tests ensuring mathematical equivalence

#### US-016: Create Dimension Pydantic Schema
**Description:** As a developer, I want a Pydantic schema for dimensions so that template inheritance and hierarchy defaults are handled during parsing.

**Acceptance Criteria:**
- [ ] Schema fields: `name`, `levels`, `hierarchies`, `default_hierarchy_name`, `label`, `description`, `info`, `role`, `cardinality`, `category`, `nonadditive`, `attributes` (alternative to levels)
- [ ] Implement `@model_validator(mode="after")` to enforce: cannot have BOTH `levels` AND `attributes`
- [ ] Implement template resolution logic in `@model_validator(mode="before")` or factory
- [ ] If no hierarchies, create default hierarchy from all levels
- [ ] If no default_hierarchy_name, use "default" or first hierarchy
- [ ] Support `template` field for dimension inheritance
- [ ] Filter template hierarchies to only those using local levels
- [ ] Port legacy dimension expansion tests including template inheritance

#### US-017: Create Dimension Domain Dataclass
**Description:** As a developer, I want a domain dataclass for dimensions so that level ownership, hierarchy management, and attribute indexing are cleanly implemented.

**Acceptance Criteria:**
- [ ] Create `Dimension` dataclass with properties: `name`, `levels`, `hierarchies`, `_levels` (dict), `_hierarchies` (dict), `_attributes` (dict), `_attributes_by_ref` (dict), `default_hierarchy_name`, `label`, `description`, `info`, `role`, `cardinality`, `category`, `master`, `nonadditive`
- [ ] Implement `__post_init__` to:
  - Build level and hierarchy dictionaries
  - Set default level roles for time dimensions
  - Collect all attributes and build lookup maps
  - Assign dimension ownership to all level attributes
  - Validate and set default hierarchy
- [ ] Implement property getters: `levels`, `hierarchies`, `level_names`, `is_flat`, `has_details`, `key_attributes`, `attributes`
- [ ] Implement lookup methods: `level(obj)`, `hierarchy(obj=None)`, `attribute(name, by_ref=False)`
- [ ] Implement `clone(hierarchies=None, exclude_hierarchies=None, nonadditive=None, default_hierarchy_name=None, cardinality=None, alias=None)` for dimension cloning with modifications
- [ ] Clone must unown level attributes from original dimension
- [ ] Implement `to_dict()` with `hierarchy_limits` support
- [ ] Port legacy Dimension tests including cloning and template inheritance

#### US-018: Create Drilldown String Parser
**Description:** As a developer, I want a utility to parse drilldown strings so that dimension/hierarchy/level specifications can be parsed from compact notation.

**Acceptance Criteria:**
- [ ] Implement `string_to_dimension_level(astring)` function
- [ ] Parse format: `dimension@hierarchy:level` (hierarchy and level optional)
- [ ] Return tuple: `(dimension, hierarchy, level)`
- [ ] Raise `ArgumentError` on invalid format with descriptive message
- [ ] Use regex pattern: `[\w\d_]` for identifiers
- [ ] Port legacy parser tests

---

### Phase 4: The Cube Object

#### US-019: Create Cube Pydantic Schema
**Description:** As a developer, I want a Pydantic schema for cubes so that dimension links, implicit aggregates, and nonadditive propagation are handled during parsing.

**Acceptance Criteria:**
- [ ] Create `src/cubes/metadata/schemas/cube.py`
- [ ] Schema fields: `name`, `dimensions` (links), `measures`, `aggregates`, `details`, `label`, `description`, `info`, `category`, `mappings`, `joins`, `fact`, `key`, `browser_options`, `store`, `implicit_aggregates`
- [ ] Implement `@model_validator(mode="after")` for:
  - Expanding dimension links from strings to dicts
  - Generating default `fact_count` aggregate if no measures/aggregates
  - Implicit aggregate generation from measures if `implicit_aggregates=True`
  - Propagating `nonadditive` from cube to measures
  - Assigning implicit aggregate labels using template strings
- [ ] Handle mutually exclusive `dimensions` vs `dimension_links` validation
- [ ] Merge model-level and cube-level mappings and joins
- [ ] Port legacy cube expansion tests

#### US-020: Create Cube Domain Dataclass
**Description:** As a developer, I want a domain dataclass for cubes so that dimension linking, attribute collection, and aggregate management are cleanly implemented.

**Acceptance Criteria:**
- [ ] Create `src/cubes/metadata/models/cube.py` with `Cube` dataclass
- [ ] Properties: `name`, `measures`, `aggregates`, `details`, `dimensions`, `_dimensions`, `_measures`, `_aggregates`, `mappings`, `joins`, `fact`, `key`, `browser_options`, `store`, `store_name`, `dimension_links`, `label`, `description`, `info`, `category`, `basename`
- [ ] Implement `__post_init__` to:
  - Validate measures and aggregates are correct types
  - Build lookup dictionaries
  - Expand dimension links
  - Handle store string vs object
- [ ] Implement property getters: `measures`, `aggregates`, `dimensions`, `all_dimension_keys`
- [ ] Implement lookup methods: `measure(name)`, `get_measures(measures)`, `aggregate(name)`, `get_aggregates(names)`, `aggregates_for_measure(name)`, `dimension(obj)`
- [ ] Implement attribute collection properties: `all_attributes`, `base_attributes`, `all_fact_attributes`, `attribute_dependencies`, `all_aggregate_attributes`
- [ ] Implement `attribute(attribute)` lookup across dimensions, details, and measures
- [ ] Implement `get_attributes(attributes, aggregated=False)` with simplified/full reference support
- [ ] Implement `collect_dependencies(attributes)` for topological sorting
- [ ] Implement `link_dimension(dimension)` with cloning support for overrides
- [ ] Implement `_add_dimension(dimension)` internal method
- [ ] Implement `distilled_hierarchies` property returning `{(dim_name, hier_name): [level_keys]}`
- [ ] Implement `to_dict()` with `expand_dimensions` and `hierarchy_limits` options
- [ ] Port legacy Cube tests including dimension linking and attribute collection

#### US-021: Create Dimension Link Expansion
**Description:** As a developer, I want utilities for expanding and validating dimension links so that cube-dimension relationships are cleanly modeled.

**Acceptance Criteria:**
- [ ] Implement `expand_dimension_links(metadata)` function
- [ ] Convert strings to `{"name": link}` dicts
- [ ] Support link dicts with: `name`, `hierarchies`, `exclude_hierarchies`, `default_hierarchy_name`, `nonadditive`, `cardinality`, `alias`
- [ ] Implement `expand_cube_metadata(metadata)` for preprocessing cube definitions
- [ ] Handle legacy `hierarchies` nested dict format (deprecation warning optional)
- [ ] Port legacy link expansion tests

---

### Phase 5: Providers and Repositories

#### US-022: Create Metadata Provider Base Class
**Description:** As a developer, I want a base class for metadata providers so that different model sources (files, APIs, databases) can be implemented consistently.

**Acceptance Criteria:**
- [ ] Create `src/cubes/metadata/provider.py` with `MetadataProvider` base class
- [ ] Methods: `list_cubes()`, `has_cube(name)`, `cube(name)`, `dimension(name, templates=None)`
- [ ] Implement template resolution caching
- [ ] Implement dimension lookup with cycle detection
- [ ] Port legacy provider tests

#### US-023: Create Static Model Provider
**Description:** As a developer, I want a concrete provider for static JSON file models so that file-based model definitions can be loaded.

**Acceptance Criteria:**
- [ ] Create `StaticModelProvider` class extending `MetadataProvider`
- [ ] Constructor accepts metadata dictionary or file path
- [ ] Implement `list_cubes()` returning list of cube summary dicts
- [ ] Implement cube and dimension caching
- [ ] Support model directory bundles
- [ ] Port legacy StaticModelProvider tests

#### US-024: Create Model Validation Layer
**Description:** As a developer, I want a validation layer replacing jsonschema so that models are validated using Pydantic's native validation.

**Acceptance Criteria:**
- [ ] Drop dependency on `jsonschema` Draft4Validator
- [ ] Implement validation through Pydantic `Model.model_validate()`
- [ ] Convert legacy validation error messages to Pydantic-compatible format
- [ ] Support all validation rules from legacy `defaults.py`:
  - Mutually exclusive `levels` vs `attributes`
  - Required `master` and `detail` in joins
  - Duplicate dimension/cube name detection
  - Default hierarchy inference
- [ ] Port legacy validation tests ensuring equivalent error detection

---

## Functional Requirements

### Parsing & Validation

- **FR-001:** All input parsing MUST use Pydantic v2 models with strict type validation
- **FR-002:** String inputs for attribute/dimension/level names MUST be coerced to dicts using `BeforeValidator`
- **FR-003:** Legacy JSON schemas (`cube.json`, `dimension.json`, `model.json`) MUST be supported as Pydantic model definitions
- **FR-004:** Template inheritance MUST resolve before Pydantic validation or in `BeforeValidator`
- **FR-005:** Cyclic template dependencies MUST be detected and raise `ModelError`
- **FR-006:** Mutually exclusive fields (levels vs attributes, measure vs expression) MUST raise validation errors
- **FR-007:** Default values (key, label_attribute, order_attribute) MUST be computed in validators or `__post_init__`

### Domain Logic

- **FR-008:** All domain classes MUST be immutable dataclasses (frozen or replace-based updates)
- **FR-009:** Attribute references (`ref`) MUST update automatically when dimension ownership changes
- **FR-010:** Hierarchy traversal methods (`rollup`, `next_level`, `previous_level`) MUST be mathematically equivalent to legacy
- **FR-011:** Topological sorting for attribute dependencies MUST detect cycles and raise `ExpressionError`
- **FR-012:** Dimension cloning MUST unown level attributes from source dimension
- **FR-013:** Implicit aggregate generation MUST follow legacy naming conventions (`{measure}_{function}`)
- **FR-014:** Nonadditive property MUST propagate from cube to measures and aggregates

### I/O & Serialization

- **FR-015:** Model bundles MUST support `dim_*.json` and `cube_*.json` file patterns
- **FR-016:** `read_model_metadata()` MUST support file paths, URLs, dicts, and directory bundles
- **FR-017:** `write_model_metadata_bundle()` MUST preserve model structure when exporting
- **FR-018:** `to_dict()` methods MUST support options: `create_label`, `full_attribute_names`, `hierarchy_limits`
- **FR-019:** Serialization MUST use `dataclasses.asdict` or Pydantic's `model_dump()`, dropping custom `IgnoringDictionary`


---

## Non-Goals (Out of Scope)

- **NG-001:** No backward compatibility with legacy Python 2/3 compatibility shims
- **NG-002:** No support for deprecated `high_cardinality` boolean (use `cardinality` string)
- **NG-003:** No support for legacy `record_count` implicit aggregate (must be explicit)
- **NG-004:** No mutable model objects (all updates via clone/replace)
- **NG-005:** No in-place model modification (functional updates only)
- **NG-006:** No SQL backend logic (handled in separate `cubes.sql` module)
- **NG-007:** No query execution or browser implementation (metadata only)
- **NG-008:** No server/HTTP API logic (handled in `cubes.server`)

---

## Design Considerations

### Architecture

```
Input (JSON/Dict) -> Pydantic Schema -> Domain Dataclass -> Business Logic
                      (Validation)        (Invariants)
```

- Pydantic schemas handle all input validation and coercion
- Domain dataclasses implement business logic and invariants
- Clear separation: parsing vs domain
- Immutable data structures throughout

### Type Safety

- Use `from __future__ import annotations` for forward references
- All functions and methods must have type annotations
- Use `mypy --strict` for type checking
- Generic types for `object_dict` and similar utilities

### Testing Strategy

- Port all legacy tests from `tests/test_model.py`
- New test structure: `tests/metadata/test_*.py`
- Test files: `test_base.py`, `test_attributes.py`, `test_dimension.py`, `test_cube.py`, `test_providers.py`
- Use exact JSON fixtures from legacy tests for equivalence verification
- Property-based testing for topological sort and hierarchy traversal

---

## Technical Considerations

### Dependencies

- **Required:** `pydantic>=2.0`, `dataclasses` (stdlib)
- **Optional:** `expressions` (for dependency parsing, if still needed)
- **Dropped:** `jsonschema`, `ordereddict` (Python 3.7+ dicts are ordered)

### Migration Strategy

1. Implement Pydantic schemas first (input validation)
2. Implement domain dataclasses second (business logic)
3. Port tests for each component immediately after implementation
4. Verify mathematical equivalence with legacy for algorithms
5. Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 dependency order

### Performance

- Use `@cached_property` for expensive computed properties in domain classes
- Attribute lookup maps (`_attributes_by_ref`) built once in `__post_init__`
- Topological sorting algorithm ported exactly for O(V+E) complexity

### Error Handling

- Validation errors: Pydantic `ValidationError` with detailed context
- Domain errors: Custom exceptions (`ModelError`, `ArgumentError`, `HierarchyError`, `ExpressionError`)
- All error messages must be descriptive and actionable

---

## Success Metrics

- **SM-001:** 100% of legacy `tests/test_model.py` tests pass with new implementation
- **SM-002:** Type checking passes with `mypy --strict` (zero errors)
- **SM-003:** All JSON fixtures from legacy tests parse identically
- **SM-004:** Hierarchy traversal algorithms produce identical output for all test cases
- **SM-005:** Topological sort produces identical ordering for dependency graphs
- **SM-006:** Template inheritance resolves to identical dimension structures
- **SM-007:** Zero breaking changes to external JSON model specifications

---

## Open Questions

1. **Q:** Should we keep the `IgnoringDictionary` behavior (omitting falsy values) or use standard dict serialization?
   - **A:** (To be decided) Standard dict serialization is preferred unless UI relies on falsy omission.

2. **Q:** How should we handle the `expressions` library dependency for `inspect_variables`?
   - **A:** (To be decided) Evaluate if we can replace with simple regex or keep dependency.

3. **Q:** Should `ModelProvider` be a true abstract base class (ABC) or just a protocol?
   - **A:** (To be decided) ABC with abstract methods for clarity.

4. **Q:** Do we need to support the `info` dictionary nesting patterns from legacy?
   - **A:** (To be decided) Yes, `info` is user-extensible metadata and must be preserved.

5. **Q:** How do we handle the Windows drive letter hack in URL parsing?
   - **A:** (To be decided) Keep for compatibility or drop if modern `pathlib` handles it.

---

## Appendix: Legacy Test Mapping

| Legacy Test Class | New Test File | Stories Covered |
|-------------------|---------------|-----------------|
| `AttributeTestCase` | `test_attributes.py` | US-004, US-005, US-006 |
| `MeasuresTestsCase` | `test_attributes.py` | US-007, US-008, US-009, US-010 |
| `LevelTestCase` | `test_dimension.py` | US-012, US-013 |
| `HierarchyTestCase` | `test_dimension.py` | US-014, US-015 |
| `DimensionTestCase` | `test_dimension.py` | US-016, US-017, US-018 |
| `CubeTestCase` | `test_cube.py` | US-019, US-020, US-021 |
| Provider Tests | `test_providers.py` | US-022, US-023, US-024 |
| ModelObject Tests | `test_base.py` | US-001, US-002, US-003 |
