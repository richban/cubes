# Audit: [defaults.py](file:///Users/melchior/Developer/cubes/cubes/metadata/defaults.py)

#### Component: [validate_model](file:///Users/melchior/Developer/cubes/cubes/metadata/defaults.py#28-33) (module function)
- **Public API Status:** Drop (Replace with Pydantic implicitly).
- **Core Domain Logic & Invariants:** Convenience wrapper to instantiate and run [ModelMetadataValidator](file:///Users/melchior/Developer/cubes/cubes/metadata/defaults.py#35-125).
- **Data Flow & Logic Flow:** Instantiate -> [validate()](file:///Users/melchior/Developer/cubes/cubes/metadata/defaults.py#48-62) -> return list of errors.
- **Validation vs. Domain Logic:** Pure validation.
- **Failure Handling:** Returns a list of generic validation errors.
- **Edge Cases:** None.
- **External Dependencies:** [ModelMetadataValidator](file:///Users/melchior/Developer/cubes/cubes/metadata/defaults.py#35-125).
- **Modernization Risk & Strategy:** Drop this completely. It relies on `jsonschema` draft4. We will replace this step entirely with Pydantic model validation `Model.model_validate(metadata)`.

#### Component: `ModelMetadataValidator.__init__`
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Loads raw `jsonschema` JSON files from package resources.
- **Data Flow & Logic Flow:** Uses `pkgutil` to read [model.json](file:///Users/melchior/Developer/cubes/cubes/schemas/model.json), [cube.json](file:///Users/melchior/Developer/cubes/cubes/schemas/cube.json), [dimension.json](file:///Users/melchior/Developer/cubes/cubes/schemas/dimension.json). Parses them.
- **Validation vs. Domain Logic:** Setup for logic.
- **Failure Handling:** None.
- **Edge Cases:** None.
- **External Dependencies:** `pkgutil`, [json](file:///Users/melchior/Developer/cubes/cubes/schemas/cube.json).
- **Modernization Risk & Strategy:** Drop. Replaced by Pydantic models.

#### Component: `ModelMetadataValidator.validate`
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Runs validation over the model, cubes, and dimensions, aggregating errors.
- **Data Flow & Logic Flow:** Calls [validate_model](file:///Users/melchior/Developer/cubes/cubes/metadata/defaults.py#28-33), then loops over `cubes` calling [validate_cube](file:///Users/melchior/Developer/cubes/cubes/metadata/defaults.py#95-100), and `dimensions` calling [validate_dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/defaults.py#101-125). Concatenates all returned error lists.
- **Validation vs. Domain Logic:** Pure validation routing.
- **Failure Handling:** Returns list of errors rather than raising exceptions directly.
- **Edge Cases:** Gracefully handles missing `cubes` or `dimensions` keys.
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** Drop. Replaced by root Pydantic [Model](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#27-84) which will recursively validate itself and its properties.

#### Component: `ModelMetadataValidator._collect_errors`
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Converts `jsonschema` yielded errors into `ValidationError` namedtuples.
- **Data Flow & Logic Flow:** Iterates `validator.iter_errors`. Reconstructs object path. Appends to list.
- **Validation vs. Domain Logic:** Adapting validation framework outputs.
- **Failure Handling:** None.
- **Edge Cases:** Handles empty paths for root errors.
- **External Dependencies:** `ValidationError` namedtuple.
- **Modernization Risk & Strategy:** Drop. Pydantic's `ValidationError` provides identical/better functionality out of the box with `e.errors()`.

#### Component: `ModelMetadataValidator.validate_model`
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Validates root model JSON against `model_schema`, plus custom logic to warn about string-only dimensions. 
- **Data Flow & Logic Flow:** Draft 4 validator checks schema. Custom iteration over `dimensions` checks if any item is a string, generating a "warning/default" message that it will create a flat single-attribute dimension.
- **Validation vs. Domain Logic:** Mixed schema validation and domain defaulting rules. 
- **Failure Handling:** Emits errors.
- **Edge Cases:** String dimensions are supported as shorthand! This is a domain invariant we MUST preserve.
- **External Dependencies:** `jsonschema`.
- **Modernization Risk & Strategy:** Pydantic will handle schema validation. For the string shorthand, we will write a Pydantic `BeforeValidator` or custom root validator on the `dimensions` list to coerce strings into flat single-attribute dimension model definitions before full validation.

#### Component: `ModelMetadataValidator.validate_cube`
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Runs `jsonschema` against the cube definition.
- **Data Flow & Logic Flow:** Extracts name and runs schema check.
- **Validation vs. Domain Logic:** Schema validation.
- **Failure Handling:** None.
- **Edge Cases:** None.
- **External Dependencies:** `jsonschema`.
- **Modernization Risk & Strategy:** Drop. Replaced by Pydantic.

#### Component: `ModelMetadataValidator.validate_dimension`
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Validates against schema, plus custom domain constraints: must not have BOTH levels and attributes at the same time. If missing both, assumes flat dimension without details. If missing default hierarchy name, outputs a warning/default info.
- **Data Flow & Logic Flow:** Draft 4 validation -> check if `default_hierarchy_name` missing -> check mutually exclusive `levels` and `attributes` -> check missing both.
- **Validation vs. Domain Logic:** Heavily embedded domain validation.
- **Failure Handling:** Yields specific error messages.
- **Edge Cases:** 
    - No default hierarchy name -> warning (defaults to first one eventually).
    - No levels and no attributes -> warning (flat dimension).
    - BOTH levels and attributes -> ERROR (mutually exclusive).
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** Pydantic models will capture these custom constraints using `@model_validator(mode='after')` on the Dimension Pydantic model to enforce the mutually exclusive `levels` vs `attributes` rule. Defaulting logic can also be captured there or in the domain conversion layer.
