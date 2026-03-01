# Audit: [attributes.py](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py)

#### Component: [expand_attribute_metadata](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#29-36)
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Normalizes raw attribute definition.
- **Data Flow & Logic Flow:** If string, turns it into `{"name": string}` dict.
- **Validation vs. Domain Logic:** Input coercion/validation.
- **Failure Handling:** None.
- **Edge Cases:** None.
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** Handled entirely by Pydantic's `BeforeValidator`.

#### Component: `AttributeBase.from_metadata`
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Factory method to instantiate attributes from dict, string, or existing object.
- **Data Flow & Logic Flow:** String -> Instantiates with string as name. Object -> copies. Dict -> passes as kwargs.
- **Validation vs. Domain Logic:** Factory / coercion.
- **Failure Handling:** Raises `ModelError` if dict lacks "name".
- **Edge Cases:** None.
- **External Dependencies:** [copy](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#340-352).
- **Modernization Risk & Strategy:** Pydantic `model_validate` completely replaces this.

#### Component: `AttributeBase.__init__`
- **Public API Status:** Modify (Dataclass fields).
- **Core Domain Logic & Invariants:** Sets up base properties. Normalizes `order` to `asc` or `desc`. Initializes [ref](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#169-187) = `name`.
- **Data Flow & Logic Flow:** calls super(), assigns properties, validates/normalizes `order` field.
- **Validation vs. Domain Logic:** Mixture of domain and validation logic (`order`).
- **Failure Handling:** Raises `ArgumentError` on invalid order.
- **Edge Cases:** Capitalization handling for `order`.
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** Pydantic will validate `order` string (Literal). In domain object (dataclass), properties are just assigned.

#### Component: `AttributeBase.__eq__`, [__ne__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#133-135), [__hash__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#136-138)
- **Public API Status:** Keep semantics.
- **Core Domain Logic & Invariants:** Equality based on properties, hash based on [ref](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#169-187).
- **Data Flow & Logic Flow:** Standard comparisons.
- **Validation vs. Domain Logic:** Domain logic.
- **Failure Handling:** None.
- **Edge Cases:** None.
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** Dataclasses provide [__eq__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#270-276) for free. If custom hashing on [ref](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#169-187) is needed, just implement [__hash__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#136-138).

#### Component: `AttributeBase.to_dict`
- **Public API Status:** Modify (use standard mechanisms).
- **Core Domain Logic & Invariants:** Serializes to dict.
- **Dependencies:** `super().to_dict()`
- **Modernization Risk & Strategy:** Use `dataclasses.asdict`.

#### Component: `AttributeBase.localizable_dictionary`, [is_localizable](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#288-290), [localize](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#160-164), [localized_ref](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#169-187)
- **Public API Status:** Keep as is.
- **Core Domain Logic & Invariants:** Handles getting/setting localized format and references.
- **Data Flow & Logic Flow:** Fallbacks, localized [ref](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#169-187) string building by appending `.locale`.
- **Validation vs. Domain Logic:** Domain logic.
- **Failure Handling:** Raises `ArgumentError` if trying to format a non-localizable attribute or unknown locale.
- **Edge Cases:** None.
- **External Dependencies:** `get_localizable_attributes`.
- **Modernization Risk & Strategy:** Cleanly map to domain dataclass methods.

#### Component: `AttributeBase.dependencies` property
- **Public API Status:** Keep as is.
- **Core Domain Logic & Invariants:** Parses arithmetic expressions to find dependencies.
- **Data Flow & Logic Flow:** If no expression, empty set. Otherwise, calls `inspect_variables(self.expression)`.
- **Validation vs. Domain Logic:** Core domain logic.
- **Failure Handling:** None.
- **Edge Cases:** None.
- **External Dependencies:** `expressions.inspect_variables`.
- **Modernization Risk & Strategy:** Directly port this property.

#### Component: `Attribute.__init__`, [__deepcopy__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#340-352), [__eq__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#270-276), [to_dict](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#280-287), [is_localizable](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#288-290)
- **Public API Status:** Modify.
- **Core Domain Logic & Invariants:** Main dimension attribute. Holds `locales`. Dis-owns dimension on copy.
- **Data Flow & Logic Flow:** Inherits base. Adds [dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#249-256) property management. [__deepcopy__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#340-352) explicitly sets `dimension=None` to detach it.
- **Validation vs. Domain Logic:** Domain structure management.
- **Failure Handling:** None.
- **Edge Cases:** Disowning on copy.
- **External Dependencies:** [copy](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#340-352).
- **Modernization Risk & Strategy:** Port logic. Be careful with parent references ([dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#249-256)) in Pydantic. Pydantic doesn't do parent references easily, but the domain class can.

#### Component: `Attribute.dimension` property / setter
- **Public API Status:** Keep as is.
- **Core Domain Logic & Invariants:** When [dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#249-256) owner is assigned, updates [ref](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#169-187) to `dimension.name + "." + name`.
- **Data Flow & Logic Flow:** String building [ref](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#169-187) on assignment.
- **Validation vs. Domain Logic:** Domain structural linkage.
- **Failure Handling:** None.
- **Edge Cases:** Fallback to name alone if [dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#249-256) is None.
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** Crucial invariant for [ref](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#169-187) rebuilding. Implement in domain dataclass property setter.

#### Component: `Measure.__init__`, [__deepcopy__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#340-352), [__eq__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#270-276), [to_dict](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#280-287)
- **Public API Status:** Modify.
- **Core Domain Logic & Invariants:** Adds properties: `formula`, [aggregates](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#372-404), `nonadditive`, `window_size`. Normalizes `nonadditive`.
- **Data Flow & Logic Flow:** Normalizes `none` to `None`, `"all"`/`"any"` to `"any"`, `"time"` to `"time"`.
- **Validation vs. Domain Logic:** Domain initialization + Enum-like validation.
- **Failure Handling:** Raises `ModelError` on unknown nonadditive type.
- **Edge Cases:** Normalizing nonadditive.
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** Port validation to Pydantic Enum/Literal.

#### Component: `Measure.default_aggregates`
- **Public API Status:** Keep as is.
- **Core Domain Logic & Invariants:** Auto-generates [MeasureAggregate](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#406-500) objects if model is lazy.
- **Data Flow & Logic Flow:** Defaults [aggregates](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#372-404) to `["sum"]`. For each, if it's `"identity"`, name = measure name, no function. Otherwise name = `{measure}_{agg}`, function = [agg](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#372-404).
- **Validation vs. Domain Logic:** Core domain logic (expansion).
- **Failure Handling:** None.
- **Edge Cases:** Magic `"identity"` aggregate logic.
- **External Dependencies:** [MeasureAggregate](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#406-500).
- **Modernization Risk & Strategy:** Keep in domain classes or run as a post-parse expansion pass mapping generated domains back into the main structure.

#### Component: `MeasureAggregate.__init__`, [__deepcopy__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#340-352), [__eq__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#270-276), [to_dict](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#280-287)
- **Public API Status:** Modify.
- **Core Domain Logic & Invariants:** Has `function`, `formula`, `measure`, `nonadditive`.
- **Data Flow & Logic Flow:** Standard assignment.
- **Validation vs. Domain Logic:** Domain definition.
- **Failure Handling:** None.
- **Edge Cases:** None.
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** Dataclasses.

#### Component: `MeasureAggregate.dependencies`
- **Public API Status:** Keep as is.
- **Core Domain Logic & Invariants:** Derived aggregates depend on `measure` OR `expression`. Cannot have both.
- **Data Flow & Logic Flow:** Checks for both -> error. If `measure` -> set containing just `measure`. If `expression` -> inspect expression variables.
- **Validation vs. Domain Logic:** Embedded Validation inside a property getter.
- **Failure Handling:** Raises `ModelError`.
- **Edge Cases:** Mutually exclusive `measure` / `expression`.
- **External Dependencies:** `inspect_variables`.
- **Modernization Risk & Strategy:** Move the mutual-exclusion check to Pydantic `@model_validator(mode='after')`. The dependency extraction stays in the domain property.

#### Component: [create_list_of](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#502-506)
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Batch applies `class_.from_metadata`.
- **Modernization Risk & Strategy:** Replaced by Pydantic lists.

#### Component: [collect_attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#508-525)
- **Public API Status:** Keep as utility.
- **Core Domain Logic & Invariants:** Flattens attributes from explicit lists and `container.all_attributes`.
- **Data Flow & Logic Flow:** Simple list concatenation.
- **Modernization Risk & Strategy:** Low risk port.

#### Component: [collect_dependencies](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#527-550)
- **Public API Status:** Keep as is.
- **Core Domain Logic & Invariants:** Gathers all attributes by ref and topologically sorts them using [depsort_attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#551-609).
- **Data Flow & Logic Flow:** Builds `{ref: deps}` map. Calls [depsort](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#551-609).
- **Modernization Risk & Strategy:** Important arithmetic core logic. Port directly.

#### Component: [depsort_attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#551-609)
- **Public API Status:** Keep as is.
- **Core Domain Logic & Invariants:** Topologically sorts attributes based on their [dependencies](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#480-500) map.
- **Data Flow & Logic Flow:** Classic graph traversal. Finds nodes with zero un-met dependencies. Errors on cycles.
- **Validation vs. Domain Logic:** Domain algorithm + graph cyclic validation.
- **Failure Handling:** Catches unknown variables, raises `ExpressionError`. Detects cycles, raises `ExpressionError(Circular attribute reference)`.
- **Edge Cases:** Fully resilient to complex DAGs.
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** Keep algorithmic implementation exactly as-is, just strongly type it.
