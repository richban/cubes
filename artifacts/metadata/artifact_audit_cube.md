# Audit: [cube.py](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py)

#### Component: `Cube.from_metadata`
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Factory method that expands metadata block into Cube, linking implicit aggregates, propagating `nonadditive` properties to measures, generating default aggregate labels, handling implicit default fact count.
- **Data Flow & Logic Flow:** Validates name, calls [expand_cube_metadata](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#781-827). Extracts [measures](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#273-276), initializes [Attribute](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#204-290)/[Measure](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#292-404) classes. If no measures/aggregates, inserts default `fact_count`. Implicitly defaults missing labels via [_measure_aggregate_label](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#743-760).
- **Validation vs. Domain Logic:** Mix of coercion, factory defaults, domain propagation (nonadditive down to children, implicit grouping).
- **Failure Handling:** Raises `ModelError` if missing [name](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#799-802) or if implicit aggregate conflicts with explicit.
- **Edge Cases:** Implicit sum/count aggregates based on measure generation. `DEFAULT_FACT_COUNT_AGGREGATE`.
- **External Dependencies:** [expand_cube_metadata](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#781-827), [create_list_of](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#502-506).
- **Modernization Risk & Strategy:** This is a heavy factory method with profound domain invariants (implicit default count, propagation of non-additive nature). Move into Pydantic `@model_validator(mode='after')` on the root Cube schema model or a dedicated domain conversion factory layer.

#### Component: `Cube.__init__`
- **Public API Status:** Modify / convert to Dataclass.
- **Core Domain Logic & Invariants:** Resolves dimension links ([dimension_links](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#762-779)). Validates homogeneous collections (all measures must be Measure instances). Checks for [store](file:///Users/melchior/Developer/cubes/cubes/metadata/providers.py#242-246). Initializes blank OrderedDict tracking for dimensions.
- **Data Flow & Logic Flow:** Assigns attributes. Checks constraint: cannot provide *both* [dimensions](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#548-551) and [dimension_links](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#762-779) kwargs. Expands [dimension_links](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#762-779). Uses [object_dict](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#86-107) to enforce unique constraints on measure arrays.
- **Validation vs. Domain Logic:** Domain structure initialization + unique array validations.
- **Failure Handling:** `ModelError` if both [dimensions](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#548-551) and [dimension_links](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#762-779) provided. `ModelError` wrapper around [object_dict](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#86-107) for duplicate names.
- **Edge Cases:** Store string vs object instantiation tracking (`store_name`).
- **External Dependencies:** [object_dict](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#86-107), [expand_dimension_links](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#762-779).
- **Modernization Risk & Strategy:** Pydantic will handle the unique assertions in list validation arrays. Dataclass `__post_init__` handles the indexing.

#### Component: `Cube.measures`, [measure](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#277-293), [get_measures](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#294-304)
- **Public API Status:** Keep semantics.
- **Core Domain Logic & Invariants:** Fetchers for measures.
- **Modernization Risk & Strategy:** Normal properties/dict lookups.

#### Component: `Cube.aggregates`, [aggregate](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#309-324), [get_aggregates](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#325-331), [aggregates_for_measure](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#332-338)
- **Public API Status:** Keep semantics.
- **Core Domain Logic & Invariants:** Filtering and lookups for aggregates. [aggregates_for_measure](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#332-338) specifically filters out derived expressions.
- **Modernization Risk & Strategy:** Normal properties/dict lookups.

#### Component: `Cube.all_dimension_keys`
- **Public API Status:** Keep.
- **Core Domain Logic & Invariants:** Flattens key attributes across all dimensions.

#### Component: `Cube.all_attributes`, [base_attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#376-390), [all_fact_attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#391-407), [attribute_dependencies](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#408-421), [all_aggregate_attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#422-434)
- **Public API Status:** Keep / Migrate.
- **Core Domain Logic & Invariants:** Categorizes attributes based on what stage of the query pipeline they belong to (base vs derived, fact vs aggregate).
- **Data Flow & Logic Flow:** Concatenates arrays ([details](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#1193-1199) + [measures](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#273-276) + [dimensions](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#548-551) + [aggregates](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#305-308)), sometimes checking `attr.is_base`. 
- **Validation vs. Domain Logic:** Domain structure traversing.
- **Failure Handling:** None.
- **External Dependencies:** Checks `attr.is_base` and `attr.dependencies`.
- **Modernization Risk & Strategy:** Direct port. Highly critical for query generation logic later.

#### Component: `Cube.attribute` (and [get_attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#461-499))
- **Public API Status:** Keep.
- **Core Domain Logic & Invariants:** Looks up ANY attribute regardless of scope. First tries dimension references (`dim.attribute`), then [details](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#1193-1199), then [measures](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#273-276).
- **Failure Handling:** Raises `NoSuchAttributeError`.
- **Modernization Risk & Strategy:** Needs exact porting of lookup order.

#### Component: `Cube.collect_dependencies`
- **Public API Status:** Keep.
- **Core Domain Logic & Invariants:** Maps attributes down to their topologically sorted base dependencies in the cube.
- **External Dependencies:** [collect_dependencies](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#527-550) tool function.
- **Modernization Risk & Strategy:** Direct port.

#### Component: `Cube.link_dimension`, [_add_dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#533-547), [dimensions](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#548-551), [dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#245-248)
- **Public API Status:** Keep.
- **Core Domain Logic & Invariants:** Resolves and attaches dimensions to the cube context. Uses [clone](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#473-560) on the dimension if there are link overrides.
- **Failure Handling:** `NoSuchDimensionError`. `ArgumentError` if linking `None`.
- **Modernization Risk & Strategy:** Move to a domain linker service / method.

#### Component: `Cube.distilled_hierarchies`
- **Public API Status:** Experimental (Keep).
- **Core Domain Logic & Invariants:** Returns a flat mapping of [(dimension_name, hierarchy_name)](file:///Users/melchior/Developer/cubes/cubes/metadata/localization.py#68-88) to a list of level keys.
- **Modernization Risk & Strategy:** Easy port.

#### Component: `Cube.to_dict`
- **Public API Status:** Modify (Standardize).
- **Core Domain Logic & Invariants:** Serialization of Cube with specific `expand_dimensions` option and `hierarchy_limits`.
- **Modernization Risk & Strategy:** Use `model_dump()`. The custom limitation logic can be a helper.

#### Component: `Cube.__eq__`
- **Public API Status:** Keep semantics.

#### Component: `Cube.validate`
- **Public API Status:** Drop (Replace with Pydantic).
- **Core Domain Logic & Invariants:** Cross checks [measures](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#273-276) vs [details](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#1193-1199). Validates types.
- **Modernization Risk & Strategy:** Use Pydantic array validations or `@model_validator`.

#### Component: `Cube.localize`, [localizable_dictionary](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#700-717)
- **Public API Status:** Keep / Modify.

#### Component: [_measure_aggregate_label](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#743-760)
- **Public API Status:** Keep.
- **Core Domain Logic & Invariants:** Lookups default formatting strings (e.g. `Sum of {measure}`).

#### Component: [expand_dimension_links](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#762-779)
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Turns lists of strings into dicts.
- **Modernization Risk & Strategy:** Pydantic `types.Union[str, list]`.

#### Component: [expand_cube_metadata](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#781-827)
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Pulls up nested metadata, merges links, overrides nonadditive arrays.
- **Modernization Risk & Strategy:** Handled entirely by Pydantic modeling preprocessing.
