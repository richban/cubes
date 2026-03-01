# Audit: [providers.py](file:///Users/melchior/Developer/cubes/cubes/metadata/providers.py)

#### Component: [link_cube](file:///Users/melchior/Developer/cubes/cubes/metadata/providers.py#31-62)
- **Public API Status:** Modify / Keep.
- **Core Domain Logic & Invariants:** Mutates a completely constructed [Cube](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#46-741) instance by fetching [Dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#49-717) instances from the `provider` and linking them based on [dimension_links](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#762-779).
- **Data Flow & Logic Flow:** Iterates `cube.dimension_links.keys()`. Looks up dimension using [find_dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/providers.py#65-127). Calls `cube.link_dimension(dim)`. Keeps track of `linked` to avoid double-linking.
- **Validation vs. Domain Logic:** Domain structure resolution.
- **Failure Handling:** Raises `ModelError` on duplicate link. Raises `CubesError` if dimension isn't found and `ignore_missing` is False.
- **Edge Cases:** Resolving `TemplateRequired`.
- **External Dependencies:** [find_dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/providers.py#65-127).
- **Modernization Risk & Strategy:** High. Pydantic validation handles linking far better. Rather than an explicit mutation pass, the Pydantic root model Validator can fetch and attach instances.

#### Component: [find_dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/providers.py#65-127)
- **Public API Status:** Drop / refactor into loader registry.
- **Core Domain Logic & Invariants:** Looks up a dimension by name, managing a template resolution cycle where if a dimension needs a template, it recursively fetches the template dimension first.
- **Data Flow & Logic Flow:** While loop tracking `missing` dimensions to fetch. Tries [_lookup_dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/providers.py#130-154). Catches `TemplateRequired` and queues the template. Detects cycles in templates. Finally applies translation logic if `namespace` is provided.
- **Validation vs. Domain Logic:** Domain resolution logic with cycle validation.
- **Failure Handling:** Raises `BackendError` if a provider ignores a passed template. Raises `ModelError` on template cycle (`"Dimension templates cycle"`).
- **Edge Cases:** Cycle detection during resolution. Localization hack at the end using `# TODO: pass lookup instead of just first found translation`.
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** Pydantic models can perform template resolution in `@model_validator(mode='before')` or be handled by a distinct domain factory `Loader` class that caches all parsed definitions.

#### Component: [_lookup_dimension](file:///Users/melchior/Developer/cubes/cubes/metadata/providers.py#130-154)
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Tries fetching from provider, then from namespace.
- **Modernization Risk & Strategy:** Drop in favor of unified loader logic.

#### Component: `ModelProvider.__init__`
- **Public API Status:** Drop / modernize into base `Loader` or `Repository`.
- **Core Domain Logic & Invariants:** Base class for resolving models. Initialized with raw metadata dictionary. Pre-pends [default_metadata()](file:///Users/melchior/Developer/cubes/cubes/metadata/providers.py#224-240). Generates structural dictionaries.
- **Data Flow & Logic Flow:** Merges dictionaries. Iterates nested [dimensions](file:///Users/melchior/Developer/cubes/cubes/metadata/cube.py#548-551) and [cubes](file:///Users/melchior/Developer/cubes/cubes/metadata/providers.py#376-386) arrays to build `dimensions_metadata` and `cubes_metadata` maps.
- **Validation vs. Domain Logic:** Data loading & indexing.
- **Failure Handling:** None.
- **Edge Cases:** Options merging is ugly ([options](file:///Users/melchior/Developer/cubes/cubes/metadata/providers.py#261-275) vs `browser_options`).
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** In the new architecture, the [ModelProvider](file:///Users/melchior/Developer/cubes/cubes/metadata/providers.py#156-429) concept maps cleanly to a `MetadataRepository` or `Store` that parses the JSON into Pydantic models. We will drop this exact parent class structure.

#### Component: `ModelProvider._merge_metadata`
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Hand-written deep dictionary merge for cubes, dimensions, joins, mappings.
- **Modernization Risk & Strategy:** Unnecessary if we use standard deeply merged structures or Pydantic validation that allows appending.

#### Component: `ModelProvider.cube_metadata`
- **Public API Status:** Drop / Refactor.
- **Core Domain Logic & Invariants:** Fetches raw metadata dictionary for a cube, then merges it with global options. Merges `browser_options`, `mappings`, and `joins`. Validates that all joins have `master` and [detail](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#364-373).
- **Data Flow & Logic Flow:** Complex nested dictionary merge logic.
- **Validation vs. Domain Logic:** Mixture of configuration overriding and schema validation (join keys).
- **Failure Handling:** Raises `NoSuchCubeError`, `ModelError` on missing keys.
- **Edge Cases:** Logic for overriding global joins by name in a loop.
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** High risk of subtle differences in deep-merging. We must write a strict unit test for join overrides during migration or explicitly translate this logic to standard Python dict updates or Pydantic `model_copy(update=...)`.

#### Component: `StaticModelProvider.list_cubes`
- **Public API Status:** Keep (adapt to static repo).
- **Core Domain Logic & Invariants:** Returns summarized info dictionaries of all cubes.
- **Data Flow & Logic Flow:** Iterates `metadata.get('cubes')`. 
- **Modernization Risk & Strategy:** Return actual `CubeSummary` dataclasses or Pydantic models.
