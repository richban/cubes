# Audit: [dimension.py](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py)

#### Component: `Conceptual.levels`, [is_flat](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#450-457)
- **Public API Status:** Keep semantics.
- **Core Domain Logic & Invariants:** Abstract class holding NotImplemented stubs. 
- **Modernization Risk & Strategy:** Standard Abstract Base Class (ABC) or Python protocol.

#### Component: `Dimension.from_metadata`
- **Public API Status:** Drop (Convert to pydantic validator).
- **Core Domain Logic & Invariants:** Complex logic to merge dimension with a "template". If `template` specified, it deepcopies the template's levels and hierarchies, overrides their properties with local ones, and restricts hierarchies only to those using local levels. Also handles implicit default hierarchy creation if missing.
- **Data Flow & Logic Flow:** Fetches template -> copies levels -> reconstructs hierarchies matching local levels -> calls [expand_dimension_metadata](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#1238-1292) for cleanup -> validates local level names against template -> filters hierarchies.
- **Validation vs. Domain Logic:** Very dense domain structure mapping and template inheritance logic.
- **Failure Handling:** Raises `TemplateRequired` or `ModelError`.
- **Edge Cases:** `"high_cardinality"` back-compat boolean mapping. Filtering unused template levels out.
- **External Dependencies:** [expand_dimension_metadata](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#1238-1292), [_create_hierarchies](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#719-743).
- **Modernization Risk & Strategy:** HIGH. Template inheritance is tricky. This logic must either live in a `ModelStore.resolve_dimension_template` method applied to raw Dicts *before* Pydantic parsing, or in a Pydantic `BeforeValidator` / Factory class layer. It mixes resolution and parsing.

#### Component: `Dimension.__init__`
- **Public API Status:** Modify / convert to Dataclass.
- **Core Domain Logic & Invariants:** Owns the levels and assigns default roles (time dimension default levels). Resolves non-additive types. Converts isolated [attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#465-472) array into a single implicit level if no levels provided. Collects all attributes down tree into [_attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#939-943) map and `_attributes_by_ref` map. Sets up default hierarchy.
- **Data Flow & Logic Flow:** Error on missing levels + attributes, or both present. Loops over default roles setting them. Loops over levels and attributes asserting dimension ownership. Validates duplicate hierarchies. 
- **Validation vs. Domain Logic:** Domain indexing, constraint checks, and defaulting.
- **Failure Handling:** Raises `ModelError` explicitly.
- **Edge Cases:** Single implicit level logic ([attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#465-472) instead of [levels](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#39-43)). Ownership validation.
- **External Dependencies:** `DEFAULT_LEVEL_ROLES`.
- **Modernization Risk & Strategy:** The post-initialization index building (`_attributes_by_ref`) maps cleanly to a `__post_init__` block in a Dataclass or `computed_field` in Pydantic. The defaulting logic ([attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#465-472) -> [Level](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#982-1211)) should be a Pydantic `model_validator`.

#### Component: `Dimension.__eq__`, [__ne__](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#820-822), [__str__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#114-116), [__repr__](file:///Users/melchior/Developer/cubes/cubes/metadata/attributes.py#117-119)
- **Public API Status:** Keep semantics.

#### Component: `Dimension.has_details`, [levels](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#39-43), [hierarchies](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#391-396), [level_names](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#799-802), [level](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#403-416), [hierarchy](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#417-433), [attribute](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#434-448), [is_flat](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#450-457), [key_attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#939-943)
- **Public API Status:** Keep.
- **Core Domain Logic & Invariants:** Mostly getters. Some logic to traverse `master` if constrained. [has_details](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#1193-1199) checks all levels for >1 attribute.
- **Modernization Risk & Strategy:** Easy direct port as property getters on the dataclass.

#### Component: `Dimension.clone`
- **Public API Status:** Keep.
- **Core Domain Logic & Invariants:** Clones dimension while selectively limiting to certain [hierarchies](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#391-396) or excluding hierarchies, while unowning all copied level attributes from the implicit new dimension so it can be reowned.
- **Data Flow & Logic Flow:** Finds specified hierarchies. Loops over used levels, un-owning attributes for safety. Resolves default hierarchy index. Returns new Dimension.
- **Validation vs. Domain Logic:** Core mutation behavior for drilldowns/links.
- **Failure Handling:** `ModelInconsistencyError` if cloning implies removing all hierarchies.
- **Modernization Risk & Strategy:** Will need care if translating to immutable dataclasses (using `dataclasses.replace` but careful with deep structure pointers).

#### Component: `Dimension.to_dict`
- **Public API Status:** Modify.
- **Core Domain Logic & Invariants:** Serializes dimension, applying `hierarchy_limits`.
- **Modernization Risk & Strategy:** Replaced by `model_dump()`.

#### Component: `Dimension.validate`
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Strict validations on level limits, missing default hierarchies, empty attributes, missing keys, duplicate references across levels, ownership mismatches.
- **Modernization Risk & Strategy:** Convert all of these checks into Pydantic `@model_validator` methods to ensure bad objects cannot be instantiated.

#### Component: [_create_hierarchies](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#719-743)
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Utility to map string references into Hierarchy classes.

#### Component: `Hierarchy.__init__`
- **Public API Status:** Modify (Dataclass).
- **Core Domain Logic & Invariants:** Maps an ordered list of levels.
- **Failure Handling:** Raises `ModelInconsistencyError` on empty levels.

#### Component: `Hierarchy.keys`, [__len__](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#829-831), [__getitem__](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#832-839), [__contains__](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#840-844), [levels_for_path](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#845-851), [levels_for_depth](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#852-865), [next_level](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#866-879), [previous_level](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#880-892), [level_index](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#893-901), [is_last](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#902-906), [rollup](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#907-931), [path_is_base](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#932-938), [key_attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#939-943), [all_attributes](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#944-953)
- **Public API Status:** Keep.
- **Core Domain Logic & Invariants:** Array mathematics / indexing on drilldown paths. Handles rolling up depths ([rollup](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#907-931)), verifying if a dimension path is a "bottom" leaf path ([path_is_base](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#932-938)).
- **Data Flow & Logic Flow:** Slices arrays safely or raises `HierarchyError`.
- **Validation vs. Domain Logic:** Business queries / traversal logic.
- **Modernization Risk & Strategy:** Critical traversal code, must be ported identically as domain instance methods.

#### Component: `Level.from_metadata`
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Inflates levels dict.
- **Modernization Risk & Strategy:** Pydantic validation handles this natively.

#### Component: `Level.__init__` 
- **Public API Status:** Modify (Dataclass).
- **Core Domain Logic & Invariants:** Creates a level. Maps [key](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#803-812), `label_attribute`, and `order_attribute` if none provided, using defaults (first attribute = key, second = label, first = order).
- **Failure Handling:** `ModelError` on empty attributes. `NoSuchAttributeError` on unknown override. 
- **Modernization Risk & Strategy:** Use dataclasses `__post_init__` for the defaulting logic.

#### Component: `Level.has_details`
- **Public API Status:** Keep.
- **Core Domain Logic & Invariants:** [len(attributes) > 1](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#829-831).

#### Component: [string_to_dimension_level](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#1213-1236)
- **Public API Status:** Keep.
- **Core Domain Logic & Invariants:** Parses a drilldown string like `dimension@hierarchy:level` into a tuple.
- **Failure Handling:** `ArgumentError` on pattern failure.
- **Modernization Risk & Strategy:** Very simple Regex port.

#### Component: [expand_dimension_metadata](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#1238-1292), [expand_hierarchy_metadata](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#1294-1307), [expand_level_metadata](file:///Users/melchior/Developer/cubes/cubes/metadata/dimension.py#1308-1343)
- **Public API Status:** Drop.
- **Core Domain Logic & Invariants:** Converts shorthand string syntax (e.g. `levels="time"`) into detailed dicts (`{"name": "time", "attributes": [{"name": "time"}]}`). 
- **Modernization Risk & Strategy:** High value, must be integrated into Pydantic `@field_validator(mode='before')` blocks.
