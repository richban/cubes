# Audit: [base.py](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py)

#### Component: `ModelObject.__init__`
- **Public API Status:** Keep as is.
- **Core Domain Logic & Invariants:** Initializes basic properties (`name`, `label`, `description`, `info`) for all model objects. Ensures `info` is at least an empty dictionary.
- **Data Flow & Logic Flow:** Takes raw arguments and assigns them to instance variables. Defaults `info` to `{}` if not provided.
- **Validation vs. Domain Logic:** No explicit validation, just defaulting. In the rewrite, Pydantic should handle the input validation (e.g., ensuring `name` is a string) while the domain class [ModelObject](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#27-84) becomes a dataclass.
- **Failure Handling:** None.
- **Edge Cases:** None, other than handling `info=None`.
- **External Dependencies:** None.
- **Modernization Risk & Strategy:** Low risk. Replace with a standard Python `@dataclass`. Pydantic models will parse the raw JSON to instantiate these dataclasses.

#### Component: `ModelObject.to_dict`
- **Public API Status:** Keep as is (dict serialization is useful).
- **Core Domain Logic & Invariants:** Converts the object back into a dictionary representation, optionally creating a label from the name if none exists.
- **Data Flow & Logic Flow:** Instantiates an `IgnoringDictionary` (from `cubes.common`), populates `name`, `info`, `label`, and `description`.
- **Validation vs. Domain Logic:** Strictly serialization. 
- **Failure Handling:** None.
- **Edge Cases:** If `create_label` is true and `label` is missing, it calls `to_label(name)`.
- **External Dependencies:** `IgnoringDictionary` and `to_label`.
- **Modernization Risk & Strategy:** Medium risk because dict representation might be expected to omit certain falsy values implicitly. Use `dataclasses.asdict` or Pydantic's `model_dump(exclude_unset=True)` equivalents on the output side, dropping the custom `IgnoringDictionary` if possible or substituting with modern Python dict comprehensions.

#### Component: `ModelObject.localized`
- **Public API Status:** Keep as is or adapt to modern domain methods.
- **Core Domain Logic & Invariants:** Returns a translated copy of the object based on a provided `context`. Translates specific attributes and lists of objects.
- **Data Flow & Logic Flow:** Creates a shallow copy of the object using `__class__.__new__` and dictionary copying. Translates attributes listed in `localizable_attributes`. Iterates over lists in `localizable_lists`, translating each item recursively.
- **Validation vs. Domain Logic:** Domain logic (localization).
- **Failure Handling:** None.
- **Edge Cases:** Uses dark magic `__class__.__new__` and direct `__dict__` manipulation.
- **External Dependencies:** The `context` object (has `get` and `object_localization`).
- **Modernization Risk & Strategy:** High risk due to `__dict__` hacking. Rewrite using Python's `copy.replace` for dataclasses or custom [localized()](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#63-84) methods that reconstruct the object natively.

#### Component: [object_dict](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#86-107)
- **Public API Status:** Keep as is (utility function).
- **Core Domain Logic & Invariants:** Converts a list of model objects into an `OrderedDict`, keyed by either `obj.name` or `obj.ref`. Ensures keys are unique.
- **Data Flow & Logic Flow:** Iterates over objects, choosing `name` or `ref` as key. Raises `ModelError` on duplicates.
- **Validation vs. Domain Logic:** Primarily validation (uniqueness check) and data restructuring.
- **Failure Handling:** Raises `ModelError` if a duplicate key is found.
- **Edge Cases:** Allows custom error messages and formatting dictionaries.
- **External Dependencies:** `ModelError` from `cubes.errors`.
- **Modernization Risk & Strategy:** Low. Can use standard standard dictionaries in Python 3.7+ (since they are ordered). Unique constraint validation belongs in the Pydantic schema validation step for the lists themselves, turning this into a purely domain grouping utility.

#### Component: [_json_from_url](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#117-140)
- **Public API Status:** Drop / Modernize.
- **Core Domain Logic & Invariants:** Fetches and parses a JSON file from either a local file path or a remote URL.
- **Data Flow & Logic Flow:** Parses the URL scheme. If file, opens locally. If URL, uses `urlopen`. Parses JSON. Re-raises JSON parsing errors as `SyntaxError`.
- **Validation vs. Domain Logic:** I/O and parsing.
- **Failure Handling:** Catches `ValueError` from `json.load` and turns it into `SyntaxError`.
- **Edge Cases:** Windows drive letter hacks (`len(parts.scheme) == 1`).
- **External Dependencies:** `urllib`, `os`, [json](file:///Users/melchior/Developer/cubes/cubes/schemas/cube.json).
- **Modernization Risk & Strategy:** Low. Replace with `pathlib` for files and `httpx` or `urllib.request` for URLs, handling parsing strictly via Pydantic on the raw string/bytes.

#### Component: [read_model_metadata](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#142-159)
- **Public API Status:** Keep as is (entrypoint).
- **Core Domain Logic & Invariants:** Entrypoint for loading a logical model from a dict, file, URL, or directory bundle.
- **Data Flow & Logic Flow:** Checks if source is a string. If it's a directory, delegates to [read_model_metadata_bundle](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#161-226). If a file/URL, delegates to [_json_from_url](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#117-140). If it's already a dict, returns it.
- **Validation vs. Domain Logic:** Routing IO/decoding logic.
- **Failure Handling:** None directly.
- **Edge Cases:** Same Windows drive letter hack.
- **External Dependencies:** `os.path`.
- **Modernization Risk & Strategy:** Low. Will be rewritten to take a `Path` or `str` and route to the correct Pydantic loader.

#### Component: [read_model_metadata_bundle](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#161-226)
- **Public API Status:** Keep as is.
- **Core Domain Logic & Invariants:** Loads a multi-file model bundle (directory containing [model.json](file:///Users/melchior/Developer/cubes/cubes/schemas/model.json), `dim_*.json`, `cube_*.json`).
- **Data Flow & Logic Flow:** Loads [model.json](file:///Users/melchior/Developer/cubes/cubes/schemas/model.json). Walks directory tree finding [.json](file:///Users/melchior/Developer/cubes/cubes/schemas/cube.json) files. If `dim_` or `dimension_` prefix, asserts it has a name and adds to model's `dimensions` list. Overlap/duplicates throw `ModelError`. Same for `cube_`.
- **Validation vs. Domain Logic:** I/O and gross validation (duplicate names).
- **Failure Handling:** Raises `ArgumentError` if not a directory, `ModelError` if missing [model.json](file:///Users/melchior/Developer/cubes/cubes/schemas/model.json), missing `name` keys, or duplicate dimensions/cubes.
- **Edge Cases:** Recursively walks subdirectories (`os.walk`), which might be unintended but explicitly coded.
- **External Dependencies:** `os`, [re](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#142-159).
- **Modernization Risk & Strategy:** Medium. The parsing strategy should be ported directly to `pathlib`. The validation (duplicate names, missing names) will happen instantly when feeding the combined dictionary into Pydantic.

#### Component: [write_model_metadata_bundle](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#228-267)
- **Public API Status:** Keep as is.
- **Core Domain Logic & Invariants:** Writes out a loaded model metadata dict to a directory bundle.
- **Data Flow & Logic Flow:** Creates directory. Pops dimensions and cubes into separate files named `dim_{name}.json` and `cube_{name}.json`. Writes the rest to [model.json](file:///Users/melchior/Developer/cubes/cubes/schemas/model.json).
- **Validation vs. Domain Logic:** Pure I/O. Safe-guards against overwriting non-model directories.
- **Failure Handling:** Raises `CubesError` if target exists and isn't a directory, isn't a previously valid model directory, or `replace=False`.
- **Edge Cases:** Modifies a copy of the metadata dict via [dict(metadata)](file:///Users/melchior/Developer/cubes/cubes/metadata/base.py#42-62) and `.pop()`.
- **External Dependencies:** `os`, `shutil`, [json](file:///Users/melchior/Developer/cubes/cubes/schemas/cube.json).
- **Modernization Risk & Strategy:** Low. Port to `pathlib` and use domain models' `.model_dump()` to generate output.
