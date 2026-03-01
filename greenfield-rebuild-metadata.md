We are executing a Greenfield Rebuild (Rip and Replace) of a legacy application into modern Python.

- **What we are discarding:** The old architecture, legacy internal APIs, and outdated design patterns. There are zero backward compatibility constraints for the codebase.

- **What we are preserving:** The application's original scope, external specifications, business value, and all domain invariants. The externally observable correctness must remain perfectly intact.

- Config Parsing & Validation: Pydantic v2 (strictly for parsing and validating inputs, not for the rich domain invariants).
    - cube.json
    - model.json
    - dimension.json
    

Plan for rewriting the `metadata` module first. This is the high level plan.

1. Parsing and Validation with Pydantic:
– Use Pydantic models to define the schema for the input metadata.
– Implement validation rules within Pydantic models to ensure data correctness.
– Parse raw input data through these Pydantic models to get validated objects.

2. Combining with Domain Models:
– Define standard Python classes or data classes to represent domain concepts like cubes, dimensions, measures, etc.
– Encapsulate complex invariants and business logic within these domain classes.
– After validation with Pydantic, convert or map these validated models into the domain model instances.

3. Integration Approach:
– Keep Pydantic models focused on schema validation and parsing.
– Keep domain models focused on business logic and operations.
– Establish clear conversion or factory methods to transform validated Pydantic objects into domain models.


- The new project will adopt a standard modern Python structure (src/cubes/).

**CRITICAL INSTRUCTION:** NO OVERSIMPLIFICATION. DO NOT SUMMARIZE. Do not give me high-level architectural fluff. We are doing a file-by-file, class-by-class, method-by-method audit and rewrite. If a file has 10 methods, you will analyze all 10. 

# REWRITE CONSTRAINTS
1. **Preserve Public APIs:** You must preserve the exact public methods and their signatures unless doing so makes zero sense because it relies on dead legacy Python paradigms.
2. **Preserve Domain Invariants:** The core operational logic and math must remain perfectly intact.
3. **Modernize the Guts:** Rewrite the internals using modern Python (type hints, dataclasses, standard modern libraries).
4. **Decouple:** Identify where validation is heavily convoluted with domain logic and separate them in the new design.

# THE PROCESSING LOOP

For EVERY SINGLE class and EVERY SINGLE method/function within that code, you MUST create a a .md file documenting the following exact Markdown schema before writing any new code. Do not skip any functions.

### TEMPLATE TO FILL OUT FOR EACH CLASS/METHOD:

#### Component: `[Class.method_name]`
- **Public API Status:** [Keep exactly as is / Modify due to legacy framework / Drop]
- **Core Domain Logic & Invariants:** [What is the actual business value/math happening here?]
- **Data Flow & Logic Flow:** [Where does the data come from, how is it mutated step-by-step, and what is returned?]
- **Validation vs. Domain Logic:** [How is validation currently handled? Is it tangled with the business logic? How will we separate it?]
- **Failure Handling:** [What exceptions are caught or thrown? Are there silent fallbacks?]
- **Edge Cases:** [What obscure conditions, null checks, or weird legacy behaviors are hiding in this method?]
- **External Dependencies:** [Does this touch the filesystem, legacy DB drivers, global state, or other modules?]
- **Modernization Risk & Strategy:** [What is the risk of breaking this during migration? How exactly will we rewrite this using modern Python concepts?]