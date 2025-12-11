# Tools Library

This directory (`ai_scientist/tools`) contains the **Raw Capability Implementations**. These are standard Python classes/functions that perform the heavy lifting. They are typically wrapped by `tool_wrappers.py` before being exposed to agents, but can also be used directly by scripts.

## Literature & Search
- **`semantic_scholar.py`**:
  - `SemanticScholarSearchTool`: Keyword search.
  - `SemanticScholarRecommendationsTool`: Graph-based recommendations (`get_lit_recommendations`).
- **`lit_data_assembly.py`**:
  - `LitDataAssemblyTool`: The workhorse for fetching, deduplicating, and excluding papers. Handles `lit_summary.json` assembly.
- **`reference_verification.py`**: Verifies if citations in a text actually exist in the known literature.
- **`manuscript_reader.py`**: PDF parsing and text extraction.

## Modeling & Simulation
- **`compartmental_sim.py`**: Solves differential equations for compartmental models (transport, autophagy, etc.).
- **`graph_builder.py`**: Generates network topologies (biophysically realistic graphs) for simulations.
- **`sensitivity_sweep.py`**: Runs parameter sweeps (e.g., varying transport rates) to find failure thresholds.
- **`biological_model.py`**: Simple analytical biological models.

## Analysis & Validation
- **`biological_stats.py`**: Statistical routines (hypergeometric enrichment, FDR correction).
- **`validation_compare.py`**: Compares simulation outputs against literature reference values.
- **`claim_graph.py`** & **`claim_graph_checker.py`**: Manages the "Claim Graph" - a structured representation of scientific arguments and their evidence status.

## core Infrastructure
- **`base_tool.py`**: The abstract base class (`BaseTool`) that all tools inherit from. Enforces standard behavior for parameter definition and execution.
