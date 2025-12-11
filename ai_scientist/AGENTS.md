# Agent Stages and Entry Points

The `ai_scientist` root directory contains the primary entry points and stage-specific scripts for the agent workflow. These scripts are typically invoked by the main orchestrator (`agents_orchestrator.py` or `perform_experiments.py`) but can also be run standalone for debugging or specific tasks.

## Core Stages

### 1. Ideation
- **`perform_ideation_temp_free.py`**: The initial creativity engine. It takes a workshop description and generates novel research ideas.
  - **Inputs**: Text description of the research area.
  - **Outputs**: `ideas/<idea_name>.json`.
  - **Tools**: Accesses literature via `SearchSemanticScholar` and `AssembleLitData` to ground ideas in reality.

### 2. Literature Assembly
- **`perform_lit_data_assembly.py`**: The "Archivist" engine. It constructs the knowledge base for a run.
  - **Key Features**:
    - **Semantic Scholar Integration**: Fetches paper metadata, abstracts, and citations.
    - **Deduplication**: Merges results from multiple queries and seeds.
    - **Idempotency**: Caches queries in `lit_search_history.json` and loads existing `lit_summary.json` to safely resume interrupted runs.
    - **Exclusion**: Supports `lit_exclusions.json` and runtime `excluded_ids` to block specific papers (e.g., pre-prints).
  - **Outputs**: `experiment_results/lit_summary.json` (and .csv).

### 3. Experiments & Modeling
- **`perform_biological_modeling.py`**: Runs core biological simulations (e.g., compartmental models).
- **`perform_biological_stats.py`**: Performs statistical analysis (enrichment, p-value adjustment).
- **`perform_plotting.py`**: Aggregates results into figures. This is often an "LLM-assisted" step where the agent writes code to call this script.

### 4. Interpretation & Writeup
- **`perform_biological_interpretation.py`**: For theoretical runs, synthesizes findings into an `interpretation.md` document.
- **`perform_writeup.py`**: Compiles the final manuscript.
  - Generates LaTeX from templates (`blank_theoretical_biology_latex`, etc.).
  - Renders PDF using `pdflatex`.
  - Injects release metadata if a release tag is provided.

### 5. Review
- **`perform_llm_review.py`**: Auto-reviewer using LLMs to critique the PDF/text.
- **`perform_vlm_review.py`**: Vision-Language Model reviewer that can "see" figures in the PDF.

## Maintenance Scripts
- **`perform_repair_sim_outputs.py`**: A utility to fix/upgrade simulation artifacts in existing runs without re-running the expensive simulation.
