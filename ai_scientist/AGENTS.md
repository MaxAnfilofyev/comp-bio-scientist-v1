# Agent Stages and Entry Points

The `ai_scientist` root directory contains the primary entry points and stage-specific scripts for the agent workflow. These scripts are typically invoked by the main orchestrator (`agents_orchestrator.py` or `perform_experiments.py`) but can also be run standalone for debugging or specific tasks.

## Core Stages

### 1. Ideation
- **`generate_ideas.py`** (was `perform_ideation_temp_free.py`): The initial creativity engine. It takes a workshop description and generates novel research ideas.
  - **Location**: `ai_scientist/tools/generate_ideas.py`
  - **Inputs**: Text description of the research area.
  - **Outputs**: `ideas/<idea_name>.json`.
  - **Tools**: Accesses literature via `SearchSemanticScholar` and `AssembleLitData` to ground ideas in reality.

### 2. Literature Assembly
- **Literature Search**: The `Archivist` agent uses `assemble_lit_data` (using `ai_scientist.tools.lit_data_assembly`) to search Semantic Scholar and aggregate results.
  - **Key Features**:
    - **Semantic Scholar Integration**: Fetches paper metadata, abstracts, and citations.
    - **Deduplication**: Merges results from multiple queries and seeds.
    - **Idempotency**: Caches queries in `lit_search_history.json` and loads existing `lit_summary.json` to safely resume interrupted runs.
    - **Exclusion**: Supports `lit_exclusions.json` and runtime `excluded_ids` to block specific papers (e.g., pre-prints).
  - **Outputs**: `experiment_results/lit_summary.json` (and .csv).

### 3. Experiments & Modeling
- **Biological Modeling**: The `Modeler` agent uses `run_biological_model` (using `ai_scientist.tools.biological_model`) to run simulations.
- **Biological Stats**: The `Analyst` agent uses `run_biological_stats` (using `ai_scientist.tools.biological_stats`) for statistical analysis (enrichment, p-value adjustment).
- **Biological Plotting**: The `Analyst` agent uses `run_biological_plotting` (using `ai_scientist.tools.biological_plotting`) to create plots from simulation outputs.

### 4. Interpretation & Writeup
- **Biological Interpretation**: The `Interpreter` agent uses `interpret_biology` (wrapping `ai_scientist.tools.biological_interpretation`) to synthesize findings into `interpretation.json` and `interpretation.md`.
- **`writeup.py`** (was `perform_writeup.py`): Compiles the final manuscript.
  - **Location**: `ai_scientist/tools/writeup.py`
  - Generates LaTeX from templates (`ai_scientist/templates/blank_theoretical_biology_latex`, etc.).
  - Renders PDF using `pdflatex`.
  - Injects release metadata if a release tag is provided.

### 5. Review
- **`llm_review.py`** (was `perform_llm_review.py`): Auto-reviewer using LLMs to critique the PDF/text.
  - **Location**: `ai_scientist/tools/llm_review.py`
- **`vlm_review.py`** (was `perform_vlm_review.py`): Vision-Language Model reviewer that can "see" figures in the PDF.
  - **Location**: `ai_scientist/tools/vlm_review.py`

## Maintenance Scripts
- **`repair_sim_outputs.py`** (was `perform_repair_sim_outputs.py`): A utility to fix/upgrade simulation artifacts in existing runs without re-running the expensive simulation.
  - **Location**: `ai_scientist/tools/repair_sim_outputs.py`
