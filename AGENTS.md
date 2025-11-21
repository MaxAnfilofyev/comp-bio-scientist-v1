# Agents and Tools Overview

This project orchestrates multiple agent flows (ideation → experiments → interpretation → writeup) with a small toolset exposed to LLMs. This document gives a practical map of what agents run, what tools they can call, and where outputs land.

## Agent Stages

1. **Ideation (`perform_ideation_temp_free.py`)**
   - Generates/refines ideas from a workshop description.
   - Tools available:
     - `SearchSemanticScholar`: Query Semantic Scholar for papers (optional `S2_API_KEY`).
     - `AssembleLitData`: Assemble literature data into `lit_summary.csv/json` (optional Semantic Scholar queries + local seeds).
   - Outputs: JSON idea file (Name, Title, Abstract, Short Hypothesis, Experiments, Risk Factors).

2. **Experiment search (`launch_scientist_bfts.py` + tree search)**
   - Drives code generation/debug loops to run the plan.
   - Configuration via `bfts_config.yaml` (per-run copy in `experiments/<ts>/.../bfts_config.yaml`).
   - Optional pre-step: `--run_lit_data_assembly` to create `experiment_results/lit_summary.csv/json`.
   - Outputs: `logs/0-run/experiment_results/` (code, .npy, plots), `journal.json`, `unified_tree_viz.html`.

3. **Plot aggregation (`perform_plotting.py`)**
   - Aggregates plots from experiment results (LLM-assisted).
   - Output directory: `figures/` under the experiment root.

4. **Interpretation (theoretical only) (`perform_biological_interpretation.py`)**
   - For `biology.research_type: theoretical`, synthesizes `interpretation.json/md`.

5. **Writeup (`perform_writeup.py` / `perform_icbinb_writeup.py`)**
   - Generates PDF(s) using LaTeX templates. Outputs to the experiment root.

6. **Review (`perform_llm_review.py`, `perform_vlm_review.py`)**
   - Optional LLM/VLM review of the generated PDF and figures.

## Tools

- **SearchSemanticScholar** (`ai_scientist/tools/semantic_scholar.py`)
  - Params: `query` (str), uses `S2_API_KEY` if set for higher limits.
- **AssembleLitData** (`ai_scientist/tools/lit_data_assembly.py`)
  - Params: `output_dir` (str, default `experiment_results`), `queries` (list[str]), `seed_paths` (list[str]), `max_results` (int), `use_semantic_scholar` (bool).
  - Uses `ai_scientist/perform_lit_data_assembly.py` under the hood.

## Environment and API Keys

- OpenAI: `OPENAI_API_KEY` (loaded from `.env` at project root if present).
- Semantic Scholar: `S2_API_KEY` (optional; improves throughput).
- Other providers (Anthropic, Bedrock, Gemini, etc.): set corresponding env vars as described in README.

## Outputs

- Per-run artifacts under `experiments/<timestamp>_<idea>_attempt_<id>/`:
  - `logs/0-run/experiment_results/` (code, arrays, intermediate plots).
  - `figures/` (aggregated plots).
  - `interpretation.json/md` (theoretical runs).
  - PDF(s) from writeup stage.
  - `journal.json`, `unified_tree_viz.html` (search trace).

## Typical CLI Invocation

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/my_idea.json \
  --research-type theoretical \
  --writeup-type theoretical_biology \
  --run_lit_data_assembly \
  --lit_seed_paths data/snc_vta_lit_seed.csv
```

Adjust `bfts_config.yaml` for search budget (num_workers, stage iters) and skip ML-centric steps for theoretical runs.***
