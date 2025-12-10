# Agents and Tools Overview

This project orchestrates multiple agent flows (ideation → experiments → interpretation → writeup) with a small toolset exposed to LLMs. This document gives a practical map of what agents run, what tools they can call, and where outputs land.

## Agent Stages

1. **Ideation (`perform_ideation_temp_free.py`)**
   - Generates/refines ideas from a workshop description.
   - Tools available:
     - `SearchSemanticScholar`: Query Semantic Scholar for papers (optional `S2_API_KEY`).
     - `AssembleLitData`: Assemble literature data into `lit_summary.csv/json` (optional Semantic Scholar queries + local seeds).
   - Outputs: JSON idea file (Name, Title, Abstract, Short Hypothesis, Experiments, Risk Factors).

2. **Tool-driven orchestration (`agents_orchestrator.py`)**
   - PI agent delegates to role agents (Archivist, Modeler, Analyst, Interpreter, Reviewer, Publisher, Coder) via tools/handoffs until the reviewer reports no gaps and a PDF exists.
   - Dynamic prompts include Title/Hypothesis/Abstract/Experiments/Risks and template alignment (default `blank_theoretical_biology_latex`). Output conventions enforced (artifacts -> `experiment_results/`, figures -> `figures/` when aggregated, PDFs at run root).
   - Each role returns structured status (status/artifacts/notes) and writes status files (e.g., `experiment_results/status_<role>.json` or appends to `tool_summary.txt`).
   - Resuming: pass `--resume` to pick the latest folder matching the idea name, or `--base_folder <experiments/...>` to restart from a specific existing run directory without creating a new timestamped folder.
   - Additional awareness: plot aggregation (`perform_plotting.py`), modeling/stats utils (`perform_biological_modeling.py`, `perform_biological_stats.py`), interpretation (`perform_biological_interpretation.py`), manuscript reading (`ai_scientist/tools/manuscript_reader.py`), and alternative templates (`blank_bioinformatics_latex`, `blank_icbinb_latex`).
   - Agent tool highlights:
     - Archivist: `AssembleLitData`, `ValidateLitSummary`, `SearchSemanticScholar`, `UpdateClaimGraph`.
     - Modeler: `BuildGraphs`, `RunBiologicalModel`, `RunCompartmentalSimulation`, `RunSensitivitySweep`, `RunInterventionTester`.
   - Analyst: `RunBiologicalPlotting`, `RunValidationCompare`, `RunBiologicalStats`.
   - Interpreter: `interpret_biological_results` wrapper for theoretical runs (`interpretation.json/md`).
   - Reviewer: `ReadManuscript`, `CheckClaimGraph`, `RunBiologicalStats`.
   - Repair: `repair_sim_outputs` to bulk run `sim_postprocess` on sim.json entries lacking exported arrays, validate per-compartment artifacts, and update manifest/tool_summary (lock-aware, idempotent). CLI: `python ai_scientist/perform_repair_sim_outputs.py --manifest <path>`.

  ## Modular Orchestrator Support Modules

  - `ai_scientist/orchestrator/tool_wrappers.py` now centralizes every `@function_tool` surface that role agents call. Each wrapper resolves canonical paths, gates operations via the dependence chain (lit/model/transport), pushes manifest entries, and exposes convenience helpers such as `inspect_manifest`, `reserve_typed_artifact`, `write_text_artifact`, and `append_run_note_tool`.
  - `ai_scientist/orchestrator/agents.py` builds the curated agent team used by `agents_orchestrator.py`. It pulls the wrapper tools, stitches the per-role instructions/prompts, and provides the PI agent with the same tooling surface plus `build_team()` and `extract_run_output()` helper methods so the orchestrator only manages flow and deployment.
  - Keep running `ruff check agents_orchestrator.py ai_scientist/orchestrator/tool_wrappers.py ai_scientist/orchestrator/agents.py` and `pyright agents_orchestrator.py ai_scientist/orchestrator/tool_wrappers.py ai_scientist/orchestrator/agents.py` after edits to keep the modularization healthy.

3. **Plot aggregation (`perform_plotting.py`)**
   - Aggregates plots from experiment results (LLM-assisted).
   - Output directory: `figures/` under the experiment root.

4. **Interpretation (theoretical only) (`perform_biological_interpretation.py`)**
   - For `biology.research_type: theoretical`, synthesizes `interpretation.json/md`.

5. **Writeup (`perform_writeup.py` / `perform_icbinb_writeup.py`)**
   - Generates PDF(s) using LaTeX templates. Outputs to the experiment root.

6. **Review (`perform_llm_review.py`, `perform_vlm_review.py`, Holistic Reviewer in agents orchestrator)**
   - Optional LLM/VLM review of the generated PDF and figures. Holistic Reviewer can read drafts (via manuscript reader) to flag gaps/citations/structure.

## Tools

- **SearchSemanticScholar** (`ai_scientist/tools/semantic_scholar.py`)
  - Params: `query` (str), uses `S2_API_KEY` if set for higher limits.
- **AssembleLitData** (`ai_scientist/tools/lit_data_assembly.py`)
  - Params: `output_dir` (str, default `experiment_results`), `queries` (list[str]), `seed_paths` (list[str]), `max_results` (int), `use_semantic_scholar` (bool). Uses `perform_lit_data_assembly.py`.
- **ValidateLitSummary** (`ai_scientist/tools/lit_validator.py`)
  - Params: `path` to lit_summary CSV/JSON; reports coverage of required fields.
- **BuildGraphs** (`ai_scientist/tools/graph_builder.py`)
  - Params: `n_nodes`, `output_dir`, `seed`; saves gpickle + adjacency.
- **RunCompartmentalSimulation** (`ai_scientist/tools/compartmental_sim.py`)
  - Params: `graph_path`, `output_dir`, `steps`, `dt`, `transport_rate`, `demand_scale`, `mitophagy_rate`, `noise_std`, `seed`. Supports `.gpickle`, `.graphml`, and `.gml` graph inputs. Emits standardized per-compartment artifacts (`per_compartment.npz` with `binary_states`/`continuous_states`/`time`, `node_index_map.json`, `topology_summary.json` with schema version/status/metrics/checksum). Mapping is canonical (`ordering`, `ordering_checksum`, bidirectional lookups); validator will fail on checksum/shape mismatches.
- **RunSensitivitySweep** (`ai_scientist/tools/sensitivity_sweep.py`)
  - Params: `graph_path` (file; supports .gpickle/.graphml/.gml/.npz/.npy), `output_dir`, `transport_vals`, `demand_vals`, `steps`, `dt`, `failure_threshold`. Each sweep point now also writes per-compartment artifacts with ordering checksum to its subfolder.
- **RunInterventionTester** (`ai_scientist/tools/intervention_tester.py`)
  - Params: `graph_path` (file; supports .gpickle/.graphml/.gml/.npz/.npy), `output_dir`, `transport_vals`, `demand_vals`, `baseline_transport`, `baseline_demand`, `failure_threshold`. Baseline and each intervention write per-compartment artifacts with ordering checksum to dedicated subfolders.
- **RunValidationCompare** (`ai_scientist/tools/validation_compare.py`)
  - Params: `lit_path`, `sim_path`; computes simple correlations.
- **RunBiologicalModel** (`ai_scientist/tools/biological_model.py`)
  - Params: `model_key`, `time_end`, `num_points`, `output_dir`; runs a built-in model and saves JSON.
- **RunBiologicalPlotting** (`ai_scientist/tools/biological_plotting.py`)
  - Params: `solution_path`, `output_dir`, `make_phase_portrait`; plots time series/phase portraits.
- **RunBiologicalStats** (`ai_scientist/tools/biological_stats.py`)
  - Params: `task` (`adjust_pvalues` or `enrichment`), plus required args. For enrichment, pass `term_to_ids_json` (JSON string mapping term -> [ids]) to avoid schema issues.
- **ReadManuscript** (`ai_scientist/tools/manuscript_reader.py`)
  - Params: `path` to PDF or txt/md; returns extracted text.
- **UpdateClaimGraph** (`ai_scientist/tools/claim_graph.py`)
  - Params: `path` to claim_graph.json, `claim_id`, `claim_text`, `parent_id` (use null for thesis), `evidence` (list), `status`, `notes`; adds/updates claim entries.
- **CheckClaimGraph** (`ai_scientist/tools/claim_graph_checker.py`)
  - Params: `path` to claim_graph.json; reports claims (and descendants) lacking support.
- **Filesystem helpers** (agents_orchestrator.py wrappers)
  - `list_artifacts` (browse experiment_results/ subdirs), `read_artifact` (with summary-only mode for large JSON), `reserve_output` (sanitizes names, rejects `..`, auto-uniques, and quarantines to `experiment_results/_unrouted` with a note if the primary path is unavailable), `resolve_path`, `get_run_paths`, `write_text_artifact` + conveniences (`write_interpretation_text`, `write_figures_readme`) which use the same sanitizer/quarantine behavior.
  - Manifest helpers: `inspect_manifest` (default summary-only; filters by role/path_glob/since; returns shard metadata) + `inspect_recent_manifest_entries` share the same sharded backend (`experiment_results/manifest/manifest_shard_*.ndjson` with `manifest_index.json`, auto-rotated ~10k entries/shard, legacy file_manifest.json auto-migrated). `append_manifest`/`read_manifest`/`read_manifest_entry`/`check_manifest`/`get_artifact_index` all use this backend and drop health reports into `experiment_results/_health/verification_missing_report_post_run.json` when gaps are found.
  - `check_status` (reads *.status.json), `coder_create_python` (safe code writes under run folder), `run_ruff`, `run_pyright`, `summarize_artifact` (lightweight heads/shapes).
  - PI/user inbox notes are canonical under `experiment_results` and maintained via the orchestrator wrappers (`read_note`, `write_pi_notes`, `check_user_inbox`) backed by `ai_scientist.utils.notes`. Root-level `pi_notes.md`/`user_inbox.md` are symlinks/copies only—agents should never write these paths directly.
  - `read_npy_artifact` (safely load small .npy arrays to JSON-friendly data; returns shape/dtype or full data if under size limits; errors for large or pickled arrays).
  - `validate_per_compartment_outputs` checks for required per-compartment artifacts (npz + map + topology summary), enforces schema version, shape/time alignment, and ordering checksum agreement; surfaces shapes/status/warnings/errors to gate completion.
  - Graph loaders accept `.gpickle` even on NetworkX>=3.5 via a pickle fallback (upstream removed `read_gpickle`).
- **FreezeRelease** (`agents_orchestrator.py`)
  - Params: `tag` (str), `description` (str, optional), `include_large_artifacts` (bool, default False).
  - Creates a release bundle under `experiment_results/releases/{tag}/`:
    - Regenerates `provenance_summary.md`, captures repo git hash + dirty diff (writes `diff.patch` when dirty), builds `env_manifest.json` (python/OS + requirements/env.yml + pip freeze), zips code into `code_release.zip` (skips heavy dirs: experiments, experiment_results, figures, venv/ caches).
    - Copies manifest/hypothesis_trace-referenced artifacts into `artifacts/` with checksums and a `release_manifest.json` (checksums, sizes, git state, skipped/missing lists).
  - Registers typed artifacts via manifest kinds: `code_release_archive`, `env_manifest`, `release_manifest`, `release_diff_patch`.
- **CheckReleaseReproducibility** (`agents_orchestrator.py`)
  - Params: `tag` (str), `quick` (bool, default True).
  - Verifies a release bundle by checksum-ing everything in `release_manifest.json`, then (optionally) runs a smoke test using available artifacts: calls `compute_model_metrics` on a sweep CSV and `run_biological_plotting` on a solution JSON if present.
  - Writes `releases/{tag}/repro_status.md` (kind `release_repro_status_md`) with status, git commit, env checksum, missing/mismatched files, and quick-test outcomes; logs a project knowledge entry if failures/partials are found.
- **GenerateReproductionSection** (`agents_orchestrator.py`)
  - Params: `tag` (str), `style` (str, default `"methods_and_supp"`).
  - Reads the release manifest + env manifest to emit manuscript-ready text: a ≤400-word Methods subsection on code/env availability and rerun instructions, plus a Supplementary protocol with concrete commands and a figure/tool mapping table grounded in the release files.
  - Writes `releases/{tag}/reproduction_methods.md` and `reproduction_protocol.md` via `write_text_artifact`, registering them under kinds `repro_methods_md` and `repro_protocol_md`.
- **run_writeup_task** (`agents_orchestrator.py`)
  - Params: `base_folder`, `page_limit`, optional `release_tag`.
  - Compiles the manuscript PDF, and when `release_tag` is provided, injects release metadata (tag/commit/DOI/env+code checksums) into the LaTeX front matter and PDF metadata, then registers the PDF manifest entry with release metadata attached.
- **Checks**: After code changes, run `ruff check agents_orchestrator.py` and `pyright agents_orchestrator.py` (ensure pyright cache is writable) to catch lint/type issues.

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

Agents orchestrator:
```bash
python agents_orchestrator.py \
  --load_idea ai_scientist/ideas/my_idea.json \
  --idea_idx 0 \
  --model gpt-5o-mini \
  --max_cycles 25 \
  --base_folder experiments/20251121_1801_axonal_arbor_percolation  # optional: restart from existing folder

# Manuscript-first orchestrator (derives idea from a draft)
python agents_orchestrator.py \
  --load_manuscript ai_scientist/ideas/manuscript_v3.md \
  --manuscript_title "A Topological Tipping Point Explains the Selective Vulnerability of Substantia Nigra Neurons" \
  --model gpt-5o-mini \
  --max_cycles 25
```
- Manuscript-first runs cache the ingested text to `experiment_results/manuscript_input.txt` and the derived seed idea to `experiment_results/seed_idea_from_manuscript.json` for provenance/resume.

### Sub-agent output visibility
- If a role agent hits `max_turns` or returns sparse text, the orchestrator now echoes the last message plus a brief summary of tool calls (`tools_called: ... (n_new_items=...)`) so the PI can see what actually ran.
