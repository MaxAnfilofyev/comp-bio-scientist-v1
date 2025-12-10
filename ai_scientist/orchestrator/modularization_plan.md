# Orchstrator Modularization Plan & Progress

This document tracks the remaining work to fully modularize the orchestrator by extracting utilities and context into separate, focused modules.

## Status Summary

**Done:** Phase 1 (Context & artifacts consolidation) is complete. The orchestrator now properly delegates to `context.py` and `artifacts.py` modules instead of duplicating logic.

**Remaining:** Phases 2-8 cover the extraction of remaining utilities into focused modules while keeping `agents_orchestrator.py` compatible and functional.

---

## Phase 1 ✅ COMPLETED - Context & Artifacts Consolidation

### What was done:
- ✅ Moved `parse_args` from `agents_orchestrator.py` to `ai_scientist/orchestrator/context.py`
- ✅ Added imports from `context.py`: `_fill_figure_dir`, `_fill_output_dir`, `_bootstrap_note_links`, `_ensure_transport_readme`, `_report_capabilities`
- ✅ Removed duplicate implementations from `agents_orchestrator.py`
- ✅ Consolidated artifact registry/reservation helpers in `artifacts.py`
- ✅ Verified syntax compilation passes

---

## Phase 2: Manifest Service Extraction

### Target Module: `ai_scientist/orchestrator/manifest_service.py`

Move manifest-related utilities and project knowledge wrappers:

#### Functions to move:
- [x] `_scan_and_auto_update_manifest`
- [x] `_normalize_manifest_entry`
- [x] `_build_metadata_for_compat`
- [x] `_load_manifest_map`
- [x] `_append_manifest_entry`
- [x] `_append_artifact_from_result`
- [x] `_append_figures_from_result`

#### `@function_tool` wrappers to delegate:
- [_] `inspect_manifest`
- [_] `inspect_recent_manifest_entries`
- [_] `append_manifest`
- [_] `read_manifest_entry`
- [_] `check_manifest`
- [_] `read_manifest`
- [_] `check_manifest_unique_paths`
- [_] `list_artifacts`
- [_] `list_artifacts_by_kind`
- [_] `get_artifact_index`
- [_] `check_project_state` (manifest-focused health check)
- [_] `manage_project_knowledge`
- [_] `append_run_note_tool`

#### Expected signature:
```python
# In manifest_service.py
from ai_scientist.utils import manifest as manifest_utils

def inspect_manifest(base_folder=None, role=None, ...):
    # Delegate to manifest_utils.inspect_manifest
    ...

# In agents_orchestrator.py (after delegates)
from ai_scientist.orchestrator.manifest_service import inspect_manifest

@function_tool
def inspect_manifest(base_folder=None, ...):  # Keep original signature
    return manifest_service.inspect_manifest(...)
```

---

## Phase 3: Hypothesis/Pvenance Module

### Target Module: `ai_scientist/orchestrator/hypothesis.py`

Move hypothesis trace, manuscript seed, gates, and provenance logic:

#### Functions to move:
- [_] Hypothesis trace: `_bootstrap_hypothesis_trace`, `_load_hypothesis_trace`, `_write_hypothesis_trace`, `_ensure_hypothesis_entry`, `_ensure_experiment_entry`, `_update_hypothesis_trace_with_sim`, `_update_hypothesis_trace_with_figures`
- [_] Manuscript processing: `_extract_markdown_section`, `_extract_markdown_title`, `_extract_subheadings`, `_extract_bullets_or_paragraph`, `_first_sentence`, `_derive_experiments_from_text`, `_derive_idea_from_manuscript`, `_persist_manuscript_seed`
- [_] Lit verification/gates: `_resolve_lit_summary_path`, `_resolve_verification_path`, `_load_verification_rows`, `_verification_row_confirmed`, `_evaluate_lit_ready`, `_log_lit_gate_decision`, `_record_lit_gate_in_provenance`, `_should_skip_lit_gate`, `_ensure_lit_gate_ready`
- [_] Model provenance: `_model_metadata_from_key`, `_ensure_model_spec_and_params`, `_load_spec_content`, `_evaluate_model_provenance`, `_record_model_provenance_in_provenance`
- [_] Claim consistency: `_resolve_claim_graph_path`, `_load_claim_graph`, `_load_hypothesis_trace_file`, `_gather_support_from_trace`, `_evaluate_claim_consistency`, `_record_claim_consistency_in_provenance`
- [_] Provenance summary: `_collect_provenance_sections`, `_render_provenance_markdown`
- [_] General: `_coerce_float`, `_coerce_int` (utility for type conversions)

#### `@function_tool` wrappers to delegate:
- [_] `update_hypothesis_trace`
- [_] `validate_lit_summary`
- [_] `verify_references`
- [_] `check_lit_ready`
- [_] `check_model_provenance`
- [_] `update_claim_graph`
- [_] `check_claim_graph`
- [_] `check_claim_consistency`
- [_] `generate_provenance_summary`

#### Dependencies & interactions:
- [_] Update sim/model wrappers (`run_comp_sim`, `run_biological_model`, `run_sensitivity_sweep`, `run_intervention_tests`) to call `hypothesis` for gates and trace updates

---

## Phase 4: Transport Module

### Target Module: `ai_scientist/orchestrator/transport.py`

Move transport manifests, baseline resolution, and run layout:

#### Functions to move:
- [_] Transport manifest core: `_transport_manifest_path`, `_acquire_manifest_lock`, `_atomic_write_json`, `_load_transport_manifest`, `_upsert_transport_manifest_entry`, `_scan_transport_runs`
- [_] Baseline/run helpers: `_resolve_baseline_path_internal`, `_build_seed_dir`, `_resolve_run_paths`, `_status_from_paths`, `_write_verification`, `_generate_run_recipe`

#### `@function_tool` wrappers to delegate:
- [_] `scan_transport_manifest`
- [_] `read_transport_manifest`
- [_] `resolve_baseline_path`
- [_] `resolve_sim_path`
- [_] `update_transport_manifest`
- [_] `run_transport_batch`
- [_] `validate_per_compartment_outputs`
- [_] `sim_postprocess` (transport-related batches)
- [_] `repair_sim_outputs` (transport-related repair)

#### Integration:
- [_] Wire transport manifest calls in sim wrappers through this module

---

## Phase 5: Release Module

### Target Module: `ai_scientist/orchestrator/release.py`

Move release bundling, reproducibility, and reproduction text:

#### Functions to move:
- [_] Low-level helpers: `_safe_sha256`, `_detect_repo_root`, `_collect_manifest_artifacts`, `_collect_paths_from_trace`, `_write_env_manifest`, `_create_code_archive`, `_gather_git_state`, `_copy_release_sources`, `_relative_to_release`, `_load_release_manifest`, `_verify_release_files`, `_read_env_manifest`
- [_] Template helpers: `_select_first`, `_build_figure_mapping_table`, `_word_limit`

#### `@function_tool` wrappers to delegate:
- [_] `freeze_release`
- [_] `check_release_reproducibility`
- [_] `generate_reproduction_section`

#### Expected dependencies:
- [_] Will call into `manifest_service` for collection, but keep implementation self-contained

---

## Phase 6: Tool Wrappers Module

### Target Module: `ai_scientist/orchestrator/tool_wrappers.py`

Consolidate all `@function_tool` decorators into focused groupings:

#### Wrapper groups to create with delegation:
- [_] **Literature tools:** `assemble_lit_data`, `validate_lit_summary`, `verify_references`, `search_semantic_scholar`
- [_] **Model/Simulation tools:** `build_graphs`, `run_biological_model`, `run_comp_sim`, `run_sensitivity_sweep`, `run_intervention_tests`, `compute_model_metrics`, `sim_postprocess`, `repair_sim_outputs`
- [_] **Analysis tools:** `run_biological_plotting`, `run_validation_compare`, `run_biological_stats`, `graph_diagnostics`, `read_npy_artifact`, `summarize_artifact`, `read_artifact`, `head_artifact`
- [_] **Governance tools:** `check_lit_ready`, `check_model_provenance`, `check_claim_graph`, `check_claim_consistency`, `generate_provenance_summary`, `check_project_state`
- [_] **Release tools:** `freeze_release`, `check_release_reproducibility`, `generate_reproduction_section`
- [_] **Notes/Governance:** `read_note`, `write_pi_notes`, `wait_for_human_review`, `check_user_inbox`

#### Integration pattern:
Each wrapper should call appropriate domain modules:
```python
@function_tool
def run_biological_model(...):
    # Call hypothesis for gate checks
    # Call manifest_service for artifact tracking
    # Call transport for any relevant manifests
    # Delegate to actual tool
    ...
```

---

## Phase 7: Agents Module

### Target Module: `ai_scientist/orchestrator/agents.py`

Move agent creation and team assembly:

#### Functions to move:
- [_] `_make_agent` (generic agent factory)
- [_] `extract_run_output` (custom output extractor)
- [_] `build_team` (full team assembly with all role prompts)

#### Expected imports:
- [_] Tools from `tool_wrappers` (via `@function_tool` wrappers)
- [_] Keep tool choice and model settings

---

## Phase 8: CLI/Main Slimming

### Target: Either `ai_scientist/orchestrator/cli.py` or refactor `agents_orchestrator.py`

#### Goal: Make main() focused on high-level flow
Either:
- [_] **Option A:** Create `cli.py` with encapsulated main logic:
  ```python
  def main():
      args = parse_args()  # from context
      ctx = bootstrap_run(args, idea)  # new helper
      # ... capability reporting
      team = build_team(args.model, idea, ctx)  # from agents
      prompt = build_initial_prompt(ctx)  # new helper
      Runner.run(team, prompt, ...)  # with timeout handling
  ```
  Make `agents_orchestrator.py` import and call it.

- [_] **Option B:** Keep main() in `agents_orchestrator.py` but factor into helpers in other modules.

---

## Testing & Verification Strategy

### After each phase:
- [_] Test compilation: `python3 -m py_compile agents_orchestrator.py`
- [_] Run phase-specific tests (see test mapping below)
- [_] Manual smoke test on simple idea

### Test mapping by phase:
- Phase 2 (manifest): `test_check_run_health.py`, `test_notes_helper.py`
- Phase 3 (hypothesis): `test_lit_gate.py`, `test_model_provenance_gate.py`, `test_claim_consistency.py`, `test_generate_reproduction_section.py`
- Phase 4 (transport): `test_transport_index.py`, `test_per_compartment.py`, `test_repair_helpers.py`
- Phase 5 (release): `test_freeze_release.py`, `test_release_repro.py`, `test_generate_reproduction_section.py`
- Phase 6-8 (final): full test suite + end-to-end orchestrator run

### End-to-end verification:
- Run with `ideas/theoretical_biology/evolution_of_cooperation.json`
- Check that experiment outputs are created correctly
- Validate manifest/tracking works end-to-end

---

## Implementation Order & Safety

**Start with:** Phase 2 (Manifest service) - Small, well-scoped, critical for others.

**Safety principles:**
1. Keep modules small and focused
2. Preserve all `@function_tool` signatures exactly
3. Test compilation after each file change
4. Use thin delegation wrappers initially (avoid major refactoring within phases)
5. Commit/verify after each phase completes

**Assumptions:**
- Existing util modules (`ai_scientist/utils/*`) are reused, not duplicated
- All imports work correctly (add missing ones as discovered)
- No functional changes to algorithmic behavior

---

## Progress Tracking
Update this section as work progresses:

- Phase 1: ✅ DONE
- Phase 2: ✅ DONE
- Phase 3: ⏳ PENDING
- Phase 4: ⏳ PENDING
- Phase 5: ⏳ PENDING
- Phase 6: ⏳ PENDING
- Phase 7: ⏳ PENDING
- Phase 8: ⏳ PENDING

## Recommended Implementation Order

Go with: **Phase 3 (Hypothesis/Pvenance) > Phase 4 (Transport)** - Both are self-contained domains with high utility. Hypothesis unlocks gating, Transport enables simulation scalability. After those, Phase 6 (Tool Wrappers) can consolidate all delegations, followed by Phase 7 (Agents) and Phase 8 (CLI).

Skip Phase 5 (Release) until reproduction features are stabilized (has more external dependencies).
