## VI-01 — Canonical Artifact Types & Naming Scheme

**Goal**
Stop agents from ad-hoc naming files. Every persistent artifact must belong to a small set of **artifact types**, each with a **canonical path pattern**.

**Key Ideas**

* Introduce an internal registry like `artifact_types.yml` or a Python map:

  ```yaml
  lit_summary:
    rel_dir: "experiment_results/lit"
    pattern: "lit_summary_main.json"
  ac_sweep:
    rel_dir: "experiment_results/simulations/ac_sweep"
    pattern: "ac_sweep__{run_id}.csv"
  tipping_fig_main:
    rel_dir: "experiment_results/figures_for_manuscript"
    pattern: "fig_{figure_id}.png"
  ```

* Add a **single tool** (e.g. `reserve_typed_artifact(kind, meta)`), which:

  * Looks up `kind` in the registry.
  * Fills in placeholders from `meta` (e.g. `run_id`, `figure_id`).
  * Returns the concrete path and calls `reserve_output` under the hood.

* Agents are **forbidden** (by prompt) from inventing filenames. They must:

  * Choose an existing `kind` from the registry.
  * Ask PI to add a new `kind` if what they need doesn’t exist.

**Scope**

* Implement typed registry.
* Implement wrapper tool `reserve_typed_artifact`.
* Update PI/Modeler/Analyst/Publisher prompts to require this tool for *all* new files except scratch logs.

**Acceptance Criteria**

1. **Code-level**:

   * There exists a single canonical registry file (YAML or Python) with ≤ 30 artifact types for the SNc project.
   * There is exactly one tool (e.g. `reserve_typed_artifact`) that all agents use to get output paths.
   * `reserve_output` is no longer called directly from agent prompts for scientific artifacts, only inside `reserve_typed_artifact` and maybe for scratch logs.

2. **Behavioral** (can be tested with a dummy run script):

   * In a full SNc pipeline run (lit → sims → sweeps → plots → manuscript), **100%** of `.json`, `.csv`, `.npy`, `.npz`, `.png`, `.svg`, `.pdf` artifacts live under directories declared in the registry and match the configured `pattern`.
   * No file in `experiment_results/` has a name that is not matched by some `artifact_types` entry (excluding temp files / logs in `_unrouted`).

3. **Guardrails**:

   * If an agent requests `reserve_typed_artifact(kind="unknown_kind")`, the tool returns an explicit error; there is **no** fallback to arbitrary path construction.

---

## VI-02 — Manifest v2: Lean, One-Row-Per-Artifact Index

**Goal**
Make the manifest a **compact artifact index**, not a stream-of-consciousness log.

**Problem Now**

* `annotations` and metadata are used as a dumping ground for notes.
* Many tools write repeatedly, so manifest grows to tens of thousands of entries.

**Key Ideas**

* **Manifest v2 schema** (JSONL or SQLite table), per artifact:

  ```json
  {
    "path": "experiment_results/simulations/ac_sweep/ac_sweep__001.csv",
    "name": "ac_sweep__001.csv",
    "kind": "ac_sweep",
    "created_at": "2025-12-09T12:34:56",
    "created_by": "Modeler",
    "size_bytes": 123456,
    "status": "ok"
  }
  ```

* No more free-form `annotations` for “notes”; instead store only short tags if needed.

* Enforce **one row per `path`** (upsert by `path`), no matter how many times it’s touched.

**Scope**

* Implement a new manifest writer (`manifest_v2.append_or_update`).
* Make `append_manifest` a thin wrapper over v2 that:

  * Accepts `kind` + `created_by` instead of arbitrary JSON blob.
  * Upserts by `path`.
* Update tools (`_scan_and_auto_update_manifest`, `mirror_artifacts`, etc.) to use v2.
* Stop using older `manifest_utils.append_manifest_entry` for anything except a migration path.

**Acceptance Criteria**

1. **Schema**:

   * Manifest file (`experiment_results/manifest/manifest_v2.jsonl` or `.db`) has **no field** that can hold > 1 KB of text (no long notes).
   * Each row has at least: `path`, `name`, `kind`, `created_at`.

2. **Size / Growth**:

   * Run a “maximal” SNc experiment that previously blew up the manifest.
   * Resulting v2 manifest file is **< 5 MB** and contains **≤ 3×** as many rows as there are distinct filesystem artifacts under `experiment_results/`.
   * Re-running the pipeline end-to-end produces **no more than +5%** additional manifest rows (i.e. idempotent upserts rather than appending endlessly).

3. **Uniqueness**:

   * Automated test: for a given run, `COUNT(DISTINCT path) == COUNT(*)` in manifest v2 (or analogous JSONL test). No duplicate `path` entries allowed.

---

## VI-03 — Separate “Notes/Reflections” Log from Manifest

**Goal**
All human/agent commentary, reflections, and status notes belong in a **knowledge log**, not in the artifact index.

You already have `project_knowledge.md`, `pi_notes.md`, `user_inbox.md`. We just need to **enforce the split**.

**Key Ideas**

* Introduce a tiny, dedicated tool for notes, e.g. `append_run_note(category, text, actor)` that writes to:

  * `experiment_results/run_notes.md` or updates `project_knowledge.md`.
* Update all prompts that currently say “log to manifest” with “log to project_knowledge / run_notes”.
* In `append_manifest`, reject `metadata_json` that contains large free-form `note`/`description` fields; allow only a short `kind` and `created_by`.

**Scope**

* Add `append_run_note` tool.
* Modify prompts for PI / Modeler / Analyst / Reviewer to:

  * Use `append_run_note` or `manage_project_knowledge` for reflections.
  * Use `append_manifest` only when a *new artifact file* is created or its status changes.
* Validate and trim `metadata_json` in `append_manifest` (e.g. warn if length > 300 chars).

**Acceptance Criteria**

1. **Behavior**:

   * Full SNc run with verbose agents produces:

     * A readable `run_notes.md` and/or expanded `project_knowledge.md`.
     * A manifest that contains **no English sentences** longer than a configurable limit (e.g. 120 characters).

2. **Static Check**:

   * Grep-style test over manifest v2: if any value matches regex for long prose (say, `\bthe\b.*\band\b` and length > 200), test fails.
   * Corresponding test over `run_notes.md` *does* find long paragraphs (notes go there).

3. **Prompt Discipline**:

   * Manual prompt audit: there is **no** instruction telling agents to “write notes into manifest”. All such instructions now name `project_knowledge`, `run_notes`, or `pi_notes` explicitly.

---

## VI-04 — “Typed Artifact” Helper API and Elimination of Ad-Hoc Paths

**Goal**
Make it *hard* for agents to go off-script with paths, by providing a **minimal helper API** and pruning everything else from their toolset.

**Key Ideas**

* Provide 2–3 primitives only:

  * `reserve_typed_artifact(kind, meta)` → path
  * `list_artifacts(kind)` → known paths of that kind
  * `read_artifact(path, ...)` (already exists)

* Hide:

  * `reserve_output` and raw path tools from most agents (only PI/coder can use them).
  * Any reference to `BaseTool.resolve_output_dir` from agent-exposed tools.

**Scope**

* Implement a thin adapter over `reserve_output` that takes `kind`, uses the registry (VI-01), and updates manifest v2 (VI-02).
* Narrow the tool list for Archivist / Modeler / Analyst / Reviewer to:

  * Only these typed helpers + reading/summary tools.
* For SNc work, seed the registry with enough kinds to cover:

  * `lit_summary`, `parameter_sets`, `ac_sweep`, `phase_portrait`, `energetic_landscape`, `manuscript_pdf`, etc.

**Acceptance Criteria**

1. **Tool Exposure**:

   * In `build_team`, the tool lists for Archivist, Modeler, Analyst, Reviewer **do not** include:

     * `reserve_output`
     * `BaseTool` proxies
     * Any tool that writes outside typed interface.
   * They *do* include the new typed helper(s).

2. **Runtime**:

   * Log instrumentation: for a full SNc run, number of direct calls to `reserve_output` from agents other than PI/Coder is **zero**.
   * All new artifacts created during the run have `kind` populated in manifest v2.

3. **Safety**:

   * If an agent tries to pass a raw filesystem path to a write tool (bypassing kind), they hit an explicit error or quarantine path, and that event is logged.

---

## VI-05 — Manifest Health Check + CI Gate

**Goal**
Prevent regressions where future changes re-introduce manifest bloat or broken naming.

**Key Ideas**

* Add a CLI command, e.g. `python -m lab_tools.check_run_health <base_folder>`, that:

  * Scans `experiment_results`.
  * Loads manifest v2.
  * Runs a battery of checks (naming, one-row-per-artifact, file existence, etc.).
* Wire this into:

  * Local `make test` / `pytest`.
  * CI pipeline for the repo.

**Scope**

* Implement `check_run_health` using existing helpers (`get_artifact_index`, `check_manifest`, new manifest v2 adapter).
* Define a baseline “golden run” for the SNc tipping point experiment (small, deterministic).
* Add tests that:

  * Run the golden experiment.
  * Run `check_run_health` and assert exit code 0.

**Acceptance Criteria**

1. **Checks Implemented**:

   * `check_run_health` at least verifies:

     * Every manifest entry `path` exists on disk, or is explicitly marked `status != "ok"`.
     * Every artifact in `experiment_results` is covered by a manifest entry **or** lives under a designated scratch directory (e.g. `_temp` / `_unrouted`).
     * All artifact `name`s match the regex defined by the registry `pattern` for their `kind` (VI-01).
     * Manifest size constraints from VI-02 hold.

2. **CI / Automation**:

   * CI job fails if `check_run_health` exits non-zero on the golden SNc run.
   * Local dev can run `check_run_health` on any run folder with a single command.

3. **Regression Safety**:

   * Intentionally breaking the naming scheme (e.g. manually dropping `foo.txt` into `experiment_results`) causes `check_run_health` to report a clear error and non-zero exit code.

---

## VI-06 — SNc Energetic Tipping Point Profile (Project Template)

**Goal**
Package everything above into a **reusable project profile** so someone can clone the repo and start a new project with the same discipline.

**Key Ideas**

* Add a template like `profiles/snc_energetic_tipping_point/` that includes:

  * `idea.json` for this project.
  * `artifact_types.yml` tailored to this workflow.
  * A sample `run.sh` that launches the PI with this profile.
* Document how to:

  * Create a new project by copying a profile and adjusting `artifact_types.yml`.

**Scope**

* Create `profiles/snc_energetic_tipping_point` directory.
* Add minimal README describing:

  * Folder layout.
  * Artifact kinds and their meaning.
  * How to run a full cycle from terminal.
* Ensure main runner can accept `--profile snc_energetic_tipping_point`.

**Acceptance Criteria**

1. **New User Experience**:

   * Given a fresh checkout of the repo:

     * `./run_snc_example.sh` (or similar) runs the full SNc pipeline using the profile.
     * The run finishes with a valid manifest, correct naming, and a compiled manuscript / figures.

2. **Reusability**:

   * Starting a second project by copying the profile and changing `idea.json` + `artifact_types.yml` does **not** require touching the core engine code.
   * Both projects can co-exist in `experiments/` with their own manifests and artifact layouts, and `check_run_health` passes for each.

3. **Documentation**:

   * There is a single README section (“Starting a new research project”) that describes:

     * How to create a new project profile.
     * How to run it.
     * How to interpret the generated folders/artifacts.