## VI-CF1 — Reproducible “release snapshot” for manuscripts

**Goal:** One command/tool that freezes *exactly* what was run (code, config, environment, manifest) into a versioned bundle suitable for Zenodo/OSF + manuscript supplementary materials.

### What it should do

* Create a **read-only snapshot** of:

  * The repo state (commit hash, dirty/uncommitted diff).
  * `project_knowledge.md`, `hypothesis_trace.json`, `claim_graph.json`, `provenance_summary.md`.
  * All simulation/figure/model artifacts referenced in the manifest and/or `hypothesis_trace.json`.
  * Environment description (`requirements.txt` or `environment.yml`, `python --version`, OS info).

* Write everything under a canonical location, e.g.
  `experiment_results/releases/<release_tag>/…`

* Register the release bundle and its key components in the manifest via `ARTIFACT_TYPE_REGISTRY`, e.g.:

  * `code_release_archive` → `experiment_results/releases/{tag}/code_release.zip`
  * `env_manifest` → `experiment_results/releases/{tag}/env_manifest.json`
  * `release_manifest` → `experiment_results/releases/{tag}/release_manifest.json`

### User-facing trigger

A single tool, e.g.:

```python
@function_tool
def freeze_release(tag: str, description: str = "", include_large_artifacts: bool = False):
    ...
```

(May internally refuse `include_large_artifacts=True` if over a size threshold and instead emit a “too big” summary.)

### Acceptance criteria

You can call `freeze_release(tag="snc_energy_tipping_v1")` and then:

1. A directory `experiment_results/releases/snc_energy_tipping_v1/` exists and contains at minimum:

   * `code_release.zip` (or `.tar.gz`) with the repo tree (minus huge junk/venv).
   * `env_manifest.json` (Python version, OS, dependency list).
   * `release_manifest.json` listing:

     * checksums + sizes for all included files
     * git commit hash and dirty flag
     * timestamp, tag, and description.
2. `reserve_typed_artifact` + `append_manifest` are used so that:

   * `list_artifacts_by_kind("code_release_archive")` returns the archive path.
   * `list_artifacts_by_kind("env_manifest")` returns the env manifest path.
3. `generate_provenance_summary()` has been run or is re-run as part of freezing, and its path is included in `release_manifest.json`.
4. If the git working tree is dirty, `release_manifest.json` clearly indicates `dirty: true` and includes a short diff summary or a pointer to a `diff.patch` file inside the archive.

---

## VI-CF2 — “Reproduce from release” dry-run check

Freezing is half the job. You also want a **sanity check** that a fresh user could re-run the core pipeline from the release bundle.

### What it should do

* Provide a tool that, given a release tag, performs a **non-destructive reproducibility check**:

  * Verifies checksums of all files in the release.
  * Optionally spins up a **minimal test run** using the environment manifest only (no interactive PI agent), e.g.:

    * load one sweep CSV
    * re-run `compute_model_metrics`
    * re-run `run_biological_plotting` on a small solution file

* Emit a structured “repro badge” summary:

  * `status: ok/failed/partial`
  * any missing files or mismatched checksums
  * command line and steps used for the test

### Example tool

```python
@function_tool
def check_release_reproducibility(tag: str, quick: bool = True):
    ...
```

### Acceptance criteria

For any existing tag:

1. `check_release_reproducibility(tag="snc_energy_tipping_v1")` returns JSON with:

   * `"status": "ok"` if:

     * All files listed in `release_manifest.json` exist and pass checksum verification.
     * The quick reproduction steps (plots + metrics) complete without uncaught exceptions.
   * `"status": "failed"` or `"partial"` plus explicit reasons if not.
2. The check logs an entry to `project_knowledge.md` with:

   * category `constraint` or `failure_pattern` when it fails
   * a short suggestion (e.g. “missing morphology graph, add to artifact registry for future releases”).
3. You can point a reviewer to a *single text artifact* (e.g. `experiment_results/releases/snc_energy_tipping_v1/repro_status.md`) summarizing:

   * the tag
   * git commit
   * env hash
   * repro status and date of last check.

---

## VI-CF3 — Manuscript-ready “How to rerun this code” section generator

You don’t just want a bundle; you want **ready-to-paste text** for the Methods / Supplementary.

### What it should do

* Read the release manifest, env manifest, provenance summary, and `hypothesis_trace.json`.
* Emit a **short, standardized Methods subsection** plus a **longer Supplementary protocol**:

  * Methods-level (for main text):

    * One paragraph on environment and code availability.
    * One paragraph on how to rerun key simulations and analysis.

  * Supplementary-level:

    * Stepwise commands:

      * `conda create -n snc_energy python=...`
      * `pip install -r requirements.txt`
      * `python run_experiment.py --load_idea ...`
    * Table or bullets linking each **main figure** to:

      * the exact script/tool calls
      * input artifact paths from the release.

### Example tool

```python
@function_tool
def generate_reproduction_section(tag: str, style: str = "methods_and_supp"):
    ...
```

### Acceptance criteria

For a given tag:

1. `generate_reproduction_section(tag="snc_energy_tipping_v1")` returns a dict with:

   * `methods_section_md`: ≤ 400 words, self-contained, references the DOI/Zenodo link field in `release_manifest.json` if present.
   * `supplementary_protocol_md`: step-by-step instructions, explicit commands, and a **table mapping figures ↔ artifact paths ↔ tools**.
2. The generated text:

   * Does **not** reference internal paths that aren’t in the release archive.
   * Uses only commands that actually exist (e.g., `run_comp_sim`, `run_sensitivity_sweep`, `run_writeup_task`, etc.).
3. A call to `write_text_artifact` writes both sections into:

   * `experiment_results/releases/<tag>/reproduction_methods.md`
   * `experiment_results/releases/<tag>/reproduction_protocol.md`
     and registers them under a `kind` like `repro_methods_md` and `repro_protocol_md`.

---

## VI-CF4 — Release tagging + cross-link in manuscript PDF (optional but strong)

If you want to be fancy (and future-proof):

### What it should do

* Allow the writeup pipeline (`run_writeup_task`) to accept a `release_tag`, and:

  * Embed that tag + git hash + DOI (if known) into the PDF footer or first page.
  * Optionally write a tiny JSON block inside the PDF metadata with:

    * tag
    * commit
    * env hash (e.g., SHA256 of `env_manifest.json`)

### Acceptance criteria

1. Calling `run_writeup_task(base_folder=..., page_limit=...)` with a `release_tag`:

   * Produces `manuscript.pdf` that, when inspected, has the tag and commit hash in the front matter (visible) and in PDF metadata.
2. `list_artifacts_by_kind("manuscript_pdf")` returns an entry with a `metadata` field that includes the `release_tag` and the hash of the associated `code_release_archive`.