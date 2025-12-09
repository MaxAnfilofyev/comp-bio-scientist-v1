## VI-02 – Enforce “Computational-Only” Scope at the System Level

**Problem / Opportunity**

Right now the guardrail “no wet lab, no new external datasets, only published lit + our own models” lives only in your head and a couple of prompts. That’s fragile. The PI and sub-agents can still propose/describe wet-lab experiments or analyses that assume new data collection.

**Scope**

* PI, Archivist, Modeler prompts
* Idea JSON validation

**Changes**

1. **Idea schema gatekeeper**

   * Add a lightweight `IdeaScopeValidatorTool` (or repurpose `manage_project_knowledge`) that:

     * Classifies each `Experiments` entry into one of:

       * `computational_modeling`
       * `secondary_data_analysis`
       * `wet_lab_or_clinical`
     * Rejects or marks “out of scope” any experiment not in the first two buckets.

2. **PI hard guardrail**

   * At the start of `build_team` / PI instructions, add:

     * “You MUST NOT plan or delegate wet-lab, animal, or clinical experiments; if an experiment requires generating new biological data, mark it as OUT_OF_SCOPE and log it to project_knowledge as such.”
   * Require the PI to:

     * Produce `implementation_plan.md` that explicitly lists which experiments are in-scope vs out-of-scope.

3. **Modeler hard guardrail**

   * Add explicit text: “If a requested experiment appears to require new biological samples, lab assays, or clinical interventions, you MUST refuse and report back to the PI; do not attempt to ‘simulate’ availability of unseen data.”

**Acceptance Criteria**

* Given an `idea.json` where one experiment clearly says “patch clamp recordings on slices” or “mouse model”, the run:

  * Produces an artifact (e.g., `experiment_filter_report.md`) listing that experiment as `wet_lab_or_clinical`.
  * Does **not** call `run_comp_sim`, `run_sensitivity_sweep`, etc., on behalf of that experiment.
* In at least 3 test ideas:

  * All in-scope (computational) experiments are kept and planned.
  * All obviously wet-lab experiments are flagged and excluded from downstream agents.
* No agent prompt contains instructions that could be reasonably interpreted as “design a lab protocol” after the change.

---

## VI-03 – Strict Literature Provenance & Reference Verification

**Problem / Opportunity**

The pipeline assembles and validates a `lit_summary`, but it doesn’t **prove** that references exist, have valid DOIs, and aren’t hallucinated. That’s directly at odds with your requirement that all claims come from verifiable published work.

**Scope**

* `assemble_lit_data`, `LitDataAssemblyTool`
* Archivist agent
* New artifact types

**Changes**

1. **Add new artifact types**

   Extend `ARTIFACT_TYPE_REGISTRY` with:

   * `lit_reference_verification_table` → `experiment_results/lit_reference_verification.csv`
   * `lit_reference_verification_json` → `experiment_results/lit_reference_verification.json`

2. **New verification tool**

   Implement a `ReferenceVerificationTool` (or wrapper) that:

   * Takes `lit_summary.json`.
   * For each reference:

     * Looks it up via `SemanticScholarSearchTool` (title, authors, year).
     * Confirms:

       * Existence.
       * DOI (or clear reason if absent).
       * Basic metadata match (title similarity, authors).
   * Outputs:

     * CSV/JSON with columns: `ref_id`, `title`, `authors`, `year`, `doi`, `found`, `match_score`, `notes`.

3. **Archivist contract**

   * Archivist must:

     * Call `ReferenceVerificationTool` after `assemble_lit_data`.
     * Refuse to call the project “ready for modeling” if:

       * `found == False` for more than N% of references (e.g., 10–20%), or
       * Any reference has `match_score` below a threshold.
   * Log a reflection via `manage_project_knowledge` if verification repeatedly fails for a specific venue/source.

**Acceptance Criteria**

* Whenever `lit_summary.json` exists, the run also produces **one** of:

  * `lit_reference_verification.csv` **or**
  * `lit_reference_verification.json`.
* In a test where you inject a clearly fake citation into the lit summary:

  * The verification artifact marks it as `found == False` or `match_score < threshold`.
  * Archivist returns a FAILURE message and logs a reflection instead of proceeding.
* For a known good reference list (e.g., 5–10 real citations):

  * ≥90% of entries are marked `found == True` with a DOI populated.
* The Reviewer, when run, can read this verification artifact and identify any “unsupported” references.

---

## VI-04 – Model Specification & Parameter Source Ledger

**Problem / Opportunity**

You want rigorous comp-bio: every parameter in a model should either be tied to literature or explicitly flagged as a modeling convenience. Right now, Modeler runs `RunBiologicalModelTool` and `RunCompartmentalSimTool` without generating a structured **param provenance ledger**.

**Scope**

* Modeler agent
* New artifact types
* Possibly a small helper tool

**Changes**

1. **New artifact types**

   Add:

   * `model_spec_yaml` → `experiment_results/models/{model_key}_spec.yaml`
   * `parameter_source_table` → `experiment_results/parameters/{model_key}_param_sources.csv`

2. **Parameter source table schema**

   Each row:

   * `param_name`
   * `value`
   * `units`
   * `source_type` ∈ {`lit_value`, `fit_to_data`, `dimensionless_scaling`, `free_hyperparameter`}
   * `lit_claim_id` (link to claim_graph, or `NA`)
   * `reference_id` (from `lit_summary`)
   * `notes`

3. **Modeler responsibilities**

   * Before first simulation of a given `model_key`, the Modeler must:

     * Generate a `model_spec_yaml` describing:

       * State variables.
       * ODEs / update equations.
       * Parameter names and initial guesses.
     * Generate a `parameter_source_table` linking each parameter to lit or modeling assumption.
   * Link to claim graph:

     * When `source_type == lit_value`, require `lit_claim_id` that exists in `claim_graph.json`.

4. **Reviewer integration**

   * Reviewer checks that:

     * For every parameter used in a final figure/simulation, there is a row in `parameter_source_table`.
     * Missing entries are reported as a gap.

**Acceptance Criteria**

* After running a built-in model (`run_biological_model`), the run produces:

  * `models/{model_key}_spec.yaml` and
  * `parameters/{model_key}_param_sources.csv`.
* In that CSV:

  * Every parameter used in the model appears exactly once.
* If you manually remove one parameter row and re-run Reviewer:

  * Reviewer reports a gap (“parameter X used without provenance”) and does **not** claim “NO GAPS”.
* If a parameter is marked `source_type=lit_value`, its `lit_claim_id` exists in `claim_graph.json`.

---

## VI-05 – Hypothesis–Experiment–Artifact Traceability

**Problem / Opportunity**

You already have a claim graph and a structured idea JSON, but you don’t have a clean, machine-readable mapping from:

> Hypothesis → Experiments → Sim runs / figures → Conclusion

That makes it hard to verify that every conclusion actually has data and code behind it.

**Scope**

* PI, Modeler, Analyst, Reviewer
* New artifact type

**Changes**

1. **New artifact type**

   Add:

   * `hypothesis_trace_json` → `experiment_results/hypothesis_trace.json`

2. **Data model**

   Structure:

   ```json
   {
     "hypotheses": [
       {
         "id": "H1",
         "text": "...",
         "experiments": [
           {
             "id": "E1",
             "description": "...",
             "sim_runs": [
               {"baseline": "...", "transport": 0.05, "seed": 0},
               ...
             ],
             "figures": ["fig_3.svg", "phase_portrait_H1_E1.png"],
             "metrics": ["frac_failed", "critical_transport"]
           }
         ],
         "status": "supported | refuted | inconclusive"
       }
     ]
   }
   ```

3. **PI responsibilities**

   * When generating `implementation_plan.md`, PI also:

     * Writes/updates `hypothesis_trace.json` skeleton (H*, E* ids, descriptions).
   * Ensures each experiment in the idea is mapped to at least one planned sim or analysis.

4. **Modeler & Analyst responsibilities**

   * After running simulations:

     * Update `hypothesis_trace.json` with actual sim run identifiers and metrics.
   * After generating figures:

     * Add figure filenames under the appropriate experiment entries.

5. **Reviewer responsibilities**

   * Reject any hypothesis marked as “supported” if:

     * No associated `sim_runs` or `figures` exist.
   * Highlight hypotheses that have only one weak/edge-case experiment.

**Acceptance Criteria**

* After a full run on an idea with at least one hypothesis and one experiment:

  * `experiment_results/hypothesis_trace.json` exists and is valid JSON.
* All final manuscript figures referenced in the text have corresponding entries in `hypothesis_trace.json`.
* If you delete a figure file and re-run Reviewer:

  * Reviewer flags the gap between the text claim and missing artifact.
* A small test script can iterate over hypotheses and assert:

  * `status != "supported"` implies **no** figures/sim runs are missing.
  * If they are, Reviewer has reported it.

---

## VI-06 – Domain-Specific Model Evaluation Metrics

**Problem / Opportunity**

Current tools mostly use generic outputs (e.g., `frac_failed`). For real comp-bio work (e.g., energetic tipping points, bifurcation geometry), you want structured domain-specific metrics, not just plots.

**Scope**

* Modeler, Analyst
* New tools + artifact types

**Changes**

1. **New artifact types**

   * `model_metrics_json` → `experiment_results/models/{model_key}_metrics.json`
   * `sweep_metrics_csv` → `experiment_results/simulations/{label}_metrics.csv`

2. **New helper tool**

   Implement `ComputeModelMetricsTool` that:

   * Given per-compartment outputs and/or sweep CSVs, computes:

     * Bifurcation proxies (e.g., approximate critical transport / load).
     * Summary stats (mean frac_failed at key parameter regimes).
     * Any model-specific metrics you care about (e.g., energy gap between equilibria).

3. **Modeler responsibilities**

   * After finishing a sweep or transport batch, call `ComputeModelMetricsTool`.
   * Save metrics via `reserve_typed_artifact(kind="parameter_set"/"sweep_metrics_csv"/"model_metrics_json", ...)`.

4. **Analyst responsibilities**

   * Build figures from **metrics** rather than raw CSV whenever possible.
   * Use metrics to check whether data actually shows a stable tipping point, not just plot something pretty.

**Acceptance Criteria**

* After a sensitivity sweep, there is a corresponding `*_metrics.csv` or `{model_key}_metrics.json` with:

  * Named columns for key quantities (`critical_transport_est`, etc.).
* Analyst can regenerate summary plots using only metrics artifacts (no need to re-read raw sweeps).
* If `ComputeModelMetricsTool` is deliberately skipped and you run Reviewer:

  * Reviewer flags that no metrics artifacts exist for the simulations referenced in the text.

---

## VI-07 – Simulation Budgeting & Chunked Execution

**Problem / Opportunity**

`run_transport_batch` can run an arbitrarily large product of `transport_values × seeds`. It’s easy for an LLM to accidentally request a huge batch and blow up runtime/memory. You also want tighter control over “academic lab-scale” resource use.

**Scope**

* `run_transport_batch`
* PI & Modeler prompts

**Changes**

1. **Hard caps in `run_transport_batch`**

   * Add a guardrail at the top of `run_transport_batch`:

     ```python
     MAX_JOBS = int(os.environ.get("AISC_MAX_TRANSPORT_JOBS", "64"))
     n_jobs = len(transport_values or []) * len(seeds or [])
     if n_jobs > MAX_JOBS:
         return {
             "error": f"Requested {n_jobs} jobs exceeds MAX_JOBS={MAX_JOBS}. "
                      "Chunk your batch and rerun.",
             "max_jobs": MAX_JOBS
         }
     ```

2. **Modeler instructions**

   * Explicitly instruct:

     * “Never request more than 64 sim runs in one `run_transport_batch` call. For larger experiments, break into chunks and update `pi_notes.md` with progress after each chunk.”

3. **PI instructions**

   * Reinforce that large experiments must be broken into multiple PI → Modeler invocations, each with:

     * Clear parameter ranges.
     * Expected number of runs below the cap.

**Acceptance Criteria**

* Calling `run_transport_batch` with `len(transports)=20`, `len(seeds)=10` (200 jobs) returns an error and does **not** start simulations.
* For a full sweep that logically needs 200 runs:

  * The PI / Modeler workflow executes it as ≥4 separate batches, each below `MAX_JOBS`.
* No single `run_transport_batch` invocation produces more completed runs than `MAX_JOBS`.

---

## VI-08 – Failure Reporting & Resumability Contract

**Problem / Opportunity**

You already have a lot of manifest machinery and `inspect_recent_manifest_entries`, but there’s no strict contract that *every* failure produces a human-readable “what got done, what’s broken, where to resume” artifact.

**Scope**

* PI and all agents
* `extract_run_output`, run_notes

**Changes**

1. **Standard failure note format**

   * Define a canonical `run_failure.md` (or `run_status.md`) under `experiment_results/` with fields:

     * `phase` (Archivist / Modeler / Analyst / etc.)
     * `last_successful_artifacts` (list of paths)
     * `error_message`
     * `suggested_next_step`

2. **PI responsibilities**

   * Whenever an agent returns an error or times out:

     * PI writes/updates `run_failure.md` via `write_text_artifact`.
     * PI calls `inspect_recent_manifest_entries` and summarizes progress in that file.

3. **Resume rule**

   * At the top of the PI instructions:

     * “On startup, read `run_failure.md` if present and resume from the last listed successful artifact instead of restarting from scratch.”

**Acceptance Criteria**

* If any agent tool call raises an exception (e.g., you simulate a failure), the next PI turn:

  * Creates or updates `run_failure.md` with a non-empty `error_message`.
* On a fresh run in the same base folder:

  * PI reads `run_failure.md` and the behavior described in that file (e.g., “skip lit assembly; start from lit_summary.json”) is followed.
* Deleting `run_failure.md` causes PI to treat the run as fresh (no resumable state).

---

## VI-09 – Manuscript-Ready Provenance Summary

**Problem / Opportunity**

You have proof-of-work notes (`*_verification.md`), but there’s no single, human-friendly summary of “what we did, with what data, and where it came from” that you can paste into Methods / Supplementary.

**Scope**

* Reviewer, Publisher
* New artifact type

**Changes**

1. **New artifact type**

   * `provenance_summary_md` → `experiment_results/provenance_summary.md`

2. **Reviewer responsibilities**

   * Aggregate, from manifest + verification notes:

     * Data sources (Semantic Scholar, built-in models).
     * Simulation set-ups (baseline graphs, parameter ranges).
     * Key analysis steps (sweeps, interventions, stats).

   * Write `provenance_summary.md` in structured sections:

     * “Literature Sources”
     * “Model Definitions”
     * “Simulation Protocols”
     * “Statistical Analyses”

3. **Publisher responsibilities**

   * Ensure this summary is referenced or integrated into the final manuscript (e.g., used to populate Methods text) before calling `run_writeup_task`.

**Acceptance Criteria**

* After a successful full pipeline run:

  * `experiment_results/provenance_summary.md` exists and is non-empty.
* This file lists:

  * At least one lit summary artifact.
  * At least one model spec file.
  * At least one simulation/sweep output.
* If you delete `provenance_summary.md` and re-run Reviewer:

  * Reviewer regenerates it.
* If a major artifact is missing from manifest (e.g., no `lit_summary.json`), Reviewer:

  * Either refuses to generate a “complete” provenance summary or flags the missing section explicitly.