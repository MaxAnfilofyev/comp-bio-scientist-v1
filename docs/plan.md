## VI-02 – Research Policy & Mode Config (Comp-Bio Only)

**Goal**
Make the “only published works + Semantic Scholar + our own math” constraint *enforceable*, not just a story in our heads.

**Rationale**
Right now, nothing in this runner prevents a future tool or agent from pulling in arbitrary datasets, web APIs, or simulating “wet-lab” data. That’s a governance hole relative to the spec we agreed on.

**Scope / Changes**

* Introduce a `research_policy.yaml` (or `bfts_config.yaml` extension) that defines:

  * Allowed data sources (Semantic Scholar, local PDFs, our sim outputs).
  * Disallowed sources (clinical EHR, proprietary datasets, arbitrary web scrapes).
  * Allowed tool classes (lit tools, modeling, sim, stats, writeup).
* Implement a **policy checker** that:

  * Wraps tool invocation (inside `Runner` or a thin wrapper here).
  * Rejects any tool marked with `requires_external_data=True` that is not whitelisted in the current policy.
* Add a `--research_mode` CLI flag:

  * `comp_bio_pubs_only` (default).
  * `unrestricted` (only for local debugging, explicitly logged).

**Acceptance Criteria**

1. New file `research_policy.yaml` is loaded at startup; if missing, run fails with a clear error.
2. Attempting to invoke a tool flagged as “external_data” in `comp_bio_pubs_only` mode:

   * Results in a hard error logged to `project_knowledge.md` as a `[CONSTRAINT]` entry.
   * The run terminates before the tool executes.
3. A small test harness (or unit test) demonstrates:

   * Same run passes in `unrestricted` mode and fails in `comp_bio_pubs_only` with identical idea JSON.
4. The active mode (`comp_bio_pubs_only` vs `unrestricted`) is recorded in:

   * `provenance_summary.md` and `hypothesis_trace.json` top-level metadata.

---

## VI-03 – Idea / Hypothesis Registry & Multi-Idea Discipline

**Goal**
Make hypotheses first-class and reproducible: every run must declare *which* idea and *which* hypothesis/experiment set it is executing.

**Rationale**
You already have `_ACTIVE_IDEA`, `_bootstrap_hypothesis_trace`, `update_hypothesis_trace`, and `--idea_idx`, but it’s still loose. There’s no canonical registry of ideas or strict linking between an idea file and the run.

**Scope / Changes**

* Introduce an `ideas/` directory under the project root with:

  * `ideas/idea_<id>.json` (canonical spec: Name, Short Hypothesis, Experiments, Tags, Status).
* On `--load_idea`:

  * Copy or symlink the idea file into `experiment_results/idea.json`.
  * Record `idea_id` and `idea_path` in `hypothesis_trace.json` and `provenance_summary.md`.
* Enforce that **every tool call that records sims/figures** must include:

  * `hypothesis_id`, `experiment_id` (already optional in several tools) or be explicitly marked as `unlinked`.

**Acceptance Criteria**

1. Every successful run writes `experiment_results/idea.json` and `hypothesis_trace.json` that both contain a matching `idea_id`.
2. A “health” check (new tool, e.g. `check_hypothesis_trace`) reports:

   * Number of sim runs and figures with valid `(hypothesis_id, experiment_id)`.
   * Number of “unlinked” artifacts (<= a configurable tolerance; default 0).
3. Running without `--load_idea` or with a non-existent idea path:

   * Fails fast before any expensive tools run.

---

## VI-04 – Literature Quality Gate Before Modeling/Sim

**Goal**
Stop the system from simulating before the literature grounding is structurally sound and references are verified to a minimum standard.

**Rationale**
You already have:

* `assemble_lit_data`
* `validate_lit_summary`
* `verify_references`
* Artifact types for lit summaries and verification tables.

But there’s no *gate* that says: “Lit is good enough, you may proceed.”

**Scope / Changes**

* Add a `check_lit_ready` tool that:

  * Reads `lit_summary.json` and the verification outputs.
  * Enforces minimal thresholds:

    * No required fields missing in lit summary.
    * At least X% of references have `status == "confirmed"` or similar.
    * At most Y “not found / unclear” entries (configurable).
* Modify any path that runs simulations/modeling from an idea to:

  * Require `check_lit_ready` to pass first (unless `--skip_lit_gate` is set).

**Acceptance Criteria**

1. `assemble_lit_data(..., run_verification=True)` followed by `check_lit_ready`:

   * Returns `status: "ready"` when:

     * Lit validator passes.
     * > = configurable threshold for confirmed references (e.g. 70%).
2. If `check_lit_ready` returns `status: "not_ready"`, any call to:

   * `run_biological_model`, `run_comp_sim`, `run_sensitivity_sweep`, `run_intervention_tests`
   * Fails with a descriptive “literature gate not satisfied” error unless `--skip_lit_gate` is set.
3. The outcome of `check_lit_ready` is logged in:

   * `project_knowledge.md` as `[DECISION]` with thresholds used.
   * `provenance_summary.md` under the “Literature” section (“Lit gate: READY / NOT READY”).

---

## VI-05 – Model Spec & Parameter Provenance Completeness Gate

**Goal**
Force every model used in sims to have a *complete, documented, and sourced* parameter ledger — no silent “free hyperparameter” usage.

**Rationale**
`_ensure_model_spec_and_params` already creates specs and param source CSVs, but default rows are labeled `free_hyperparameter`. That’s fine for bootstrapping but unacceptable for a “finished” run meant to be published.

**Scope / Changes**

* Add a `check_model_provenance(model_key)` tool that:

  * Reads `{model_key}_spec.yaml` and `{model_key}_param_sources.csv`.
  * Fails if any parameter referenced in the spec:

    * Has no row in the CSV; or
    * Has `source_type == "free_hyperparameter"` in final mode.
* Introduce a `--enforce_param_provenance` CLI flag (default `True` for “publishable” mode).

**Acceptance Criteria**

1. Before any call to `run_biological_model(model_key=...)` with `--enforce_param_provenance`:

   * `check_model_provenance(model_key)` is run automatically.
   * If any parameter is still `free_hyperparameter`, call fails with a list of missing params.
2. For models used in a “publishable” run:

   * `check_model_provenance` passes and writes a short summary to `provenance_summary.md` (which parameters are lit-derived vs fit vs scaling).
3. A unit test fixtures a dummy `model_key` with one missing/unsourced parameter and verifies:

   * `run_biological_model(..., enforce=True)` fails.
   * `run_biological_model(..., enforce=False)` passes but logs a `[FAILURE_PATTERN]` entry in `project_knowledge.md`.

---

## VI-06 – Simulation Plan & Budgeting (Plan vs Execution)

**Goal**
Move from “tools called ad hoc” to **explicit sim plans** with runtime budgets and coverage guarantees.

**Rationale**
You already have:

* `run_transport_batch` with manifest integration.
* `_generate_run_recipe` for transport runs.
  But there is no single, top-level concept of a *simulation plan* tied to an idea/hypothesis, nor any resource budgeting (number of sims, time, token ceilings).

**Scope / Changes**

* Create `experiment_results/sim_plan.json` with schema like:

  * Which simulations: baseline, sweeps, interventions.
  * Parameter ranges and seeds.
  * Expected artifact types.
  * Resource limits: max runs, max CPU time estimate, etc.
* Add a `plan_vs_execution` checker that:

  * Compares `sim_plan.json` vs `transport_runs/manifest.json` and sensitivity/intervention outputs.
  * Marks each plan entry as `pending / running / complete / failed / skipped`.

**Acceptance Criteria**

1. Running a “full” experiment from idea writes:

   * `sim_plan.json` *before* any heavy sim tools are called.
2. After sim execution, calling `plan_vs_execution` returns:

   * A JSON report with counts of `complete`, `failed`, `skipped`.
   * Coverage: at least 90% of planned combinations either `complete` or `failed` (no silent missing runs).
3. If the number of simulations exceeds a configured budget (e.g. 10,000 runs or some CPU estimate):

   * Execution halts and logs a `[CONSTRAINT]` entry suggesting plan trimming.

---

## VI-07 – Agent Budget & Safety Guardrails (Time, Tokens, Calls)

**Goal**
Prevent uncontrolled cost/time blowups when you let agents iterate (max cycles, runaway Semantic Scholar queries, massive sim sweeps).

**Rationale**
You already have:

* `--max_cycles`
* `--timeout`
  But there’s no per-tool/per-phase budget. A mis-prompted agent could still flood Semantic Scholar or spawn absurd numbers of sims.

**Scope / Changes**

* Add a `budgets` section either to `research_policy.yaml` or a separate `budgets.yaml`:

  * E.g. `max_semantic_scholar_calls`, `max_sim_runs`, `max_lit_results`, `max_tokens_llm_small`, `max_tokens_llm_big`.
* Wrap key tools (`SemanticScholarSearchTool`, `RunCompartmentalSimTool`, sweeps, interventions) with a small budget manager that:

  * Decrements counters and fails when limits are hit.
* Ensure every budget breach is recorded via `manage_project_knowledge(action="add", category="constraint", ...)`.

**Acceptance Criteria**

1. If a run tries to:

   * Call Semantic Scholar > `max_semantic_scholar_calls`, or
   * Launch > `max_sim_runs` individual sim runs,
     the system:
   * Aborts further calls to those tools with a clear error message.
2. A `check_budgets` tool reports:

   * Utilization of each budget for the current run (`used`, `remaining`, `limit`).
3. At least one end-to-end test demonstrates:

   * An intentionally over-ambitious sim plan getting cut off with budgets enforced and a clearly logged constraint.

---

## VI-08 – Health Dashboard for Manifests & Artifacts

**Goal**
Give yourself a **one-shot health report** of the experiment folder: missing files, untyped artifacts, orphaned sims, etc.

**Rationale**
You already have:

* `_scan_and_auto_update_manifest`
* `check_manifest`
* `repair_sim_outputs`
* `validate_per_compartment_outputs`
  But they’re separate knobs. You want a single “is this run clean enough to publish?” button.

**Scope / Changes**

* Introduce `generate_health_report` that:

  * Runs:

    * `check_project_state`
    * `check_manifest`
    * `read_transport_manifest`
    * `repair_sim_outputs` (optional dry-run)
    * `validate_per_compartment_outputs` on each sim dir
  * Aggregates results into `experiment_results/health_report.json` + small markdown summary.

**Acceptance Criteria**

1. `generate_health_report()` produces:

   * `health_report.json` with sections: `manifest`, `transport_runs`, `sims`, `lit`, `models`, `writeup`.
   * `health_report.md` summarizing key issues.
2. If critical issues exist (e.g. missing sim arrays, broken JSON, empty metrics):

   * `health_report.json["status"]` is `"not_ok"` with a machine-readable list of problems.
3. `run_writeup_task` in “publishable” mode must:

   * Check `health_report.status == "ok"` or log a `[FAILURE_PATTERN]` explaining why it proceeded despite issues (e.g. `--ignore_health`).

---

## VI-09 – Claim Graph, Hypothesis Trace, and Manuscript Consistency Check

**Goal**
Enforce that **every major claim in the manuscript** is:

* Represented in `claim_graph.json`.
* Backed by at least one experiment or metric in `hypothesis_trace.json`.

**Rationale**
You already have:

* `update_claim_graph`, `check_claim_graph`
* `update_hypothesis_trace`, `generate_provenance_summary`
  But there’s no cross-check tying claims → hypotheses → experiments → figures.

**Scope / Changes**

* Add `check_claim_consistency` tool that:

  * Reads `claim_graph.json`, `hypothesis_trace.json`, and `provenance_summary.md`.
  * For each claim:

    * Checks if it references at least one experiment/metric.
    * Flags claims with `status == "unlinked"` or no supporting experiments.
* Integrate this check before `run_writeup_task` completes:

  * If serious inconsistencies exist, write a prominent warning into the manuscript or block the “publishable” mode.

**Acceptance Criteria**

1. `check_claim_consistency` outputs:

   * A list of claims with `support_status: "ok" | "weak" | "missing"`.
2. A run with missing support for any non-trivial claim:

   * Sets overall status to `"not_ready_for_publication"`.
3. In a clean run:

   * All `claim_graph` entries used in the abstract/introduction have at least one `sim_run` or `metric` in `hypothesis_trace.json`.

---

## VI-10 – Human-in-the-Loop Checkpoints (Where It Actually Matters)

**Goal**
Make `--human_in_the_loop` *real*, not just a flag — introduce **critical checkpoints** where you explicitly review and approve plans.

**Rationale**
The CLI exposes `--human_in_the_loop`, but this code doesn’t yet plug it into tooling. Right now, agents can decide to spend a lot of compute or adopt a sketchy hypothesis without you reviewing.

**Scope / Changes**

Define a small set of **blocking checkpoints** when `human_in_the_loop` is true:

1. After literature assembly + verification:

   * Present top-N key papers and parameter candidates (from `parameter_source_table`).
2. Before first heavy sim sweep / batch:

   * Show `sim_plan.json` summary and estimated budget usage.
3. Before final writeup:

   * Show `health_report.md` and `claim_consistency` summary.

At each checkpoint, an interactive function (or simple `y/n` prompt in CLI mode) must be called; if you decline, the system logs a pivot via `log_strategic_pivot`.

**Acceptance Criteria**

1. With `--human_in_the_loop`:

   * The system pauses at least at the three checkpoints above and requires explicit approval.
2. Refusing at any checkpoint:

   * Aborts downstream tasks.
   * Logs a `[DECISION]` or `[STRATEGIC PIVOT]` entry in `project_knowledge.md`.
3. With `--human_in_the_loop` **off**:

   * The run bypasses checkpoints but still records their would-be decisions in a dry-run section of `health_report.json` for auditability.
