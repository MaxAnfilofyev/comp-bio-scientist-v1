## 1. Firehose Isolation & Context Scoping

1. **No Direct History Exposure**

   * No agent ever receives the full raw project history or global log as context.
   * Every agent invocation uses a *filtered subset* of project state.

2. **Context = Artifacts, Not Chats**

   * Inputs to agents are derived from structured artifacts (and their summaries), not from unstructured chat transcripts or raw logs.
   * Raw logs, if stored, are only used to *create or update* artifacts, not fed directly into working agents.

3. **Task-Local Context Only**

   * Each agent invocation receives only:

     * A brief project overview.
     * The subset of artifacts relevant to the specific task/module.
   * There is a hard upper bound on context size per call, enforced by the system (not by convention).

---

## 2. Artifact-Centric State Model

4. **Typed Artifacts as Single Source of Truth**

   * All persistent project knowledge exists as typed artifacts (e.g., `project_brief`, `paper_note`, `model_spec`, `analysis_result`, `integrated_memo`, `manuscript_section`, etc.).
   * There is no other persistent state that can change project logic or conclusions.

5. **Minimal Required Fields**

   * Each artifact has, at minimum:

     * `id`
     * `type`
     * `version`
     * `parent_version` (for updates)
     * `status` (e.g., draft/canonical/deprecated/etc.)
     * `module` (which part of the project it belongs to)
     * `summary` (short, human-readable)
     * `content` (full details)
     * `metadata` (creator, timestamps, tags)
   * These fields are consistently populated and non-optional where defined.

6. **Links & Usage Traceability**

   * Artifacts can reference other artifacts (e.g., memo cites paper_notes, model_spec refers to analysis_results).
   * For any major artifact (e.g., integrated memo, manuscript section), you can trace which underlying artifacts it depends on.

---

## 3. Role- & Agent-Specific Views

7. **Role-Based Read Scope**

   * For each agent type, there is a defined policy specifying which artifact types it may *read* by default.
   * This policy is enforced by the system; agents cannot “opt around” it.

8. **Role-Based Write Scope**

   * For each agent type, there is a defined policy specifying which artifact types it may *create or modify* (via new versions).
   * Agents cannot create or update artifacts outside their allowed types.

9. **Context View Specifications**

   * For each agent type, there exists a formal “context view spec” describing:

     * Which artifact types it receives.
     * Which fields/variants (summary vs. full content).
     * How many artifacts of each type (max K).
   * The system uses these specs to assemble inputs; they are not only documentation.

---

## 4. Summaries as First-Class Citizens

10. **Hierarchical Summarization**

    * Large or complex areas of the project have:

      * Fine-grained artifacts (e.g., individual paper_notes).
      * Module-level integrated memos.
      * A short project-level summary.
    * Higher-level work (e.g., global design, writing) primarily consumes summaries and memos, not raw artifacts.

11. **Summaries Always Present**

    * Every artifact that might be used in context includes a `summary` field that is:

      * Short.
      * Up-to-date relative to `content`.
    * It is possible to construct a context view using only summaries when needed.

12. **Mandatory Compression Steps**

    * When a threshold of new or updated artifacts in a module is exceeded, a summarization/integration step (e.g., updating an integrated memo) is required before major cross-module decisions or rewrites proceed.

---

## 5. Versioning, Change Tracking, and Canonicality

13. **No In-Place Overwrites**

    * Updating an artifact always creates a *new version* with a `parent_version` reference.
    * Previous versions remain available for inspection.

14. **Explicit Canonical Status**

    * At any time, for each key artifact type and module, there is a clearly identifiable *canonical* version.
    * Non-canonical versions are explicitly marked (draft, alternative, deprecated, needs_rebase).

15. **Change Logs & Deltas**

    * Significant updates (especially to core artifacts like `project_brief`, `integrated_memo`, `model_spec`) are accompanied by machine- or human-readable change descriptions (what changed and why).
    * It is possible to answer: “What changed between version X and version Y?” without diffing raw text manually.

16. **Version-Aware Task Specs**

    * Each task specification records the exact versions of its input artifacts.
    * When the task completes, the system can check whether those inputs have changed since the task started.

---

## 6. Asynchronous Safety & Consistency

17. **Input Version Validation**

    * When a task produces new artifacts, the system validates:

      * Whether the referenced input artifacts are still at the same versions.
    * If not, the output is flagged (e.g., `needs_rebase`) rather than silently treated as consistent.

18. **No Silent Promotion**

    * New or updated artifacts are not automatically treated as canonical.
    * There is a clear promotion step (by PI or other governance logic) to mark an artifact as canonical.

19. **Fork Detection**

    * The system can detect when multiple alternative versions of a “core” artifact (e.g., model_spec, integrated_memo) exist.
    * Such forks are visible and not silently hidden; resolution/merge can be scheduled as an explicit task.

---

## 7. Guardrails Against Context Bloat

20. **Hard Bounds on Context Size**

    * Each agent type has a maximum allowed context size (in whatever units are relevant to the model, e.g., model input length).
    * The system enforces these bounds; if candidate artifacts exceed them, some are pruned according to defined priorities.

21. **Top-K Artifact Selection**

    * When more relevant artifacts exist than can fit in context:

      * The system selects a limited number (Top-K) based on an explicit relevance strategy (recency, tags, references, etc.).
    * This selection is deterministic or at least explainable, not arbitrary.

22. **Summaries-First Strategy**

    * By default, agents receive summaries instead of full contents where possible.
    * Full content is only included when the role/spec explicitly requires it (e.g., detailed paper review).

23. **Module-Level Scoping**

    * Context is scoped to one or a small number of modules by default.
    * Cross-module context is only assembled when a task is explicitly cross-cutting (e.g., global consistency review).

---

## 8. Drift & Quality Control

24. **Alignment to Project Brief**

    * Core artifacts (integrated memos, model_specs, manuscript_sections) can be systematically checked against the `project_brief` for:

      * Scope creep.
      * Contradictions with core goals/constraints.
    * Misalignments can be flagged by a critic / reviewer process.

25. **Assumption Consistency Checks**

    * There is a mechanism to identify conflicting assumptions across modules:

      * e.g., “Module A assumes X, Module B assumes not-X.”
    * These conflicts are recorded in review/critique artifacts, not just buried in logs.

26. **Staleness Detection**

    * When a canonical artifact changes, dependent artifacts (integrated memos, manuscript sections) are marked as potentially stale (e.g., `needs_update`).
    * The system can list which artifacts depend on outdated inputs.

---

## 9. Artifact Lifecycle & Cleanup

27. **Deprecation of Unused Artifacts**

    * Artifacts that are not referenced by any canonical or in-use artifacts for a defined period can be marked as deprecated.
    * Deprecated artifacts are excluded from default retrieval but remain accessible for audit.

28. **Regular Consolidation**

    * There is a periodic process (manual or automated tasking) that:

      * Merges redundant/overlapping artifacts.
      * Cleans up or reclassifies low-quality or unused drafts.

29. **Stable “Spine” of the Project**

    * Despite many artifacts, there is always a small, clearly identified set of “spine” artifacts:

      * `project_brief` (canonical)
      * per-module `integrated_memo` (canonical)
      * current high-level `project_summary`
    * These form the minimum consistent view of the project.

---

## 10. Observability & Explainability

30. **Traceable Agent Calls**

    * For every agent invocation, it is possible to see:

      * Which artifacts (IDs, versions, types) were used as context.
      * Which artifacts were produced or updated as a result.
    * This trace is available for audit and debugging.

31. **Reconstructable Reasoning Chain**

    * Given a particular manuscript section or integrated memo, it is possible to:

      * Trace back to the underlying artifacts (paper_notes, analysis_results, model_specs).
      * See which tasks and agents contributed to them.

32. **Human-Understandable State Snapshots**

    * At any point, it is possible to produce a human-readable snapshot of:

      * The current project question and scope.
      * The current hypotheses and models.
      * The main supporting evidence and open questions.
    * This snapshot is generated from canonical artifacts, not ad hoc reconstruction.