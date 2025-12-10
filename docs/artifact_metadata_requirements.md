# Artifact Metadata Requirements

This reference documents the canonical metadata that every typed artifact must carry. Agents and helpers in `ai_scientist/orchestrator` treat these attributes as the single source of truth; they correspond to principles 4–6, 13–16, and 30–32 in `docs/system_design_principles.md`. Always consult this page when reserving, creating, or updating artifacts.

## Required Fields

1. `id` — a unique identifier for the artifact (e.g., `lit_summary_main_v2`, `claim_graph_main_2025-10-01`). This may include the kind plus semantic suffixes, and it ties entries together when versions change.
2. `type` — the artifact kind (must match one of the keys listed in `ai_scientist/orchestrator/artifacts.py` such as `lit_summary_main`, `transport_sim_json`, etc.).
3. `version` — a monotonically increasing string or integer that describes the current variant. When an artifact is first created, start at `v1` or a timestamped label; each update must supply a new `version`.
4. `parent_version` — the `version` value of the artifact this entry evolved from. If this is the first creation, set to `null` or omit (but the field must exist). This enforces principle 13 and enables change tracking.
5. `status` — one of `draft`, `pending`, `canonical`, `deprecated`, `needs_review`, etc. Updates never overwrite a canonical artifact; instead they set a new `version` with the appropriate status and keep the old entry accessible.
6. `module` — the subsystem the artifact belongs to (`literature`, `modeling`, `analysis`, `writeup`, `plumbing`, etc.). This enables module scoping (Principles 7–9, 20–23) and artifact selection.
7. `summary` — a short, up-to-date human description of what the artifact contains. Summaries should be kept current with the artifact’s `content` field to support the summaries-first strategy (principles 10–12, 32).
8. `content` — the full details, either embedded in the manifest (for small artifacts) or reachable via the reserved path (for larger JSON/CSV/arrays). Always ensure the manifest entry references the file produced.
9. `metadata` — a dictionary describing provenance, timestamps, tags, authors, and any other structured attributes that aid tracing (Principles 6, 19, 30). Include at least `created_by`, `created_at`, `tags`, and `tools_used` when available.

## Supporting Conventions

- **Canonical promotion**: Only a PI or governance helper may mark an artifact `canonical`. New versions start as `draft` or `pending` and require a promotion step documented with `append_manifest` or `write_pi_notes` before being treated as the spine.
- **Change summaries**: Every update should include a short `change_summary` string explaining what changed between `parent_version` and the new version. This makes Principle 15 auditable.
- **Dependencies and tracing**: Use `metadata.dependencies` to list other artifact IDs/versions. It must be clear which underlying artifacts support each higher-level entry, fulfilling Principles 6 and 31.
- **Verification proof**: When writing `_verification.md` files or proofs of work, link them via the `metadata.verification_id` so downstream readers can easily confirm assumptions.

## Agent Guidance

Agents must: 

- Consult this document before reserving artifacts via `reserve_typed_artifact` or `reserve_and_register_artifact`.
- Populate every required field; manifests without `version`, `summary`, or `module` should be rejected during reservation (future helper enforcement will validate this).
- Reference the `metadata` entries when describing why an artifact exists or how it relates to other artifacts (Principle 30).

The plan in `docs/compliance_plan.md` tracks when the registry helper validates this metadata and when prompts remind agents of these expectations.
