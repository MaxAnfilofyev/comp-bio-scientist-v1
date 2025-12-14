"""initial_schema

Revision ID: 27707676918b
Revises: 
Create Date: 2025-12-13 21:57:20.188907

"""
from typing import Sequence, Union
from alembic import op, context
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '27707676918b'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    sql = """
    -- ============================================
    -- Claim Ledger + SearchRun pipeline (SQLite)
    -- WITH project_id integrated cleanly
    -- ============================================

    -- Recommended connection-time settings (execute once per connection/session)
    PRAGMA journal_mode = WAL;
    PRAGMA foreign_keys = ON;
    PRAGMA busy_timeout = 5000;

    -- ============================================
    -- 0) Schema migration tracker (optional but recommended)
    -- ============================================
    CREATE TABLE IF NOT EXISTS schema_migration (
      migration_id   TEXT PRIMARY KEY,
      applied_at     TEXT NOT NULL
    );

    -- ============================================
    -- 1) Policy snapshots (so scoring/promotions are reproducible)
    -- ============================================
    CREATE TABLE IF NOT EXISTS policy_snapshot (
      policy_version            TEXT PRIMARY KEY,          -- e.g., "v1.0"
      created_at                TEXT NOT NULL,             -- ISO8601
      policy_yaml               TEXT NOT NULL,             -- full policy config (string)
      policy_hash               TEXT NOT NULL              -- sha256(policy_yaml)
    );

    -- ============================================
    -- 2) Projects (unit of a research program / manuscript)
    -- ============================================
    CREATE TABLE IF NOT EXISTS project (
      project_id                TEXT PRIMARY KEY,          -- e.g. "PRJ_SNc_Energetics_2025"
      title                     TEXT NOT NULL,
      description               TEXT,
      status                    TEXT NOT NULL DEFAULT 'active', -- 'active','archived','abandoned','published'
      owner                     TEXT,                      -- human/PI identifier
      created_at                TEXT NOT NULL,
      updated_at                TEXT NOT NULL,
      policy_version            TEXT NOT NULL,
      FOREIGN KEY (policy_version) REFERENCES policy_snapshot(policy_version)
    );

    CREATE INDEX IF NOT EXISTS idx_project_status ON project(status);
    CREATE INDEX IF NOT EXISTS idx_project_updated_at ON project(updated_at);

    -- ============================================
    -- 3) Canonical work identity (dedupe by DOI)  [GLOBAL, not project-scoped]
    -- ============================================
    CREATE TABLE IF NOT EXISTS work (
      doi                       TEXT PRIMARY KEY,          -- canonical normalized DOI
      title                     TEXT,
      year                      INTEGER,
      venue                     TEXT,                      -- journal/conference name
      publisher                 TEXT,
      article_type_raw          TEXT,                      -- as observed from metadata/source
      article_type_norm         TEXT,                      -- normalized enum-like string (optional)
      is_peer_reviewed          INTEGER NOT NULL DEFAULT 1, -- 1=true, 0=false/unknown
      indexing_json             TEXT,                      -- JSON: {pubmed:true, scopus:false,...}
      core_work_id              TEXT,                      -- CORE identifier if available
      crossref_work_id          TEXT,                      -- Crossref identifier if available
      full_text_available       INTEGER NOT NULL DEFAULT 0, -- 1/0
      full_text_hash            TEXT,                      -- sha256 of stored full text (if you store it)
      full_text_source          TEXT,                      -- e.g. "CORE"
      retraction_status         TEXT NOT NULL DEFAULT 'unknown', -- 'not_retracted','retracted','unknown'
      eoc_status                TEXT NOT NULL DEFAULT 'unknown', -- 'none','eoc','unknown'
      created_at                TEXT NOT NULL,
      updated_at                TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_work_year ON work(year);
    CREATE INDEX IF NOT EXISTS idx_work_venue ON work(venue);
    CREATE INDEX IF NOT EXISTS idx_work_publisher ON work(publisher);

    -- ============================================
    -- 4) Search runs (one query execution / retrieval event)  [PROJECT-SCOPED]
    -- ============================================
    CREATE TABLE IF NOT EXISTS search_run (
      search_run_id             TEXT PRIMARY KEY,          -- e.g. "SR_2025_12_13_001"
      project_id                TEXT NOT NULL,
      provider                  TEXT NOT NULL,             -- "CORE"
      query_text                TEXT NOT NULL,
      created_at                TEXT NOT NULL,
      policy_version            TEXT NOT NULL,
      filters_json              TEXT NOT NULL,             -- JSON describing applied filters
      result_count_total        INTEGER NOT NULL DEFAULT 0,
      top_k_stored              INTEGER NOT NULL DEFAULT 0,
      notes                     TEXT,
      FOREIGN KEY (project_id) REFERENCES project(project_id) ON DELETE CASCADE,
      FOREIGN KEY (policy_version) REFERENCES policy_snapshot(policy_version)
    );

    CREATE INDEX IF NOT EXISTS idx_search_run_project_created_at ON search_run(project_id, created_at);
    CREATE INDEX IF NOT EXISTS idx_search_run_project ON search_run(project_id);

    -- ============================================
    -- 5) Candidates (work as returned in a specific search run)  [PROJECT-INFERRED via search_run]
    -- ============================================
    CREATE TABLE IF NOT EXISTS candidate (
      candidate_id              TEXT PRIMARY KEY,          -- e.g. "CAND_SR_..._0001"
      search_run_id             TEXT NOT NULL,
      doi                       TEXT NOT NULL,
      rank_in_results           INTEGER NOT NULL,
      retrieval_score_raw       REAL,                      -- from provider, if any
      features_json             TEXT,                      -- JSON: {relevance:..., directness:..., ...}
      composite_score           REAL,                      -- computed using policy/scoring version
      scoring_version           TEXT NOT NULL DEFAULT 'v1', -- scoring logic version
      policy_version            TEXT NOT NULL,
      shortlist_status          TEXT NOT NULL DEFAULT 'none', -- 'none','shortlisted','rejected'
      shortlist_reason          TEXT,
      created_at                TEXT NOT NULL,
      FOREIGN KEY (search_run_id) REFERENCES search_run(search_run_id) ON DELETE CASCADE,
      FOREIGN KEY (doi) REFERENCES work(doi) ON DELETE CASCADE,
      FOREIGN KEY (policy_version) REFERENCES policy_snapshot(policy_version),
      UNIQUE (search_run_id, doi)                           -- dedupe within a run
    );

    CREATE INDEX IF NOT EXISTS idx_candidate_run ON candidate(search_run_id);
    CREATE INDEX IF NOT EXISTS idx_candidate_doi ON candidate(doi);
    CREATE INDEX IF NOT EXISTS idx_candidate_score ON candidate(search_run_id, composite_score DESC);
    CREATE INDEX IF NOT EXISTS idx_candidate_shortlist ON candidate(search_run_id, shortlist_status);

    -- ============================================
    -- 6) Claims (thin canonical ledger objects)  [PROJECT-SCOPED]
    -- ============================================
    CREATE TABLE IF NOT EXISTS claim (
      claim_id                  TEXT PRIMARY KEY,          -- e.g. "CLM_000123"
      project_id                TEXT NOT NULL,
      module                    TEXT NOT NULL,             -- e.g. "Morphology"
      statement                 TEXT NOT NULL,
      claim_type                TEXT NOT NULL,             -- enum-like string
      strength_target           TEXT NOT NULL DEFAULT 'medium', -- 'strong','medium','weak'
      evidence_required         TEXT NOT NULL DEFAULT 'citation', -- 'citation','analysis','both','none'
      priority                  TEXT NOT NULL DEFAULT 'P1', -- 'P0','P1','P2'
      status                    TEXT NOT NULL DEFAULT 'proposed', -- 'proposed','in_scope','out_of_scope','retired'
      disposition               TEXT NOT NULL DEFAULT 'undecided', -- 'keep','weaken','rephrase','move_to_discussion','drop','undecided'
      canonical_for_manuscript  INTEGER NOT NULL DEFAULT 0, -- PI-controlled gate
      created_by                TEXT NOT NULL,
      created_at                TEXT NOT NULL,
      updated_at                TEXT NOT NULL,
      etag                      INTEGER NOT NULL DEFAULT 1, -- optimistic concurrency
      policy_version            TEXT NOT NULL,
      FOREIGN KEY (project_id) REFERENCES project(project_id) ON DELETE CASCADE,
      FOREIGN KEY (policy_version) REFERENCES policy_snapshot(policy_version)
    );

    CREATE INDEX IF NOT EXISTS idx_claim_project_module ON claim(project_id, module);
    CREATE INDEX IF NOT EXISTS idx_claim_project_status ON claim(project_id, status);
    CREATE INDEX IF NOT EXISTS idx_claim_project_priority ON claim(project_id, priority);
    CREATE INDEX IF NOT EXISTS idx_claim_project_canonical ON claim(project_id, canonical_for_manuscript);

    -- Helpful uniqueness to prevent accidental duplicates within a project (optional; keep if you like)
    -- If you're iterating heavily and expect near-duplicates, you can drop this constraint.
    CREATE UNIQUE INDEX IF NOT EXISTS uidx_claim_project_statement
    ON claim(project_id, statement);

    -- ============================================
    -- 7) Claim supports (promoted evidence links, with anchors)  [PROJECT-SCOPED]
    -- ============================================
    CREATE TABLE IF NOT EXISTS claim_support (
      support_id                TEXT PRIMARY KEY,          -- e.g. "SUP_000987"
      project_id                TEXT NOT NULL,
      claim_id                  TEXT NOT NULL,
      doi                       TEXT NOT NULL,
      support_type              TEXT NOT NULL,             -- 'citation','primary_measurement','meta_analysis','systematic_review','analysis_result','figure'
      verification_status       TEXT NOT NULL DEFAULT 'unverified', -- 'verified','partial','failed','unverified'
      anchor_excerpt            TEXT,                      -- <=25 words enforced by app/policy
      anchor_location_json      TEXT,                      -- JSON: {section:"...", paragraph:..., offsets:{start:..,end:..}}
      promoted_from_candidate_id TEXT,                     -- optional provenance link
      promotion_reason          TEXT NOT NULL,
      created_by                TEXT NOT NULL,
      created_at                TEXT NOT NULL,
      policy_version_at_promotion TEXT NOT NULL,
      FOREIGN KEY (project_id) REFERENCES project(project_id) ON DELETE CASCADE,
      FOREIGN KEY (claim_id) REFERENCES claim(claim_id) ON DELETE CASCADE,
      FOREIGN KEY (doi) REFERENCES work(doi) ON DELETE CASCADE,
      FOREIGN KEY (promoted_from_candidate_id) REFERENCES candidate(candidate_id) ON DELETE SET NULL,
      FOREIGN KEY (policy_version_at_promotion) REFERENCES policy_snapshot(policy_version),
      UNIQUE (claim_id, doi)                               -- no duplicate supports per claim
    );

    CREATE INDEX IF NOT EXISTS idx_support_project_claim ON claim_support(project_id, claim_id);
    CREATE INDEX IF NOT EXISTS idx_support_project_doi ON claim_support(project_id, doi);
    CREATE INDEX IF NOT EXISTS idx_support_project_verification ON claim_support(project_id, verification_status);

    -- ============================================
    -- 8) Claim gaps (structured “what’s missing”)  [PROJECT-SCOPED]
    -- ============================================
    CREATE TABLE IF NOT EXISTS claim_gap (
      gap_id                    TEXT PRIMARY KEY,          -- e.g. "GAP_000321"
      project_id                TEXT NOT NULL,
      claim_id                  TEXT NOT NULL,
      gap_type                  TEXT NOT NULL,             -- 'no_source_found','needs_analysis','overstated','conflicting_sources',...
      recommendation            TEXT NOT NULL,
      assigned_to               TEXT,
      resolved                  INTEGER NOT NULL DEFAULT 0,
      resolution_note           TEXT,
      created_by                TEXT NOT NULL,
      created_at                TEXT NOT NULL,
      resolved_at               TEXT,
      FOREIGN KEY (project_id) REFERENCES project(project_id) ON DELETE CASCADE,
      FOREIGN KEY (claim_id) REFERENCES claim(claim_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_gap_project_claim ON claim_gap(project_id, claim_id);
    CREATE INDEX IF NOT EXISTS idx_gap_project_resolved ON claim_gap(project_id, resolved);

    -- ============================================
    -- 9) Audit events (append-only change history)  [PROJECT-SCOPED]
    -- ============================================
    CREATE TABLE IF NOT EXISTS audit_event (
      event_id                  TEXT PRIMARY KEY,          -- e.g. "EVT_2025_12_13_..."
      project_id                TEXT NOT NULL,
      entity_type               TEXT NOT NULL,             -- 'claim','claim_support','claim_gap','candidate','work','search_run','project'
      entity_id                 TEXT NOT NULL,
      action                    TEXT NOT NULL,             -- 'create','update','delete','promote','verify','resolve_gap',...
      actor_role                TEXT NOT NULL,             -- 'archivist','pi','reviewer','modeler','analyst'
      actor_id                  TEXT NOT NULL,             -- optional stable identifier
      created_at                TEXT NOT NULL,
      policy_version            TEXT,                      -- policy in effect for this event (if relevant)
      before_json               TEXT,                      -- JSON snapshot or minimal diff
      after_json                TEXT,                      -- JSON snapshot or minimal diff
      reason                    TEXT,                      -- short justification
      refs_json                 TEXT,                      -- JSON: pointers (search_run_id, candidate_id, artifact_ids, etc.)
      FOREIGN KEY (project_id) REFERENCES project(project_id) ON DELETE CASCADE,
      FOREIGN KEY (policy_version) REFERENCES policy_snapshot(policy_version)
    );

    CREATE INDEX IF NOT EXISTS idx_audit_project_entity ON audit_event(project_id, entity_type, entity_id);
    CREATE INDEX IF NOT EXISTS idx_audit_project_created ON audit_event(project_id, created_at);

    -- ============================================
    -- 10) Integrity views (audits)  [PROJECT-AWARE]
    -- ============================================

    -- Canonical claims missing any verified supports (should block "ready to ship")
    CREATE VIEW IF NOT EXISTS v_project_canonical_claims_missing_verified_support AS
    SELECT
      c.project_id,
      c.claim_id,
      c.module,
      c.statement,
      c.evidence_required,
      c.strength_target
    FROM claim c
    LEFT JOIN claim_support s
      ON s.claim_id = c.claim_id
     AND s.project_id = c.project_id
     AND s.verification_status = 'verified'
    WHERE c.canonical_for_manuscript = 1
    GROUP BY c.project_id, c.claim_id
    HAVING COUNT(s.support_id) = 0;

    -- Supports whose linked works violate core integrity rules (should be prevented by gates)
    CREATE VIEW IF NOT EXISTS v_project_supports_with_work_integrity_issues AS
    SELECT
      s.project_id,
      s.support_id,
      s.claim_id,
      s.doi,
      w.full_text_available,
      w.retraction_status,
      w.eoc_status
    FROM claim_support s
    JOIN work w ON w.doi = s.doi
    WHERE w.full_text_available = 0
       OR w.retraction_status = 'retracted';

    -- Search runs with unusually high stored candidate volume (a canary for "Archivist firehose")
    CREATE VIEW IF NOT EXISTS v_project_search_runs_high_candidate_volume AS
    SELECT
      sr.project_id,
      sr.search_run_id,
      sr.created_at,
      sr.query_text,
      sr.result_count_total,
      sr.top_k_stored,
      COUNT(c.candidate_id) AS candidates_stored
    FROM search_run sr
    LEFT JOIN candidate c ON c.search_run_id = sr.search_run_id
    GROUP BY sr.project_id, sr.search_run_id
    HAVING COUNT(c.candidate_id) > 500;
    """
    bind = op.get_bind()
    if context.is_offline_mode():
        op.execute(sql)
    else:
        if bind.dialect.name == 'sqlite':
            raw_conn = None
            # Try to get the underlying DBAPI connection
            if hasattr(bind, 'driver_connection'):
                raw_conn = bind.driver_connection
            elif hasattr(bind, 'connection'):
                # Legacy or different wrapper
                raw_conn = bind.connection
            elif hasattr(bind, '_dbapi_connection'):
                # Internal attribute suggested by error
                raw_conn = bind._dbapi_connection
            
            if raw_conn and hasattr(raw_conn, 'executescript'):
                raw_conn.executescript(sql)
            else:
                op.execute(sql)
        else:
            op.execute(sql)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("""
    DROP VIEW IF EXISTS v_project_search_runs_high_candidate_volume;
    DROP VIEW IF EXISTS v_project_supports_with_work_integrity_issues;
    DROP VIEW IF EXISTS v_project_canonical_claims_missing_verified_support;
    DROP TABLE IF EXISTS audit_event;
    DROP TABLE IF EXISTS claim_gap;
    DROP TABLE IF EXISTS claim_support;
    DROP TABLE IF EXISTS claim;
    DROP TABLE IF EXISTS candidate;
    DROP TABLE IF EXISTS search_run;
    DROP TABLE IF EXISTS work;
    DROP TABLE IF EXISTS project;
    DROP TABLE IF EXISTS policy_snapshot;
    DROP TABLE IF EXISTS schema_migration;
    """)
