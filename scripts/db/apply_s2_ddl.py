import sqlite3
import os

DB_PATH = "ai_scientist.sqlite"

def table_has_column(cursor, table, column):
    try:
        cols = [r[1] for r in cursor.execute(f"PRAGMA table_info({table})").fetchall()]
        return column in cols
    except:
        return False

def apply_ddl():
    if not os.path.exists(DB_PATH):
        print(f"DB not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        # 1. Update work table
        print("Updating work table...")
        if not table_has_column(c, "work", "s2_paper_id"):
            c.execute("ALTER TABLE work ADD COLUMN s2_paper_id TEXT")
        if not table_has_column(c, "work", "is_open_access"):
             c.execute("ALTER TABLE work ADD COLUMN is_open_access INTEGER NOT NULL DEFAULT 0")
        if not table_has_column(c, "work", "open_access_pdf_url"):
             c.execute("ALTER TABLE work ADD COLUMN open_access_pdf_url TEXT")
        if not table_has_column(c, "work", "license"):
             c.execute("ALTER TABLE work ADD COLUMN license TEXT")
        
        # Check Project ID issue
        print("Checking project_id columns...")
        if not table_has_column(c, "search_run", "project_id"):
             print("Adding project_id to search_run")
             c.execute("ALTER TABLE search_run ADD COLUMN project_id TEXT")
        
        if not table_has_column(c, "claim", "project_id"):
             print("Adding project_id to claim")
             c.execute("ALTER TABLE claim ADD COLUMN project_id TEXT")
        
        # 2. Create search_round
        print("Creating search_round table...")
        c.execute("""
        CREATE TABLE IF NOT EXISTS search_round (
          search_round_id     TEXT PRIMARY KEY,
          search_run_id       TEXT NOT NULL,
          claim_id            TEXT,                 -- nullable for global exploration
          round_index         INTEGER NOT NULL,
          provider            TEXT NOT NULL,         -- "S2"
          base_query          TEXT NOT NULL,
          compiled_query      TEXT NOT NULL,
          filters_json        TEXT NOT NULL,
          summary_json        TEXT NOT NULL,
          next_action         TEXT NOT NULL,
          created_at          TEXT NOT NULL,
          FOREIGN KEY (search_run_id) REFERENCES search_run(search_run_id) ON DELETE CASCADE,
          FOREIGN KEY (claim_id) REFERENCES claim(claim_id) ON DELETE CASCADE,
          UNIQUE(search_run_id, claim_id, round_index)
        );
        """)
        # Note: SQLITE unique constraint with NULLs can be tricky in older versions but normally NULL != NULL. 
        # User requested UNIQUE(search_run_id, COALESCE(claim_id, 'GLOBAL'), round_index)
        # We can implement this via a unique index on expression if sqlite supports it, or just standard unique and handle nulls manually?
        # Standard SQLite: multiple NULLs are allowed in UNIQUE columns.
        # User wants strict uniqueness even for NULL claim_id.
        # Let's try to create the index as requested.
        try:
             c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_search_round_uniq ON search_round(search_run_id, IFNULL(claim_id, 'GLOBAL'), round_index)")
        except Exception as e:
             print(f"Warning creating complex index: {e}. Fallback to standard index.")
             c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_search_round_std ON search_round(search_run_id, claim_id, round_index)")

        # 3. Create candidate_quality_check
        print("Creating candidate_quality_check table...")
        c.execute("""
        CREATE TABLE IF NOT EXISTS candidate_quality_check (
          check_id        TEXT PRIMARY KEY,
          candidate_id    TEXT NOT NULL,
          claim_id        TEXT,  -- nullable for global checks
          check_type      TEXT NOT NULL,
          verdict         TEXT NOT NULL, -- PASS|FAIL|UNKNOWN|NA
          policy_id       TEXT NOT NULL,
          policy_hash     TEXT,
          details_json    TEXT,
          executed_by     TEXT NOT NULL,
          executed_at     TEXT NOT NULL,
          FOREIGN KEY (candidate_id) REFERENCES candidate(candidate_id) ON DELETE CASCADE,
          FOREIGN KEY (claim_id) REFERENCES claim(claim_id) ON DELETE CASCADE
        );
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_cqc_candidate_claim_type ON candidate_quality_check(candidate_id, IFNULL(claim_id,'GLOBAL'), check_type, executed_at)")

        # 4. Create candidate_decision
        print("Creating candidate_decision table...")
        c.execute("""
        CREATE TABLE IF NOT EXISTS candidate_decision (
          decision_id     TEXT PRIMARY KEY,
          candidate_id    TEXT NOT NULL,
          claim_id        TEXT, -- nullable = global reject
          outcome         TEXT NOT NULL, -- PROMOTED|REJECTED|HOLD|ELIGIBLE_SUPPORT|SELECTED_AS_SUPPORT
          basis_json      TEXT NOT NULL, -- pointers to latest checks + short note
          policy_id       TEXT NOT NULL,
          decided_by      TEXT NOT NULL,
          decided_at      TEXT NOT NULL,
          FOREIGN KEY (candidate_id) REFERENCES candidate(candidate_id) ON DELETE CASCADE,
          FOREIGN KEY (claim_id) REFERENCES claim(claim_id) ON DELETE CASCADE
        );
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_cd_candidate_claim_time ON candidate_decision(candidate_id, IFNULL(claim_id,'GLOBAL'), decided_at)")

        # 5. Create work_fulltext_cache
        print("Creating work_fulltext_cache table...")
        c.execute("""
        CREATE TABLE IF NOT EXISTS work_fulltext_cache (
          cache_id        TEXT PRIMARY KEY,
          doi             TEXT,
          s2_paper_id     TEXT,
          source          TEXT NOT NULL, -- "S2_OPEN_ACCESS_PDF"
          content_url     TEXT NOT NULL,
          content_sha256  TEXT NOT NULL,
          content_bytes   INTEGER NOT NULL,
          extracted_text  TEXT,          -- truncated (policy cap)
          extracted_sha256 TEXT,
          retrieved_at    TEXT NOT NULL,
          FOREIGN KEY (doi) REFERENCES work(doi) ON DELETE SET NULL
        );
        """)

        conn.commit()
        print("Schema applied successfully.")

    except Exception as e:
        print(f"Error applying schema: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    apply_ddl()
