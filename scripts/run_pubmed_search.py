
import os
import re
import json
import sqlite3
from sqlalchemy import create_engine, text
from ai_scientist.database.config import DATABASE_URL
from ai_scientist.medline_gate import MedlinePmcEvidenceGate, PubMedSearchRequest

# DB Setup to ensure hydration
MIGRATION_FILE = "ai_scientist/database/migrations/versions/27707676918b_initial_schema.py"

# Override DB path to avoid schema conflicts with existing stale DB
DB_PATH = "ai_scientist_manual_search.sqlite"
DATABASE_URL = f"sqlite:///{DB_PATH}"

def get_ddl():
    with open(MIGRATION_FILE, "r") as f:
        content = f.read()
    match = re.search(r'sql = """(.*?)"""', content, re.DOTALL)
    if not match:
        raise ValueError("Could not extract SQL from migration file")
    return match.group(1)

def setup_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    print(f"Setting up fresh DB at {DATABASE_URL}...")
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Apply Schema
    ddl = get_ddl()
    conn.executescript(ddl)
    
    # 2. Insert Dummy Policy and Project for FK constraints
    cursor = conn.cursor()
    
    # Policy
    cursor.execute("""
    INSERT OR IGNORE INTO policy_snapshot (policy_version, created_at, policy_yaml, policy_hash)
    VALUES ('medline_gate_v1', datetime('now'), 'dummy_yaml', 'dummy_hash')
    """)
    
    # Project
    cursor.execute("""
    INSERT OR IGNORE INTO project (project_id, title, status, created_at, updated_at, policy_version)
    VALUES ('manual_test', 'Manual Test Project', 'active', datetime('now'), datetime('now'), 'medline_gate_v1')
    """)
    
    conn.close()
    print("DB setup complete.")

def run_search():
    # Pass custom DB URL
    from ai_scientist.evidence_service import EvidenceService
    svc = EvidenceService(db_url=DATABASE_URL)
    gate = MedlinePmcEvidenceGate(evidence_service=svc) 
    
    query = "(ATP diffusion) AND (neuron OR axon OR dendrite)"
    print(f"Executing search for: {query}")
    
    req = PubMedSearchRequest(
        project_id="manual_test",
        query=query,
        retmax=20 # Request small batch for manual test
    )
    
    resp = gate.pubmed_search(req)
    
    print(f"Search Run ID: {resp.search_run_id}")
    print(f"Total Count: {resp.count_total}")
    print(f"PMIDs found: {len(resp.pmids)}")
    print(f"PMIDs: {resp.pmids}")
    
    # Save result
    output_file = "pubmed_search_results.json"
    with open(output_file, "w") as f:
        f.write(resp.model_dump_json(indent=2))
    print(f"Results saved to {output_file}")
    
    # Also fetch metadata for these PMIDs to verify full loop?
    # User just asked to "run the pubmed search and save the result".
    # I'll stick to search result.

if __name__ == "__main__":
    setup_db()
    run_search()
