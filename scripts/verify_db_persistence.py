
import sqlite3
import glob
import os

# Find most recent DB
db_files = glob.glob("ai_scientist_manual_search_*.sqlite")
latest_db = max(db_files, key=os.path.getctime)
print(f"Checking DB: {latest_db}")

conn = sqlite3.connect(latest_db)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("\n--- Work Fulltext Cache ---")
rows = cursor.execute("SELECT doi, source, length(content) as size, retrieved_at FROM work_fulltext_cache").fetchall()
for r in rows:
    print(dict(r))

print("\n--- Claim Support ---")
rows = cursor.execute("SELECT support_id, doi, verification_status, anchor_excerpt, anchor_location_json FROM claim_support").fetchall()
for r in rows:
    print(dict(r))

print("\n--- Quality Checks (New Types) ---")
rows = cursor.execute("SELECT check_type, verdict, details_json FROM candidate_quality_check WHERE check_type IN ('anchor_extraction', 'claim_entailment_llm')").fetchall()
for r in rows:
    print(dict(r))

conn.close()
