import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

db_url = "sqlite:///orchestrator_trace.sqlite"
engine = sa.create_engine(db_url)
Session = sessionmaker(bind=engine)
session = Session()

# Inspect search_run table
print(">>> Inspecting search_run table query_text:")
try:
    results = session.execute(sa.text("SELECT search_run_id, query_text FROM search_run")).fetchall()
    for row in results:
        print(f"ID: {row[0]}")
        print(f"Query: {row[1]}")
        print("-" * 20)
except Exception as e:
    print(f"Error reading search_run: {e}")

# Inspect claim_search_round table (summary might contain query)
print("\n>>> Inspecting claim_search_round table summary_json -> query:")
try:
    import json
    results = session.execute(sa.text("SELECT round_id, summary_json FROM claim_search_round")).fetchall()
    for row in results:
        rid = row[0]
        summary = json.loads(row[1])
        q = summary.get("query", "N/A")
        print(f"Round: {rid}")
        print(f"Query: {q}")
        print("-" * 20)
except Exception as e:
    print(f"Error reading round summary: {e}")
