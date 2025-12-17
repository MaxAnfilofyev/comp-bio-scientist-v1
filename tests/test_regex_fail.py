import re

# Exact Regex from evidence_control_loop.py
forbidden_patterns = [
    r'(?i)"pubmed\s+pmc"\s*\[sb\]',
    r'(?i)"pubmed\s+pmc"', 
    r'(?i)test\s+"pubmed\s+pmc"',
    r'(?i)english\s*\[la\]', 
    r'(?i)pmc\s*\[filter\]',
    r'(?i)hasabstract', 
    r'(?i)sort\s*=', 
    r'(?i)retmax\s*=', 
    r'(?i)retstart\s*='
]

# Exact String from DB Inspection (Search Run ID: SR_20251216_7a0a82b1)
# Query: (("adenosine triphosphate"[tiab] OR ATP[tiab]) AND (axon*[tiab] OR axonal[tiab] OR "axonal transport"[tiab] OR "Axons"[mh]) AND (diffusion[tiab] OR diffusive[tiab] OR "passive diffusion"[tiab]) NOT (glia[tiab] OR retinal[tiab]) AND "pubmed pmc"[sb]) AND "pubmed pmc"[sb] english[la]
# Note: This is the Full Query (Base + Policy).
# The Base Part is logic inside parens?
# Wait. `run_orchestrator_real.py` compiles keys.
# The `query_text` in DB is the `compiled` query.
# If `query_text` = `(Base) AND Policy`.
# And Base = `... AND "pubmed pmc"[sb]`.
# Then `Base` has it.

dirty_string = '("adenosine triphosphate"[tiab] OR ATP[tiab]) AND (axon*[tiab] OR axonal[tiab] OR "axonal transport"[tiab] OR "Axons"[mh]) AND (diffusion[tiab] OR diffusive[tiab] OR "passive diffusion"[tiab]) NOT (glia[tiab] OR retinal[tiab]) AND "pubmed pmc"[sb]'

print(f"Testing String: {dirty_string}")

cleaned = dirty_string
for pat in forbidden_patterns:
    print(f"Applying Pat: {pat}")
    match = re.search(pat, cleaned)
    if match:
        print(f"  MATCH FOUND: '{match.group(0)}'")
        cleaned = re.sub(pat, "", cleaned)
        print(f"  Result: {cleaned}")
    else:
        print("  No Match")

print(f"\nFinal: {cleaned}")
if '"pubmed pmc"[sb]' in cleaned:
    print("FAIL: Still dirty.")
else:
    print("PASS: Cleaned.")
