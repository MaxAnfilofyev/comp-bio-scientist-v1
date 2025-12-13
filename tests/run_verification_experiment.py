
from ai_scientist.tools.reference_verification import ReferenceVerificationTool
from pathlib import Path
import json

def run_experiment_verification():
    base_path = Path("/Users/maxa/AI-Scientist-v2/experiments/20251212_1658_A_Topological_Tipping_Point_Explains_the_Selective/experiment_results")
    
    # Try to find lit_summary.json
    lit_path = base_path / "literature" / "lit_summary.json"
    if not lit_path.exists():
        lit_path = base_path / "lit_summary.json"
        
    print(f"Using lit_summary at: {lit_path}")
    if not lit_path.exists():
        print("Error: lit_summary.json not found.")
        return

    # Set output dir to same place
    output_dir = base_path / "literature"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running verification with CrossRef fallback...")
    tool = ReferenceVerificationTool()
    result = tool.use_tool(
        lit_path=str(lit_path),
        output_dir=str(output_dir),
        max_results=5,
        score_threshold=0.65
    )
    
    print("--- Verification Complete ---")
    print(json.dumps(result, indent=2))
    
    # Check for CrossRef improvements
    json_out = output_dir / "lit_reference_verification.json"
    if json_out.exists():
        data = json.loads(json_out.read_text())
        crossref_hits = [r for r in data if "via CrossRef" in r.get("notes", "")]
        print(f"\nReferences found via CrossRef: {len(crossref_hits)}")
        for i, hit in enumerate(crossref_hits[:5]):
             print(f"{i+1}. {hit['title'][:50]}... (DOI: {hit.get('doi')})")

if __name__ == "__main__":
    run_experiment_verification()
