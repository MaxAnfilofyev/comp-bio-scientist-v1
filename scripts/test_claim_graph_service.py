"""
Test script for ClaimGraphService - Phase 1 integration
"""

import sys
sys.path.insert(0, '/Users/maxa/AI-Scientist-v2')

from sqlalchemy import create_engine
from ai_scientist.database.claim_graph_service import ClaimGraphService

def main():
    # Connect to database
    engine = create_engine("sqlite:///ai_scientist.sqlite")
    service = ClaimGraphService(engine)
    
    print("=" * 80)
    print("Claim Graph Service - Integration Test")
    print("=" * 80)
    
    # Test 1: Create a graph
    print("\n1. Creating claim graph for test project...")
    project_id = "TEST_PROJECT_001"
    graph_id = service.create_graph(project_id)
    print(f"   ✓ Created graph: {graph_id}")
    
    # Test 2: Create claims
    print("\n2. Creating test claims...")
    from sqlalchemy import text
    with engine.connect() as conn:
        # Create thesis claim
        conn.execute(
            text("""
                INSERT OR REPLACE INTO claim 
                (claim_id, project_id, module, statement, claim_type, status, created_by, policy_version)
                VALUES (:cid, :pid, :module, :stmt, :claim_type, :status, :created_by, :policy)
            """),
            {
                "cid": "thesis",
                "pid": project_id,
                "module": "test",
                "stmt": "Axonal transport is essential for neuronal function.",
                "claim_type": "thesis",
                "status": "proposed",
                "created_by": "test_script",
                "policy": "test_v1"
            }
        )
        
        # Create sub-claim
        conn.execute(
            text("""
                INSERT OR REPLACE INTO claim 
                (claim_id, project_id, module, statement, claim_type, status, created_by, policy_version)
                VALUES (:cid, :pid, :module, :stmt, :claim_type, :status, :created_by, :policy)
            """),
            {
                "cid": "c1",
                "pid": project_id,
                "module": "test",
                "stmt": "Mitochondria are transported along axons.",
                "claim_type": "hypothesis",
                "status": "proposed",
                "created_by": "test_script",
                "policy": "test_v1"
            }
        )
        conn.commit()
    print("   ✓ Created 2 claims")
    
    # Test 3: Add nodes to graph
    print("\n3. Adding nodes to graph...")
    thesis_node_id = service.add_node("thesis")
    c1_node_id = service.add_node("c1", parent_node_id=thesis_node_id)
    print(f"   ✓ Thesis node: {thesis_node_id}")
    print(f"   ✓ Sub-claim node: {c1_node_id}")
    
    # Test 4: Add supports
    print("\n4. Adding support references...")
    service.add_support(c1_node_id, "citation", "10.1016/j.neuron.2020.01.001", "Example citation")
    service.add_support(c1_node_id, "artifact", "experiments/mitochondria_tracking.csv", "Tracking data")
    print("   ✓ Added 2 supports")
    
    # Test 5: Query claim with evidence
    print("\n5. Querying claim with evidence...")
    claim_data = service.get_claim_with_evidence("c1")
    print(f"   Claim: {claim_data['claim_text']}")
    print(f"   Support count: {claim_data['support_count']}")
    print(f"   Supports:")
    for sup in claim_data['supports']:
        print(f"     - {sup['support_type']}: {sup['reference']}")
    
    # Test 6: Find unsupported claims
    print("\n6. Finding unsupported claims...")
    unsupported = service.find_unsupported_claims(project_id, min_support=3)
    print(f"   Found {len(unsupported)} claims needing more support:")
    for claim in unsupported:
        print(f"     - {claim['claim_id']}: {claim['support_count']} supports")
    
    # Test 7: Export to JSON
    print("\n7. Exporting to JSON format...")
    json_data = service.export_to_json(project_id)
    print(f"   ✓ Exported {len(json_data)} claims")
    import json
    print(json.dumps(json_data, indent=2))
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
