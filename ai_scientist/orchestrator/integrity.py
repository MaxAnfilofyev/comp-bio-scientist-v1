from typing import Any, Dict
from ai_scientist.utils.manifest import find_manifest_entry
from ai_scientist.tools.base_tool import BaseTool

def check_dependency_staleness(artifact_name_or_path: str) -> Dict[str, Any]:
    """
    Check if an artifact is stale by comparing its 'tools_used' (inputs) against 
    the current canonical versions in the manifest.
    
    Returns:
        {
            "stale": bool,
            "artifact": ...,
            "stale_dependencies": [
                {"name": ..., "used_version": ..., "current_scanned_version": ...}
            ]
        }
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    entry = find_manifest_entry(artifact_name_or_path, base_folder=exp_dir)
    
    if not entry:
        return {"error": f"Artifact not found: {artifact_name_or_path}"}
        
    meta = entry.get("metadata", {})
    # Handle potential double nesting from reserve_and_register_artifact
    # Structure might be entry['metadata']['metadata']['tools_used']
    tools_used = meta.get("tools_used")
    if not tools_used:
        inner_meta = meta.get("metadata")
        if isinstance(inner_meta, dict):
            tools_used = inner_meta.get("tools_used")
    tools_used = tools_used or entry.get("tools_used", [])
    # Or sometimes just a list of IDs.
    # The requirement says "uses artifact metadata (tools_used, provenance)". 
    # Current implementation of 'tools_used' in agents might be sparse. 
    # We will assume that if an artifact lists a dependency with a version, we use that.
    
    stale_deps = []
    
    for dep in tools_used:
        if not isinstance(dep, dict):
            continue
            
        dep_path = dep.get("path")
        dep_name = dep.get("name") # or os.path.basename(dep_path)
        if not dep_path and not dep_name:
            continue
            
        lookup_target = dep_path or dep_name
        current_dep_entry = find_manifest_entry(lookup_target, base_folder=exp_dir)
        print(f"DEBUG: Looking up '{lookup_target}' -> Found: {current_dep_entry is not None}")
        
        if not current_dep_entry:
            continue
            
        # To check staleness, we want to know if there is a NEWER version of this artifact kind/module.
        dep_kind = current_dep_entry.get("kind") or current_dep_entry.get("metadata", {}).get("kind")
        dep_module = current_dep_entry.get("module") or current_dep_entry.get("metadata", {}).get("module")
        
        from ai_scientist.orchestrator.artifacts import _get_latest_artifact_entry
        latest_entry = _get_latest_artifact_entry(dep_kind, dep_module)
        
        if not latest_entry:
            continue
            
        used_ver = dep.get("version")
        latest_ver = latest_entry.get("version") or latest_entry.get("metadata", {}).get("version")
        
        # If used version is not the latest, it's stale.
        # We assume lexically sortable versions (v1, v2, ...)
        if used_ver and latest_ver and used_ver != latest_ver:
            print(f"DEBUG: Stale! used={used_ver}, latest={latest_ver}")
            stale_deps.append({
                "name": latest_entry.get("name"),
                "path": latest_entry.get("path"),
                "used_version": used_ver,
                "current_version": latest_ver
            })
            
    return {
        "stale": len(stale_deps) > 0,
        "name": entry.get("name"),
        "version": meta.get("version"),
        "stale_dependencies": stale_deps
    }
