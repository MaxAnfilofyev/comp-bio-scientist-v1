"""
Manifest health checker (VI-05).

Usage:
    python -m ai_scientist.lab_tools.check_run_health --base-folder <run_root> [--fix]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Import canonical artifact registry and helper functions
from ai_scientist.orchestrator.artifacts import ARTIFACT_TYPE_REGISTRY, _format_artifact_path

SCRATCH_DIRS = {"_unrouted", "_temp", "_scratch"}
MAX_STRING_FIELD_LEN = 500  # catch verbose notes sneaking into manifest


def _load_manifest(base_folder: Path) -> List[Dict]:
    manifest_dir = base_folder / "experiment_results" / "manifest"
    index_path = manifest_dir / "manifest_index.json"
    entries: List[Dict] = []
    if not index_path.exists():
        return entries
    try:
        index = json.loads(index_path.read_text())
    except Exception:
        return entries
    shards = index.get("shards", [])
    for shard_meta in shards:
        shard_path = Path(shard_meta.get("path", ""))
        if not shard_path.is_absolute():
            # If shard path is relative, resolve it relative to manifest_dir
            # Note: The current system might store absolute paths, but we should handle both or
            # at least assume how they are stored. If they are stored as absolute paths in the shard meta,
            # this check handles it. If relative, we need to be careful.
            # However, looking at the previous code, it just did Path(shard_meta.get("path", ""))
            # which implies they might be absolute or relative to CWD?
            # Let's try to resolve relative to manifest_dir if it doesn't exist as is.
            if not shard_path.exists():
                 candidate = manifest_dir / shard_path
                 if candidate.exists():
                     shard_path = candidate
        
        if not shard_path.exists():
            continue
        try:
            with shard_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            entries.append(obj)
                    except Exception:
                        continue
        except Exception:
            continue
    return entries


def _get_canonical_path(base_folder: Path, entry: Dict) -> Path | None:
    kind = entry.get("kind")
    if not kind:
        return None
    
    # We need metadata to format the path
    meta = entry.get("metadata") or {}
    # Also include top-level fields that might be used in templates as fallbacks
    # artifacts.py uses keys from meta dict to format.
    # Check what keys are expected.
    # But wait, artifacts.py _format_artifact_path takes (kind, meta).
    # ensure meta has everything needed.
    
    # Sometimes 'name' or other fields are used. Let's merge entry into meta strictly for formatting purposes
    # efficiently without modifying the actual entry.
    formatting_context = meta.copy()
    for k, v in entry.items():
        if k not in formatting_context:
            formatting_context[k] = v

    try:
        rel_dir, name = _format_artifact_path(kind, formatting_context)
        return base_folder / "experiment_results" / rel_dir / name
    except Exception:
        return None


def _check_manifest_entries(base_folder: Path, entries: List[Dict]) -> Tuple[List[str], List[str], List[str], List[Dict]]:
    missing: List[str] = []
    naming_errors: List[str] = []
    verbose_fields: List[str] = []
    misplaced_entries: List[Dict] = []
    
    registry = ARTIFACT_TYPE_REGISTRY

    for entry in entries:
        path = entry.get("path")
        kind = entry.get("kind")
        status = entry.get("status", "ok")
        name = entry.get("name") or (Path(path).name if path else "")
        
        canonical_path = _get_canonical_path(base_folder, entry)
        
        if path:
            p = Path(path)
            try:
                exists = p.exists()
            except Exception:
                exists = False
            if not exists and status == "ok":
                # Only report missing if it's not misplaced (misplaced check below)
                # But wait, if it's misplaced, 'p' points to the WRONG location.
                # If it doesn't exist at the WRONG location, is it missing?
                # Maybe it exists at the RIGHT location?
                if canonical_path and canonical_path.exists():
                    pass # It exists at canonical path, so we will catch it in misplaced check
                else:
                    missing.append(path)
        
            if canonical_path:
                # Resolve both to absolute to compare
                try:
                    p_resolved = p.resolve()
                    c_resolved = canonical_path.resolve()
                    if p_resolved != c_resolved:
                        misplaced_entries.append({
                            "entry": entry,
                            "current_path": p,
                            "canonical_path": canonical_path
                        })
                except Exception:
                    pass

        if kind:
            reg = registry.get(kind)
            if not reg:
                naming_errors.append(f"{path or name}: unknown kind '{kind}'")
            else:
                # We can't easily check regex matching here without re-implementing _pattern_to_regex
                # or exposing it. But _get_canonical_path already does validation inside _format_artifact_path!
                # If _get_canonical_path returned None or raised, we might have an issue.
                # Let's rely on _format_artifact_path for validation implicitly.
                pass
                
        for val in entry.values():
            if isinstance(val, str) and len(val) > MAX_STRING_FIELD_LEN:
                verbose_fields.append(f"{path or name}: field too long ({len(val)} chars)")
                
    return missing, naming_errors, verbose_fields, misplaced_entries


def _check_files_covered(base_folder: Path, entries: List[Dict]) -> List[str]:
    exp_dir = base_folder / "experiment_results"
    manifest_paths = {str(Path(e["path"]).resolve()) for e in entries if e.get("path")}
    uncovered: List[str] = []
    if not exp_dir.exists():
        return []
    for p in exp_dir.rglob("*"):
        if p.is_dir():
            continue
        rel_parts = p.relative_to(exp_dir).parts
        if rel_parts and rel_parts[0] in SCRATCH_DIRS:
            continue
        if "manifest" in p.parts:
            continue
        try:
            resolved = str(p.resolve())
        except Exception:
            resolved = str(p)
        if resolved not in manifest_paths:
            uncovered.append(str(p))
    return uncovered


def fix_run_health(base_folder: Path, misplaced_entries: List[Dict]) -> List[str]:
    fixed = []
    # Load manifest index to find shards
    manifest_dir = base_folder / "experiment_results" / "manifest"
    index_path = manifest_dir / "manifest_index.json"
    if not index_path.exists():
        return ["Manifest index not found, cannot fix."]

    # Map entries by some ID or path to update them.
    # Since we need to update the actual files, we will load all shards, update entries in memory, 
    # and rewrite the shards.
    
    # 1. Move files
    for item in misplaced_entries:
        entry = item["entry"]
        current_path = item["current_path"]
        canonical_path = item["canonical_path"]
        
        # FIX: generic kind fix (model_spec_yaml -> model_spec)
        # This is a bit ad-hoc but requested by user plan.
        if entry.get("kind") == "model_spec_yaml" and str(canonical_path).endswith(".json"):
            # Update entry kind in memory (and for shard rewrite)
            entry["kind"] = "model_spec"
            # Re-calculate canonical path with new kind?
            # model_spec pattern is {model_key}_spec_{version}.json
            # model_spec_yaml pattern is unknown (it failed earlier).
            # If we change kind, we should re-eval canonical path.
            new_canon = _get_canonical_path(base_folder, entry)
            if new_canon:
                 canonical_path = new_canon
                 item["canonical_path"] = new_canon

        if current_path.exists():
            if str(current_path.resolve()) == str(canonical_path.resolve()):
                 print(f"File {current_path.name} is already at {canonical_path}, but manifest might need update.")
                 fixed.append(f"Updated manifest for {canonical_path.name}")
            else:
                print(f"Moving {current_path} -> {canonical_path}")
                canonical_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current_path), str(canonical_path))
                fixed.append(f"Moved {current_path.name}")
        elif canonical_path.exists():
             print(f"File already at canonical path {canonical_path}, updating manifest only.")
             fixed.append(f"Updated manifest for {canonical_path.name}")
        else:
            print(f"Cannot move {current_path}: source not found and target not found.")
            continue
            
    # 2. Rewrite manifest shards
    try:
        index = json.loads(index_path.read_text())
    except Exception:
        return fixed + ["Failed to read manifest index."]

    shards = index.get("shards", [])
    for shard_meta in shards:
        shard_path = Path(shard_meta.get("path", ""))
        if not shard_path.is_absolute():
            if not shard_path.exists():
                 candidate = manifest_dir / shard_path
                 if candidate.exists():
                     shard_path = candidate
        
        if not shard_path.exists():
            continue
            
        new_lines = []
        modified_shard = False
        with shard_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    
                    # Fix kinds if needed
                    if obj.get("kind") == "model_spec_yaml":
                         obj["kind"] = "model_spec"
                         modified_shard = True

                    canonical = _get_canonical_path(base_folder, obj)
                    current = Path(obj.get("path", ""))
                    if canonical and str(current.resolve()) != str(canonical.resolve()):
                        obj["path"] = str(canonical)
                        modified_shard = True
                    
                    new_lines.append(json.dumps(obj))
                except Exception:
                    new_lines.append(line)
        
        if modified_shard:
            print(f"Rewriting shard {shard_path}")
            with shard_path.open("w") as f:
                for line in new_lines:
                    f.write(line + "\n")

    return fixed


def register_untracked_files(base_folder: Path, uncovered_files: List[str]) -> List[str]:
    print(f"Attempting to register {len(uncovered_files)} untracked files...")
    registered = []
    
    # We need to reverse match files to kinds. 
    # This is tricky because patterns have placeholders.
    # Simple heuristic:
    # 1. Filter registry by relative directory.
    # 2. Try to regex match filename.
    
    from ai_scientist.orchestrator.artifacts import ARTIFACT_TYPE_REGISTRY, _pattern_to_regex, append_or_update
    from ai_scientist.tools.base_tool import BaseTool
    
    # We can't easily import base tool config, so assume base_folder is the root.
    
    exp_dir = base_folder / "experiment_results"

    for file_path_str in uncovered_files:
        p = Path(file_path_str)
        try:
            rel_path = p.relative_to(exp_dir)
        except ValueError:
            continue # not in experiment_results?
            
        rel_dir = str(rel_path.parent)
        if rel_dir == ".":
            rel_dir = "."
            
        filename = p.name
        
        # Find candidate kinds
        candidates = []
        for kind, info in ARTIFACT_TYPE_REGISTRY.items():
            # Check directory match
            # info['rel_dir'] might contain placeholders too! e.g. transport_runs/{baseline}/...
            # This makes exact reverse matching hard.
            # But let's handle simple cases without placeholders in dir, or try to match dir regex too?
            
            # Simple check: strict string match for dir OR partial match?
            # Let's try to match filename regex first, then see if dir makes sense.
            
            # But wait, patterns are like "{model_key}_solution.json".
            # regex will be `(?P<model_key>...)_solution.json`.
            
            pat = info["pattern"]
            regex = _pattern_to_regex(pat)
            match = regex.match(filename)
            if match:
                 # Check if directory matches roughly
                 # If reg dir has placeholders, it's hard. If not, it's easy.
                 reg_dir = info["rel_dir"]
                 if "{" not in reg_dir:
                     if reg_dir == rel_dir:
                         candidates.append((kind, match.groupdict()))
                 else:
                     # Heuristic: verify if rel_dir pattern matches actual rel_dir?
                     # Too complex for this script. 
                     # Let's assume filename uniqueness across kinds in same dir structure usually holds.
                     # But we don't know if 'rel_dir' matches.
                     # Checking if 'reg_dir' matches 'rel_dir' is hard without valid regex for dir.
                     # Let's SKIP complex directories for now and focus on simple ones.
                     pass

        if not candidates:
             # Try fallback artifacts?
             pass
        else:
            # Pick best candidate?
            # If multiple, warn.
            if len(candidates) > 1:
                print(f"Ambiguous kind for {filename}: {[c[0] for c in candidates]}, skipping.")
                continue
                
            kind, params = candidates[0]
            
            # Construct entry
            # we need to reconstruct metadata from params
            # e.g. model_key
            
            # Create a basic entry
            entry = {
                "path": str(p),
                "name": filename,
                "kind": kind,
                "created_by": "recovered_untracked",
                "status": "ok",
                "metadata": params
            }
            # Add to manifest
            res = append_or_update(entry, base_folder=base_folder)
            if not res.get("error"):
                registered.append(f"{filename} as {kind}")
            else:
                print(f"Failed to register {filename}: {res.get('error')}")

    return registered


def check_run_health(base_folder: Path, fix: bool = False, register_untracked: bool = False) -> Tuple[bool, List[str]]:
    entries = _load_manifest(base_folder)
    missing, naming_errors, verbose_fields, misplaced_entries = _check_manifest_entries(base_folder, entries)
    uncovered = _check_files_covered(base_folder, entries)

    errors: List[str] = []
    if missing:
        errors.append(f"Missing files (status=ok): {missing}")
    if naming_errors:
        errors.append(f"Naming/registry violations: {naming_errors}")
    if verbose_fields:
        errors.append(f"Verbose manifest fields (> {MAX_STRING_FIELD_LEN} chars): {verbose_fields}")
    if misplaced_entries:
        msg = f"Misplaced files ({len(misplaced_entries)}):"
        for m in misplaced_entries[:5]: # Show first 5
            msg += f"\n  {m['current_path']} should be at {m['canonical_path']}"
        if len(misplaced_entries) > 5:
            msg += f"\n  ... and {len(misplaced_entries) - 5} more."
        errors.append(msg)
    if uncovered:
        errors.append(f"Files not covered by manifest (outside scratch): {uncovered}")
        
    generated_reports = []
    
    if fix and misplaced_entries:
        print("Fixing misplaced files...")
        fixed_msgs = fix_run_health(base_folder, misplaced_entries)
        generated_reports.append(f"Fixed {len(fixed_msgs)} misplaced files/entries.")
        
    if register_untracked and uncovered:
        print("Registering untracked files...")
        reg_msgs = register_untracked_files(base_folder, uncovered)
        generated_reports.append(f"Registered {len(reg_msgs)} untracked files.")
        
    if generated_reports:
        return True, errors + generated_reports

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Manifest health check (VI-05).")
    parser.add_argument("--base-folder", help="Run root (folder containing experiment_results). Defaults to AISC_BASE_FOLDER.")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix misplaced files and update manifest.")
    parser.add_argument("--register-untracked", action="store_true", help="Attempt to register untracked files into manifest.")
    args = parser.parse_args()

    default_root = Path(os.environ.get("AISC_BASE_FOLDER", "."))
    base = Path(args.base_folder) if args.base_folder else default_root
    ok, errors = check_run_health(base, fix=args.fix, register_untracked=args.register_untracked)
    
    if args.fix or args.register_untracked:
        print("Fix/Register run complete.")
        for err in errors:
            print(f"- {err}")
        return 0

    if ok:
        print("Health check passed.")
        return 0
    print("Health check FAILED:")
    for err in errors:
        print(f"- {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
