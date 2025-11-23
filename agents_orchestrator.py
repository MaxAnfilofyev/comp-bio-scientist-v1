# pyright: reportMissingImports=false
import argparse
import json
import os
import os.path as osp
from pathlib import Path
from datetime import datetime, timedelta
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import ast
import difflib
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

try:
    from agents.types import RunResult
except ImportError:
    if TYPE_CHECKING:
        from agents.types import RunResult  # type: ignore  # noqa: F401
    else:
        class RunResult:  # minimal stub
            def __init__(self, output=None, error=None, status=None):
                self.output = output
                self.error = error
                self.status = status

# --- Framework Imports ---
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from agents import Agent, Runner, function_tool, ModelSettings

# --- Underlying Tool Imports ---
from ai_scientist.tools.lit_data_assembly import LitDataAssemblyTool
from ai_scientist.tools.lit_validator import LitSummaryValidatorTool
from ai_scientist.tools.compartmental_sim import RunCompartmentalSimTool
from ai_scientist.tools.biological_plotting import RunBiologicalPlottingTool
from ai_scientist.tools.biological_model import RunBiologicalModelTool
from ai_scientist.tools.sensitivity_sweep import RunSensitivitySweepTool
from ai_scientist.tools.intervention_tester import RunInterventionTesterTool
from ai_scientist.tools.validation_compare import RunValidationCompareTool
from ai_scientist.tools.biological_stats import RunBiologicalStatsTool
from ai_scientist.tools.graph_builder import BuildGraphsTool
from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool
from ai_scientist.tools.sim_postprocess import SimPostprocessTool
from ai_scientist.tools.graph_diagnostics import GraphDiagnosticsTool
from ai_scientist.tools.claim_graph import ClaimGraphTool
from ai_scientist.tools.claim_graph_checker import ClaimGraphCheckTool
from ai_scientist.tools.manuscript_reader import ManuscriptReaderTool
from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.repair_sim_outputs import RepairSimOutputsTool
from ai_scientist.tools.per_compartment_validator import validate_per_compartment_outputs as validate_per_compartment_outputs_internal
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_biological_interpretation import interpret_biological_results
from ai_scientist.utils.notes import ensure_note_files, read_note_file, write_note_file
from ai_scientist.utils.pathing import resolve_output_path
from ai_scientist.utils.health import log_missing_or_corrupt

# --- Configuration & Helpers ---

def parse_args():
    p = argparse.ArgumentParser(description="ADCRT: Argument-Driven Computational Research Team")
    p.add_argument("--load_idea", required=True, help="Path to idea JSON.")
    p.add_argument("--model", default="gpt-5-mini-2025-08-07", help="LLM model id.")
    p.add_argument("--max_cycles", type=int, default=30, help="Max Orchestrator turns.")
    p.add_argument("--timeout", type=float, default=1800.0, help="Timeout in seconds (default: 30 mins).") 
    p.add_argument("--base_folder", default=None, help="Existing experiment directory to restart from (overrides timestamped creation).")
    p.add_argument("--resume", action="store_true", help="Don't overwrite existing experiment folder.")
    p.add_argument("--idea_idx", type=int, default=0, help="Index if the idea file contains a list of ideas.")
    p.add_argument("--input", default=None, help="Initial input message to the PI agent.")
    p.add_argument("--human_in_the_loop", action="store_true", help="Enable interactive mode where agents ask for confirmation before expensive tasks.")
    return p.parse_args()

def _fill_output_dir(output_dir: Optional[str]) -> str:
    """
    Resolve output dir to the run-specific folder using sanitized resolver.
    """
    path, _, _ = resolve_output_path(subdir=None, name="", allow_quarantine=False, unique=False)
    base = str(path)
    if output_dir:
        if os.path.isabs(output_dir):
            return output_dir
        # Anchor relative to experiment_results
        anchored, _, _ = resolve_output_path(subdir=None, name=output_dir, allow_quarantine=False, unique=False)
        return str(anchored)
    return base


def _fill_figure_dir(output_dir: Optional[str]) -> str:
    """
    Resolve a figure output dir, preferring the run-root figures/ folder, sanitized.
    """
    if output_dir and os.path.isabs(output_dir):
        return output_dir
    name = output_dir or "figures"
    path, _, _ = resolve_output_path(subdir=None, name=name, allow_quarantine=False, unique=False)
    return str(path)


def _build_metadata_for_compat(entry_type: Optional[str], annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if entry_type:
        meta["type"] = entry_type
    return meta


def _normalize_manifest_entry(entry: Dict[str, Any], fallback_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    def _add_annotation(raw: Any, annotations_list: List[Dict[str, Any]]):
        if not isinstance(raw, dict):
            return
        cleaned = {k: v for k, v in raw.items() if k not in {"type", "annotations"}}
        if cleaned and cleaned not in annotations_list:
            annotations_list.append(cleaned)

    path = entry.get("path") or fallback_path or entry.get("name")
    if not path:
        return None
    name = os.path.basename(entry.get("name") or path)
    base_type = entry.get("type")
    annotations: List[Dict[str, Any]] = []

    meta = entry.get("metadata")
    if isinstance(meta, dict):
        base_type = meta.get("type", base_type)
        if not annotations:
            _add_annotation(meta, annotations)
        nested = meta.get("annotations")
        if isinstance(nested, list):
            for ann in nested:
                _add_annotation(ann, annotations)
    elif isinstance(meta, list):
        for m in meta:
            if isinstance(m, dict):
                if m.get("type") and not base_type:
                    base_type = m.get("type")
                if not annotations:
                    _add_annotation(m, annotations)
                nested = m.get("annotations")
                if isinstance(nested, list):
                    for ann in nested:
                        _add_annotation(ann, annotations)

    existing_annotations = entry.get("annotations")
    if isinstance(existing_annotations, list):
        for ann in existing_annotations:
            _add_annotation(ann, annotations)

    normalized = {
        "name": name,
        "path": path,
        "type": base_type,
        "annotations": annotations,
        "timestamp": entry.get("timestamp")
    }
    compat_meta = _build_metadata_for_compat(base_type, annotations)
    if compat_meta:
        normalized["metadata"] = compat_meta
    return normalized


def _load_manifest_map(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path) as f:
            data = json.load(f)
    except Exception:
        return {}

    manifest_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict):
                norm = _normalize_manifest_entry(val, fallback_path=key)
                if norm:
                    manifest_map[norm["path"]] = norm
        return manifest_map

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                norm = _normalize_manifest_entry(item)
                if norm:
                    manifest_map[norm["path"]] = norm
    return manifest_map

# --- WATCHER: Auto-Scan Logic ---
def _scan_and_auto_update_manifest(exp_dir: Path) -> List[str]:
    """
    Background Watcher: Scans experiment_results for orphaned files (files not in manifest).
    Adds them with inferred types and 'auto_watcher' annotation.
    Returns list of added filenames.
    """
    manifest_path = exp_dir / "file_manifest.json"
    manifest_map = _load_manifest_map(manifest_path)
    added_files = []
    
    # We normalized existing keys in _load_manifest_map, but we need to check against 
    # the exact string format used in the manifest (usually absolute paths or relative to execution).
    # To be safe, we check if the path_str matches any key in manifest_map.
    
    for root, _, files in os.walk(exp_dir):
        for name in files:
            if name == "file_manifest.json":
                continue
            
            full_path = Path(root) / name
            path_str = str(full_path)
            
            # Check for existence
            if path_str in manifest_map:
                continue
                
            # Double check relative path just in case
            try:
                full_path.relative_to(exp_dir)
                # Some agents might store relative paths? We compute once to ensure it is relative.
            except ValueError:
                pass

            # Infer type
            suffix = full_path.suffix.lower()
            etype = "unknown"
            if suffix in [".png", ".pdf", ".svg"]:
                etype = "figure"
            elif suffix in [".csv", ".json", ".npy", ".npz"]:
                etype = "data"
            elif suffix in [".py"]:
                etype = "code"
            elif suffix in [".md", ".txt", ".log"]:
                etype = "text"
            
            # Create Entry
            entry = {
                "name": name,
                "path": path_str,
                "type": etype,
                "annotations": [{"source": "auto_watcher", "note": "Recovered orphaned file"}],
                "timestamp": datetime.now().isoformat()
            }
            manifest_map[path_str] = entry
            added_files.append(name)

    if added_files:
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest_map, f, indent=2)
        except Exception:
            pass # Fail silently (watcher shouldn't crash run)

    return added_files


def _append_manifest_entry(name: str, metadata_json: Optional[str] = None, allow_missing: bool = False):
    exp_dir = BaseTool.resolve_output_dir(None)
    manifest_path = exp_dir / "file_manifest.json"
    manifest_map = _load_manifest_map(manifest_path)

    try:
        target_path = BaseTool.resolve_input_path(name, allow_dir=True)
    except FileNotFoundError:
        if not allow_missing:
            return {"error": f"Referenced file not found: {name}. Use reserve_output + append_manifest after creation, or set allow_missing=True if intentional."}
        target_path = BaseTool.resolve_output_dir(None) / name

    meta: Dict[str, Any] = {}
    if metadata_json:
        try:
            meta = json.loads(metadata_json)
        except Exception:
            meta = {"raw": metadata_json}
    path_str = str(target_path)
    name_only = os.path.basename(name or path_str)
    entry = manifest_map.get(path_str, {"name": name_only, "path": path_str, "type": None, "annotations": []})
    entry["name"] = name_only
    entry_type = entry.get("type") or meta.get("type")
    annotations: List[Dict[str, Any]] = entry.get("annotations") or []
    annotation = {k: v for k, v in meta.items() if k != "type"}
    if annotation and annotation not in annotations:
        annotations.append(annotation)
    entry["type"] = entry_type
    entry["annotations"] = annotations
    compat_meta = _build_metadata_for_compat(entry_type, annotations)
    if compat_meta:
        entry["metadata"] = compat_meta
    elif "metadata" in entry:
        entry.pop("metadata", None)
     
    entry["timestamp"] = datetime.now().isoformat()
     
    manifest_map[path_str] = entry
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest_map, f, indent=2)
    except Exception as e:
        return {"error": f"Failed to write manifest: {str(e)}"}
         
    return {"manifest_path": str(manifest_path), "n_entries": len(manifest_map)}


def _append_artifact_from_result(result: Any, key: str, metadata_json: Optional[str], allow_missing: bool = True):
    if not metadata_json or not isinstance(result, dict):
        return
    out = result.get(key)
    if isinstance(out, str):
        _append_manifest_entry(name=out, metadata_json=metadata_json, allow_missing=allow_missing)


def _append_figures_from_result(result: Any, metadata_json: Optional[str]):
    if not metadata_json or not isinstance(result, dict):
        return
    for _, v in result.items():
        if isinstance(v, str) and v.endswith((".png", ".pdf", ".svg")):
            _append_manifest_entry(name=v, metadata_json=metadata_json, allow_missing=True)


# --- Transport Run Manifest Helpers ---
def _transport_manifest_path() -> Path:
    exp_dir = BaseTool.resolve_output_dir(None)
    manifest_path = exp_dir / "simulations" / "transport_runs" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    return manifest_path


def _acquire_manifest_lock(manifest_path: Path, timeout: float = 5.0, poll: float = 0.2) -> Optional[Path]:
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    start = time.time()
    while time.time() - start < timeout:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock_path
        except FileExistsError:
            time.sleep(poll)
    return None


def _atomic_write_json(target: Path, data: Any) -> Optional[str]:
    lock = _acquire_manifest_lock(target)
    if lock is None:
        return "Failed to acquire manifest lock; concurrent write in progress."

    tmp_path = target.with_suffix(target.suffix + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, target)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        finally:
            try:
                lock.unlink()
            except Exception:
                pass
        return f"Failed to write manifest: {exc}"

    try:
        lock.unlink()
    except Exception:
        pass
    return None


def _load_transport_manifest() -> Dict[str, Any]:
    manifest_path = _transport_manifest_path()
    if not manifest_path.exists():
        return {"schema_version": 1, "runs": []}
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"schema_version": 1, "runs": []}
        data.setdefault("schema_version", 1)
        data.setdefault("runs", [])
        return data
    except Exception:
        return {"schema_version": 1, "runs": []}


def _upsert_transport_manifest_entry(
    baseline: str,
    transport: float,
    seed: int,
    status: str,
    paths: Dict[str, Optional[str]],
    notes: str = "",
    actor: Optional[str] = None,
) -> Dict[str, Any]:
    manifest = _load_transport_manifest()
    runs: List[Dict[str, Any]] = manifest.get("runs", [])
    actor_name = actor or os.environ.get("AISC_ACTIVE_ROLE", "") or "unknown"

    found = None
    for entry in runs:
        if (
            entry.get("baseline") == baseline
            and entry.get("transport") == transport
            and entry.get("seed") == seed
        ):
            found = entry
            break

    if found is None:
        found = {"baseline": baseline, "transport": transport, "seed": seed}
        runs.append(found)

    found.update(
        {
            "status": status,
            "paths": paths,
            "updated_at": datetime.now().isoformat(),
            "notes": notes or "",
            "actor": actor_name,
        }
    )
    manifest["runs"] = runs
    manifest.setdefault("schema_version", 1)

    err = _atomic_write_json(_transport_manifest_path(), manifest)
    if err:
        return {"error": err}
    return {"manifest_path": str(_transport_manifest_path()), "entry": found}


def _scan_transport_runs(root: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not root.exists():
        return entries

    for baseline_dir in root.iterdir():
        if not baseline_dir.is_dir():
            continue
        baseline = baseline_dir.name
        for transport_dir in baseline_dir.iterdir():
            if not transport_dir.is_dir() or not transport_dir.name.startswith("transport_"):
                continue
            t_str = transport_dir.name.split("transport_", 1)[-1]
            try:
                transport_val = float(t_str)
            except Exception:
                continue
            for seed_dir in transport_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                s_str = seed_dir.name.split("seed_", 1)[-1]
                try:
                    seed_val = int(s_str)
                except Exception:
                    continue

                def _maybe(path: Path) -> Optional[str]:
                    return str(path) if path.exists() else None

                base_prefix = baseline
                paths = {
                    "failure_matrix": _maybe(seed_dir / f"{base_prefix}_sim_failure_matrix.npy"),
                    "time_vector": _maybe(seed_dir / f"{base_prefix}_sim_time_vector.npy"),
                    "nodes_order": _maybe(seed_dir / f"nodes_order_{base_prefix}_sim.txt"),
                    "sim_json": _maybe(seed_dir / f"{base_prefix}_sim.json"),
                    "sim_status": _maybe(seed_dir / f"{base_prefix}_sim.status.json"),
                }
                missing = [k for k, v in paths.items() if v is None]
                status = "complete" if not missing else "partial"
                notes = ""
                if missing:
                    notes = f"missing: {', '.join(missing)}"

                entries.append(
                    {
                        "baseline": baseline,
                        "transport": transport_val,
                        "seed": seed_val,
                        "status": status,
                        "paths": paths,
                        "updated_at": datetime.now().isoformat(),
                        "notes": notes,
                        "actor": "system-scan",
                    }
                )
    return entries


def _build_seed_dir(baseline: str, transport: float, seed: int) -> Path:
    root = BaseTool.resolve_output_dir(None) / "simulations" / "transport_runs"
    seed_dir = root / baseline / f"transport_{transport}" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    return seed_dir


def _resolve_run_paths(seed_dir: Path, baseline: str) -> Dict[str, Path]:
    return {
        "failure_matrix": seed_dir / f"{baseline}_sim_failure_matrix.npy",
        "time_vector": seed_dir / f"{baseline}_sim_time_vector.npy",
        "nodes_order": seed_dir / f"nodes_order_{baseline}_sim.txt",
        "sim_json": seed_dir / f"{baseline}_sim.json",
        "sim_status": seed_dir / f"{baseline}_sim.status.json",
        "verification": seed_dir / f"{baseline}_sim_verification.md",
    }

def _run_root() -> Path:
    base = os.environ.get("AISC_BASE_FOLDER", "")
    return Path(base) if base else Path(".")


def _bootstrap_note_links() -> None:
    for name in ("pi_notes.md", "user_inbox.md"):
        try:
            ensure_note_files(name)
        except Exception as exc:
            print(f"⚠️ Failed to ensure {name}: {exc}")


def format_list_field(data: Any) -> str:
    if isinstance(data, list):
        return "\n".join([f"- {item}" for item in data])
    return str(data)


def _render_pdf_or_markdown(path: Path, content: str) -> Tuple[Path, Optional[str]]:
    warning: Optional[str] = None
    try:
        import pypandoc  # type: ignore

        if shutil.which("pandoc"):
            pypandoc.convert_text(
                content,
                "pdf",
                format="md",
                outputfile=str(path),
                extra_args=["--pdf-engine=pdflatex", "--standalone", "-V", "geometry:margin=1in"],
            )
            return path, None
        warning = "pandoc not found; saved Markdown fallback instead of PDF."
    except Exception as exc:
        warning = f"PDF generation failed ({exc}); saved Markdown fallback."

    fallback = path.with_suffix(".md")
    fallback.write_text(content, encoding="utf-8")
    return fallback, warning


def _run_cli_tool(tool_name: str, args: str = "") -> Any:
    cwd = os.getcwd()
    if not shutil.which(tool_name):
        return f"{tool_name} not found in PATH."
    cmd = f"cd {cwd} && {tool_name} {args}".strip()
    try:
        return os.popen(cmd).read()
    except Exception as exc:
        return {"error": str(exc)}


def _report_capabilities() -> Dict[str, Any]:
    tools = {name: bool(shutil.which(name)) for name in ("pandoc", "pdflatex", "ruff", "pyright")}
    return {"tools": tools, "pdf_engine_ready": tools.get("pandoc") and tools.get("pdflatex")}


def _ensure_transport_readme(base_folder: str):
    """
    Write a standard transport_runs/README.md describing layout and manifest usage for this run.
    Safe to call multiple times; overwrites with the canonical content.
    """
    transport_dir, _, _ = resolve_output_path(
        subdir="simulations/transport_runs",
        name="",
        run_root=Path(base_folder) / "experiment_results",
        allow_quarantine=False,
        unique=False,
    )
    transport_dir.mkdir(parents=True, exist_ok=True)
    readme_path = transport_dir / "README.md"
    template_path = Path(__file__).parent / "docs" / "transport_runs_README.md"
    if template_path.exists():
        try:
            content = template_path.read_text()
        except Exception:
            content = ""
    else:
        content = (
            "Transport run layout and naming\n\n"
            "- Root: experiment_results/simulations/transport_runs\n"
            "- Baseline folders: transport_runs/<baseline>/\n"
            "- Transport folders: transport_runs/<baseline>/transport_<transport>/\n"
            "- Seed folders: transport_runs/<baseline>/transport_<transport>/seed_<seed>/\n"
            "- Files in each seed folder:\n"
            "  - <baseline>_sim_failure_matrix.npy\n"
            "  - <baseline>_sim_time_vector.npy\n"
            "  - nodes_order_<baseline>_sim.txt\n"
            "  - <baseline>_sim.json\n"
            "  - <baseline>_sim.status.json\n\n"
            "Completion rule\n"
            "- A run is considered complete only when the arrays (failure_matrix, time_vector, nodes_order) and sim.json + sim.status.json exist.\n"
            "- Prefer exporting arrays during the sim run; otherwise run sim_postprocess immediately on the sim.json so arrays are present before marking complete.\n\n"
            "Canonical manifest\n"
            "- Manifest path: experiment_results/simulations/transport_runs/manifest.json\n"
            "- Each entry keyed by (baseline, transport, seed) with fields: status (complete|partial|error), paths, updated_at, notes, actor.\n"
            "- Use the manifest as the source of truth for skip/verify; if missing or stale, run scan_transport_manifest to rebuild from disk.\n"
        )
    try:
        readme_path.write_text(content)
    except Exception:
        pass

def _make_agent(name: str, instructions: str, tools: List[Any], model: str, settings: ModelSettings) -> Agent:
    return Agent(name=name, instructions=instructions, model=model, tools=tools, model_settings=settings)


async def extract_run_output(run_result: RunResult) -> str:
    parts: List[str] = []
     
    def get_attr(obj, attr):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if hasattr(obj, "get"):
            return obj.get(attr)
        return None

    err = get_attr(run_result, "error")
    if err:
        parts.append(f"❌ TERMINATION: {err}")

    status_val = get_attr(run_result, "status")
    if status_val:
        parts.append(f"STATUS: {status_val}")

    candidate_fields = ["final_output", "output", "final_message", "content", "message"]
    out: Any = None
    for field in candidate_fields:
        out = get_attr(run_result, field)
        if out:
            break

    if not out and hasattr(run_result, "messages"):
        msgs = getattr(run_result, "messages")
        try:
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                out = getattr(last, "content", None) if not isinstance(last, dict) else last.get("content")
        except Exception:
            pass

    if not out and hasattr(run_result, "raw_responses"):
        try:
            raw = getattr(run_result, "raw_responses")
            if isinstance(raw, list) and raw:
                last = raw[-1]
                out = getattr(last, "content", None) or getattr(last, "text", None)
        except Exception:
            pass

    if not out and hasattr(run_result, "new_items"):
        try:
            new_items = getattr(run_result, "new_items")
            if isinstance(new_items, list) and new_items:
                last_item = new_items[-1]
                if hasattr(last_item, "content"):
                    out = f"last_item: {getattr(last_item, 'content')}"
                elif hasattr(last_item, "tool_name"):
                    out = f"last_tool: {getattr(last_item, 'tool_name')}({getattr(last_item, 'tool_input', '')})"
        except Exception:
            pass

    if out:
        parts.append(f"FINAL MSG: {str(out)[:500]}...")

    try:
        ni = getattr(run_result, "new_items", None)
        if isinstance(ni, list) and ni:
            tool_trace: List[str] = []
            for item in ni:
                t_name = None
                t_input = None
                if hasattr(item, "tool_name"):
                    t_name = getattr(item, "tool_name")
                    t_input = getattr(item, "tool_input", "")
                elif isinstance(item, dict) and "tool_name" in item:
                    t_name = str(item["tool_name"])
                    t_input = str(item.get("tool_input", ""))
                 
                if t_name:
                    inp_str = str(t_input).replace('\n', ' ')[:20]
                    tool_trace.append(f"{t_name}({inp_str}...)")
             
            if tool_trace:
                parts.append("\n📋 TOOL TRACE (Execution History):")
                for i in range(0, len(tool_trace), 3):
                    parts.append(" -> ".join(tool_trace[i:i+3]))
            else:
                parts.append("(No tool calls recorded)")
    except Exception:
        pass

    if not parts:
        return str(run_result)
    return "\n".join(parts)

# --- Tool Definitions (Wrappers for Agents SDK) ---

@function_tool
def inspect_recent_manifest_entries(limit: int = 20, since_minutes: int = 1440, role: Optional[str] = None):
    """
    Forensic Tool: Check the file manifest for updated entries.
    Crucial for recovering work after a sub-agent timeout or crash.
    - limit: Max number of files to return (default 20).
    - since_minutes: Only return files updated in the last N minutes (default 1440 = 24 hours).
    - role: If provided (e.g. 'modeler', 'analyst'), only show files where metadata['source'] matches this role.
    Returns the files in reverse chronological order (newest first).
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    manifest_path = exp_dir / "file_manifest.json"
    manifest_map = _load_manifest_map(manifest_path)
    
    if not manifest_map:
        return "Manifest is empty."

    cutoff = datetime.now() - timedelta(minutes=since_minutes)
    valid_entries = []
    
    for entry in manifest_map.values():
        ts_str = entry.get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts < cutoff:
                    continue
            except ValueError:
                pass 
        
        if role:
            meta = entry.get("metadata", {})
            source = meta.get("source", "").lower()
            if role.lower() not in source and role.lower() not in str(meta).lower():
                continue

        valid_entries.append(entry)

    sorted_entries = sorted(
        valid_entries, 
        # FIXED: Ensure None values are treated as empty strings to avoid sorting TypeError
        key=lambda x: x.get("timestamp") or "", 
        reverse=True
    )
    
    recent = sorted_entries[:limit]
    
    if not recent:
        msg = f"No manifest updates found in the last {since_minutes} minutes"
        if role:
            msg += f" for role '{role}'"
        return msg + "."

    output = [f"Last {len(recent)} manifest updates (since {since_minutes} min ago):"]
    for entry in recent:
        ts = entry.get("timestamp", "N/A")
        name = entry.get("name", "Unknown")
        etype = entry.get("type", "Unknown")
        output.append(f"- [{ts}] {name} (Type: {etype})")
        
    return "\n".join(output)

@function_tool
def manage_project_knowledge(
    action: str,
    category: str = "general",
    observation: str = "",
    solution: str = "",
    actor: str = "",
):
    """
    Manage the persistent Project Knowledge Base (project_knowledge.md).
    Use this to store constraints, decisions, failure patterns, and REFLECTIONS that persist across sessions.
     
    Args:
        action: 'add' to log new info, 'read' to retrieve all knowledge.
        category: 'constraint', 'decision', 'failure_pattern', or 'reflection'.
        observation: Context of the problem, inefficiency, or constraint (Required for 'add').
        solution: The fix, decision, or proposed improvement (Required for 'add').
        actor: Optional role/agent name to auto-log who created the record. If omitted, falls back to env AISC_ACTIVE_ROLE or 'unknown'.
    """
    base = os.environ.get("AISC_BASE_FOLDER", "")
    kb_path = os.path.join(base, "project_knowledge.md")
    actor_name = (
        actor.strip()
        or os.environ.get("AISC_ACTIVE_ROLE", "").strip()
        or "unknown"
    )
     
    if action == "read":
        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading knowledge base: {str(e)}"
        return "Project Knowledge Base is empty."
        
    if action == "add":
        if not observation or not solution:
            return "Error: Both 'observation' and 'solution' are required for 'add' action."
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"\n## [{category.upper()}] {timestamp}\n"
            f"**Actor:** {actor_name}\n"
            f"**Observation/Problem:** {observation}\n"
            f"**Solution/Insight:** {solution}\n"
            f"{'-'*40}\n"
        )
        
        try:
            with open(kb_path, 'a') as f:
                f.write(entry)
            return f"Added new {category} entry to project_knowledge.md"
        except Exception as e:
            return f"Error writing to knowledge base: {str(e)}"
        
    return "Invalid action. Use 'add' or 'read'."

@function_tool
def check_project_state(base_folder: str) -> str:
    """
    Reads the project state to see what artifacts exist.
    UPDATED: Automatically scans for orphaned files and updates the manifest.
    """
    status_msg = "Folder existed"
    
    if not os.path.exists(base_folder):
        try:
            os.makedirs(base_folder, exist_ok=True)
            exp_results = os.path.join(base_folder, "experiment_results")
            os.makedirs(exp_results, exist_ok=True)
            status_msg = f"Created new directory: {base_folder}"
        except Exception as e:
            return json.dumps({"error": f"Failed to create folder {base_folder}: {str(e)}"})
        
    exists = os.listdir(base_folder)
    exp_results = os.path.join(base_folder, "experiment_results")
    
    # --- AUTO-WATCHER TRIGGER ---
    orphans = []
    if os.path.exists(exp_results):
        orphans = _scan_and_auto_update_manifest(Path(exp_results))
    
    artifacts = os.listdir(exp_results) if os.path.exists(exp_results) else []
    has_plots = False
    has_data = any(x.endswith('.csv') for x in artifacts)
    has_lit_review = "lit_summary.json" in artifacts or "lit_summary.csv" in artifacts
    if os.path.exists(exp_results):
        for root, dirs, files in os.walk(exp_results):
            for f in files:
                lf = f.lower()
                if lf.endswith((".png", ".svg", ".pdf")):
                    has_plots = True
                if lf.endswith(".csv"):
                    has_data = True
                if lf in {"lit_summary.json", "lit_summary.csv"}:
                    has_lit_review = True
            if has_plots and has_data and has_lit_review:
                break

    return json.dumps({
        "status_message": status_msg,
        "orphaned_files_recovered": len(orphans),
        "root_files": exists,
        "artifacts": artifacts,
        "has_lit_review": has_lit_review,
        "has_data": has_data,
        "has_plots": has_plots,
        "has_draft": "manuscript.pdf" in exists or "manuscript.tex" in exists
    })


@function_tool
def scan_transport_manifest(write: bool = True):
    """
    Scan transport_runs and (optionally) write manifest.json with status of (baseline, transport, seed).
    """
    root = BaseTool.resolve_output_dir(None) / "simulations" / "transport_runs"
    entries = _scan_transport_runs(root)
    manifest = {"schema_version": 1, "runs": entries, "updated_at": datetime.now().isoformat()}
    if write:
        err = _atomic_write_json(_transport_manifest_path(), manifest)
        if err:
            return {"error": err, "manifest_path": str(_transport_manifest_path())}
    return {"manifest_path": str(_transport_manifest_path()), "runs": entries}


def _status_from_paths(paths: Dict[str, Optional[str]], required_keys: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    required = required_keys or ["failure_matrix", "time_vector", "nodes_order", "sim_json", "sim_status"]
    missing = [k for k in required if not paths.get(k)]
    status = "complete" if not missing else "partial"
    return status, missing


def _write_verification(seed_dir: Path, baseline: str, transport: float, seed: int, status: str, notes: str):
    ver_path = _resolve_run_paths(seed_dir, baseline)["verification"]
    lines = [
        f"baseline: {baseline}",
        f"transport: {transport}",
        f"seed: {seed}",
        f"status: {status}",
        f"notes: {notes}",
    ]
    try:
        ver_path.write_text("\n".join(lines))
    except Exception:
        pass


def _generate_run_recipe(base_folder: str):
    """
    Generate a per-run run_recipe.json under experiment_results/simulations/transport_runs by scanning available morphologies
    and seeding template/nodes_order from existing transport_runs if present. Overwrites safely each run startup.
    """
    base_path = Path(base_folder)
    morph_dir, _, _ = resolve_output_path(subdir="morphologies", name="", run_root=base_path / "experiment_results", allow_quarantine=False, unique=False)
    transport_runs_dir, _, _ = resolve_output_path(subdir="simulations/transport_runs", name="", run_root=base_path / "experiment_results", allow_quarantine=False, unique=False)
    recipe_path = transport_runs_dir / "run_recipe.json"
    transport_runs_dir.mkdir(parents=True, exist_ok=True)

    allowed_suffixes = [".npy", ".npz", ".graphml", ".gpickle", ".gml"]
    entries: List[Dict[str, str]] = []

    # Helper to find an existing template in transport_runs for a baseline
    def _find_template(baseline: str) -> Tuple[Optional[str], Optional[str]]:
        root = transport_runs_dir / baseline
        if not root.exists():
            return None, None
        for tdir in root.iterdir():
            if not tdir.is_dir() or not tdir.name.startswith("transport_"):
                continue
            seed_dir = tdir / "seed_0"
            if not seed_dir.exists():
                continue
            sim_json = seed_dir / f"{baseline}_sim.json"
            nodes = seed_dir / f"nodes_order_{baseline}_sim.txt"
            sim_str = str(sim_json) if sim_json.exists() else None
            nodes_str = str(nodes) if nodes.exists() else None
            if sim_str or nodes_str:
                return sim_str, nodes_str
        return None, None

    if morph_dir.exists():
        for f in morph_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in allowed_suffixes:
                continue
            baseline = f.stem
            tpl_json, tpl_nodes = _find_template(baseline)
            entries.append(
                {
                    "baseline": baseline,
                    "morphology_path": str(f),
                    "template_json": tpl_json or "",
                    "nodes_order_template": tpl_nodes or "",
                    "output_root": str(
                        base_path
                        / "experiment_results"
                        / "simulations"
                        / "transport_sweep"
                        / "transport_{transport}"
                        / "seed_{seed}"
                        / ""
                    ),
                }
            )

    try:
        with open(recipe_path, "w") as fp:
            json.dump(entries, fp, indent=2)
    except Exception:
        pass

@function_tool
def mirror_artifacts(
    src_paths: List[str],
    dest_dir: str = "experiment_results/figures_for_manuscript",
    mode: str = "copy",
    prefix: str = "",
    suffix: str = "",
):
    """
    Copy or move artifacts into a canonical figures directory under the run root.
    - src_paths: list of files to mirror.
    - dest_dir: relative or absolute dest; defaults to experiment_results/figures_for_manuscript.
    - mode: 'copy' (default) or 'move'.
    - prefix/suffix: optional disambiguation added to filename stem.
    Refuses to write outside AISC_BASE_FOLDER. Uses temp+atomic rename to avoid partial writes.
    """
    base_env = os.environ.get("AISC_BASE_FOLDER", ".")
    base = Path(base_env).resolve()
    if not base.exists():
        return {"error": "AISC_BASE_FOLDER not set or missing."}
    dest_path, _, note = resolve_output_path(subdir=None, name=dest_dir, run_root=base / "experiment_results", allow_quarantine=True, unique=False)
    try:
        dest_path.relative_to(base)
    except Exception:
        return {"error": f"Destination outside run root: {dest_path}"}
    dest = dest_path
    dest.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    skipped: List[str] = []
    errors: List[str] = []

    for src in src_paths:
        try:
            p = BaseTool.resolve_input_path(src, allow_dir=False)
        except Exception as exc:
            errors.append(f"{src}: {exc}")
            continue
        try:
            p.relative_to(base)
        except Exception:
            errors.append(f"{p}: outside run root")
            continue
        if not p.exists():
            errors.append(f"{p}: missing")
            continue

        stem = p.stem
        name = stem
        if prefix:
            name = f"{prefix}{name}"
        if suffix:
            name = f"{name}{suffix}"
        name = f"{name}{p.suffix}"
        target = dest / name

        if target.exists():
            errors.append(f"{p}: target exists {target}; use prefix/suffix to disambiguate.")
            continue

        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            if mode == "move":
                shutil.copy2(p, tmp)
                os.replace(tmp, target)
                p.unlink()
            else:
                shutil.copy2(p, tmp)
                os.replace(tmp, target)
            copied.append(str(target))
        except Exception as exc:
            errors.append(f"{p} -> {target}: {exc}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
    return {"dest_dir": str(dest), "copied": copied, "skipped": skipped, "errors": errors}


def _resolve_baseline_path_internal(baseline: str) -> Tuple[Optional[Path], List[str], Optional[str]]:
    """
    Resolve a baseline path. Accepts absolute/relative path or a name under experiment_results/morphologies/<baseline>.*.
    Only allows graph formats: .npy, .npz, .graphml, .gpickle, .gml.
    Returns (path_or_none, available_baselines, error_message_or_none).
    """
    allowed_suffixes = [".npy", ".npz", ".graphml", ".gpickle", ".gml"]
    try:
        p = BaseTool.resolve_input_path(baseline, allow_dir=False)
        if p.suffix.lower() in allowed_suffixes:
            return p, [], None
        return None, [], f"Unsupported baseline format '{p.suffix}'. Allowed: {', '.join(allowed_suffixes)}"
    except Exception:
        pass

    morph_dir = BaseTool.resolve_output_dir(None) / "morphologies"
    candidates: List[Path] = []
    if morph_dir.exists():
        for f in morph_dir.iterdir():
            if f.is_file():
                candidates.append(f)
    available = sorted({c.stem for c in candidates})

    for suff in allowed_suffixes:
        candidate = morph_dir / f"{baseline}{suff}"
        if candidate.exists():
            return candidate, available, None

    return None, available, f"Baseline '{baseline}' not found. Provide a valid graph path or one of: {', '.join(available)}"


@function_tool
def resolve_baseline_path(baseline: str):
    """
    Resolve a baseline path by name or explicit path (searches experiment_results/morphologies for <baseline> with common suffixes).
    Returns {path} or an error with available baselines.
    """
    path, available, err = _resolve_baseline_path_internal(baseline)
    if path:
        return {"path": str(path)}
    return {"error": err or "baseline not found", "available_baselines": available}


@function_tool
def resolve_sim_path(baseline: str, transport: float, seed: int):
    """
    Resolve a sim.json path for (baseline, transport, seed) using the transport manifest with a scan fallback.
    Returns {path, status, missing, available_transports, available_pairs} or an error with suggestions.
    """
    data = _load_transport_manifest()
    runs = data.get("runs", [])
    if not runs:
        root = BaseTool.resolve_output_dir(None) / "simulations" / "transport_runs"
        runs = _scan_transport_runs(root)

    candidates = [r for r in runs if r.get("baseline") == baseline]
    available_transports = sorted(
        [t for t in (r.get("transport") for r in candidates if "transport" in r) if isinstance(t, (int, float))]
    )
    available_pairs = sorted(
        [
            (t, s)
            for t, s in (
                (r.get("transport"), r.get("seed")) for r in candidates if "transport" in r and "seed" in r
            )
            if isinstance(t, (int, float)) and isinstance(s, (int, float))
        ]
    )

    entry = next(
        (r for r in candidates if r.get("transport") == transport and r.get("seed") == seed),
        None,
    )
    if entry is None:
        return {
            "error": f"Run not found for baseline={baseline}, transport={transport}, seed={seed}",
            "available_transports": available_transports,
            "available_pairs": available_pairs,
        }

    paths = entry.get("paths", {}) if isinstance(entry, dict) else {}
    sim_path = paths.get("sim_json")
    # Reconstruct expected path as fallback
    expected = (
        BaseTool.resolve_output_dir(None)
        / "simulations"
        / "transport_runs"
        / baseline
        / f"transport_{transport}"
        / f"seed_{seed}"
        / f"{baseline}_sim.json"
    )
    resolved_path: Optional[Path] = None
    if sim_path:
        p = Path(sim_path)
        if not p.is_absolute():
            p = Path(sim_path)
        if p.exists():
            resolved_path = p
    if resolved_path is None and expected.exists():
        resolved_path = expected

    missing = [k for k in ["failure_matrix", "time_vector", "nodes_order", "sim_json", "sim_status"] if not paths.get(k)]
    if resolved_path is None:
        return {
            "error": f"sim.json missing for baseline={baseline}, transport={transport}, seed={seed}",
            "status": entry.get("status"),
            "paths": paths,
            "missing": missing,
            "available_transports": available_transports,
            "available_pairs": available_pairs,
        }

    if missing:
        note = f"warning: missing {', '.join(missing)}"
    else:
        note = ""
    return {
        "path": str(resolved_path),
        "status": entry.get("status"),
        "missing": missing,
        "note": note,
        "available_transports": available_transports,
        "available_pairs": available_pairs,
    }


@function_tool
def run_transport_batch(
    baseline_path: str,
    transport_values: List[float],
    seeds: List[int],
    steps: int = 150,
    dt: float = 0.1,
    export_arrays: bool = True,
    max_workers: int = 1,
    mitophagy_rate: float = 0.02,
    demand_scale: float = 0.5,
    noise_std: float = 0.0,
    downsample: int = 1,
):
    """
    Batch wrapper: run transport sims for a baseline across transports/seeds, postprocess arrays if missing, and update the transport manifest.
    """
    baseline_path_resolved, available_bases, err = _resolve_baseline_path_internal(baseline_path)
    if baseline_path_resolved is None or err:
        return {
            "error": err or "Baseline not found",
            "available_baselines": available_bases,
            "hint": "Provide a valid baseline path or use resolve_baseline_path.",
        }
    baseline_name = baseline_path_resolved.stem
    tasks = []
    results: List[Dict[str, Any]] = []

    manifest_data = _load_transport_manifest()
    existing = manifest_data.get("runs", [])

    def should_skip(entry: Optional[Dict[str, Any]], path_map: Dict[str, Path]) -> bool:
        entry_paths = entry.get("paths", {}) if isinstance(entry, dict) else {}
        paths_now = {k: entry_paths.get(k) or (str(v) if v.exists() else None) for k, v in path_map.items() if k != "verification"}
        status_now, missing = _status_from_paths(paths_now)
        return status_now == "complete" and not missing

    for t in transport_values:
        for s in seeds:
            seed_dir = _build_seed_dir(baseline_name, t, s)
            path_map = _resolve_run_paths(seed_dir, baseline_name)
            entry = next((e for e in existing if e.get("baseline") == baseline_name and e.get("transport") == t and e.get("seed") == s), None)
            if should_skip(entry, path_map):
                results.append({"baseline": baseline_name, "transport": t, "seed": s, "skipped": True, "reason": "manifest_complete"})
                continue
            tasks.append((t, s, seed_dir, path_map))

    def run_one(t: float, s: int, seed_dir: Path, path_map: Dict[str, Path]):
        actor = os.environ.get("AISC_ACTIVE_ROLE", "") or "batch_runner"
        notes = ""
        status = "partial"
        try:
            RunCompartmentalSimTool().use_tool(
                graph_path=str(baseline_path_resolved),
                output_dir=str(seed_dir),
                steps=steps,
                dt=dt,
                transport_rate=t,
                demand_scale=demand_scale,
                mitophagy_rate=mitophagy_rate,
                noise_std=noise_std,
                seed=s,
                store_timeseries=True,
                downsample=downsample,
                export_arrays=export_arrays,
            )
            paths_now = {k: (str(v) if v.exists() else None) for k, v in path_map.items() if k != "verification"}
            status_now, missing = _status_from_paths(paths_now)
            if missing:
                sim_json = path_map["sim_json"]
                if sim_json.exists():
                    SimPostprocessTool().use_tool(
                        sim_json_path=str(sim_json),
                        output_dir=str(seed_dir),
                        graph_path=str(baseline_path_resolved),
                    )
                    paths_now = {k: (str(v) if v.exists() else None) for k, v in path_map.items() if k != "verification"}
                    status_now, missing = _status_from_paths(paths_now)
            status = status_now
            if missing:
                notes = f"missing after postprocess: {', '.join(missing)}"
            _write_verification(seed_dir, baseline_name, t, s, status, notes)
            upd = _upsert_transport_manifest_entry(
                baseline=baseline_name,
                transport=t,
                seed=s,
                status=status,
                paths=paths_now,
                notes=notes,
                actor=actor,
            )
            return {"baseline": baseline_name, "transport": t, "seed": s, "status": status, "notes": notes, "manifest": upd}
        except Exception as exc:
            notes = f"error: {exc}"
            _write_verification(seed_dir, baseline_name, t, s, "error", notes)
            _upsert_transport_manifest_entry(
                baseline=baseline_name,
                transport=t,
                seed=s,
                status="error",
                paths={k: (str(v) if isinstance(v, Path) else str(v)) for k, v in path_map.items() if k != "verification"},
                notes=notes,
                actor=actor,
            )
            return {"baseline": baseline_name, "transport": t, "seed": s, "status": "error", "notes": notes}

    if max_workers <= 1:
        for t, s, seed_dir, path_map in tasks:
            results.append(run_one(t, s, seed_dir, path_map))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(run_one, t, s, seed_dir, path_map) for t, s, seed_dir, path_map in tasks]
            for fut in as_completed(futs):
                results.append(fut.result())

    return {"baseline": baseline_name, "requested": len(transport_values) * len(seeds), "scheduled": len(tasks), "results": results}


@function_tool
def read_transport_manifest(baseline: Optional[str] = None, transport: Optional[float] = None, seed: Optional[int] = None):
    """
    Read transport_runs manifest (filters optional).
    If missing, auto-scan without writing.
    """
    data = _load_transport_manifest()
    runs = data.get("runs", [])
    filtered = []
    for entry in runs:
        if baseline is not None and entry.get("baseline") != baseline:
            continue
        if transport is not None and entry.get("transport") != transport:
            continue
        if seed is not None and entry.get("seed") != seed:
            continue
        filtered.append(entry)
    if not runs:
        # fall back to a read-only scan
        root = BaseTool.resolve_output_dir(None) / "simulations" / "transport_runs"
        runs = _scan_transport_runs(root)
        filtered = [e for e in runs if (baseline is None or e["baseline"] == baseline) and (transport is None or e["transport"] == transport) and (seed is None or e["seed"] == seed)]
    return {"manifest_path": str(_transport_manifest_path()), "runs": filtered, "schema_version": data.get("schema_version", 1)}


@function_tool
def update_transport_manifest(
    baseline: str,
    transport: float,
    seed: int,
    status: str,
    paths_json: Optional[str] = None,
    notes: str = "",
    actor: str = "",
):
    """
    Upsert a single transport run entry. Paths not provided will be inferred from standard filenames in the seed folder.
    """
    root = BaseTool.resolve_output_dir(None) / "simulations" / "transport_runs"
    seed_dir = root / baseline / f"transport_{transport}" / f"seed_{seed}"

    def _infer(path_name: str) -> Optional[str]:
        try:
            parsed_paths = json.loads(paths_json) if paths_json else {}
        except Exception:
            parsed_paths = {}
        if isinstance(parsed_paths, dict) and path_name in parsed_paths:
            return parsed_paths[path_name]
        candidates = {
            "failure_matrix": seed_dir / f"{baseline}_sim_failure_matrix.npy",
            "time_vector": seed_dir / f"{baseline}_sim_time_vector.npy",
            "nodes_order": seed_dir / f"nodes_order_{baseline}_sim.txt",
            "sim_json": seed_dir / f"{baseline}_sim.json",
            "sim_status": seed_dir / f"{baseline}_sim.status.json",
        }
        p = candidates.get(path_name)
        return str(p) if p and p.exists() else None

    resolved_paths = {
        "failure_matrix": _infer("failure_matrix"),
        "time_vector": _infer("time_vector"),
        "nodes_order": _infer("nodes_order"),
        "sim_json": _infer("sim_json"),
        "sim_status": _infer("sim_status"),
    }
    missing = [k for k, v in resolved_paths.items() if v is None]
    if status == "complete" and missing:
        status = "partial"
        if notes:
            notes = notes + f"; missing: {', '.join(missing)}"
        else:
            notes = f"missing: {', '.join(missing)}"

    actor_name = actor or os.environ.get("AISC_ACTIVE_ROLE", "") or "unknown"
    return _upsert_transport_manifest_entry(
        baseline=baseline,
        transport=transport,
        seed=seed,
        status=status,
        paths=resolved_paths,
        notes=notes,
        actor=actor_name,
    )
@function_tool
def log_strategic_pivot(reason: str, new_plan: str):
    """Logs a major change in direction to the system logs."""
    print(f"\n[STRATEGIC PIVOT] {reason}\nPlan: {new_plan}\n")
    return "Pivot logged."

@function_tool
def assemble_lit_data(
    queries: Optional[List[str]] = None,
    seed_paths: Optional[List[str]] = None,
    max_results: int = 25,
    use_semantic_scholar: bool = True,
):
    """Searches for literature and creates a lit_summary."""
    if not queries and not seed_paths:
        return "Error: You provided no 'queries' or 'seed_paths'. Please provide specific keywords or paper IDs."
        
    return LitDataAssemblyTool().use_tool(
        queries=queries,
        seed_paths=seed_paths,
        max_results=max_results,
        use_semantic_scholar=use_semantic_scholar,
    )

@function_tool
def validate_lit_summary(path: str):
    """Validates the structure of the literature summary."""
    return LitSummaryValidatorTool().use_tool(path=path)

@function_tool
def run_comp_sim(
    graph_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    steps: int = 200,
    dt: float = 0.1,
    transport_rate: float = 0.05,
    demand_scale: float = 0.5,
    mitophagy_rate: float = 0.02,
    noise_std: float = 0.0,
    seed: int = 0,
    store_timeseries: bool = True,
    downsample: int = 1,
    max_elements: int = 5_000_000,
    metadata_json: Optional[str] = None,
):
    """Runs a compartmental simulation and saves CSV data."""
    res = RunCompartmentalSimTool().use_tool(
        graph_path=graph_path,
        output_dir=_fill_output_dir(output_dir),
        steps=steps,
        dt=dt,
        transport_rate=transport_rate,
        demand_scale=demand_scale,
        mitophagy_rate=mitophagy_rate,
        noise_std=noise_std,
        seed=seed,
        store_timeseries=store_timeseries,
        downsample=downsample,
        max_elements=max_elements,
    )
    _append_artifact_from_result(res, "output_json", metadata_json)
    return res

@function_tool
def run_biological_plotting(solution_path: str, output_dir: Optional[str] = None, make_phase_portrait: bool = True):
    """Generates plots from simulation data."""
    out_dir = _fill_figure_dir(output_dir)
    res = RunBiologicalPlottingTool().use_tool(
        solution_path=solution_path,
        output_dir=out_dir,
        make_phase_portrait=make_phase_portrait,
        make_combined_svg=True,
    )
    _append_figures_from_result(res, '{"type":"figure","source":"analyst"}')
    return res

@function_tool
def sim_postprocess(
    sim_json_path: str,
    output_dir: Optional[str] = None,
    graph_path: Optional[str] = None,
    failure_threshold: float = 0.2,
):
    """Convert sim.json into failure_matrix.npy, time_vector.npy, and nodes_order.txt."""
    return SimPostprocessTool().use_tool(
        sim_json_path=sim_json_path,
        output_dir=_fill_output_dir(output_dir),
        graph_path=graph_path,
        failure_threshold=failure_threshold,
    )


@function_tool
def repair_sim_outputs(manifest_paths: Optional[List[str]] = None, batch_size: int = 10, force: bool = False):
    """
    Bulk repair sim.json entries missing exported arrays; validates per-compartment artifacts and updates manifest/tool_summary.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    manifest_path = exp_dir / "file_manifest.json"
    run_root = exp_dir.parent if exp_dir.name == "experiment_results" else exp_dir
    return RepairSimOutputsTool().use_tool(
        manifest_paths=manifest_paths,
        batch_size=batch_size,
        force=force,
        manifest_path=str(manifest_path),
        run_root=str(run_root),
    )


@function_tool
def graph_diagnostics(
    graph_path: str,
    output_dir: Optional[str] = None,
    make_plots: bool = True,
    max_nodes_for_layout: int = 2000,
):
    """Compute graph stats and optionally degree/layout plots."""
    return GraphDiagnosticsTool().use_tool(
        graph_path=graph_path,
        output_dir=_fill_output_dir(output_dir),
        make_plots=make_plots,
        max_nodes_for_layout=max_nodes_for_layout,
    )

@function_tool
def read_manuscript(path: str):
    """Reads the PDF or text of the manuscript."""
    return ManuscriptReaderTool().use_tool(path=path)

@function_tool
def run_writeup_task(
    base_folder: Optional[str] = None,
    page_limit: int = 8
):
    """Compiles the manuscript using the theoretical biology template."""
    base_folder = base_folder or os.environ.get("AISC_BASE_FOLDER", "")
    ok = perform_writeup(
        base_folder=base_folder,
        no_writing=False,
        num_cite_rounds=10,
        small_model="gpt-4o-mini", 
        big_model="gpt-4o",
        n_writeup_reflections=2,
        page_limit=page_limit,
    )
    return {"success": ok}

@function_tool
def search_semantic_scholar(query: str):
    """Directly search Semantic Scholar for papers."""
    return SemanticScholarSearchTool().use_tool(query=query)

@function_tool
def build_graphs(n_nodes: int = 100, output_dir: Optional[str] = None, seed: int = 0):
    """Generate canonical graphs (binary tree, heavy-tailed, random tree)."""
    res = BuildGraphsTool().use_tool(n_nodes=n_nodes, output_dir=_fill_output_dir(output_dir), seed=seed)
    for graph_type, paths in res.items():
        for k, v in paths.items():
            _append_manifest_entry(
                name=v,
                metadata_json=json.dumps({"type": "graph", "source": "modeler", "graph_type": graph_type, "format": k}),
                allow_missing=True,
            )
    return res

@function_tool
def run_biological_model(
    model_key: str = "cooperation_evolution",
    time_end: float = 20.0,
    num_points: int = 200,
    output_dir: Optional[str] = None,
    metadata_json: Optional[str] = None,
):
    """Run a built-in biological ODE/replicator model and save JSON results."""
    res = RunBiologicalModelTool().use_tool(
        model_key=model_key,
        time_end=time_end,
        num_points=num_points,
        output_dir=_fill_output_dir(output_dir),
    )
    _append_artifact_from_result(res, "output_json", metadata_json)
    return res

@function_tool
def run_sensitivity_sweep(
    graph_path: str,
    output_dir: Optional[str] = None,
    transport_vals: Optional[List[float]] = None,
    demand_vals: Optional[List[float]] = None,
    steps: int = 150,
    dt: float = 0.1,
    metadata_json: Optional[str] = None,
):
    """Sweep transport_rate and demand_scale over a graph and log frac_failed."""
    res = RunSensitivitySweepTool().use_tool(
        graph_path=graph_path,
        output_dir=_fill_output_dir(output_dir),
        transport_vals=transport_vals,
        demand_vals=demand_vals,
        steps=steps,
        dt=dt,
    )
    _append_artifact_from_result(res, "output_csv", metadata_json)
    return res

@function_tool
def run_intervention_tests(
    graph_path: str,
    output_dir: Optional[str] = None,
    transport_vals: Optional[List[float]] = None,
    demand_vals: Optional[List[float]] = None,
    baseline_transport: float = 0.05,
    baseline_demand: float = 0.5,
    metadata_json: Optional[str] = None,
):
    """Test parameter interventions vs a baseline and report delta frac_failed."""
    res = RunInterventionTesterTool().use_tool(
        graph_path=graph_path,
        output_dir=_fill_output_dir(output_dir),
        transport_vals=transport_vals,
        demand_vals=demand_vals,
        baseline_transport=baseline_transport,
        baseline_demand=baseline_demand,
    )
    _append_artifact_from_result(res, "output_json", metadata_json)
    return res

@function_tool
def run_validation_compare(lit_path: str, sim_path: str):
    """Correlate lit_summary metrics with simulation frac_failed."""
    res = RunValidationCompareTool().use_tool(lit_path=lit_path, sim_path=sim_path)
    if isinstance(res, dict):
        _append_manifest_entry(
            name="validation_compare.json",
            metadata_json=json.dumps({"type": "validation", "source": "analyst", "lit_path": lit_path, "sim_path": sim_path}),
            allow_missing=True,
        )
    return res

@function_tool
def run_biological_stats(
    task: str,
    pvalues: Optional[List[float]] = None,
    alpha: float = 0.05,
    test_ids: Optional[List[str]] = None,
    background_ids: Optional[List[str]] = None,
    term_to_ids_json: Optional[str] = None,
):
    """Run BH correction or enrichment analysis. term_to_ids_json: JSON mapping term -> [ids]."""
    term_to_ids: Optional[Dict[str, List[str]]] = None
    if term_to_ids_json:
        try:
            term_to_ids = json.loads(term_to_ids_json)
        except Exception as exc:
            raise ValueError(f"term_to_ids_json must be JSON mapping term -> [ids]; got error: {exc}") from exc
    return RunBiologicalStatsTool().use_tool(
        task=task,
        pvalues=pvalues,
        alpha=alpha,
        test_ids=test_ids,
        background_ids=background_ids,
        term_to_ids=term_to_ids,
    )

@function_tool
def update_claim_graph(
    path: Optional[str] = None,
    claim_id: str = "thesis",
    claim_text: str = "",
    parent_id: Optional[str] = None,
    support: Optional[List[str]] = None,
    status: str = "unlinked",
    notes: str = "",
):
    """Add or update a claim entry with support references."""
    claim_path = path or os.path.join(os.environ.get("AISC_BASE_FOLDER", ""), "claim_graph.json")
    return ClaimGraphTool().use_tool(
        path=claim_path,
        claim_id=claim_id,
        claim_text=claim_text,
        parent_id=parent_id,
        support=support,
        status=status,
        notes=notes,
    )

@function_tool
def check_claim_graph(path: Optional[str] = None):
    """Check claim_graph.json for claims lacking supporting evidence."""
    claim_path = path or os.path.join(os.environ.get("AISC_BASE_FOLDER", ""), "claim_graph.json")
    return ClaimGraphCheckTool().use_tool(path=claim_path)

@function_tool
def interpret_biology(base_folder: Optional[str] = None, config_path: Optional[str] = None):
    """Generate interpretation.json/md for theoretical biology runs."""
    base = base_folder or os.environ.get("AISC_BASE_FOLDER", "")
    base_path = Path(base)
    # If someone passes experiment_results/, lift to run root
    if base_path.name == "experiment_results":
        base_path = base_path.parent
    base = str(base_path)

    # Prefer explicit config path; otherwise try env, then repo-root default.
    repo_root = Path(__file__).resolve().parent
    cfg_default = base_path / "bfts_config.yaml"
    cfg_candidates = [
        config_path,
        os.environ.get("AISC_CONFIG_PATH", ""),
        str(cfg_default),
        str(repo_root / "bfts_config.yaml"),
        "bfts_config.yaml",
    ]
    cfg = next((c for c in cfg_candidates if c and os.path.exists(c)), cfg_candidates[-1])

    return {
        "success": interpret_biological_results(
            base_folder=base,
            config_path=cfg,
        ),
        "base_folder": base,
        "config_path": cfg,
    }


@function_tool
def get_run_paths():
    """Return canonical paths for the active run so agents avoid guessing directories."""
    base = os.environ.get("AISC_BASE_FOLDER", "")
    exp = os.environ.get("AISC_EXP_RESULTS", "")
    return {
        "base_folder": base,
        "experiment_results": exp,
        "figures": os.path.join(base, "figures") if base else "",
        "graphs": os.path.join(exp, "graphs") if exp else "",
        "claim_graph": os.path.join(base, "claim_graph.json") if base else "",
    }


@function_tool
def resolve_path(path: str, must_exist: bool = True, allow_dir: bool = False):
    """
    Resolve a filename against the current run folders (experiment_results/base).
    GUARDRAIL: If file not found, scan directory for fuzzy matches to suggest alternatives.
    """
    try:
        p = BaseTool.resolve_input_path(path, must_exist=must_exist, allow_dir=allow_dir)
        return {"resolved_path": str(p)}
    except FileNotFoundError as e:
        # Fuzzy matching logic
        if must_exist:
            d, f = os.path.split(path)
            # Search in experiment_results by default if d is empty
            search_dir = BaseTool.resolve_output_dir(d if d else None)
            if search_dir.exists():
                candidates = os.listdir(search_dir)
                matches = difflib.get_close_matches(f, candidates, n=3, cutoff=0.6)
                if matches:
                    return {"error": f"File '{path}' not found. Did you mean: {', '.join(matches)}?"}
        raise e


@function_tool
def list_artifacts(suffix: Optional[str] = None, subdir: Optional[str] = None):
    """
    List artifacts under experiment_results (optionally a subdir) with optional suffix filter.
    Agents should use this before selecting files.
    """
    # Default root is the run's experiment_results
    exp_root = BaseTool.resolve_output_dir(None)
    roots: List[Path] = []
    if subdir:
        sub_path = Path(subdir)
        roots.append(sub_path if sub_path.is_absolute() else exp_root / sub_path)
        # Also try under base folder if provided
        base = os.environ.get("AISC_BASE_FOLDER", "")
        if base:
            roots.append(Path(base) / sub_path)
            roots.append(Path(base) / "experiment_results" / sub_path)
    else:
        roots.append(exp_root)

    root = next((r for r in roots if r.exists()), roots[0])
    files: List[str] = []
    try:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file():
                    if suffix and not str(p).endswith(suffix):
                        continue
                    try:
                        rel = p.relative_to(root)
                        files.append(str(rel))
                    except Exception:
                        files.append(str(p))
        else:
            return {"root": str(root), "files": files, "warning": "root_missing"}
        return {"root": str(root), "files": files}
    except Exception as exc:
        return {"root": str(root), "files": files, "error": f"list_artifacts failed: {exc}"}


@function_tool
def read_artifact(path: str, summary_only: bool = False, head_lines: Optional[int] = None):
    """
    Resolve and read a small artifact (json/text). Use for configs/metadata, not large binaries.
    - summary_only=True: for large JSON, return top-level keys + types instead of full payload.
    - head_lines: return only the first N lines/items (bypasses size guard for text/JSON).
    """
    p = BaseTool.resolve_input_path(path, allow_dir=False)
    max_bytes = 1_000_000  # ~1 MB

    try:
        size = p.stat().st_size
        if size > max_bytes and head_lines is None:
            if summary_only and p.suffix.lower() == ".json":
                with open(p) as f:
                    try:
                        data = json.load(f)
                    except Exception as exc:
                        return {"error": f"Failed to parse JSON for summary: {exc}"}
                if isinstance(data, dict):
                    summary = {k: type(v).__name__ for k, v in list(data.items())[:20]}
                    return {
                        "path": str(p),
                        "size_bytes": size,
                        "summary": summary,
                        "note": "Summary only; file exceeds inline limit."
                    }
            return {
                "error": f"Artifact too large to inline ({size} bytes > {max_bytes}). "
                         "Use summary_only=True, head_lines, or a dedicated tool."
            }
    except Exception:
        size = None

    suffix = p.suffix.lower()
    if suffix == ".json":
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception as exc:
            return {"error": f"Failed to parse JSON: {exc}"}

        if head_lines is not None:
            if isinstance(data, list):
                return {
                    "path": str(p),
                    "type": "json_list_head",
                    "items": data[: head_lines],
                    "total_items": len(data),
                }
            if isinstance(data, dict):
                return {
                    "path": str(p),
                    "type": "json_dict_keys",
                    "keys": list(data.keys())[: max(head_lines, 1)],
                }
        if summary_only and isinstance(data, dict):
            return {k: type(v).__name__ for k, v in data.items()}
        return data

    # Text-like
    if head_lines is not None:
        lines: List[str] = []
        consumed = 0
        try:
            with open(p, "r", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= head_lines:
                        break
                    if consumed >= max_bytes:
                        break
                    line = line.rstrip("\n")
                    consumed += len(line.encode("utf-8")) + 1
                    lines.append(line)
            result: Dict[str, Any] = {"path": str(p), "type": "text_head", "head": lines}
            if size is not None and consumed < size:
                result["note"] = "truncated"
            return result
        except Exception as exc:
            return {"error": str(exc)}

    with open(p) as f:
        return f.read()


@function_tool
def head_artifact(path: str, max_lines: int = 20, max_bytes: int = 200_000):
    """
    Return the top of a file without loading it fully. Supports text/CSV/JSON.
    - For text/CSV: returns first max_lines lines (trimmed to max_bytes total).
    - For JSON list: returns first items up to max_lines.
    - For JSON dict: returns top-level keys.
    """
    p = BaseTool.resolve_input_path(path, allow_dir=False)
    info: Dict[str, Any] = {"path": str(p)}
    try:
        size = p.stat().st_size
        info["size_bytes"] = size
    except Exception:
        size = None

    suffix = p.suffix.lower()
    if suffix == ".json":
        try:
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, list):
                info["type"] = "json_list_head"
                info["items"] = data[:max_lines]
            elif isinstance(data, dict):
                info["type"] = "json_dict_keys"
                info["keys"] = list(data.keys())[: max_lines * 2]
            else:
                info["type"] = "json_scalar"
                info["value"] = data
            return info
        except Exception as exc:
            info["error"] = f"Failed to parse JSON: {exc}"
            return info

    # Text-like fallback (CSV, txt, md, log, yaml, etc.)
    lines: List[str] = []
    consumed = 0
    try:
        with open(p, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines or consumed >= max_bytes:
                    break
                line = line.rstrip("\n")
                consumed += len(line.encode("utf-8")) + 1
                lines.append(line)
        info["type"] = "text_head"
        info["head"] = lines
        if size is not None and consumed < size:
            info["note"] = "truncated"
        return info
    except Exception as exc:
        info["error"] = str(exc)
        return info


@function_tool
def read_npy_artifact(path: str, max_elements: int = 50_000, max_bytes: int = 5_000_000):
    """
    Load a small .npy file safely and return its data or shape.
    - If file too large (elements or bytes), returns metadata (shape, dtype) instead of full data.
    - Only supports .npy; refuses pickled objects.
    """
    p = BaseTool.resolve_input_path(path, allow_dir=False)
    try:
        size = p.stat().st_size
    except Exception:
        size = None
    if size is not None and size > max_bytes:
        return {"path": str(p), "error": f"file too large ({size} bytes)", "size_bytes": size}
    try:
        import numpy as np  # type: ignore
        arr = np.load(p, allow_pickle=False)
        shape = tuple(arr.shape)
        dtype = str(arr.dtype)
        if arr.size > max_elements:
            return {"path": str(p), "shape": shape, "dtype": dtype, "note": "too large, data omitted"}
        return {"path": str(p), "shape": shape, "dtype": dtype, "data": arr.tolist()}
    except Exception as exc:
        return {"path": str(p), "error": str(exc)}


@function_tool
def validate_per_compartment_outputs(sim_dir: str):
    """
    Validate standardized per-compartment outputs under a simulation folder.
    Expects per_compartment.npz (binary_states, continuous_states, time), node_index_map.json, topology_summary.json.
    Returns shapes/status and any detected errors.
    """
    return validate_per_compartment_outputs_internal(sim_dir)


@function_tool
def summarize_artifact(path: str, max_lines: int = 5):
    """
    Return a lightweight summary of a file without loading full contents.
    Supports: .json (keys), .csv (first lines), .npy/.npz (shape/keys), .gpickle (nodes/edges), .txt (head).
    """
    p = BaseTool.resolve_input_path(path, allow_dir=False)
    info: Dict[str, Any] = {"path": str(p)}
    try:
        size = p.stat().st_size
        info["size_bytes"] = size
    except Exception:
        pass

    suffix = p.suffix.lower()
    try:
        if suffix in {".json"}:
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, dict):
                info["type"] = "json"
                info["keys"] = list(data.keys())[:20]
            elif isinstance(data, list):
                info["type"] = "json_array"
                info["length"] = len(data)
                if data and isinstance(data[0], dict):
                    info["first_keys"] = list(data[0].keys())[:20]
        elif suffix in {".csv"}:
            info["type"] = "csv"
            lines = []
            with open(p) as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip("\n"))
            info["head"] = lines
        elif suffix in {".npy", ".npz"}:
            import numpy as np  # type: ignore
            arr = np.load(p, allow_pickle=True)
            # np.savez returns an NpzFile; otherwise numpy array
            if hasattr(arr, "files"):
                info["type"] = "npz"
                info["keys"] = list(arr.files)
                if arr.files:
                    key = arr.files[0]
                    info["first_array_shape"] = np.shape(arr[key])
            else:
                info["type"] = "npy"
                info["shape"] = np.shape(arr)
        elif suffix in {".gpickle", ".pkl", ".pickle"}:
            import networkx as nx  # type: ignore
            G = nx.read_gpickle(p)  # type: ignore[attr-defined]
            info["type"] = "gpickle_graph"
            info["nodes"] = G.number_of_nodes()
            info["edges"] = G.number_of_edges()
        elif suffix in {".txt", ".md"}:
            info["type"] = "text"
            lines = []
            with open(p) as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip("\n"))
            info["head"] = lines
        else:
            info["type"] = "unknown"
    except Exception as exc:
        info["error"] = str(exc)
    return info


@function_tool
def reserve_output(name: str, subdir: Optional[str] = None):
    """
    Return a canonical output path under experiment_results (or a subdir), auto-uniqued and sanitized.
    Agents must use this instead of manual path joining. Rejects traversal and routes to _unrouted on failure.
    """
    target, quarantined, note = resolve_output_path(subdir=subdir, name=name)
    result = {"reserved_path": str(target), "quarantined": quarantined}
    if note:
        result["note"] = note
    return result


@function_tool
def append_manifest(name: str, metadata_json: Optional[str] = None, allow_missing: bool = False):
    """
    Append an entry to the run's file manifest (experiment_results/file_manifest.json).
    Pass metadata as a JSON string (e.g., '{"type":"figure","source":"analyst"}').
    Creates the manifest file if missing.
    """
    return _append_manifest_entry(name=name, metadata_json=metadata_json, allow_missing=allow_missing)


def _read_manifest_entries() -> List[Dict[str, Any]]:
    """Helper to read manifest entries without tool wrapper."""
    exp_dir = BaseTool.resolve_output_dir(None)
    manifest_path = exp_dir / "file_manifest.json"
    manifest_map = _load_manifest_map(manifest_path)
    entries = list(manifest_map.values())
    entries.sort(key=lambda e: e.get("path", ""))
    return entries


@function_tool
def read_manifest_entry(path_or_name: str):
    """
    Read a single manifest entry by path key or filename (basename).
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    manifest_path = exp_dir / "file_manifest.json"
    manifest_map = _load_manifest_map(manifest_path)
    if not manifest_map:
        return {"error": "Manifest empty or missing", "path": str(manifest_path)}

    # Exact path match first
    entry = manifest_map.get(path_or_name)
    if entry:
        return {"path": path_or_name, "entry": entry}

    # Fallback: match by basename
    matches = []
    for p, e in manifest_map.items():
        if os.path.basename(p) == path_or_name:
            matches.append({"path": p, "entry": e})
    if matches:
        return {"matches": matches}
    return {"error": "Not found", "path_or_name": path_or_name}


@function_tool
def check_manifest():
    """
    Validate manifest entries: report missing files, entries lacking type, and duplicate basenames.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    manifest_path = exp_dir / "file_manifest.json"
    manifest_map = _load_manifest_map(manifest_path)
    if not manifest_map:
        return {"error": "Manifest empty or missing", "path": str(manifest_path)}

    missing = []
    missing_type = []
    by_name: Dict[str, list[str]] = {}
    for path, entry in manifest_map.items():
        try:
            exists = Path(path).exists()
        except Exception:
            exists = False
        if not exists:
            missing.append(path)
        if not entry.get("type"):
            missing_type.append(path)
        name = entry.get("name") or os.path.basename(path)
        by_name.setdefault(name, []).append(path)

    duplicates = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    health_entries: List[Dict[str, Any]] = []
    if missing:
        health_entries.append({"missing_files": missing})
    if missing_type:
        health_entries.append({"missing_type": missing_type})
    if duplicates:
        health_entries.append({"duplicate_names": duplicates})
    if health_entries:
        log_missing_or_corrupt(health_entries)
    return {
        "manifest_path": str(manifest_path),
        "n_entries": len(manifest_map),
        "missing_files": missing,
        "missing_type": missing_type,
        "duplicate_names": duplicates,
    }


@function_tool
def read_manifest():
    """Read the run's file manifest if present (experiment_results/file_manifest.json)."""
    exp_dir = BaseTool.resolve_output_dir(None)
    manifest_path = exp_dir / "file_manifest.json"
    return {"manifest_path": str(manifest_path), "entries": _read_manifest_entries()}


def _write_text_artifact_raw(name: str, content: str, subdir: Optional[str] = None, metadata_json: Optional[str] = None) -> Dict[str, Any]:
    # Normalize common nested paths (e.g., figures/figures, graphs/graphs)
    norm_name = name
    for dup in ("figures/figures/", "graphs/graphs/", "derived/derived/", "processed/processed/"):
        if norm_name.startswith(dup):
            norm_name = norm_name[len(dup) - len(dup.split('/')[-1]) - 1 :]  # strip duplicate prefix
    # If name already starts with figures/ or graphs/ and we're writing into those roots, strip the prefix
    if subdir == "figures" and norm_name.startswith("figures/"):
        norm_name = norm_name[len("figures/") :]
    if subdir in ("graphs", "derived", "processed") and norm_name.startswith(f"{subdir}/"):
        norm_name = norm_name[len(f"{subdir}/") :]

    path, quarantined, note = resolve_output_path(subdir=subdir, name=norm_name)
    if str(path).lower().endswith(".pdf"):
        path, warning = _render_pdf_or_markdown(path, content)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    if metadata_json:
        _append_manifest_entry(name=str(path), metadata_json=metadata_json, allow_missing=True)
    result: Dict[str, Any] = {"path": str(path), "quarantined": quarantined}
    if note:
        result["note"] = note
    if "warning" in locals() and warning:  # type: ignore[name-defined]
        result["warning"] = warning
    return result


@function_tool
def write_text_artifact(name: str, content: str, subdir: Optional[str] = None, metadata_json: Optional[str] = None):
    """
    Write text content to a file under the run (default experiment_results or a subdir) and return its path.
    """
    return _write_text_artifact_raw(name=name, content=content, subdir=subdir, metadata_json=metadata_json)


@function_tool
def write_interpretation_text(content: str, filename: str = "theory_interpretation.txt"):
    """
    Convenience: save interpretation text to experiment_results/<filename> (default theory_interpretation.txt).
    """
    return _write_text_artifact_raw(name=filename, content=content, subdir=None, metadata_json='{"type":"interpretation","source":"interpreter"}')


@function_tool
def write_figures_readme(content: str, filename: str = "README.md"):
    """
    Convenience: save a figures README under figures/ (default README.md).
    """
    # Force figures root
    return _write_text_artifact_raw(
        name=os.path.join("figures", filename),
        content=content,
        subdir=None,
        metadata_json='{"type":"readme","source":"analyst"}',
    )


@function_tool
def read_note(name: str = "pi_notes.md"):
    """
    Read a note file from the canonical run location (pi_notes.md or user_inbox.md). Returns empty string if missing.
    """
    result = read_note_file(name=name)
    if result.get("error"):
        return result
    return result


@function_tool
def write_pi_notes(content: str, name: str = "pi_notes.md"):
    """
    Overwrite PI notes in the canonical location (experiment_results/pi_notes.md) and refresh root symlink.
    """
    return write_note_file(content=content, name=name, append=False)


@function_tool
def check_status(status_path: Optional[str] = None, glob_pattern: str = "*.status.json"):
    """
    Inspect simulation/status files. If a path is provided, return that file's JSON. Otherwise, list all matching status files under experiment_results.
    """
    root = BaseTool.resolve_output_dir(None)
    if status_path:
        p = BaseTool.resolve_input_path(status_path, allow_dir=False)
        with open(p) as f:
            return {"path": str(p), "status": json.load(f)}
    matches = list(root.rglob(glob_pattern))
    statuses = []
    for m in matches:
        try:
            with open(m) as f:
                statuses.append({"path": str(m), "status": json.load(f)})
        except Exception as exc:
            statuses.append({"path": str(m), "error": str(exc)})
    return {"root": str(root), "matches": statuses}


@function_tool
def get_artifact_index(max_entries: int = 2000):
    """
    Build a lightweight index of artifacts under experiment_results and include manifest entries if present.
    """
    root = BaseTool.resolve_output_dir(None)
    manifest = _read_manifest_entries()
    files: List[Dict[str, Any]] = []
    try:
        count = 0
        for p in root.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(root))
                try:
                    size = p.stat().st_size
                except Exception:
                    size = None
                files.append({"path": rel, "suffix": p.suffix.lower(), "size": size})
                count += 1
                if count >= max_entries:
                    break
    except Exception as exc:
        return {"root": str(root), "manifest": manifest, "error": f"index failed: {exc}"}
    return {"root": str(root), "manifest": manifest, "files": files}


@function_tool
def run_ruff():
    """Run ruff check . from repo root and return output (non-fatal if missing)."""
    return _run_cli_tool("ruff", "check .")


@function_tool
def run_pyright():
    """Run pyright from repo root and return output (non-fatal if missing)."""
    return _run_cli_tool("pyright")


@function_tool
def coder_create_python(file_path: str, content: str):
    """
    Safely create/update a Python file under the current run folder. Paths are anchored to AISC_BASE_FOLDER to avoid writing elsewhere.
    GUARDRAIL: Check syntax via AST before saving.
    """
    # GUARDRAIL: Check syntax first
    try:
        ast.parse(content)
    except SyntaxError as e:
        return {"error": f"SyntaxError in python code at line {e.lineno}: {e.msg}. File not saved."}

    base = os.environ.get("AISC_BASE_FOLDER", "")
    if not base:
        raise ValueError("AISC_BASE_FOLDER is not set; cannot determine safe write location.")
    base_path = Path(base).resolve()
    fp = Path(file_path)
    # Resolve relative paths against CWD to detect if they already include the base folder
    if not fp.is_absolute():
        fp = (Path.cwd() / fp).resolve()
    # If the provided path is already inside the base, use it; otherwise anchor to base_path
    if fp.is_relative_to(base_path):
        target = fp
    else:
        target = (base_path / file_path).resolve()
        try:
            target.relative_to(base_path)
        except Exception:
            raise ValueError(f"Refusing to write outside run folder: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        f.write(content)
    return {"path": str(target), "bytes_written": len(content)}

# --- NEW TOOLS for Interactive Governance ---

@function_tool
def wait_for_human_review(artifact_name: str, description: str = ""):
    """
    Pause execution to request human review of a specific artifact (e.g., implementation plan).
    If human-in-the-loop mode is off, this just logs the request and auto-approves.
    """
    interactive = os.environ.get("AISC_INTERACTIVE", "false").lower() == "true"
    msg = f"⏸️  REVIEW REQUESTED: {artifact_name}\n   Context: {description}"
    
    if not interactive:
        return f"Auto-approved (Interactive mode OFF). Logged review request for {artifact_name}."
    
    print(f"\n{msg}")
    print("   >> Press ENTER to approve and continue, or Ctrl+C to abort.")
    try:
        input()
        return f"User approved {artifact_name}."
    except KeyboardInterrupt:
        raise SystemExit("User aborted execution during review.")

@function_tool
def check_user_inbox():
    """
    Check for new asynchronous feedback from the user in 'user_inbox.md'.
    Returns the content of the inbox if present, or 'Inbox empty'.
    """
    inbox = read_note_file("user_inbox.md")
    if inbox.get("error"):
        return f"Error reading inbox: {inbox['error']}"
    content = inbox.get("content", "").strip()
    if not content:
        return "Inbox empty."
    return f"USER MESSAGE: {content}"

# --- Agent Definitions ---

def build_team(model: str, idea: Dict[str, Any], dirs: Dict[str, str]):
    """
    Constructs the agents with strict context partitioning.
    """
    common_settings = ModelSettings(tool_choice="auto")
    role_max_turns = 40  # cap for sub-agent turn depth when invoked as a tool
    # Extract richer context from Idea JSON and format lists for Prompt ingestion
    title = idea.get('Title', 'Project')
    abstract = idea.get('Abstract', '')
    hypothesis = idea.get('Short Hypothesis', 'None')
    related_work = idea.get('Related Work', 'None provided.')
    
    experiments_plan = format_list_field(idea.get('Experiments', []))
    risk_factors = format_list_field(idea.get('Risk Factors and Limitations', []))

    # --- PRE-CALCULATE PATHS FOR PROMPT INJECTION (Saves ~1 turn/agent) ---
    # Note: dirs['base'] and dirs['results'] are already resolved strings.
    path_context = (
        f"SYSTEM CONTEXT: Run Root='{dirs['base']}', Exp Results='{dirs['results']}'. "
        f"Figures='{os.path.join(dirs['base'], 'figures')}'. "
        "Use these paths directly; do NOT call get_run_paths. "
        "Assume provided input paths exist; only list_artifacts if path is missing."
    )
    path_guardrails = (
        "FILE IO POLICY: Build write paths via 'reserve_output' or 'write_text_artifact'; do not hand-roll paths or use '../'. "
        "Outputs are anchored to experiment_results; if a directory is unavailable, writes are auto-rerouted to experiment_results/_unrouted with a manifest note."
    )
    
    # --- STANDARD REFLECTION PROMPT ---
    reflection_instruction = (
        "SELF-REFLECTION: When finished (or if stuck), ask: 'What missing tool or knowledge would have made this trivial?' "
        "If you have a concrete, new insight, log it via manage_project_knowledge(action='add', category='reflection', "
        "observation='<your specific friction>', solution='<your specific fix>', actor='<your role name>'). "
        "Do NOT log boilerplate or repeated reflections; skip logging if nothing new. Use your actual role name (e.g., 'PI', 'Modeler')."
    )

    # --- PROOF OF WORK INSTRUCTION ---
    proof_of_work_instruction = (
        "PROOF OF WORK: For every significant result (data or figure), you must write a corresponding "
        "`_verification.md` file. This file must list: 1) Input files used, 2) Key parameters/filters applied, "
        "3) Explicit validation checks (e.g., 'Checked x > 0: Pass'). Do not output the artifact without this proof."
    )

    # 1. The Archivist (Scope: Literature & Claims)
    archivist = _make_agent(
        name="Archivist",
        instructions=(
            f"You are an expert Literature Curator.\n"
            f"Goal: Verify novelty of '{title}' and map claims to citations.\n"
            f"Context: {abstract}\n"
            f"Related Work to Consider: {related_work}\n"
            f"{path_context}\n{path_guardrails}\n"  # INJECT PATHS
            "Directives:\n"
            "1. Use 'assemble_lit_data' or 'search_semantic_scholar' to gather papers.\n"
            "2. Maintain a claim graph via 'update_claim_graph' when mapping evidence.\n"
            "3. Output 'lit_summary.json' to experiment_results/.\n"
            "4. If you create or deeply analyze artifacts not yet in the manifest, log them with 'append_manifest' (name + metadata/description).\n"
            "5. CRITICAL: If no papers are found, report FAILURE. Do not invent 'TBD' citations.\n"
            f"6. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            list_artifacts,
            read_artifact,
            reserve_output,
            append_manifest,
            read_manifest,
            read_manifest_entry,
            check_manifest,
            assemble_lit_data,
            validate_lit_summary,
            search_semantic_scholar,
            update_claim_graph,
            manage_project_knowledge,
        ],
        model=model,
        settings=common_settings,
    )

    # 2. The Modeler (Scope: Python & Simulation)
    modeler = _make_agent(
        name="Modeler",
        instructions=(
            f"You are an expert Computational Biologist.\n"
            f"Goal: Execute simulations for '{title}'.\n"
            f"Hypothesis: {hypothesis}\n"
            f"Experimental Plan:\n{experiments_plan}\n"
            f"{path_context}\n{path_guardrails}\n"  # INJECT PATHS
            "Directives:\n"
            "1. You do NOT care about LaTeX or writing styles. Focus on DATA.\n"
            "2. Build graphs ('build_graphs'), run baselines ('run_biological_model') or custom sims ('run_comp_sim').\n"
            "3. Explore parameter space using 'run_sensitivity_sweep' and 'run_intervention_tests'.\n"
            "4. Ensure parameter sweeps cover the range specified in the hypothesis.\n"
            "5. Save raw outputs to experiment_results/.\n"
            "6. Always produce arrays for each (baseline, transport, seed): prefer export_arrays during sim; otherwise immediately run 'sim_postprocess' on the produced sim.json so failure_matrix.npy/time_vector.npy/nodes_order_*.txt exist before marking the run complete. Every run must also emit per_compartment.npz + node_index_map.json + topology_summary.json (binary_states, continuous_states/time); validate with validate_per_compartment_outputs before marking status=complete.\n"
            "7. Use the transport run manifest (read_transport_manifest / update_transport_manifest); consult it before reruns and update it after completing or failing a run. Do not mark status=complete unless arrays + sim.json + sim.status.json all exist; otherwise mark partial and note missing files.\n"
            "7b. Resolve baselines via resolve_baseline_path before running batches; only pass graph baselines (.npy/.npz/.graphml/.gpickle/.gml), never sim.json.\n"
            "7c. Process one baseline per call; if run_recipe.json exists under experiment_results/simulations/transport_runs, load it for template/output roots instead of embedding long paths. Append ensemble CSV incrementally and write per-baseline status to pi_notes/user_inbox.\n"
            "8. Before calling 'append_manifest', ask if appending new info to the artifact's record in the manifest adds new value (new file or materially new analysis/description). Log only when yes, with name + metadata/description.\n"
            "9. If you encounter simulation failures or parameter issues, log them to Project Knowledge via 'manage_project_knowledge'.\n"
            f"10. {proof_of_work_instruction}\n"
            f"11. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            get_artifact_index,
            list_artifacts,
            read_artifact,
            summarize_artifact,
            reserve_output,
            append_manifest,
            read_manifest,
            read_manifest_entry,
            check_manifest,
            build_graphs,
            run_biological_model,
            run_comp_sim,
            sim_postprocess,
            run_sensitivity_sweep,
            run_intervention_tests,
            run_transport_batch,
            scan_transport_manifest,
            read_transport_manifest,
            resolve_baseline_path,
            resolve_sim_path,
            update_transport_manifest,
            mirror_artifacts,
            read_npy_artifact,
            validate_per_compartment_outputs,
            manage_project_knowledge,
            write_text_artifact, # Added for writing logs
        ], 
        model=model,
        settings=common_settings,
    )

    # 3. The Analyst (Scope: Visualization & Validation)
    analyst = _make_agent(
        name="Analyst",
        instructions=(
            "You are an expert Scientific Visualization Expert.\n"
            "Goal: Convert simulation data into PLOS-quality figures.\n"
            f"{path_context}\n{path_guardrails}\n"  # INJECT PATHS
            "Directives:\n"
            "1. Read data from provided input paths. Do NOT list files to find them; assume the path is correct.\n"
            "2. Assert that the data supports the hypothesis BEFORE plotting. If data contradicts hypothesis, report this back immediately.\n"
            "3. Generate PNG/SVG files using 'run_biological_plotting'. Use 'sim_postprocess' if you need failure_matrix/time_vector/node order from sim.json before plotting.\n"
            "3b. Use the transport run manifest (read_transport_manifest / resolve_sim_path / update_transport_manifest) to decide what to plot or skip; resolve sim.json via resolve_sim_path instead of guessing paths, and error if the requested transport/seed is missing.\n"
            "3c. After plotting, mirror outputs into experiment_results/figures_for_manuscript using 'mirror_artifacts' (use prefix/suffix if name collisions occur). Do not leave final plots only in nested subfolders.\n"
            "3d. Before computing cluster/finite-size metrics, run validate_per_compartment_outputs on the sim folder; if per_compartment artifacts are missing or invalid, report and request rerun instead of plotting placeholders.\n"
            "4. Validate models vs lit via 'run_validation_compare' and use 'run_biological_stats' for significance/enrichment.\n"
            "5. Before calling 'append_manifest', ask if the artifact adds new value (new figure/analysis). Log only when yes, with name + metadata/description.\n"
            "6. Check Project Knowledge for visualization standards (e.g., colormaps) before starting.\n"
            f"7. {proof_of_work_instruction}\n"
            f"8. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            get_artifact_index,
            list_artifacts,
            read_artifact,
            summarize_artifact,
            reserve_output,
            append_manifest,
            read_manifest,
            read_manifest_entry,
            check_manifest,
            scan_transport_manifest,
            read_transport_manifest,
            resolve_baseline_path,
            run_biological_plotting,
            run_validation_compare,
            run_biological_stats,
            sim_postprocess,
            run_transport_batch,
            resolve_sim_path,
            update_transport_manifest,
            write_figures_readme,
            write_text_artifact,
            graph_diagnostics,
            mirror_artifacts,
            read_npy_artifact,
            validate_per_compartment_outputs,
            manage_project_knowledge,
        ],
        model=model,
        settings=common_settings,
    )

    # 4. The Reviewer (Scope: Logic & Completeness)
    reviewer = _make_agent(
        name="Reviewer",
        instructions=(
            "You are an expert Holistic Reviewer.\n"
            "Goal: Identify logical gaps and structural flaws.\n"
            f"Risk Factors & Limitations to Check:\n{risk_factors}\n"
            f"{path_context}\n{path_guardrails}\n"  # INJECT PATHS
            "Directives:\n"
            "1. Read the manuscript draft using 'read_manuscript'.\n"
            "2. Check claim support using 'check_claim_graph' and sanity-check stats with 'run_biological_stats' if needed.\n"
            "3. Check consistency: Does Figure 3 actually support the claim in paragraph 2?\n"
            "4. If gaps exist, report them clearly to the PI.\n"
            "5. Only report 'NO GAPS' if the PDF validates completely.\n"
            "6. If you create or materially analyze artifacts, log them with 'append_manifest' (name + metadata/description) only when it adds value.\n"
            "7. VERIFY PROOF OF WORK: Reject any result artifact that lacks a corresponding `_verification.md` or `_log.md` file explaining how it was derived.\n"
            f"8. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            get_artifact_index,
            list_artifacts,
            read_artifact,
            summarize_artifact,
            reserve_output,
            append_manifest,
            read_manifest,
            read_manifest_entry,
            check_manifest,
            read_manuscript,
            check_claim_graph,
            run_biological_stats,
            write_text_artifact,
            manage_project_knowledge,
        ],
        model=model,
        settings=common_settings,
    )

    # 5. The Interpreter (Scope: Theoretical Interpretation)
    interpreter = _make_agent(
        name="Interpreter",
        instructions=(
            "You are an expert Mathematical-Biological Interpreter.\n"
            "Goal: Produce interpretation.json/md for theoretical biology projects.\n"
            f"{path_context}\n{path_guardrails}\n"  # INJECT PATHS
            "Directives:\n"
            "1. Call 'interpret_biology' only when biology.research_type == theoretical.\n"
            "2. Use experiment summaries and idea text; do NOT hallucinate unsupported claims.\n"
            "3. If interpretation fails, report the error clearly.\n"
            "4. Use 'write_text_artifact' to save interpretations (e.g., theory_interpretation.txt) under experiment_results/.\n"
            "5. Before calling 'append_manifest', ask if the artifact adds new value (new interpretation or substantial edit). Log only when yes, with name + metadata/description.\n"
            f"6. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            get_artifact_index,
            list_artifacts,
            read_artifact,
            summarize_artifact,
            reserve_output,
            append_manifest,
            read_manifest,
            read_manifest_entry,
            check_manifest,
            write_text_artifact,
            write_interpretation_text,
            write_figures_readme,
            check_status,
            interpret_biology,
            manage_project_knowledge,
        ],
        model=model,
        settings=common_settings,
    )

    # 6. The Coder (Scope: Utility Code & Tooling)
    coder = _make_agent(
        name="Coder",
        instructions=(
            "You are an expert Utility Engineer.\n"
            "Goal: Write or update lightweight Python helpers/tools confined to this run folder.\n"
            f"{path_context}\n{path_guardrails}\n"  # INJECT PATHS
            "Directives:\n"
            "1. Use 'coder_create_python' to create/update files under the run root; do NOT write outside AISC_BASE_FOLDER.\n"
            "2. If you add tools/helpers, document them briefly and log via 'append_manifest'.\n"
            "3. Prefer small, dependency-light snippets; avoid large libraries or network access.\n"
            "4. If you need existing artifacts, list them with 'list_artifacts' or read via 'read_artifact' (use summary_only for large files).\n"
            "5. Log code patterns or library constraints to Project Knowledge.\n"
            f"6. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            list_artifacts,
            read_artifact,
            reserve_output,
            append_manifest,
            read_manifest,
            write_text_artifact,
            coder_create_python,
            run_ruff,
            run_pyright,
            manage_project_knowledge,
        ],
        model=model,
        settings=common_settings,
    )

    # 6. The Publisher (Scope: LaTeX & Compilation)
    publisher = _make_agent(
        name="Publisher",
        instructions=(
            "You are an expert Production Editor.\n"
            "Goal: Compile final PDF.\n"
            f"{path_context}\n{path_guardrails}\n"  # INJECT PATHS
            "Directives:\n"
            "1. Target the 'blank_theoretical_biology_latex' template.\n"
            "2. Integrate 'lit_summary.json' and figures into the text.\n"
            "3. Ensure compile success. Debug LaTeX errors autonomously.\n"
            f"4. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            list_artifacts,
            read_artifact,
            summarize_artifact,
            reserve_output,
            append_manifest,
            read_manifest,
            write_text_artifact,
            write_figures_readme,
            check_status,
            run_writeup_task,
            manage_project_knowledge,
        ],
        model=model,
        settings=common_settings,
    )

    # 7. The PI (Scope: Strategy & Handoffs)
    pi = Agent(
        name="Principal Investigator",
        instructions=(
            f"You are an expert Principal Investigator for project: {title}.\n"
            f"Hypothesis: {hypothesis}\n"
            "Responsibilities:\n"
            "0. Agents are stateless tools with a hard ~40-turn budget (including their tool calls). Do NOT send 'prepare' or 'wait until X' tasks. Delegate small, end-to-end units with concrete paths; if a job is large, split it into multiple invocations (e.g., one per batch of sims/plots) and ask the agent to persist outputs plus a brief status note to user_inbox.md/pi_notes.md before returning. You may spawn multiple parallel calls to the same role if that speeds work, as long as each request is end-to-end and self-contained. If you already know the relevant file paths or artifact names, include them in the prompt to save turn budget.\n"
            "1. STATE CHECK: First, use any injected context about 'pi_notes.md', 'user_inbox.md', or prior 'check_project_state' runs that appears in your system/user message. Only call 'read_note' or 'check_project_state' if you need a fresh snapshot beyond what is already provided.\n"
            "2. REVIEW KNOWLEDGE: Check 'manage_project_knowledge' for constraints or decisions before delegating.\n"
            "3. MITIGATE ITERATIVE GAP: Before complex phases (e.g., large simulations, drafting full sections), write an `implementation_plan.md` using `write_text_artifact` (default path: experiment_results/implementation_plan.md). Update the plan when priorities or completion status change—do not carry a stale plan forward. If `--human_in_the_loop` is active, call `wait_for_human_review` on this plan before proceeding.\n"
            "4. DELEGATE: Handoff to specialized agents based on missing artifacts. **MANDATORY: When calling a sub-agent, lookup the exact file paths first (via inspect_recent_manifest_entries or list_artifacts) and pass the EXACT PATH in the prompt. Do not ask them to 'find the file'.**\n"
            "   - Missing Lit Review -> Archivist\n"
            "   - Missing Data -> Modeler\n"
            "   - Missing Plots -> Analyst\n"
            "   - Theoretical Interpretation -> Interpreter\n"
            "   - Draft Exists -> Reviewer\n"
            "   - Validated & Ready -> Publisher\n"
            "5. ASYNC FEEDBACK: Call `check_user_inbox` frequently (e.g., between tasks) to see if the user has steered the project.\n"
            "6. HANDLE FAILURES: If a sub-agent reports error or max turns, call 'inspect_recent_manifest_entries(role=...)' to see what they accomplished before crashing. If artifacts exist, instruct the next run to continue from there rather than restarting.\n"
            "7. END OF RUN: Write a concise summary and next actions to 'pi_notes.md' using 'write_pi_notes' so it persists across resumes.\n"
            "8. TERMINATE: Stop only when Reviewer confirms 'NO GAPS' and PDF is generated."
        ),
        model=model,
        tools=[
            check_project_state,
            log_strategic_pivot,
            inspect_recent_manifest_entries,
            get_run_paths,
            resolve_path,
            list_artifacts,
            read_artifact,
            head_artifact,
            summarize_artifact,
            reserve_output,
            append_manifest,
            read_manifest,
            check_status,
            read_note,
            write_pi_notes,
            manage_project_knowledge,
            scan_transport_manifest,
            read_transport_manifest,
            resolve_baseline_path,
            resolve_sim_path,
            update_transport_manifest,
            mirror_artifacts,
            write_text_artifact,  # Added this tool to allow PI to write plans directly
            # New interactive tools
            wait_for_human_review,
            check_user_inbox,
            archivist.as_tool(tool_name="archivist", tool_description="Search literature.", max_turns=role_max_turns),
            modeler.as_tool(tool_name="modeler", tool_description="Run simulations.", max_turns=role_max_turns, custom_output_extractor=extract_run_output),
            analyst.as_tool(tool_name="analyst", tool_description="Create figures.", max_turns=role_max_turns, custom_output_extractor=extract_run_output),
            coder.as_tool(tool_name="coder", tool_description="Write/update helper code in run folder.", max_turns=role_max_turns, custom_output_extractor=extract_run_output),
            interpreter.as_tool(tool_name="interpreter", tool_description="Generate theoretical interpretation.", max_turns=role_max_turns, custom_output_extractor=extract_run_output),
            publisher.as_tool(tool_name="publisher", tool_description="Write and compile final publishable manuscript.", max_turns=role_max_turns, custom_output_extractor=extract_run_output),
            reviewer.as_tool(tool_name="reviewer", tool_description="Critique the draft.", max_turns=role_max_turns, custom_output_extractor=extract_run_output),
        ],
        model_settings=ModelSettings(tool_choice="required"),
    )

    return pi

# --- Main Execution Block ---

def main():
    args = parse_args()
    
    # Load Env
    if load_dotenv:
        load_dotenv(override=True)
    
    # Set Interactive Flag for Tools
    os.environ["AISC_INTERACTIVE"] = str(args.human_in_the_loop).lower()
    
    # Load Idea
    with open(args.load_idea) as f:
        idea_data = json.load(f)
    
    if isinstance(idea_data, list):
        if args.idea_idx < 0 or args.idea_idx >= len(idea_data):
            raise IndexError(f"idea_idx {args.idea_idx} out of range for {len(idea_data)} ideas")
        idea = idea_data[args.idea_idx]
    elif isinstance(idea_data, dict):
        idea = idea_data
    else:
        raise ValueError("Idea file must contain a JSON object or a list of objects")

    # Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_folder = args.base_folder

    if base_folder:
        if not os.path.exists(base_folder):
            raise FileNotFoundError(f"--base_folder '{base_folder}' does not exist")
        print(f"Restarting from existing folder: {base_folder}")
    else:
        base_folder = f"experiments/{timestamp}_{idea.get('Name', 'Project')}"
        
        # Handle resume
        if args.resume:
            # Simple logic: look for the most recent folder matching the name
            parent_dir = "experiments"
            if os.path.exists(parent_dir):
                candidates = sorted([d for d in os.listdir(parent_dir) if idea.get('Name', 'Project') in d])
                if candidates:
                    base_folder = os.path.join(parent_dir, candidates[-1])
                    print(f"Resuming from: {base_folder}")

    # Note: Directory creation is also handled in check_project_state, 
    # but we verify here for initial context setup.
    os.makedirs(base_folder, exist_ok=True)
    exp_results = osp.join(base_folder, "experiment_results")
    os.makedirs(exp_results, exist_ok=True)

    # Environment Variables for Tools
    os.environ["AISC_EXP_RESULTS"] = exp_results
    os.environ["AISC_BASE_FOLDER"] = base_folder
    os.environ["AISC_CONFIG_PATH"] = osp.join(base_folder, "bfts_config.yaml")

    # Ensure standard transport_runs README exists for this run
    _ensure_transport_readme(base_folder)
    # Generate per-run run_recipe.json from morphologies/templates
    _generate_run_recipe(base_folder)
    # Ensure canonical PI/user inbox files live under experiment_results with root-level symlinks
    _bootstrap_note_links()

    # Surface tool availability so PDF/analysis fallbacks are explicit
    caps = _report_capabilities()
    print(f"🛠️  Tool availability: pandoc={caps['tools'].get('pandoc')}, pdflatex={caps['tools'].get('pdflatex')}, "
          f"ruff={caps['tools'].get('ruff')}, pyright={caps['tools'].get('pyright')}, "
          f"pdf_engine_ready={caps['pdf_engine_ready']}")

    # Directories Dict for Agent Context
    dirs = {
        "base": base_folder,
        "results": exp_results
    }

    # Initialize Team
    pi_agent = build_team(args.model, idea, dirs)

    print(f"🧪 Launching ADCRT for '{idea.get('Title', 'Project')}'...")
    print(f"📂 Context: {base_folder}")
    if args.human_in_the_loop:
        print("🛑 Interactive Mode: Agents will pause for plan reviews.")

    # Input Prompt: Inject persistent memory (pi_notes) and user feedback (user_inbox)
    initial_prompt_parts = []
    
    # 1. Base instruction
    if args.input:
        initial_prompt_parts.append(args.input)
    else:
        initial_prompt_parts.append(
            f"Begin project '{idea.get('Name', 'Project')}'. \n"
            f"Current working directory is: {base_folder}\n"
            "Assess current state, preferring the injected summaries of pi_notes.md, user_inbox.md, and any prior check_project_state output above; only call tools again if you need a fresh snapshot, then begin delegation."
        )

    # 2. Inject Persistent Memory (PI Notes)
    pi_notes_snapshot = read_note_file("pi_notes.md")
    if pi_notes_snapshot.get("error"):
        print(f"⚠️ Failed to read pi_notes.md: {pi_notes_snapshot['error']}")
    else:
        notes_content = pi_notes_snapshot.get("content", "").strip()
        if notes_content:
            initial_prompt_parts.append(f"\n\n--- PERSISTENT MEMORY (pi_notes.md) ---\n{notes_content}")

    # 3. Inject User Inbox (Sticky Notes)
    user_inbox_snapshot = read_note_file("user_inbox.md")
    if user_inbox_snapshot.get("error"):
        print(f"⚠️ Failed to read user_inbox.md: {user_inbox_snapshot['error']}")
    else:
        inbox_content = user_inbox_snapshot.get("content", "").strip()
        if inbox_content:
            initial_prompt_parts.append(f"\n\n--- USER FEEDBACK (user_inbox.md) ---\n{inbox_content}")

    # 4. Inject implementation plan if present (avoid extra tool calls on resume)
    impl_plan_path = os.path.join(exp_results, "implementation_plan.md")
    if os.path.exists(impl_plan_path):
        try:
            with open(impl_plan_path, "r") as f:
                plan_content = f.read().strip()
            if plan_content:
                initial_prompt_parts.append(f"\n\n--- IMPLEMENTATION PLAN (experiment_results/implementation_plan.md) ---\n{plan_content}")
        except Exception as e:
            print(f"⚠️ Failed to read implementation_plan.md: {e}")

    initial_prompt = "\n".join(initial_prompt_parts)

    # Execution with Robust Timeout
    async def run_lab():
        return await Runner.run(
            pi_agent, 
            input=initial_prompt, 
            context={"work_dir": base_folder}, 
            max_turns=args.max_cycles
        )

    try:
        result = asyncio.run(asyncio.wait_for(run_lab(), timeout=args.timeout))
        print("✅ Experiment Cycle Completed.")
        # Log result
        with open(osp.join(base_folder, "run_log.txt"), "w") as f:
            f.write(str(result))
    except asyncio.TimeoutError:
        print(f"❌ Timeout reached ({args.timeout}s). System halted safely.")
        print("   Check 'experiment_results' for partial artifacts.")
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        # Dump error to log
        with open(osp.join(base_folder, "error_log.txt"), "w") as f:
            f.write(str(e))

if __name__ == "__main__":
    main()
