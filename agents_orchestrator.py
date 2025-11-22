# pyright: reportMissingImports=false
import argparse
import json
import os
import os.path as osp
from pathlib import Path
from datetime import datetime
import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING
try:
    from agents.types import RunResult
except ImportError:
    if TYPE_CHECKING:
        from agents.types import RunResult  # type: ignore  # noqa: F401
    else:
        class RunResult:  # minimal stub
            error: Any = None
            status: Any = None
            output: Any = None

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
from ai_scientist.tools.claim_graph import ClaimGraphTool
from ai_scientist.tools.claim_graph_checker import ClaimGraphCheckTool
from ai_scientist.tools.manuscript_reader import ManuscriptReaderTool
from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_biological_interpretation import interpret_biological_results

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
    return p.parse_args()

def _fill_output_dir(output_dir: Optional[str]) -> str:
    """
    Resolve output dir to the run-specific folder.
    - If explicit path is provided, use it.
    - Otherwise fall back to AISC_EXP_RESULTS.
    - If the chosen path is relative, anchor it under AISC_BASE_FOLDER.
    """
    target = output_dir or os.environ.get("AISC_EXP_RESULTS", "experiment_results")
    if not os.path.isabs(target):
        base = os.environ.get("AISC_BASE_FOLDER")
        if base and not target.startswith(base) and not target.startswith("experiments/"):
            target = os.path.join(base, target)
    return target


def _append_manifest_entry(name: str, metadata_json: Optional[str] = None, allow_missing: bool = False):
    """
    Internal helper to append to the run's manifest without invoking the tool wrapper.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    manifest_path = exp_dir / "file_manifest.json"
    manifest: List[Dict[str, Any]] = []
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
                if not isinstance(manifest, list):
                    manifest = []
        except Exception:
            manifest = []

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
    entry = {"name": name, "path": str(target_path), "metadata": meta}
    manifest.append(entry)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return {"manifest_path": str(manifest_path), "n_entries": len(manifest)}

def format_list_field(data: Any) -> str:
    """Helper to format JSON lists (like Experiments) into a clean string block for LLMs."""
    if isinstance(data, list):
        return "\n".join([f"- {item}" for item in data])
    return str(data)

# --- Tool Definitions (Wrappers for Agents SDK) ---

@function_tool
def check_project_state(base_folder: str) -> str:
    """
    Reads the project state to see what artifacts exist. 
    Creates the folder structure if it does not exist.
    """
    status_msg = "Folder existed"
    
    # Auto-creation logic
    if not os.path.exists(base_folder):
        try:
            os.makedirs(base_folder, exist_ok=True)
            # Also create the standard subfolder to save the agents a step
            exp_results = os.path.join(base_folder, "experiment_results")
            os.makedirs(exp_results, exist_ok=True)
            status_msg = f"Created new directory: {base_folder}"
        except Exception as e:
            return json.dumps({"error": f"Failed to create folder {base_folder}: {str(e)}"})
        
    exists = os.listdir(base_folder)
    exp_results = os.path.join(base_folder, "experiment_results")
    artifacts = os.listdir(exp_results) if os.path.exists(exp_results) else []
    
    return json.dumps({
        "status_message": status_msg,
        "root_files": exists,
        "artifacts": artifacts,
        "has_lit_review": "lit_summary.json" in artifacts,
        "has_data": any(x.endswith('.csv') for x in artifacts),
        "has_plots": any(x.endswith('.png') for x in artifacts),
        "has_draft": "manuscript.pdf" in exists or "manuscript.tex" in exists
    })

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
    graph_path: Optional[str] = None, # Optional because tool might auto-gen
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
        graph_path=graph_path,  # Require explicit path; agent must build/provide a graph
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
    if metadata_json and isinstance(res, dict):
        out = res.get("output_json")
        if isinstance(out, str):
            _append_manifest_entry(name=out, metadata_json=metadata_json, allow_missing=True)
    return res

@function_tool
def run_biological_plotting(solution_path: str, output_dir: Optional[str] = None, make_phase_portrait: bool = True):
    """Generates plots from simulation data."""
    res = RunBiologicalPlottingTool().use_tool(
        solution_path=solution_path,
        output_dir=_fill_output_dir(output_dir),
        make_phase_portrait=make_phase_portrait,
    )
    # Append outputs to manifest if possible
    for k, v in res.items():
        if isinstance(v, str) and v.endswith((".png", ".pdf", ".svg")):
            _append_manifest_entry(name=v, metadata_json='{"type":"figure","source":"analyst"}', allow_missing=True)
    return res

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
    # Log outputs to manifest
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
    if metadata_json and isinstance(res, dict):
        out = res.get("output_json")
        if isinstance(out, str):
            _append_manifest_entry(name=out, metadata_json=metadata_json, allow_missing=True)
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
    if metadata_json and isinstance(res, dict):
        out = res.get("output_csv")
        if isinstance(out, str):
            _append_manifest_entry(name=out, metadata_json=metadata_json, allow_missing=True)
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
    if metadata_json and isinstance(res, dict):
        out = res.get("output_json")
        if isinstance(out, str):
            _append_manifest_entry(name=out, metadata_json=metadata_json, allow_missing=True)
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
    """Resolve a filename against the current run folders (experiment_results/base)."""
    p = BaseTool.resolve_input_path(path, must_exist=must_exist, allow_dir=allow_dir)
    return {"resolved_path": str(p)}


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
def read_artifact(path: str, summary_only: bool = False):
    """
    Resolve and read a small artifact (json/text). Use for configs/metadata, not large binaries.
    If summary_only=True and JSON is detected, return top-level keys and basic shape info instead of full payload.
    """
    p = BaseTool.resolve_input_path(path, allow_dir=False)
    # Guardrail: avoid returning huge payloads
    max_bytes = 1_000_000  # ~1 MB
    try:
        size = p.stat().st_size
        if size > max_bytes:
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
                         "Use summary_only=True or process via a dedicated tool."
            }
    except Exception:
        pass
    if p.suffix.lower() in {".json"}:
        with open(p) as f:
            return json.load(f)
    with open(p) as f:
        return f.read()


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
    Return a canonical output path under experiment_results (or a subdir) without creating it.
    Agents should use this to avoid constructing paths manually.
    """
    base = BaseTool.resolve_output_dir(subdir or os.environ.get("AISC_EXP_RESULTS", "experiment_results"))
    path = base / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return {"reserved_path": str(path)}


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
    if not manifest_path.exists():
        return []
    with open(manifest_path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


@function_tool
def read_manifest():
    """Read the run's file manifest if present (experiment_results/file_manifest.json)."""
    exp_dir = BaseTool.resolve_output_dir(None)
    manifest_path = exp_dir / "file_manifest.json"
    return {"manifest_path": str(manifest_path), "entries": _read_manifest_entries()}


def _write_text_artifact_raw(name: str, content: str, subdir: Optional[str] = None, metadata_json: Optional[str] = None) -> Dict[str, Any]:
    root = BaseTool.resolve_output_dir(subdir or None)
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

    path = root / norm_name
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).lower().endswith(".pdf"):
        # Minimal PDF writer: single-page, Helvetica 10pt, text laid out top-down.
        lines = content.splitlines()
        x, y = 50, 750
        line_height = 14
        content_lines: list[str] = []
        for line in lines:
            if y < 50:
                break  # single-page fallback
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            content_lines.append(f"BT /F1 10 Tf {x} {y} Td ({escaped}) Tj ET")
            y -= line_height
        stream = "\n".join(content_lines).encode("latin-1", errors="replace")

        objects: list[bytes] = []
        objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
        objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
        objects.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n")
        objects.append(f"4 0 obj << /Length {len(stream)} >> stream\n".encode() + stream + b"\nendstream\nendobj\n")
        objects.append(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")

        pdf_parts = [b"%PDF-1.4\n"]
        offsets: list[int] = []
        for obj in objects:
            offsets.append(sum(len(p) for p in pdf_parts))
            pdf_parts.append(obj)
        xref_start = sum(len(p) for p in pdf_parts)
        pdf_parts.append(b"xref\n0 6\n0000000000 65535 f \n")
        for off in offsets:
            pdf_parts.append(f"{off:010d} 00000 n \n".encode())
        pdf_parts.append(b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n")
        pdf_parts.append(f"{xref_start}\n".encode())
        pdf_parts.append(b"%%EOF\n")

        path.write_bytes(b"".join(pdf_parts))
    else:
        with open(path, "w") as f:
            f.write(content)
    if metadata_json:
        _append_manifest_entry(name=str(path), metadata_json=metadata_json, allow_missing=True)
    return {"path": str(path)}


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
    return _write_text_artifact_raw(name=filename, content=content, subdir="figures", metadata_json='{"type":"readme","source":"analyst"}')


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
    try:
        return os.popen("cd /Users/maxa/AI-Scientist-v2 && ruff check .").read()
    except Exception as exc:
        return {"error": str(exc)}


@function_tool
def run_pyright():
    """Run pyright from repo root and return output (non-fatal if missing)."""
    try:
        return os.popen("cd /Users/maxa/AI-Scientist-v2 && pyright").read()
    except Exception as exc:
        return {"error": str(exc)}


@function_tool
def coder_create_python(file_path: str, content: str):
    """
    Safely create/update a Python file under the current run folder. Paths are anchored to AISC_BASE_FOLDER to avoid writing elsewhere.
    """
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

# --- Agent Definitions ---

def build_team(model: str, idea: Dict[str, Any], dirs: Dict[str, str]):
    """
    Constructs the agents with strict context partitioning.
    """
    common_settings = ModelSettings(tool_choice="auto")
    role_max_turns = 40  # cap for sub-agent turn depth when invoked as a tool

    async def extract_run_output(run_result: RunResult) -> str:
        """
        Custom extractor to return a concise summary for sub-agent tool calls.
        Includes status/error if present and the final message text.
        """
        parts: List[str] = []
        # Guard against different RunResult shapes (object or dict-like)
        err = getattr(run_result, "error", None)
        if err is None and hasattr(run_result, "get"):
            err = run_result.get("error", None)  # type: ignore
        if err:
            parts.append(f"error: {err}")

        status_val = getattr(run_result, "status", None)
        if status_val is None and hasattr(run_result, "get"):
            status_val = run_result.get("status", None)  # type: ignore
        if status_val:
            parts.append(f"status: {status_val}")

        out = getattr(run_result, "output", None)
        if out is None and hasattr(run_result, "get"):
            out = run_result.get("output", None)  # type: ignore
        if out:
            parts.append(str(out))
        return "\n".join(parts) if parts else ""
    
    # Extract richer context from Idea JSON and format lists for Prompt ingestion
    title = idea.get('Title', 'Project')
    abstract = idea.get('Abstract', '')
    hypothesis = idea.get('Short Hypothesis', 'None')
    related_work = idea.get('Related Work', 'None provided.')
    
    experiments_plan = format_list_field(idea.get('Experiments', []))
    risk_factors = format_list_field(idea.get('Risk Factors and Limitations', []))

    # 1. The Archivist (Scope: Literature & Claims)
    archivist = Agent(
        name="Archivist",
        instructions=(
            f"Role: Senior Literature Curator.\n"
            f"Goal: Verify novelty of '{title}' and map claims to citations.\n"
            f"Context: {abstract}\n"
            f"Related Work to Consider: {related_work}\n"
            "Directives:\n"
            "0. Call 'get_run_paths' once; use 'resolve_path' for any file inputs to avoid path errors.\n"
            "1. Use 'assemble_lit_data' or 'search_semantic_scholar' to gather papers.\n"
            "2. Maintain a claim graph via 'update_claim_graph' when mapping evidence.\n"
            "3. Output 'lit_summary.json' to experiment_results/.\n"
            "4. If you create or deeply analyze artifacts not yet in the manifest, log them with 'append_manifest' (name + metadata/description).\n"
            "5. CRITICAL: If no papers are found, report FAILURE. Do not invent 'TBD' citations."
        ),
        model=model,
        tools=[
            get_run_paths,
            resolve_path,
            list_artifacts,
            read_artifact,
            reserve_output,
            append_manifest,
            read_manifest,
            assemble_lit_data,
            validate_lit_summary,
            search_semantic_scholar,
            update_claim_graph,
        ],
        model_settings=common_settings
    )

    # 2. The Modeler (Scope: Python & Simulation)
    modeler = Agent(
        name="Modeler",
        instructions=(
            f"Role: Computational Biologist.\n"
            f"Goal: Execute simulations for '{title}'.\n"
            f"Hypothesis: {hypothesis}\n"
            f"Experimental Plan:\n{experiments_plan}\n"
            "Directives:\n"
            "0. Call 'get_run_paths' once; use 'resolve_path' for any file inputs to avoid path errors.\n"
            "1. You do NOT care about LaTeX or writing styles. Focus on DATA.\n"
            "2. Build graphs ('build_graphs'), run baselines ('run_biological_model') or custom sims ('run_comp_sim').\n"
            "3. Explore parameter space using 'run_sensitivity_sweep' and 'run_intervention_tests'.\n"
            "4. Ensure parameter sweeps cover the range specified in the hypothesis.\n"
            "5. Save raw outputs to experiment_results/.\n"
            "6. Log any new or newly analyzed artifacts with 'append_manifest' (name + metadata/description)."
        ),
        model=model,
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
            build_graphs,
            run_biological_model,
            run_comp_sim,
            run_sensitivity_sweep,
            run_intervention_tests,
        ], 
        model_settings=common_settings
    )

    # 3. The Analyst (Scope: Visualization & Validation)
    analyst = Agent(
        name="Analyst",
        instructions=(
            "Role: Scientific Visualization Expert.\n"
            "Goal: Convert simulation data into PLOS-quality figures.\n"
            "Directives:\n"
            "0. Call 'get_run_paths' once; use 'resolve_path' for any file inputs (lit, sims, morphologies) to avoid path errors.\n"
            "1. Read data from experiment_results/.\n"
            "2. Assert that the data supports the hypothesis BEFORE plotting. If data contradicts hypothesis, report this back immediately.\n"
            "3. Generate PNG/SVG files using 'run_biological_plotting'.\n"
            "4. Validate models vs lit via 'run_validation_compare' and use 'run_biological_stats' for significance/enrichment.\n"
            "5. Log any new or newly analyzed artifacts with 'append_manifest' (name + metadata/description)."
        ),
        model=model,
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
            run_biological_plotting,
            run_validation_compare,
            run_biological_stats,
            write_figures_readme,
            write_text_artifact,
        ],
        model_settings=common_settings
    )

    # 4. The Reviewer (Scope: Logic & Completeness)
    reviewer = Agent(
        name="Reviewer",
        instructions=(
            "Role: Holistic Reviewer (Reviewer #2).\n"
            "Goal: Identify logical gaps and structural flaws.\n"
            f"Risk Factors & Limitations to Check:\n{risk_factors}\n"
            "Directives:\n"
            "0. Call 'get_run_paths' once; use 'resolve_path' for any file inputs to avoid path errors.\n"
            "1. Read the manuscript draft using 'read_manuscript'.\n"
            "2. Check claim support using 'check_claim_graph' and sanity-check stats with 'run_biological_stats' if needed.\n"
            "3. Check consistency: Does Figure 3 actually support the claim in paragraph 2?\n"
            "4. If gaps exist, report them clearly to the PI.\n"
            "5. Only report 'NO GAPS' if the PDF validates completely.\n"
            "6. If you create or newly analyze artifacts, log them with 'append_manifest' (name + metadata/description)."
        ),
        model=model,
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
            read_manuscript,
            check_claim_graph,
            run_biological_stats,
            write_text_artifact,
        ],
        model_settings=common_settings
    )

    # 5. The Interpreter (Scope: Theoretical Interpretation)
    interpreter = Agent(
        name="Interpreter",
        instructions=(
            "Role: Mathematical–Biological Interpreter.\n"
            "Goal: Produce interpretation.json/md for theoretical biology projects.\n"
            "Directives:\n"
            "0. Call 'get_run_paths' once; use 'resolve_path' for any file inputs to avoid path errors.\n"
            "1. Call 'interpret_biology' only when biology.research_type == theoretical.\n"
            "2. Use experiment summaries and idea text; do NOT hallucinate unsupported claims.\n"
            "3. If interpretation fails, report the error clearly.\n"
            "4. Use 'write_text_artifact' to save interpretations (e.g., theory_interpretation.txt) under experiment_results/.\n"
            "5. Log any new or newly analyzed artifacts with 'append_manifest' (name + metadata/description)."
        ),
        model=model,
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
            write_text_artifact,
            write_interpretation_text,
            write_figures_readme,
            check_status,
            interpret_biology,
        ],
        model_settings=common_settings,
    )

    # 6. The Coder (Scope: Utility Code & Tooling)
    coder = Agent(
        name="Coder",
        instructions=(
            "Role: Utility Engineer.\n"
            "Goal: Write or update lightweight Python helpers/tools confined to this run folder.\n"
            "Directives:\n"
            "0. Call 'get_run_paths' and use 'resolve_path' before reading/writing.\n"
            "1. Use 'coder_create_python' to create/update files under the run root; do NOT write outside AISC_BASE_FOLDER.\n"
            "2. If you add tools/helpers, document them briefly and log via 'append_manifest'.\n"
            "3. Prefer small, dependency-light snippets; avoid large libraries or network access.\n"
            "4. If you need existing artifacts, list them with 'list_artifacts' or read via 'read_artifact' (use summary_only for large files).\n"
        ),
        model=model,
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
        ],
        model_settings=common_settings,
    )

    # 6. The Publisher (Scope: LaTeX & Compilation)
    publisher = Agent(
        name="Publisher",
        instructions=(
            "Role: Production Editor.\n"
            "Goal: Compile final PDF.\n"
            "Directives:\n"
            "1. Target the 'blank_theoretical_biology_latex' template.\n"
            "2. Integrate 'lit_summary.json' and figures into the text.\n"
            "3. Ensure compile success. Debug LaTeX errors autonomously."
        ),
        model=model,
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
        ],
        model_settings=common_settings
    )

    # 7. The PI (Scope: Strategy & Handoffs)
    pi = Agent(
        name="PI",
        instructions=(
            f"Role: Principal Investigator for project: {title}.\n"
            f"Hypothesis: {hypothesis}\n"
            "Responsibilities:\n"
            "1. STATE CHECK: First, call 'check_project_state' to see what has been done.\n"
            "2. DELEGATE: Handoff to specialized agents based on missing artifacts.\n"
            "   - Missing Lit Review -> Archivist\n"
            "   - Missing Data -> Modeler\n"
            "   - Missing Plots -> Analyst\n"
            "   - Theoretical Interpretation -> Interpreter\n"
            "   - Draft Exists -> Reviewer\n"
            "   - Validated & Ready -> Publisher\n"
            "3. ITERATE: If Reviewer finds gaps, translate them into new tasks for Modeler/Archivist.\n"
            "4. TERMINATE: Stop only when Reviewer confirms 'NO GAPS' and PDF is generated."
        ),
        model=model,
        tools=[
            check_project_state,
            log_strategic_pivot,
            get_run_paths,
            resolve_path,
            list_artifacts,
            read_artifact,
            summarize_artifact,
            reserve_output,
            append_manifest,
            read_manifest,
            check_status,
            archivist.as_tool(tool_name="archivist", tool_description="Search literature.", max_turns=role_max_turns),
            modeler.as_tool(tool_name="modeler", tool_description="Run simulations.", max_turns=role_max_turns, custom_output_extractor=extract_run_output),
            analyst.as_tool(tool_name="analyst", tool_description="Create figures.", max_turns=role_max_turns, custom_output_extractor=extract_run_output),
            coder.as_tool(tool_name="coder", tool_description="Write/update helper code in run folder.", max_turns=role_max_turns, custom_output_extractor=extract_run_output),
            interpreter.as_tool(tool_name="interpreter", tool_description="Generate theoretical interpretation.", max_turns=role_max_turns, custom_output_extractor=extract_run_output),
            publisher.as_tool(tool_name="publisher", tool_description="Write and compile text.", max_turns=role_max_turns, custom_output_extractor=extract_run_output),
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

    # Directories Dict for Agent Context
    dirs = {
        "base": base_folder,
        "results": exp_results
    }

    # Initialize Team
    pi_agent = build_team(args.model, idea, dirs)

    print(f"🧪 Launching ADCRT for '{idea.get('Title', 'Project')}'...")
    print(f"📂 Context: {base_folder}")

    # Input Prompt
    if args.input:
        initial_prompt = args.input
    else:
        initial_prompt = (
            f"Begin project '{idea.get('Name', 'Project')}'. \n"
            f"Current working directory is: {base_folder}\n"
            "Assess current state via check_project_state and begin delegation."
        )

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
