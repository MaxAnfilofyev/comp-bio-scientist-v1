# pyright: reportMissingImports=false
import json
import os
import re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import ast
import difflib
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING, cast

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

from agents import Agent, function_tool as _function_tool

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
from ai_scientist.tools.reference_verification import ReferenceVerificationTool
from ai_scientist.tools.compute_model_metrics import ComputeModelMetricsTool

from ai_scientist.tools.writeup import perform_writeup
from ai_scientist.tools.biological_interpretation import generate_biological_interpretation
from ai_scientist.utils.notes import NOTE_NAMES, read_note_file, write_note_file, append_run_note
from ai_scientist.utils.pathing import resolve_output_path
from ai_scientist.utils.transport_index import resolve_transport_sim
from ai_scientist.utils import manifest as manifest_utils

from ai_scientist.orchestrator.artifacts import (
    reserve_typed_artifact as _reserve_typed_artifact_impl,
    reserve_and_register_artifact as _reserve_and_register_impl,
)

from ai_scientist.orchestrator.summarization import ensure_module_summary_current

from ai_scientist.orchestrator.context import (
    _fill_figure_dir,
    _fill_output_dir,
)
from ai_scientist.orchestrator.context_specs import (
    ModuleName,
    VALID_MODULE_NAMES,
    active_role,
    truncate_paths_for_role,
)

from ai_scientist.orchestrator.hypothesis import (
    _update_hypothesis_trace_with_sim,
    _update_hypothesis_trace_with_figures,
    update_hypothesis_trace_impl,
    resolve_lit_summary_path,
    evaluate_lit_ready,
    log_lit_gate_decision,
    ensure_lit_gate_ready,
    ensure_model_spec_and_params,
    evaluate_model_provenance,
    record_model_provenance_in_provenance,
    evaluate_claim_consistency,
    record_claim_consistency_in_provenance,
    resolve_claim_graph_path,
    generate_provenance_summary_impl,
)

from ai_scientist.orchestrator.release import (
    freeze_release as freeze_release_impl,
    check_release_reproducibility as check_release_reproducibility_impl,
    generate_reproduction_section as generate_reproduction_section_impl,
    safe_sha256,
    load_release_manifest as load_release_manifest_impl,
)

from ai_scientist.orchestrator.manifest_service import (
    _append_artifact_from_result,
    _append_figures_from_result,
    _append_manifest_entry,
    manage_project_knowledge as manifest_manage_project_knowledge,
)

from ai_scientist.orchestrator.transport import (
    load_transport_manifest,
    upsert_transport_manifest_entry,
    build_seed_dir,
    resolve_run_paths,
    status_from_paths,
    write_verification,
    resolve_baseline_path_internal,
    scan_transport_manifest as scan_transport_manifest_impl,
    read_transport_manifest as read_transport_manifest_impl,
    update_transport_manifest as update_transport_manifest_impl,
    resolve_baseline_path as resolve_baseline_path_impl,
    resolve_sim_path as resolve_sim_path_impl,
)

_CHECKPOINTS_SEEN: set[str] = set()

function_tool: Callable[..., Any] = cast(Callable[..., Any], _function_tool)


@function_tool  # type: ignore[reportCallIssue]
def inspect_manifest(base_folder: str) -> str:
    from ai_scientist.utils import manifest as manifest_utils
    try:
        result = manifest_utils.inspect_manifest(
            base_folder=base_folder,
            summary_only=False,
            limit=100,
        )
        return json.dumps(result.get("entries", result))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@function_tool  # type: ignore[reportCallIssue]
def inspect_recent_manifest_entries(base_folder: str) -> str:
    from ai_scientist.utils import manifest as manifest_utils
    try:
        result = manifest_utils.inspect_manifest(
            base_folder=base_folder,
            limit=50,
        )
        return json.dumps(result.get("entries", result))
    except Exception as exc:
        return json.dumps({"error": str(exc)})

@function_tool
def append_manifest(name: str, metadata_json: Optional[str] = None, allow_missing: bool = False, change_summary: str = ""):
    return _append_manifest_entry(name=name, metadata_json=metadata_json, allow_missing=allow_missing, change_summary=change_summary)

@function_tool
def promote_artifact_to_canonical(name: str, kind: str, notes: str = ""):
    from ai_scientist.orchestrator.artifacts import promote_artifact_to_canonical as _promote
    return _promote(name, kind, notes)

@function_tool
def check_dependency_staleness(artifact_name: str):
    from ai_scientist.orchestrator.integrity import check_dependency_staleness as _check
    return json.dumps(_check(artifact_name))

@function_tool
def generate_project_snapshot():
    from ai_scientist.orchestrator.snapshots import generate_project_snapshot as _gen
    return _gen()


@function_tool
def read_manifest_entry(path_or_name: str):
    from .manifest_service import read_manifest_entry as raw_read_manifest_entry
    return raw_read_manifest_entry(path_or_name)

@function_tool
def check_manifest():
    from .manifest_service import check_manifest as check_manifest_impl
    return check_manifest_impl()

@function_tool
def read_manifest():
    from .manifest_service import read_manifest as read_manifest_impl
    return read_manifest_impl()

@function_tool
def check_manifest_unique_paths():
    from .manifest_service import check_manifest_unique_paths as check_manifest_unique_paths_impl
    return check_manifest_unique_paths_impl()

@function_tool
def list_artifacts(suffix: Optional[str] = None, subdir: Optional[str] = None):
    from .manifest_service import list_artifacts as _list_artifacts
    result = _list_artifacts(suffix, subdir)
    role = active_role()
    files = result.get("files", []) or []
    truncated = truncate_paths_for_role(role, files)
    if len(truncated) < len(files):
        note = result.get("note", "")
        extra = "Context view limited by role spec."
        result["note"] = f"{note + ' ' if note else ''}{extra}"
    result["files"] = truncated
    return result

@function_tool
def list_artifacts_by_kind(kind: str, limit: int = 100):
    from .manifest_service import list_artifacts_by_kind as _list_artifacts_by_kind
    result = _list_artifacts_by_kind(kind, limit)
    role = active_role()
    paths = result.get("paths", []) or []
    truncated = truncate_paths_for_role(role, paths, kind=kind)
    if len(truncated) < len(paths):
        note = result.get("note", "")
        extra = "Context view limited by role spec."
        result["note"] = f"{note + ' ' if note else ''}{extra}"
    result["paths"] = truncated
    return result


@function_tool
def ensure_module_summary(module: ModuleName) -> str:
    if module not in VALID_MODULE_NAMES:
        allowed = ", ".join(VALID_MODULE_NAMES)
        return json.dumps(
            {
                "status": "unknown_module",
                "message": f"Unknown module '{module}'. Valid module names: {allowed}.",
            }
        )
    result = ensure_module_summary_current(module)
    return json.dumps(result)

@function_tool
def get_artifact_index(max_entries: int = 2000):
    from .manifest_service import get_artifact_index
    return get_artifact_index(max_entries)

@function_tool
def check_project_state(base_folder: str) -> str:
    from .manifest_service import check_project_state
    return check_project_state(base_folder)

@function_tool
def manage_project_knowledge(
    action: Literal["add", "read"],
    category: Literal["general", "constraint", "decision", "failure_pattern", "reflection"] = "general",
    observation: str = "",
    solution: str = "",
    actor: str = "",
) -> str:
    return manifest_manage_project_knowledge(action, category, observation, solution, actor)

@function_tool
def append_run_note_tool(message: str, actor: str = "") -> str:
    append_run_note(message, actor or "tool")
    return "Note appended"

def format_list_field(data: Any) -> str:
    if isinstance(data, list):
        return "\n".join([f"- {item}" for item in data])
    return str(data)


def _truncate_text_response(
    text: str,
    *,
    path: Optional[str],
    threshold: int,
    total_bytes: Optional[int] = None,
    hint_tool: str = "head_artifact",
) -> Dict[str, Any]:
    """
    Standardized truncation message for tools returning text content.
    """
    total_chars = len(text)
    snippet = text[:threshold]
    note = (
        f"Content exceeds threshold ({threshold} chars); returned first {threshold} of {total_chars}"
        + (f" (~{total_bytes} bytes)" if total_bytes is not None else "")
        + f". To inspect more, use {hint_tool} or raise return_size_threshold_chars carefully (watch context limits)."
    )
    return {"path": path, "content": snippet, "truncated": True, "note": note, "total_chars": total_chars}


def _extract_markdown_section(text: str, headers: List[str]) -> str:
    """
    Return the text of the first markdown section matching any header label (case-insensitive).
    """
    if not text or not headers:
        return ""
    header_pattern = "|".join([re.escape(h) for h in headers])
    header_re = re.compile(rf"^\s*#{{1,6}}\s*(?:{header_pattern})\s*$", re.IGNORECASE | re.MULTILINE)
    match = header_re.search(text)
    if not match:
        return ""
    start = match.end()
    remainder = text[start:]
    next_header = re.search(r"^\s*#{1,6}\s+.+$", remainder, re.MULTILINE)
    end = start + next_header.start() if next_header else len(text)
    return remainder[: end - start].strip()


def _extract_markdown_title(text: str) -> str:
    """
    Extract the first H1 markdown title as a fallback.
    """
    match = re.search(r"^\s*#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_subheadings(block: str) -> List[str]:
    """
    Collect subheadings (H2+) from a markdown block.
    """
    return [m.group(1).strip() for m in re.finditer(r"^\s*#{2,6}\s+(.+)$", block, re.MULTILINE)]


def _extract_bullets_or_paragraph(block: str) -> List[str]:
    """
    Prefer bullet items; otherwise return the first paragraph as a single item.
    """
    items: List[str] = []
    for line in block.splitlines():
        bullet = re.match(r"\s*[-*]\s+(.*)", line)
        if bullet:
            cleaned = bullet.group(1).strip()
            if cleaned:
                items.append(cleaned)
    if not items:
        paragraph = " ".join(block.strip().split())
        if paragraph:
            items.append(paragraph)
    return items


def _first_sentence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return parts[0].strip() if parts else cleaned


def _derive_experiments_from_text(text: str) -> List[str]:
    """
    Heuristic extraction of experiment placeholders from manuscript sections.
    """
    experiments: List[str] = []
    results_block = _extract_markdown_section(text, ["results", "experiments"])
    methods_block = _extract_markdown_section(text, ["methods", "materials and methods"])
    for block in [results_block, methods_block]:
        for heading in _extract_subheadings(block):
            experiments.append(heading)
        if not experiments and block:
            experiments.extend(_extract_bullets_or_paragraph(block))
        if experiments:
            break
    if not experiments:
        experiments.append("Extract experiments/figures from manuscript sections (Results/Methods).")
    return experiments


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


# _report_capabilities and _ensure_transport_readme are imported from context

def _make_agent(name: str, instructions: str, tools: List[Any], model: str, settings: Any) -> Any:
    return Agent(name=name, instructions=instructions, model=model, tools=tools, model_settings=settings)  # type: ignore


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
def scan_transport_manifest(write: bool = True):
    """
    Scan transport_runs and (optionally) write manifest.json with status of (baseline, transport, seed).
    """
    return scan_transport_manifest_impl(write)


@function_tool
def resolve_transport_run(baseline: str, transport: str, seed: str):
    """
    Resolve paths for a transport run using the shared index (refreshes if stale).
    """
    res = resolve_transport_sim(baseline=baseline, transport=float(transport), seed=int(seed))
    return res
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


@function_tool
def resolve_baseline_path(baseline: str):
    """
    Resolve a baseline path by name or explicit path (searches experiment_results/morphologies for <baseline> with common suffixes).
    Returns {path} or an error with available baselines.
    """
    return resolve_baseline_path_impl(baseline)


@function_tool
def resolve_sim_path(baseline: str, transport: float, seed: int):
    """
    Resolve a sim.json path for (baseline, transport, seed) using the transport manifest with a scan fallback.
    Returns {path, status, missing, available_transports, available_pairs} or an error with suggestions.
    """
    return resolve_sim_path_impl(baseline, transport, seed)


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
    baseline_path_resolved, available_bases, err = resolve_baseline_path_internal(baseline_path)
    if baseline_path_resolved is None or err:
        return {
            "error": err or "Baseline not found",
            "available_baselines": available_bases,
            "hint": "Provide a valid baseline path or use resolve_baseline_path.",
        }
    baseline_name = baseline_path_resolved.stem
    tasks = []
    results: List[Dict[str, Any]] = []

    manifest_data = load_transport_manifest()
    existing = manifest_data.get("runs", [])

    def should_skip(entry: Optional[Dict[str, Any]], path_map: Dict[str, Path]) -> bool:
        entry_paths = entry.get("paths", {}) if isinstance(entry, dict) else {}
        paths_now = {k: entry_paths.get(k) or (str(v) if v.exists() else None) for k, v in path_map.items() if k != "verification"}
        status_now, missing = status_from_paths(paths_now)
        return status_now == "complete" and not missing

    for t in transport_values:
        for s in seeds:
            seed_dir = build_seed_dir(baseline_name, t, s)
            path_map = resolve_run_paths(seed_dir, baseline_name)
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
            status_now, missing = status_from_paths(paths_now)
            if missing:
                sim_json = path_map["sim_json"]
                if sim_json.exists():
                    SimPostprocessTool().use_tool(
                        sim_json_path=str(sim_json),
                        output_dir=str(seed_dir),
                        graph_path=str(baseline_path_resolved),
                    )
                    paths_now = {k: (str(v) if v.exists() else None) for k, v in path_map.items() if k != "verification"}
                    status_now, missing = status_from_paths(paths_now)
            status = status_now
            if missing:
                notes = f"missing after postprocess: {', '.join(missing)}"
            write_verification(seed_dir, baseline_name, t, s, status, notes)
            upd = upsert_transport_manifest_entry(
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
            write_verification(seed_dir, baseline_name, t, s, "error", notes)
            upsert_transport_manifest_entry(
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
    return read_transport_manifest_impl(baseline, transport, seed)


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
    return update_transport_manifest_impl(
        baseline=baseline,
        transport=transport,
        seed=seed,
        status=status,
        paths_json=paths_json,
        notes=notes,
        actor=actor,
    )


@function_tool(strict_mode=False)
def update_hypothesis_trace(
    hypothesis_id: str,
    experiment_id: str,
    sim_runs: Optional[List[Dict[str, Any]]] = None,
    figures: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    status: Optional[str] = None,
):
    """
    Update hypothesis_trace.json with sim runs, figures, metrics, or status for a hypothesis/experiment.
    """
    return update_hypothesis_trace_impl(
        hypothesis_id,
        experiment_id,
        sim_runs,
        figures,
        metrics,
        status,
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
    run_verification: bool = True,
    verification_max_results: int = 5,
    excluded_ids: Optional[List[str]] = None,
):
    """
    Assemble lit data, deduplicate, and optionally verify with Semantic Scholar.
    If 'excluded_ids' is provided, skips any papers matching those IDs/DOIs/titles (case-insensitive).
    """
    if not queries and not seed_paths:
        return "Error: You provided no 'queries' or 'seed_paths'. Please provide specific keywords or paper IDs."
        
    result = LitDataAssemblyTool().use_tool(
        queries=queries,
        seed_paths=seed_paths,
        max_results=max_results,
        use_semantic_scholar=use_semantic_scholar,
        excluded_ids=excluded_ids,
    )
    if run_verification:
        try:
            verify_res = ReferenceVerificationTool().use_tool(
                lit_path=(result.get("json") if isinstance(result, dict) else None),
                max_results=verification_max_results,
            )
            if isinstance(result, dict):
                result["reference_verification"] = verify_res
        except Exception as exc:
            if isinstance(result, dict):
                result["reference_verification_error"] = f"Verification failed: {exc}"
    try:
        human_in_loop = os.environ.get("AISC_INTERACTIVE", "false").strip().lower() in {"true", "1", "yes"}
        if human_in_loop:
            top_refs = ""
            if isinstance(result, dict):
                json_path = result.get("json") or ""
                top_refs = f"lit_summary: {json_path}, verification: {result.get('reference_verification', {}).get('json', '')}"
            summary = f"Literature assembled. {top_refs}"
            _checkpoint_required("literature_verification", summary, human_in_loop)
        else:
            _record_checkpoint_decision(
                "literature_verification",
                approved=True,
                summary="Lit assembled (dry-run approval).",
                dry_run=True,
            )
    except Exception:
        pass
    return result

@function_tool
def validate_lit_summary():
    """Validates the structure of the literature summary."""
    path = resolve_lit_summary_path(None)
    return LitSummaryValidatorTool().use_tool(path=str(path))


def _record_checkpoint_decision(name: str, approved: bool, summary: str, dry_run: bool = False) -> None:
    exp_dir = BaseTool.resolve_output_dir(None)
    health_dir = exp_dir / "_health"
    health_dir.mkdir(parents=True, exist_ok=True)
    report_path = health_dir / "health_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            report = {}
    else:
        report = {}
    key = "decisions_dry_run" if dry_run else "decisions"
    entries = report.setdefault(key, [])
    entries.append(
        {
            "name": name,
            "approved": approved,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }
    )
    report_path.write_text(json.dumps(report, indent=2))


def _checkpoint_required(name: str, summary: str, human_in_loop: bool) -> None:
    """
    If human_in_loop is True, prompt for approval once per checkpoint name.
    Otherwise, record a dry-run decision.
    """
    if name in _CHECKPOINTS_SEEN:
        return
    _CHECKPOINTS_SEEN.add(name)

    if not human_in_loop:
        _record_checkpoint_decision(name, approved=True, summary=summary, dry_run=True)
        return

    print(f"\n=== HUMAN CHECKPOINT: {name} ===\n{summary}\nApprove to proceed? [y/N]: ", end="", flush=True)
    try:
        resp = input().strip().lower()
    except Exception:
        resp = ""
    approved = resp in {"y", "yes"}
    _record_checkpoint_decision(name, approved=approved, summary=summary, dry_run=False)
    if not approved:
        try:
            cast(Any, manage_project_knowledge)(
                action="add",
                category="decision",
                observation=f"Checkpoint '{name}' declined.",
                solution=summary,
            )
        except Exception:
            pass
        raise RuntimeError(f"Checkpoint '{name}' declined by human.")



@function_tool
def check_model_provenance(model_key: str, allow_free_hyperparameters: bool = False):
    """
    Validate that a model's spec and parameter ledger are complete and sourced.
    """
    result = evaluate_model_provenance(model_key=model_key, allow_free=allow_free_hyperparameters)
    status = result.get("status", "not_ready")
    counts = result.get("counts_by_source_type", {})
    lit_count = counts.get("lit_value", 0)
    fit_count = counts.get("fit_to_data", 0)
    scale_count = counts.get("dimensionless_scaling", 0)
    summary_line = (
        f"{status.upper()} (lit={lit_count}, fit={fit_count}, scaling={scale_count}, "
        f"missing={len(result.get('missing_params', []))}, free={len(result.get('unsourced_params', []))})"
    )
    try:
        record_model_provenance_in_provenance(model_key, summary_line)
    except Exception:
        pass
    try:
        cast(Any, manage_project_knowledge)(
            action="add",
            category="decision",
            observation=f"Model provenance check for {model_key}",
            solution=summary_line,
        )
    except Exception:
        pass
    return result


@function_tool
def check_claim_consistency():
    """
    Cross-check claim_graph.json and hypothesis_trace.json for supporting experiments/metrics.
    """
    result = evaluate_claim_consistency()
    summary_line = (
        f"{result.get('overall_status')} "
        f"(missing={result.get('n_missing')}, weak={result.get('n_weak')})"
    )
    try:
        record_claim_consistency_in_provenance(summary_line)
    except Exception:
        pass
    try:
        cast(Any, manage_project_knowledge)(
            action="add",
            category="decision",
            observation="Claim consistency check.",
            solution=summary_line,
        )
    except Exception:
        pass
    return result


@function_tool
def check_lit_ready(
    lit_path: Optional[str] = None,
    verification_path: Optional[str] = None,
    confirmed_threshold: float = 0.7,
    max_unverified: int = 3,
):
    """
    Gatekeeper for modeling/sims: validates lit_summary and reference verification coverage.
    """
    result = evaluate_lit_ready(
        lit_path=lit_path,
        verification_path=verification_path,
        confirmed_threshold=confirmed_threshold,
        max_unverified=max_unverified,
    )
    log_lit_gate_decision(
        status=result.get("status", "not_ready"),
        confirmed_pct=result.get("confirmed_pct", 0.0),
        n_unverified=result.get("n_unverified", 0),
        thresholds=result.get("thresholds", {"confirmed_threshold": confirmed_threshold, "max_unverified": max_unverified}),
        reasons=result.get("reasons", []),
    )
    return result


@function_tool
def verify_references(
    max_results: int = 5,
    score_threshold: float = 0.65,
):
    """
    Verify lit_summary entries via Semantic Scholar; writes lit_reference_verification.csv/json
    using canonical lit_summary and output locations.
    """
    # Resolve canonical inputs/outputs to avoid path gymnastics.
    lit_summary_path = resolve_lit_summary_path(None)
    out_dir = BaseTool.resolve_output_dir(None)

    res = ReferenceVerificationTool().use_tool(
        lit_path=str(lit_summary_path),
        output_dir=str(out_dir),
        max_results=max_results,
        score_threshold=score_threshold,
    )
    _append_artifact_from_result(res, "csv", '{"kind":"lit_reference_verification_table","created_by":"archivist"}', allow_missing=False)
    _append_artifact_from_result(res, "json", '{"kind":"lit_reference_verification_json","created_by":"archivist"}', allow_missing=False)
    return res

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
    hypothesis_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    skip_lit_gate: bool = False,
):
    """Runs a compartmental simulation and saves CSV data."""
    try:
        human_in_loop = os.environ.get("AISC_INTERACTIVE", "false").strip().lower() in {"true", "1", "yes"}
        _checkpoint_required(
            "sim_plan",
            "About to launch compartmental simulation (potentially heavy compute).",
            human_in_loop,
        )
    except Exception:
        pass
    ensure_lit_gate_ready(skip_gate=skip_lit_gate)
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
    try:
        if hypothesis_id and experiment_id:
            _update_hypothesis_trace_with_sim(
                hypothesis_id=hypothesis_id,
                experiment_id=experiment_id,
                sim_entry={"baseline": graph_path, "transport": transport_rate, "seed": seed},
                metrics=metrics or [],
            )
    except Exception:
        pass
    return res

@function_tool
def run_biological_plotting(
    solution_path: str,
    output_dir: Optional[str] = None,
    make_phase_portrait: bool = True,
    hypothesis_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    metrics: Optional[List[str]] = None,
):
    """Generates plots from simulation data."""
    out_dir = _fill_figure_dir(output_dir)
    res = RunBiologicalPlottingTool().use_tool(
        solution_path=solution_path,
        output_dir=out_dir,
        make_phase_portrait=make_phase_portrait,
        make_combined_svg=True,
    )
    _append_figures_from_result(res, '{"type":"figure","source":"analyst"}')
    try:
        if hypothesis_id and experiment_id:
            fig_paths = [v for v in res.values() if isinstance(v, str) and v.endswith((".png", ".svg"))]
            if fig_paths:
                _update_hypothesis_trace_with_figures(
                    hypothesis_id=hypothesis_id,
                    experiment_id=experiment_id,
                    figures=fig_paths,
                    metrics=metrics or [],
                )
    except Exception:
        pass
    return res


@function_tool
def compute_model_metrics(
    input_path: str,
    label: Optional[str] = None,
    model_key: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """
    Compute domain-specific metrics from a sweep CSV/JSON or model output; writes *_metrics.csv and optional {model_key}_metrics.json.
    """
    res = ComputeModelMetricsTool().use_tool(
        input_path=input_path,
        label=label,
        model_key=model_key,
        output_dir=_fill_output_dir(output_dir),
    )
    if isinstance(res, dict):
        if res.get("output_csv"):
            _append_manifest_entry(
                name=res["output_csv"],
                metadata_json=json.dumps({"kind": "sweep_metrics_csv", "created_by": os.environ.get("AISC_ACTIVE_ROLE", "modeler"), "status": "ok"}),
                allow_missing=False,
            )
        if res.get("model_metrics_json"):
            _append_manifest_entry(
                name=res["model_metrics_json"],
                metadata_json=json.dumps({"kind": "model_metrics_json", "created_by": os.environ.get("AISC_ACTIVE_ROLE", "modeler"), "status": "ok"}),
                allow_missing=False,
            )
    return res


@function_tool
def generate_provenance_summary():
    """
    Aggregate provenance from manifest and write experiment_results/provenance_summary.md.
    """
    return generate_provenance_summary_impl()


@function_tool
def freeze_release(tag: str, description: str = "", include_large_artifacts: bool = False):
    """
    Freeze the current run into experiment_results/releases/{tag}, capturing code, environment, manifest-linked artifacts, and git state.
    """
    return freeze_release_impl(
        tag=tag,
        description=description,
        include_large_artifacts=include_large_artifacts,
    )


@function_tool
def check_release_reproducibility(tag: str, quick: bool = True):
    """
    Verify a frozen release bundle: checksum all files listed in release_manifest.json and optionally run a quick repro smoke test.
    """
    return check_release_reproducibility_impl(tag=tag, quick=quick)


@function_tool
def generate_reproduction_section(tag: str, style: str = "methods_and_supp"):
    """
    Generate manuscript-ready reproduction instructions (methods paragraph + supplementary protocol) for a release tag.
    """
    result = generate_reproduction_section_impl(tag=tag, style=style)
    if result.get("error"):
        return result

    subdir = result.get("subdir")
    methods_res = cast(Any, write_text_artifact)(
        name="reproduction_methods.md",
        content=result.get("methods_section_md", ""),
        subdir=subdir,
        metadata_json=json.dumps({"kind": "repro_methods_md", "created_by": os.environ.get("AISC_ACTIVE_ROLE", "publisher"), "status": "ok"}),
    )
    protocol_res = cast(Any, write_text_artifact)(
        name="reproduction_protocol.md",
        content=result.get("supplementary_protocol_md", ""),
        subdir=subdir,
        metadata_json=json.dumps({"kind": "repro_protocol_md", "created_by": os.environ.get("AISC_ACTIVE_ROLE", "publisher"), "status": "ok"}),
    )
    return {
        "methods_section_md": result.get("methods_section_md"),
        "supplementary_protocol_md": result.get("supplementary_protocol_md"),
        "methods_path": methods_res.get("path"),
        "protocol_path": protocol_res.get("path"),
        "env_checksum": result.get("env_checksum"),
        "git_commit": result.get("git_commit"),
        "doi": result.get("doi"),
    }


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
    manifest_path = exp_dir / "manifest" / "manifest_index.json"
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
def read_manuscript(path: str, return_size_threshold_chars: int = 2000):
    """
    Reads the PDF or text of the manuscript. Truncates text over the threshold to avoid context blowups.
    """
    result = ManuscriptReaderTool().use_tool(path=path)
    text = result.get("text")
    if isinstance(text, str) and len(text) > return_size_threshold_chars:
        truncated = _truncate_text_response(
            text,
            path=str(path),
            threshold=return_size_threshold_chars,
            total_bytes=None,
            hint_tool="head_artifact",
        )
        result.update(truncated)
    return result

@function_tool
def run_writeup_task(
    base_folder: Optional[str] = None,
    page_limit: int = 8,
    release_tag: Optional[str] = None,
):
    """Compiles the manuscript using the theoretical biology template."""
    human_in_loop = os.environ.get("AISC_INTERACTIVE", "false").strip().lower() in {"true", "1", "yes"}
    try:
        consistency = evaluate_claim_consistency()
        summary_line = (
            f"Claim consistency before writeup: {consistency.get('overall_status')} "
            f"(missing={consistency.get('n_missing')}, weak={consistency.get('n_weak')})"
        )
        _checkpoint_required("pre_writeup", summary_line, human_in_loop)
    except Exception:
        pass

    base_folder = base_folder or os.environ.get("AISC_BASE_FOLDER", "")
    release_meta: Dict[str, Any] = {}
    if release_tag:
        exp_dir = BaseTool.resolve_output_dir(None)
        release_root = exp_dir / "releases" / release_tag
        manifest = load_release_manifest_impl(release_root, release_tag)
        if not manifest.get("error"):
            release_meta = {
                "release_tag": release_tag,
                "release_commit": manifest.get("git", {}).get("commit"),
                "release_doi": manifest.get("doi") or manifest.get("zenodo_doi"),
            }
            env_rel = manifest.get("env_manifest_path") or "env_manifest.json"
            env_path = release_root / env_rel if not Path(env_rel).is_absolute() else Path(env_rel)
            if env_path.exists():
                release_meta["release_env_checksum"] = safe_sha256(env_path)
            code_checksum = None
            for entry in manifest.get("files", []) or []:
                p = entry.get("path", "")
                if "code_release" in p:
                    code_checksum = entry.get("checksum")
                    break
            release_meta["release_code_checksum"] = code_checksum

    ok = perform_writeup(
        base_folder=base_folder,
        no_writing=False,
        num_cite_rounds=10,
        small_model="gpt-4o-mini", 
        big_model="gpt-4o",
        n_writeup_reflections=2,
        page_limit=page_limit,
        release_tag=release_meta.get("release_tag"),
        release_commit=release_meta.get("release_commit"),
        release_doi=release_meta.get("release_doi"),
        release_env_checksum=release_meta.get("release_env_checksum"),
        release_code_checksum=release_meta.get("release_code_checksum"),
    )
    # Post-write consistency gate
    enforce_claims = os.environ.get("AISC_ENFORCE_CLAIM_CONSISTENCY", "true").strip().lower() not in {"0", "false", "no"}
    if enforce_claims:
        consistency = evaluate_claim_consistency()
        if consistency.get("overall_status") == "not_ready_for_publication":
            raise RuntimeError("Claim consistency check failed: missing support for one or more claims.")
    else:
        try:
            consistency = evaluate_claim_consistency()
            if consistency.get("overall_status") == "not_ready_for_publication":
                cast(Any, manage_project_knowledge)(
                    action="add",
                    category="failure_pattern",
                    observation="Claim consistency not enforced for writeup.",
                    solution="Proceeding despite missing claim support.",
                )
        except Exception:
            pass
    pdf_candidates = sorted(
        Path(base_folder or BaseTool.resolve_output_dir(None)).glob(f"{Path(base_folder or BaseTool.resolve_output_dir(None)).name}_*.pdf"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
    )
    pdf_path = pdf_candidates[-1] if pdf_candidates else Path(base_folder or BaseTool.resolve_output_dir(None)) / "manuscript.pdf"
    meta_entry: Dict[str, Any] = {
        "path": str(pdf_path),
        "name": pdf_path.name,
        "kind": "manuscript_pdf",
        "created_by": os.environ.get("AISC_ACTIVE_ROLE", "publisher"),
        "status": "ok",
    }
    if release_tag:
        meta_entry["metadata"] = {
            "release_tag": release_tag,
            "code_release_checksum": release_meta.get("release_code_checksum"),
            "release_commit": release_meta.get("release_commit"),
        }
    manifest_utils.append_or_update(meta_entry, base_folder=BaseTool.resolve_output_dir(None))
    return {"success": ok, "pdf_path": str(pdf_path)}

@function_tool
def search_semantic_scholar(query: str):
    """Directly search Semantic Scholar for papers."""
    return SemanticScholarSearchTool().use_tool(query=query)


@function_tool
def get_lit_recommendations(
    positive_paper_ids: Optional[List[str]] = None,
    limit: int = 20,
):
    """
    Get recommended papers using the Semantic Scholar Recommendations API.
    If positive_paper_ids is None, pulls up to 20 IDs from the current lit_summary.json.
    """
    from ai_scientist.tools.semantic_scholar import SemanticScholarRecommendationsTool
    
    if not positive_paper_ids:
        # Auto-discover from lit_summary
        try:
            lit_path = resolve_lit_summary_path(None)
            with lit_path.open() as f:
                data = json.load(f)
            
            # Extract up to 20 valid paper IDs (corpusId or sha)
            # Prefer 'paperId' (sha) or 'corpusId'
            ids = []
            for r in data:
                # Need an S2 ID. S2AG API uses 'paperId' (sha) or 'CorpusId:<int>'
                # lit_summary often normalizes fields. 'paperId' is standard.
                pid = r.get("paperId")
                if pid:
                    ids.append(pid)
                else:
                    # Fallback if we have corpusId
                    cid = r.get("corpusId")
                    if cid:
                        ids.append(f"CorpusId:{cid}")
            
            # Filter duplicates and limit
            positive_paper_ids = list(set(ids))[:20]
        except Exception as e:
            return f"Error reading lit_summary for seed IDs: {e}"

    if not positive_paper_ids:
        return "No positive paper IDs found (provided or in lit_summary) to generate recommendations."

    res = SemanticScholarRecommendationsTool(max_results=limit).use_tool(positive_paper_ids=positive_paper_ids)
    
    # Use proper artifact registration
    from ai_scientist.orchestrator.artifacts import reserve_and_register_artifact
    
    meta = {
        "module": "literature",
        "summary": f"S2 recommendations based on {len(positive_paper_ids)} seed papers.",
        "content": {"count": len(res), "seeds": positive_paper_ids},
        "metadata": {
            "created_by": "archivist",
            "source_tool": "get_lit_recommendations"
        }
    }
    
    reg = reserve_and_register_artifact(
        kind="lit_recommendations",
        meta_json=json.dumps(meta),
        status="done",
        unique=False # Reuse the same file if possible, or version it? "unique=False" means overwrite if name matches pattern without unique placeholders?
        # Actually pattern is "lit_recommendations.json" which is static. unique=True would fail or force quarantine.
        # But reserve_and_register_artifact handles versioning if unique=True is not helping?
        # artifacts.py: _reserve_typed_artifact_impl calls resolve_output_path(unique=unique).
        # Since pattern is static "lit_recommendations.json", we probably want to overwrite it (unique=False) or it will conflict.
        # Ideally we want a single recommendations list we update. Let's stick to unique=False (overwrite).
    )
    
    if reg.get("error"):
         return f"Failed to register artifact: {reg['error']}"

    out_path = Path(reg["reserved_path"])
    with out_path.open("w") as f:
        json.dump(res, f, indent=2)

    return f"Found {len(res)} recommendations. Saved to {out_path.name}. Top 3: {[r.get('title') for r in res[:3]]}"


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
    hypothesis_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    compute_metrics: bool = False,
    enforce_param_provenance: Optional[bool] = None,
    skip_lit_gate: bool = False,
):
    """Run a built-in biological ODE/replicator model and save JSON results."""
    ensure_lit_gate_ready(skip_gate=skip_lit_gate)
    ledger = ensure_model_spec_and_params(model_key)
    if ledger.get("created_spec"):
        _append_manifest_entry(
            name=ledger["spec_path"],
            metadata_json=json.dumps({"kind": "model_spec_yaml", "created_by": "modeler", "status": "ok"}),
            allow_missing=False,
        )
    if ledger.get("created_params"):
        _append_manifest_entry(
            name=ledger["param_path"],
            metadata_json=json.dumps({"kind": "parameter_source_table", "created_by": "modeler", "status": "ok"}),
            allow_missing=False,
        )

    enforce = enforce_param_provenance
    if enforce is None:
        enforce = os.environ.get("AISC_ENFORCE_PARAM_PROVENANCE", "true").strip().lower() not in {"0", "false", "no"}
    provenance_result = cast(Any, check_model_provenance)(model_key=model_key, allow_free_hyperparameters=not enforce)
    if enforce and provenance_result.get("status") != "ready":
        reasons = []
        if provenance_result.get("missing_params"):
            reasons.append(f"missing params: {', '.join(provenance_result['missing_params'])}")
        if provenance_result.get("unsourced_params"):
            reasons.append(f"unsourced params: {', '.join(provenance_result['unsourced_params'])}")
        raise RuntimeError(f"Model provenance gate not satisfied for {model_key}: {', '.join(reasons)}")
    if not enforce and provenance_result.get("status") != "ready":
        try:
            cast(Any, manage_project_knowledge)(
                action="add",
                category="failure_pattern",
                observation=f"Model provenance incomplete for {model_key}",
                solution=(
                    f"Missing: {provenance_result.get('missing_params', [])}; "
                    f"Unsourced: {provenance_result.get('unsourced_params', [])}. "
                    "Allowing execution because enforce_param_provenance=False."
                ),
            )
        except Exception:
            pass

    res = RunBiologicalModelTool().use_tool(
        model_key=model_key,
        time_end=time_end,
        num_points=num_points,
        output_dir=_fill_output_dir(output_dir),
    )
    _append_artifact_from_result(res, "output_json", metadata_json)
    try:
        if hypothesis_id and experiment_id:
            _update_hypothesis_trace_with_sim(
                hypothesis_id=hypothesis_id,
                experiment_id=experiment_id,
                sim_entry={"baseline": model_key, "transport": None, "seed": None},
                metrics=metrics or [],
            )
        if compute_metrics and isinstance(res, dict) and res.get("output_json"):
            try:
                ComputeModelMetricsTool().use_tool(
                    input_path=res["output_json"],
                    model_key=model_key,
                    label=model_key,
                )
            except Exception:
                pass
    except Exception:
        pass
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
    hypothesis_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    compute_metrics: bool = True,
    skip_lit_gate: bool = False,
):
    """Sweep transport_rate and demand_scale over a graph and log frac_failed."""
    try:
        human_in_loop = os.environ.get("AISC_INTERACTIVE", "false").strip().lower() in {"true", "1", "yes"}
        _checkpoint_required(
            "sim_plan",
            "About to launch sensitivity sweep (potentially heavy compute).",
            human_in_loop,
        )
    except Exception:
        pass
    ensure_lit_gate_ready(skip_gate=skip_lit_gate)
    res = RunSensitivitySweepTool().use_tool(
        graph_path=graph_path,
        output_dir=_fill_output_dir(output_dir),
        transport_vals=transport_vals,
        demand_vals=demand_vals,
        steps=steps,
        dt=dt,
    )
    _append_artifact_from_result(res, "output_csv", metadata_json)
    try:
        if hypothesis_id and experiment_id:
            _update_hypothesis_trace_with_sim(
                hypothesis_id=hypothesis_id,
                experiment_id=experiment_id,
                sim_entry={
                    "baseline": graph_path,
                    "transport": transport_vals,
                    "seed": None,
                },
                metrics=["sensitivity_sweep"],
            )
        if compute_metrics and isinstance(res, dict) and res.get("output_csv"):
            try:
                ComputeModelMetricsTool().use_tool(
                    input_path=res["output_csv"],
                    label=Path(res["output_csv"]).stem.replace(".csv", ""),
                )
            except Exception:
                pass
    except Exception:
        pass
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
    hypothesis_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    compute_metrics: bool = True,
    skip_lit_gate: bool = False,
):
    """Test parameter interventions vs a baseline and report delta frac_failed."""
    try:
        human_in_loop = os.environ.get("AISC_INTERACTIVE", "false").strip().lower() in {"true", "1", "yes"}
        _checkpoint_required(
            "sim_plan",
            "About to launch intervention tests (potentially heavy compute).",
            human_in_loop,
        )
    except Exception:
        pass
    ensure_lit_gate_ready(skip_gate=skip_lit_gate)
    res = RunInterventionTesterTool().use_tool(
        graph_path=graph_path,
        output_dir=_fill_output_dir(output_dir),
        transport_vals=transport_vals,
        demand_vals=demand_vals,
        baseline_transport=baseline_transport,
        baseline_demand=baseline_demand,
    )
    _append_artifact_from_result(res, "output_json", metadata_json)
    try:
        if hypothesis_id and experiment_id:
            _update_hypothesis_trace_with_sim(
                hypothesis_id=hypothesis_id,
                experiment_id=experiment_id,
                sim_entry={
                    "baseline": graph_path,
                    "transport": transport_vals,
                    "seed": None,
                },
                metrics=["intervention_tester"],
            )
        if compute_metrics and isinstance(res, dict) and res.get("output_csv"):
            try:
                ComputeModelMetricsTool().use_tool(
                    input_path=res["output_csv"],
                    label=Path(res["output_csv"]).stem.replace(".csv", ""),
                )
            except Exception:
                pass
    except Exception:
        pass
    return res

@function_tool
def run_validation_compare(lit_path: Optional[str] = None, sim_path: str = ""):
    """Correlate lit_summary metrics with simulation frac_failed."""
    lit_resolved = resolve_lit_summary_path(lit_path)
    res = RunValidationCompareTool().use_tool(lit_path=str(lit_resolved), sim_path=sim_path)
    if isinstance(res, dict):
        _append_manifest_entry(
            name="validation_compare.json",
            metadata_json=json.dumps({"type": "validation", "source": "analyst", "lit_path": str(lit_resolved), "sim_path": sim_path}),
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
    claim_id: str = "thesis",
    claim_text: str = "",
    parent_id: Optional[str] = None,
    support: Optional[List[str]] = None,
    status: str = "unlinked",
    notes: str = "",
):
    """Add or update a claim entry with support references."""
    claim_path = resolve_claim_graph_path()
    return ClaimGraphTool().use_tool(
        path=str(claim_path),
        claim_id=claim_id,
        claim_text=claim_text,
        parent_id=parent_id,
        support=support,
        status=status,
        notes=notes,
    )

@function_tool
def check_claim_graph():
    """Check claim_graph.json for claims lacking supporting evidence."""
    claim_path = resolve_claim_graph_path()
    return ClaimGraphCheckTool().use_tool(path=str(claim_path))

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

    result = generate_biological_interpretation(
        base_folder=base,
        config_path=cfg,
    )
    return result


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
def read_artifact(path: str, summary_only: bool = False, head_lines: Optional[int] = None, return_size_threshold_chars: int = 2000):
    """
    Resolve and read a small artifact (json/text). Use for configs/metadata, not large binaries.
    - summary_only=True: for large JSON, return top-level keys + types instead of full payload.
    - head_lines: return only the first N lines/items (bypasses size guard for text/JSON).
    - return_size_threshold_chars: if text output exceeds this length, it is truncated with a note (default 2000).
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
        content = f.read()
    if len(content) > return_size_threshold_chars:
        return _truncate_text_response(
            content,
            path=str(p),
            threshold=return_size_threshold_chars,
            total_bytes=size,
            hint_tool="head_artifact",
        )
    return content


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
def read_npy_artifact(
    path: str,
    max_elements: int = 100_000,
    max_bytes: int = 50_000_000,
    summary_only: bool = True,
    slice_spec_json: Optional[str] = None,
    full_data: bool = False,
):
    """
    Load .npy safely with hard caps and structured errors.
    Defaults to summary-only (shape/dtype/estimated_bytes + small sample stats).
    - For full data, set full_data=True or summary_only=False; requests exceeding caps return an error with a suggested sliced call.
    - Supports an optional slice JSON string: {"axis": int, "start": int, "stop": int, "step": int}.
    """
    p = BaseTool.resolve_input_path(path, allow_dir=False)
    base: Dict[str, Any] = {"path": str(p)}
    try:
        base["size_bytes"] = p.stat().st_size
    except Exception:
        pass

    if p.suffix.lower() != ".npy":
        return {**base, "status": "error", "error_type": "unsupported_type", "message": "read_npy_artifact only supports .npy files"}

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        return {**base, "status": "error", "error_type": "import_error", "message": f"numpy unavailable: {exc}"}

    try:
        arr = np.load(p, mmap_mode="r", allow_pickle=False)
    except FileNotFoundError:
        return {**base, "status": "error", "error_type": "missing_file", "message": "file not found"}
    except ValueError as exc:
        return {**base, "status": "error", "error_type": "parse_error", "message": str(exc)}
    except Exception as exc:
        return {**base, "status": "error", "error_type": "load_error", "message": str(exc)}

    try:
        total_elements = int(np.prod(arr.shape, dtype=np.int64))
    except Exception:
        total_elements = int(arr.size)
    itemsize = int(getattr(arr, "dtype", np.dtype("float64")).itemsize)
    estimated_bytes = int(total_elements * itemsize)

    meta: Dict[str, Any] = {
        **base,
        "status": "ok",
        "shape": tuple(int(x) for x in arr.shape),
        "dtype": str(arr.dtype),
        "elements": total_elements,
        "estimated_bytes": estimated_bytes,
    }

    view = arr
    view_shape = arr.shape
    view_elements = total_elements
    view_bytes = estimated_bytes
    slice_info: Optional[Dict[str, Optional[int]]] = None

    if slice_spec_json:
        try:
            import json as _json
            slice_spec = _json.loads(slice_spec_json)
        except Exception as exc:
            return {**meta, "status": "error", "error_type": "invalid_slice", "message": f"failed to parse slice_spec_json: {exc}"}
    else:
        slice_spec = None

    if slice_spec is not None:
        if not isinstance(slice_spec, dict) or "axis" not in slice_spec:
            return {**meta, "status": "error", "error_type": "invalid_slice", "message": "slice must be a dict with axis/start/stop/step"}
        try:
            axis_raw = slice_spec.get("axis", 0)
            start_raw = slice_spec.get("start", 0)
            stop_raw = slice_spec.get("stop", None)
            step_raw = slice_spec.get("step", 1)

            axis = int(0 if axis_raw is None else axis_raw)
            start = int(0 if start_raw is None else start_raw)
            stop = None if stop_raw is None else int(stop_raw)
            step = int(1 if step_raw is None else step_raw)
        except Exception:
            return {**meta, "status": "error", "error_type": "invalid_slice", "message": "slice values must be integers"}
        if axis < 0 or axis >= arr.ndim:
            return {**meta, "status": "error", "error_type": "invalid_slice", "message": f"axis {axis} out of bounds for array with ndim={arr.ndim}"}
        indexers: List[Any] = [slice(None)] * arr.ndim
        indexers[axis] = slice(start, stop, step)
        try:
            view = arr[tuple(indexers)]
            slice_info = {"axis": axis, "start": start, "stop": stop, "step": step}
            view_shape = tuple(int(x) for x in view.shape)
            try:
                view_elements = int(np.prod(view_shape, dtype=np.int64))
            except Exception:
                view_elements = int(view.size)
            view_bytes = int(view_elements * itemsize)
        except Exception as exc:
            return {**meta, "status": "error", "error_type": "slice_error", "message": str(exc)}

    # Helper: suggest a smaller slice along axis 0 when data exceeds caps
    def _suggest_slice() -> Dict[str, Any]:
        if arr.ndim == 0 or arr.shape[0] == 0:
            return {"path": str(p), "summary_only": True}
        try:
            per_row = int(np.prod(arr.shape[1:], dtype=np.int64)) if arr.ndim > 1 else 1
            rows = max(1, max_elements // max(1, per_row))
            stop = min(arr.shape[0], rows)
        except Exception:
            stop = min(arr.shape[0], 1)
        return {
            "path": str(p),
            "summary_only": True,
            "slice": {"axis": 0, "start": 0, "stop": int(stop), "step": 1},
        }

    # Determine response mode
    returning_full = full_data or not summary_only
    size_exceeds_caps = view_elements > max_elements or view_bytes > max_bytes

    if returning_full:
        if size_exceeds_caps:
            return {
                **meta,
                "status": "error",
                "error_type": "size_cap",
                "message": f"refused full_data: slice has {view_elements} elements / {view_bytes} bytes over caps (max_elements={max_elements}, max_bytes={max_bytes})",
                "slice_shape": view_shape,
                "view_elements": view_elements,
                "view_bytes": view_bytes,
                "suggested_call": _suggest_slice(),
            }
        try:
            data = view.tolist()
            resp: Dict[str, Any] = {
                **meta,
                "status": "ok",
                "mode": "full",
                "data": data,
                "view_elements": view_elements,
                "view_bytes": view_bytes,
            }
            if slice_info:
                resp["slice"] = slice_info
            return resp
        except Exception as exc:
            return {**meta, "status": "error", "error_type": "convert_error", "message": str(exc)}

    # Summary path (default)
    sample_cap = min(2048, max_elements)
    try:
        flat_view = np.asarray(view).reshape(-1)
        sample_count = int(min(sample_cap, flat_view.shape[0]))
        sample = np.asarray(flat_view[:sample_count])
    except Exception as exc:
        return {**meta, "status": "error", "error_type": "summary_error", "message": str(exc)}

    summary: Dict[str, Any] = {
        "mode": "summary",
        "sample_count": sample_count,
        "first_values": sample[: min(10, sample_count)].tolist() if sample_count else [],
    }
    if sample_count:
        try:
            numeric = np.issubdtype(sample.dtype, np.number) or np.issubdtype(sample.dtype, np.bool_)
            if numeric:
                sample_numeric = sample.astype(np.float64, copy=False)
                summary["min"] = float(np.min(sample_numeric))
                summary["max"] = float(np.max(sample_numeric))
                summary["mean"] = float(np.mean(sample_numeric))
                summary["std"] = float(np.std(sample_numeric))
                try:
                    summary["percentiles"] = [float(x) for x in np.percentile(sample_numeric, [0, 25, 50, 75, 100])]
                except Exception:
                    pass
        except Exception:
            pass

    resp: Dict[str, Any] = {
        **meta,
        "mode": "summary",
        "summary": summary,
        "slice_shape": view_shape,
        "view_elements": view_elements,
        "view_bytes": view_bytes,
    }
    if slice_info:
        resp["slice"] = slice_info
    if size_exceeds_caps:
        resp["note"] = "data omitted due to caps; request a slice or lower max_elements/max_bytes for full data"
    return resp


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
def reserve_typed_artifact(kind: str, meta_json: Optional[str] = None, unique: bool = True):
    """
    Reserve a canonical artifact path using the artifact type registry (VI-01).
    Provide meta_json to fill any {placeholders} in rel_dir/pattern. Errors on unknown kinds.
    """
    return _reserve_typed_artifact_impl(kind=kind, meta_json=meta_json, unique=unique)


@function_tool
def reserve_and_register_artifact(
    kind: str,
    meta_json: Optional[str] = None,
    status: str = "pending",
    unique: bool = True,
    skip_summary: bool = False,
):
    """
    Reserve a canonical artifact path using the registry and immediately register it in manifest v2 with provided status.
    """
    return _reserve_and_register_impl(
        kind=kind,
        meta_json=meta_json,
        status=status,
        unique=unique,
        skip_summary=skip_summary,
    )


@function_tool
def reserve_output(name: str, subdir: Optional[str] = None):
    """
    Return a canonical output path under experiment_results (or a subdir), auto-uniqued and sanitized.
    Prefer reserve_typed_artifact for persistent artifacts; use reserve_output only for scratch logs.
    Rejects traversal and routes to _unrouted on failure.
    """
    target, quarantined, note = resolve_output_path(subdir=subdir, name=name)
    result = {"reserved_path": str(target), "quarantined": quarantined}
    if note:
        result["note"] = note
    return result


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

    note_target = Path(norm_name).name
    if note_target in NOTE_NAMES and subdir in (None, "", ".", "experiment_results"):
        # Route note writes through the canonical note helper to avoid uuid-suffixed duplicates.
        write_result = write_note_file(content=content, name=note_target, append=False)
        result: Dict[str, Any] = {"path": write_result.get("path", ""), "quarantined": False}
        if metadata_json and result["path"]:
            _append_manifest_entry(name=result["path"], metadata_json=metadata_json, allow_missing=True)
        if write_result.get("warning"):
            result["warning"] = write_result["warning"]
        return result

    unique = True
    if note_target == "implementation_plan.md" and subdir in (None, "", ".", "experiment_results"):
        # Keep the implementation plan singleton; allow overwrite rather than uuid suffix.
        unique = False

    path, quarantined, note = resolve_output_path(subdir=subdir, name=norm_name, unique=unique)
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
def read_note(name: str = "pi_notes.md", return_size_threshold_chars: int = 2000):
    """
    Read a note file from the canonical run location (pi_notes.md or user_inbox.md). Returns empty string if missing.
    Truncates content over the threshold to limit context usage.
    """
    result = read_note_file(name=name)
    if result.get("error"):
        return result
    content = result.get("content", "")
    if isinstance(content, str) and len(content) > return_size_threshold_chars:
        truncated = _truncate_text_response(
            content,
            path=result.get("path"),
            threshold=return_size_threshold_chars,
            total_bytes=None,
            hint_tool="head_artifact",
        )
        result.update(truncated)
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

# --- ARCHIVIST SPECIALIZED TOOLS ---

@function_tool
def create_lit_summary_artifact(module: str = "lit", **kwargs: Any):
    """
    Create and register a new 'lit_summary_main' artifact.
    Use this when updating or creating the primary literature summary.
    """
    return reserve_and_register_artifact(
        kind="lit_summary_main",
        meta_json=json.dumps({"module": module}),
        status="pending",
        unique=True,
    )


@function_tool
def create_claim_graph_artifact(module: str = "lit", **kwargs: Any):
    """
    Create and register a new 'claim_graph_main' artifact.
    Use this when creating the claim graph.
    """
    return reserve_and_register_artifact(
        kind="claim_graph_main",
        meta_json=json.dumps({"module": module}),
        status="pending",
        unique=True,
    )


@function_tool
def list_lit_summaries(module: str = "lit", **kwargs: Any):
    """
    List existing literature summary artifacts for the given module.
    """
    return list_artifacts_by_kind(kind="lit_summary_main", limit=50)


@function_tool
def list_claim_graphs(module: str = "lit", **kwargs: Any):
    """
    List existing claim graph artifacts for the given module.
    """
    return list_artifacts_by_kind(kind="claim_graph_main", limit=50)


@function_tool
def read_archivist_artifact(name: str, **kwargs: Any):
    """
    Read a literature-related artifact (lit_summary, claim_graph).
    Restricted to specific allowed kinds.
    """
    # 1. Resolve artifact to get kind
    from ai_scientist.utils import manifest as manifest_utils
    from ai_scientist.tools.base_tool import BaseTool
    
    entry = manifest_utils.find_manifest_entry(name, base_folder=BaseTool.resolve_output_dir(None))
    if not entry:
        return {"error": f"Artifact '{name}' not found in manifest. Archivist can only read registered artifacts."}
    
    kind = entry.get("kind")
    allowed_kinds = {
        "lit_summary_main",
        "lit_summary_csv",
        "lit_reference_verification_table",
        "lit_reference_verification_json",
        "lit_review_md",
        "lit_bibliography_bib",
        "lit_coverage_json",
        "integration_memo_md",
        "claim_graph_main",
    }
    
    if kind not in allowed_kinds:
        return {
            "error": f"Permission denied: Archivist cannot read artifact kind '{kind}'. Allowed: {sorted(allowed_kinds)}"
        }
        
    if kind == "integration_memo_md":
        # Enforce module='lit'
        if entry.get("module") != "lit":
            return {"error": "Permission denied: Archivist can only read integration memos for module='lit'."}
        
    # 2. Delegate to read_artifact
    return read_artifact(path=entry.get("path") or name)


# --- Specialized Modeler Wrappers ---

@function_tool
def create_transport_artifact(baseline: str, transport: float, seed: int, artifact_type: str = "sim_json", **kwargs: Any):
    """
    Register a transport run artifact.
    artifact_type options: 'sim_json', 'status', 'failure_matrix', 'time_vector', 'nodes_order', 'per_compartment', 'node_index_map', 'topology_summary'.
    """
    kind_map = {
        "sim_json": "transport_sim_json",
        "status": "transport_sim_status",
        "failure_matrix": "transport_failure_matrix",
        "time_vector": "transport_time_vector",
        "nodes_order": "transport_nodes_order",
        "per_compartment": "transport_per_compartment",
        "node_index_map": "transport_node_index_map",
        "topology_summary": "transport_topology_summary",
    }
    kind = kind_map.get(artifact_type)
    if not kind:
        return {"error": f"Unknown artifact_type '{artifact_type}'. Options: {list(kind_map.keys())}"}

    meta = {
        "baseline": baseline,
        "transport": transport,
        "seed": seed,
        "module": "modeling",
    }
    
    # We use reserve_typed_artifact to get the path
    # unique=False because transport file paths are deterministic based on params
    return reserve_typed_artifact(kind=kind, meta_json=json.dumps(meta), unique=False)


@function_tool
def create_sensitivity_table_artifact(label: str, **kwargs: Any):
    """
    Register a sensitivity sweep CSV table.
    """
    meta = {"label": label, "module": "modeling"}
    return reserve_and_register_artifact(
        kind="sensitivity_sweep_table",
        meta_json=json.dumps(meta),
        status="pending",
        unique=True
    )


@function_tool
def create_intervention_table_artifact(label: str, **kwargs: Any):
    """
    Register an intervention test CSV table.
    """
    meta = {"label": label, "module": "modeling"}
    return reserve_and_register_artifact(
        kind="intervention_table",
        meta_json=json.dumps(meta),
        status="pending",
        unique=True
    )


@function_tool
def create_verification_note_artifact(experiment_id: str, **kwargs: Any):
    """
    Register a verification note (proof-of-work) for an experiment.
    """
    meta = {"experiment_id": experiment_id, "module": "modeling"}
    return reserve_and_register_artifact(
        kind="verification_note",
        meta_json=json.dumps(meta),
        status="pending",
        unique=True
    )


@function_tool
def list_model_specs(module: str = "modeling"):
    """
    List available model parameter sets (parameter_set artifacts).
    """
    return list_artifacts_by_kind(kind="parameter_set", limit=100)


@function_tool
def get_latest_model_spec(module: str = "modeling", model_key: Optional[str] = None):
    """
    Get the latest model specification artifact.
    """
    # from .artifacts import _get_latest_artifact_entry
    # _get_latest_artifact_entry only filters by kind/module.
    # To filter by model_key, we need to inspect metadata manually.
    # But for now, let's just use list_artifacts_by_kind and filter.
    
    res = list_artifacts_by_kind(kind="parameter_set", limit=100)
    paths = res.get("paths", [])
    if not paths:
        return {"error": "No model specs found."}
    
    # If model_key is provided, try to find a match in the filename?
    # Pattern is "{name}_params.json". So if model_key matches name.
    if model_key:
        matches = [p for p in paths if f"{model_key}_params.json" in p]
        if matches:
            return matches[0]
            
    return paths[0] # Return most recent (list is sorted by default? No, list_artifacts_by_kind just returns paths)
    # Actually list_artifacts_by_kind calls _list_artifacts_by_kind which pulls from manifest.
    # The manifest is appended to, so last entries are newest.
    # But list_artifacts_by_kind might not reverse it.
    # Let's trust list_artifacts_by_kind for now or check implementation.
    # _list_artifacts_by_kind in manifest_service returns entries.
    
    
@function_tool
def list_experiment_results(experiment_id: Optional[str] = None):
    """
    List transport simulation statuses.
    """
    # This might return too many results.
    return list_artifacts_by_kind(kind="transport_sim_status", limit=50)


@function_tool
def get_latest_metrics(model_key: str):
    """
    Get the latest metrics artifact for a model.
    """
    # We look for model_metrics_json
    # Pattern: {model_key}_metrics.json
    # Since filename is fixed per model_key (unless strictly unique?), 
    # reserve_typed_artifact uses unique=True by default for some?
    # ARTIFACT_TYPE_REGISTRY doesn't specify unique.
    
    # We use list to find it.
    res = list_artifacts_by_kind(kind="model_metrics_json", limit=50)
    paths = res.get("paths", [])
    target = f"{model_key}_metrics.json"
    matches = [p for p in paths if target in p]
    if matches:
        return matches[-1] # Assume last is newest
    return {"error": f"No metrics found for {model_key}"}


@function_tool
def read_model_spec(artifact_id_or_path: str):
    """
    Read a model specification (parameter_set).
    """
    # We should strictly enforce kind, but we only get a path/name.
    # We can lookup in manifest.
    return _safe_read_artifact(artifact_id_or_path, allowed_kinds=["parameter_set"])


@function_tool
def read_experiment_config(artifact_id_or_path: str):
    """
    Read an experiment config (e.g. from a note or plan).
    """
    # Maybe allowed implementation_plan or similar?
    # User said: "read_experiment_config(experiment_id)"
    # This might mean reading the hypothesis trace or similar?
    # Let's allow generic text/json reading for now but maybe wrap it?
    # Actually, let's skip strict enforcement unless we are sure.
    # User plan: "read_experiment_config(experiment_id)"
    
    return _safe_read_artifact(artifact_id_or_path, allowed_kinds=["implementation_plan", "hypothesis_trace_json"])


@function_tool
def read_metrics(artifact_id_or_path: str):
    """
    Read a metrics artifact.
    """
    return _safe_read_artifact(artifact_id_or_path, allowed_kinds=["model_metrics_json", "model_metrics_csv"])


def _safe_read_artifact(name_or_path: str, allowed_kinds: List[str]):
    from ai_scientist.utils import manifest as manifest_utils
    from ai_scientist.tools.base_tool import BaseTool
    
    entry = manifest_utils.find_manifest_entry(name_or_path, base_folder=BaseTool.resolve_output_dir(None))
    if not entry:
        # If passed an absolute path, we might allow it if it matches pattern?
        # But safer to require manifest entry.
        return {"error": f"Artifact '{name_or_path}' not found in manifest."}
        
    kind = entry.get("kind")
    if kind not in allowed_kinds:
         return {"error": f"Permission denied: Modeler cannot read kind '{kind}'. Allowed: {allowed_kinds}"}

    return read_artifact(path=entry.get("path") or name_or_path)


@function_tool
def create_model_spec_artifact(model_key: str, content_json: str, **kwargs: Any):
    """
    Register a model specification (parameter_set).
    content_json: JSON string of the model parameters.
    """
    # Assuming content_json is the file content.
    # We need to wrap it in a meta dict for strict typing if needed, 
    # but parameter_set uses {name}_params.json.
    # meta needs "name" = model_key.
    
    # Check if content_json is valid JSON?
    try:
        content = json.loads(content_json)
    except Exception as e:
        return {"error": f"Invalid content_json: {e}"}

    meta = {
        "name": model_key,
        "module": "modeling",
        "content": content,
        "summary": f"Model specification for {model_key}"
    }
    
    # If the content is passed in meta['content'], artifacts system might handle it?
    # reserve_typed_artifact just reserves the path. It doesn't write the content unless we use `write_to_file` or similar?
    # Wait, reserve_and_register_artifact documentation says:
    # "Preferred flow: 'reserve_and_register_artifact' -> write -> (optional) update status..."
    # But `create_transport_artifact` I implemented just calls `reserve_typed_artifact`.
    # Does Modeler write the file?
    # Directive 5b: "Reserve every persistent artifact...".
    # Directive 10 in Modeler (original): "Reserve every persistent artifact...".
    
    # If I use `create_model_spec_artifact`, should it write the file?
    # Modeler usually: reserves, then writes.
    # But if I wrap it, I can do both?
    # `run_biological_model` does the simulation.
    # For `parameter_set`, Modeler generates it.
    
    # If I follow the pattern: "create_X_artifact" returns artifact_id/path. Modeler then uses `write_text_artifact`?
    # The Prompt says: "When writing text, pass the reserved path into write_text_artifact".
    
    # So `create_model_spec_artifact` should just reserve (and register).
    # Since `parameter_set` is a JSON file, Modeler can use `write_text_artifact` or `write_file`?
    # `write_text_artifact` is in the keep list: "write_text_artifact (for structured notes...)".
    # Modeler also has `write_text_artifact` in its tool list.
    
    return reserve_and_register_artifact(
        kind="parameter_set",
        meta_json=json.dumps(meta),
        status="pending", # Modeler will write it next
        unique=True
    )


@function_tool
def create_plot_artifact(
    kind: Literal["plot_intermediate", "manuscript_figure_png", "manuscript_figure_svg"],
    figure_name: str,
    change_summary: str = "",
    metadata_json: Optional[str] = None
):
    """
    Create a registered plot artifact.
    kind: Must be one of 'plot_intermediate', 'manuscript_figure_png', 'manuscript_figure_svg'.
    figure_name: Base name for the figure (e.g. 'fig_1').
    """
    if kind not in ["plot_intermediate", "manuscript_figure_png", "manuscript_figure_svg"]:
        return {"error": f"Invalid plot kind '{kind}'. Must be one of: plot_intermediate, manuscript_figure_png, manuscript_figure_svg."}
    
    meta = {}
    if metadata_json:
        try:
            meta = json.loads(metadata_json)
        except Exception:
            return {"error": "Invalid metadata_json"}
            
    if kind == "plot_intermediate":
        if "slug" not in meta:
            meta["slug"] = figure_name
    elif kind.startswith("manuscript_figure"):
        if "figure_id" not in meta:
            clean_name = figure_name
            if clean_name.startswith("fig_"):
                clean_name = clean_name[4:]
            meta["figure_id"] = clean_name
            
    return reserve_and_register_artifact(
        kind=kind,
        meta_json=json.dumps(meta),
        change_summary=change_summary,
        unique=True 
    )

@function_tool
def publish_figure_to_manuscript_gallery(
    artifact_id: str,
    name_suffix: str = "",
):
    """
    Publish a figure artifact to the manuscript gallery (experiment_results/figures_for_manuscript).
    Wrapper around mirror_artifacts.
    """
    return mirror_artifacts(
        src_paths=[artifact_id],
        dest_dir="experiment_results/figures_for_manuscript",
        mode="copy",
        suffix=name_suffix
    )

@function_tool
def list_available_runs_for_plotting(experiment_id: Optional[str] = None):
    """
    List available simulation runs specifically for plotting.
    Filters the transport manifest for completed runs.
    """
    manifest_data = read_transport_manifest()
    runs = manifest_data.get("runs", [])
    valid_runs = []
    if isinstance(runs, list):
        for r in runs:
            if isinstance(r, dict) and r.get("status") == "complete":
                run_id = f"{r.get('baseline')}_transport_{r.get('transport')}_seed_{r.get('seed')}"
                valid_runs.append(run_id)
    return valid_runs

@function_tool
def get_metrics_for_plotting(experiment_id: Optional[str] = None, model_key: Optional[str] = None):
    """
    Find metrics artifacts (CSV/JSON) for a given model or experiment.
    """
    candidates = []
    
    r1 = list_artifacts_by_kind("model_metrics_json")
    if isinstance(r1, dict):
        candidates.extend(r1.get("paths", []) or [])
    
    r2 = list_artifacts_by_kind("model_metrics_csv")
    if isinstance(r2, dict):
        candidates.extend(r2.get("paths", []) or [])
    
    if model_key:
        filtered = [str(c) for c in candidates if model_key in str(c)]
        if filtered:
            json_matches = [f for f in filtered if f.endswith(".json")]
            if json_matches:
                return json_matches[0]
            return filtered[0]
            
    return None



# --- Specialized Reviewer Tools (VI-Reviewer) ---

@function_tool
def create_review_note_artifact(
    kind: Literal["verification_note", "review_report"],
    content: str,
    manuscript_version: Optional[str] = None,
    experiment_id: Optional[str] = None,
    change_summary: str = "",
):
    """
    Create a specialized review artifact (verification note or review report).
    - kind: "verification_note" (needs experiment_id) or "review_report" (needs manuscript_version).
    - content: The markdown content of the note/report.
    """
    if kind == "verification_note":
        if not experiment_id:
            return {"error": "experiment_id is required for verification_note"}
        meta = {"experiment_id": experiment_id, "module": "modeling"}
    elif kind == "review_report":
        if not manuscript_version:
            manuscript_version = "v1" # Default
        meta = {"manuscript_version": manuscript_version, "module": "review"}
    else:
        return {"error": f"Invalid kind '{kind}'. Must be 'verification_note' or 'review_report'."}

    # Reserve and register
    res = reserve_and_register_artifact(
        kind=kind,
        meta_json=json.dumps(meta),
        status="canonical", # Review notes are usually final for that version
        unique=True,
        change_summary=change_summary,
    )
    
    if res.get("error"):
        return res
        
    path_str = res.get("reserved_path")
    if not path_str:
        return {"error": "Failed to resolve path for review artifact."}
        
    # Write content
    try:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"status": "success", "path": path_str, "kind": kind}
    except Exception as exc:
        return {"error": f"Failed to write content: {exc}"}


@function_tool
def check_parameter_sources_for_manuscript(manuscript_id: Optional[str] = None):
    """
    Validate that all parameter sets have declared source types and references.
    Scans 'parameter_set' artifacts and checks for 'source_type' and 'reference_id'/'lit_claim_id'.
    """
    # 1. List parameter sets
    param_sets = list_artifacts_by_kind("parameter_set", limit=100)
    paths = param_sets.get("paths", [])
    
    report = []
    
    for p_path in paths:
        try:
            content = Path(p_path).read_text(encoding="utf-8")
            data = json.loads(content)
        except Exception as e:
            report.append(f"❌ {os.path.basename(p_path)}: Failed to read/parse ({e})")
            continue
            
        # Check structure (assuming dict of key -> {value, source_type, ...})
        # Or list of dicts? Adapting to common format.
        # Assuming format: { "param_name": { "value": ..., "source_type": ..., ... } }
        
        if not isinstance(data, dict):
             report.append(f"⚠️ {os.path.basename(p_path)}: Invalid format (not a dict)")
             continue

        file_issues = []
        for param, details in data.items():
            if not isinstance(details, dict):
                continue # Skip metadata keys if any
            
            source_type = details.get("source_type")
            if not source_type:
                file_issues.append(f"Missing source_type for '{param}'")
            elif source_type == "lit_value":
                if not (details.get("lit_claim_id") or details.get("reference_id")):
                     file_issues.append(f"'{param}' is lit_value but missing lit_claim_id/reference_id")
        
        if file_issues:
            report.append(f"❌ {os.path.basename(p_path)}:")
            for issue in file_issues:
                report.append(f"  - {issue}")
        else:
            report.append(f"✅ {os.path.basename(p_path)}: Valid.")

    if not paths:
        return "No parameter_set artifacts found."
        
    return "\n".join(report)


@function_tool
def check_metrics_for_referenced_models(manuscript_id: Optional[str] = None):
    """
    Check availability of metrics for referenced models.
    Scans for model_metrics_json/csv and cross-references with available model_specs (or similar).
    (Ideally would parse manuscript, but here we scan what exists).
    """
    # Simply list all metrics artifacts available
    metrics_json = list_artifacts_by_kind("model_metrics_json", limit=100)
    metrics_csv = list_artifacts_by_kind("model_metrics_csv", limit=100)
    
    found_metrics = [os.path.basename(p) for p in metrics_json.get("paths", [])]
    found_metrics += [os.path.basename(p) for p in metrics_csv.get("paths", [])]
    
    if not found_metrics:
        return "⚠️ No model metrics artifacts found (model_metrics_json/csv)."
        
    return f"✅ Found {len(found_metrics)} metrics artifacts:\n" + "\n".join([f"- {m}" for m in found_metrics])


@function_tool(strict_mode=False)
def check_hypothesis_trace_consistency():
    """
    Validate hypothesis_trace.json: checks if supported experiments have real runs/figures.
    """
    # Read trace
    trace_path = "experiment_results/hypothesis_trace.json"
    if not os.path.exists(trace_path):
        return "❌ hypothesis_trace.json not found."
        
    try:
        trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    except Exception as e:
        return f"❌ Failed to parse hypothesis_trace.json: {e}"

    report = []
    
    # Trace structure: list of { "hypothesis_id": ..., "experiments": [ { "experiment_id": ..., "status": ..., "runs": [], "figures": [] } ] }
    # Or flat list of hypotheses? Assuming common structure. 
    # Let's assume list of hypotheses.
    
    if isinstance(trace, dict) and "hypotheses" in trace:
        trace = trace["hypotheses"]
    
    if not isinstance(trace, list):
        return "❌ Invalid trace format (expected list or dict with 'hypotheses')"

    for hypo in trace:
        h_id = hypo.get("id") or "unknown"
        exps = hypo.get("experiments", [])
        for exp in exps:
            e_id = exp.get("id") or "unknown"
            status = exp.get("status")
            
            if status == "supported":
                # Check runs (params/sims existence) - currently implicit in figures check or requires more logic
                # runs = exp.get("runs", [])
                
                # Check figures
                figures = exp.get("figures", [])
                
                missing_figs = []
                for f in figures:
                    if not os.path.exists(f) and not os.path.exists(os.path.join("experiment_results", f)): 
                         # Try resolving path
                         missing_figs.append(f)
                
                if missing_figs:
                    report.append(f"❌ {h_id}/{e_id} (supported): Missing referenced figures: {missing_figs}")
                else:
                    report.append(f"✅ {h_id}/{e_id} (supported): Figures verified.")
                    
    if not report:
        return "⚠️ Trace appears empty or no supported experiments found to check."
        
    return "\n".join(report)


@function_tool
def check_proof_of_work_for_results():
    """
    Check that results (figures, tables) have corresponding verification notes.
    """
    # 1. List important results (figures)
    figs = list_artifacts_by_kind("manuscript_figure_png", limit=100)
    fig_paths = figs.get("paths", [])
    
    # 2. List verification notes
    notes = list_artifacts_by_kind("verification_note", limit=100)
    note_paths = notes.get("paths", [])
    if not note_paths:
        return "⚠️ No verification_note artifacts found."

    # Naive check: does the project have verification notes?
    # Ideally link specific figure to specific note.
    # For now, report counts.
    return f"Found {len(fig_paths)} manuscript figures and {len(note_paths)} verification notes. (Detailed linkage check not implemented)"


@function_tool
def get_lit_reference_verification():
    """
    Retrieve the Literature Reference Verification Table.
    """
    # Try CSV first, then JSON
    csv_art = list_artifacts_by_kind("lit_reference_verification_table", limit=1)
    if csv_art.get("paths"):
        path = csv_art["paths"][0]
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading verification CSV: {e}"
            
    json_art = list_artifacts_by_kind("lit_reference_verification_json", limit=1)
    if json_art.get("paths"):
        path = json_art["paths"][0]
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading verification JSON: {e}"
            
    return "❌ No literature reference verification artifacts found."


@function_tool
def check_references_completeness():
    """
    Analyze the literature reference verification stats.
    Returns a summary report of missing/invalid references.
    """
    content = get_lit_reference_verification()
    if content.startswith("❌") or content.startswith("Error"):
        return content
        
    # Check if CSV or JSON
    # Simple text scan for "False" in found column or low match scores
    # This is a heuristic validity check
    
    lines = content.splitlines()
    header = lines[0].lower() if lines else ""
    
    missing_count = 0
    total = 0
    
    if "," in header: # CSV
        # Assuming cols: citation_key, title, found, match_score...
        # Let's count "False" occurrence
        missing_count = content.count(",False,") + content.count(",false,")
        total = len(lines) - 1
    else:
        # JSON or other
        missing_count = content.count('"found": false') + content.count('"found": False')
        total = content.count('"citation_key"')
        
    status = "✅ References look complete."
    if missing_count > 0:
        status = f"❌ Found {missing_count} missing references."
        
    return f"Checked {total} references.\n{status}"


# --- PI PLANNING TOOLS ---

@function_tool
def get_or_create_implementation_plan():
    """
    Get or create the structured implementation plan for the current run.
    Returns the path to implementation_plan.md and whether it was newly created.
    """
    from ai_scientist.orchestrator.pi_planning_helpers import get_or_create_implementation_plan as _get_or_create_impl
    
    run_root = BaseTool.resolve_output_dir(None).parent
    try:
        plan_path, created = _get_or_create_impl(run_root)
        return {
            "path": str(plan_path),
            "created": created,
            "status": "created" if created else "exists"
        }
    except Exception as e:
        return {"error": str(e)}


@function_tool
def update_implementation_plan_from_state(
    hypothesis: str,
    current_phase: str,
    experiments_json: str,
    tasks_json: str,
    decisions_json: str
):
    """
    Update the implementation plan with structured state.
    
    Args:
        hypothesis: Current hypothesis text
        current_phase: One of: planning, modeling, analysis, writeup, publication
        experiments_json: JSON array of experiments, each with: experiment_id, description, owner_role, status, inputs, outputs, notes
        tasks_json: JSON array of tasks, each with: task_id, experiment_id, description, assigned_to, status, linked_artifacts, last_updated
        decisions_json: JSON array of decision strings (will be formatted as bullet points)
    """
    from ai_scientist.orchestrator.pi_planning_helpers import (
        update_implementation_plan_from_state as _update_impl,
        PlanState,
        ExperimentPlan,
        TaskPlan,
    )
    
    try:
        # Parse JSON inputs
        experiments_data = json.loads(experiments_json)
        tasks_data = json.loads(tasks_json)
        decisions_data = json.loads(decisions_json)
        
        # Backward compatibility: convert string fields to lists
        for exp in experiments_data:
            # Convert inputs to list if it's a string
            if isinstance(exp.get("inputs"), str):
                exp["inputs"] = [s.strip() for s in exp["inputs"].split(",") if s.strip()]
            elif not isinstance(exp.get("inputs"), list):
                exp["inputs"] = []
            
            # Convert outputs to list if it's a string
            if isinstance(exp.get("outputs"), str):
                exp["outputs"] = [s.strip() for s in exp["outputs"].split(",") if s.strip()]
            elif not isinstance(exp.get("outputs"), list):
                exp["outputs"] = []
        
        for task in tasks_data:
            # Convert linked_artifacts to list if it's a string
            if isinstance(task.get("linked_artifacts"), str):
                task["linked_artifacts"] = [s.strip() for s in task["linked_artifacts"].split(",") if s.strip()]
            elif not isinstance(task.get("linked_artifacts"), list):
                task["linked_artifacts"] = []
        
        # Convert to dataclasses
        experiments = [ExperimentPlan(**exp) for exp in experiments_data]
        tasks = [TaskPlan(**task) for task in tasks_data]
        
        # Create PlanState
        now = datetime.utcnow().isoformat()
        state = PlanState(
            hypothesis=hypothesis,
            current_phase=current_phase,
            last_updated=now,
            experiments=experiments,
            tasks=tasks,
            decisions=decisions_data
        )
        
        # Get plan path
        run_root = BaseTool.resolve_output_dir(None).parent
        from ai_scientist.orchestrator.pi_planning_helpers import get_or_create_implementation_plan as _get_or_create_impl
        plan_path, _ = _get_or_create_impl(run_root)
        
        # Update the plan
        _update_impl(plan_path, state)
        
        return {
            "status": "updated",
            "path": str(plan_path),
            "experiments_count": len(experiments),
            "tasks_count": len(tasks),
            "decisions_count": len(decisions_data)
        }
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}
    except Exception as e:
        return {"error": str(e)}


@function_tool
def log_status_to_user_inbox(status_block: str):
    """
    Append a timestamped status update to user_inbox.md.
    Use this to persist important status updates so the user can see them outside the chat.
    
    Args:
        status_block: The status message to log (will be timestamped automatically)
    """
    from ai_scientist.orchestrator.pi_planning_helpers import log_status_to_user_inbox as _log_status_impl
    
    run_root = BaseTool.resolve_output_dir(None).parent
    try:
        _log_status_impl(run_root, status_block)
        return {
            "status": "logged",
            "message": "Status appended to user_inbox.md"
        }
    except Exception as e:
        return {"error": str(e)}

