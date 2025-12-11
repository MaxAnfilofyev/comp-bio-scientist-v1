import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from ai_scientist.utils.notes import ensure_note_files
from ai_scientist.utils.pathing import resolve_output_path


def _fill_output_dir(output_dir: Optional[str]) -> str:
    """
    Resolve output dir to the run-specific folder using sanitized resolver.
    """
    path, _, _ = resolve_output_path(subdir=None, name="", allow_quarantine=False, unique=False)
    base = str(path)
    if output_dir:
        if os.path.isabs(output_dir):
            return output_dir
        # Fix: If agent provides "experiment_results/foo", strip the prefix to avoid
        # "experiment_results/experiment_results/foo" via resolve_output_path.
        clean_name = output_dir
        if clean_name.startswith("experiment_results/") or clean_name == "experiment_results":
            clean_name = clean_name.replace("experiment_results/", "", 1)
            if clean_name == "experiment_results":
                clean_name = ""
        
        if not clean_name:
            return base

        anchored, _, _ = resolve_output_path(subdir=None, name=clean_name, allow_quarantine=False, unique=False)
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


def parse_args():
    p = argparse.ArgumentParser(description="ADCRT: Argument-Driven Computational Research Team")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--load_idea", help="Path to idea JSON.")
    src.add_argument("--load_manuscript", help="Path to manuscript (md/pdf/txt) used to derive an idea seed.")
    p.add_argument("--manuscript_title", help="Override title when deriving idea seed from manuscript.")
    p.add_argument("--model", default="gpt-5-mini-2025-08-07", help="LLM model id.")
    p.add_argument("--max_cycles", type=int, default=30, help="Max Orchestrator turns.")
    p.add_argument("--timeout", type=float, default=1800.0, help="Timeout in seconds (default: 30 mins).")
    p.add_argument("--base_folder", default=None, help="Existing experiment directory to restart from (overrides timestamped creation).")
    p.add_argument("--resume", action="store_true", help="Don't overwrite existing experiment folder.")
    p.add_argument("--idea_idx", type=int, default=0, help="Index if the idea file contains a list of ideas.")
    p.add_argument("--input", default=None, help="Initial input message to the PI agent.")
    p.add_argument("--human_in_the_loop", action="store_true", help="Enable interactive mode where agents ask for confirmation before expensive tasks.")
    p.add_argument("--skip_lit_gate", action="store_true", help="Allow modeling/sim tools to bypass literature readiness gate.")
    p.add_argument(
        "--enforce_param_provenance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require parameter provenance to be complete before running models (default: True).",
    )
    p.add_argument(
        "--enforce_claim_consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require claim_graph/hypothesis_trace consistency before finishing writeup (default: True).",
    )
    return p.parse_args()


def _bootstrap_note_links() -> None:
    for name in ("pi_notes.md", "user_inbox.md"):
        try:
            ensure_note_files(name)
        except Exception as exc:
            print(f"⚠️ Failed to ensure {name}: {exc}")


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
    template_path = Path(__file__).parent.parent / "docs" / "transport_runs_README.md"
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


def _report_capabilities() -> Dict[str, Any]:
    tools = {name: bool(shutil.which(name)) for name in ("pandoc", "pdflatex", "ruff", "pyright")}
    return {"tools": tools, "pdf_engine_ready": tools.get("pandoc") and tools.get("pdflatex")}
