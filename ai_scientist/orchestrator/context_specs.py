from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Literal

from ai_scientist.tools.base_tool import BaseTool

ACTIVE_ROLE_ENV = "AISC_ACTIVE_ROLE"

ModuleName = Literal[
    "analysis",
    "general",
    "interpretation",
    "literature",
    "modeling",
    "oversight",
    "plumbing",
    "review",
    "writeup",
]
VALID_MODULE_NAMES: Tuple[ModuleName, ...] = (
    "general",
    "literature",
    "modeling",
    "analysis",
    "review",
    "interpretation",
    "plumbing",
    "writeup",
    "oversight",
)

# Minimal context view spec describing what each role may read/write and how much context to consume.
@dataclass(frozen=True)
class ContextViewSpec:
    role: str
    read_scope: Sequence[str]
    write_scope: Sequence[str]
    max_artifacts: int
    summary_preferred: bool
    summary_limit: int
    description: str = ""
    module_name: ModuleName = "general"

    def allows_read(self, kind: Optional[str]) -> bool:
        if not kind:
            return True
        return "*" in self.read_scope or kind in self.read_scope

    def allows_write(self, kind: Optional[str]) -> bool:
        if not kind:
            return True
        return "*" in self.write_scope or kind in self.write_scope


# Canonical context view specs for each role.
ROLE_CONTEXT_SPECS: Dict[str, ContextViewSpec] = {
    "Archivist": ContextViewSpec(
        role="Archivist",
        read_scope=[
            "lit_summary_main",
            "lit_summary_csv",
            "lit_reference_verification_table",
            "lit_reference_verification_json",
            "claim_graph_main",
            "manuscript_input_text",
        ],
        write_scope=[
            "lit_summary_main",
            "lit_summary_csv",
            "lit_reference_verification_table",
            "lit_reference_verification_json",
            "claim_graph_main",
            "lit_review_md",
            "lit_bibliography_bib",
            "lit_coverage_json",
            "integration_memo_md",
        ],
        max_artifacts=6,
        summary_preferred=True,
        summary_limit=3,
        description="Focus on paper notes, lit_summary, and the claim graph.",
        module_name="literature"
    ),
    "Modeler": ContextViewSpec(
        role="Modeler",
        read_scope=[
            "graph_pickle",
            "graph_topology_json",
            "parameter_set",
            "biological_model_solution",
            "transport_manifest",
            "transport_sim_json",
            "transport_sim_status",
            "transport_failure_matrix",
            "transport_time_vector",
            "transport_nodes_order",
            "transport_per_compartment",
            "transport_node_index_map",
            "transport_topology_summary",
            "hypothesis_trace_json",
        ],
        write_scope=[
            "transport_manifest",
            "transport_sim_json",
            "transport_sim_status",
            "transport_failure_matrix",
            "transport_time_vector",
            "transport_nodes_order",
            "transport_per_compartment",
            "transport_node_index_map",
            "transport_topology_summary",
            "sensitivity_sweep_table",
            "intervention_table",
            "parameter_set",
            "hypothesis_trace_json",
            "verification_note",
            "model_metrics_json",
            "model_metrics_csv",
        ],
        max_artifacts=14,
        summary_preferred=False,
        summary_limit=0,
        description="Read transport manifests and graph specs; reserve transport outputs and sweep tables.",
        module_name="modeling"
    ),
    "Analyst": ContextViewSpec(
        role="Analyst",
        read_scope=[
            "transport_sim_json",
            "transport_sim_status",
            "transport_per_compartment",
            "transport_node_index_map",
            "transport_topology_summary",
            "sensitivity_sweep_table",
            "intervention_table",
            "plot_intermediate",
            "manuscript_figure_png",
            "manuscript_figure_svg",
            "model_metrics_json",
            "sweep_metrics_csv",
        ],
        write_scope=[
            "plot_intermediate",
            "manuscript_figure_png",
            "manuscript_figure_svg",
            "verification_note",
        ],
        max_artifacts=10,
        summary_preferred=True,
        summary_limit=5,
        description="Consume validated sim outputs and metrics; produce manuscript figures.",
        module_name="analysis"
    ),
    "Reviewer": ContextViewSpec(
        role="Reviewer",
        read_scope=[
            "manuscript_input_text",
            "lit_summary_main",
            "lit_reference_verification_json",
            "claim_graph_main",
            "provenance_summary_md",
            "hypothesis_trace_json",
            "transport_sim_json",
            "transport_manifest",
        ],
        write_scope=[
            "verification_note",
            "review_report",
            "provenance_summary_md",
        ],
        max_artifacts=12,
        summary_preferred=True,
        summary_limit=4,
        description="Survey manuscripts, provenance, claim graphs, and hypothesis traces to flag gaps.",
        module_name="review"
    ),
    "Interpreter": ContextViewSpec(
        role="Interpreter",
        read_scope=[
            "lit_summary_main",
            "hypothesis_trace_json",
            "manuscript_input_text",
            "interpretation_json",
            "interpretation_md",
        ],
        write_scope=[
            "interpretation_json",
            "interpretation_md",
        ],
        max_artifacts=8,
        summary_preferred=True,
        summary_limit=4,
        description="Produce theoretical interpretations from summaries and idea text.",
        module_name="interpretation"
    ),
    "Coder": ContextViewSpec(
        role="Coder",
        read_scope=["*"],
        write_scope=["verification_note"],
        max_artifacts=20,
        summary_preferred=False,
        summary_limit=0,
        description="Full read/write scope limited by PI oversight.",
        module_name="plumbing"
    ),
    "Publisher": ContextViewSpec(
        role="Publisher",
        read_scope=["*"],
        write_scope=[
            "provenance_summary_md",
            "release_manifest",
            "code_release_archive",
            "env_manifest",
            "repro_methods_md",
            "repro_protocol_md",
            "release_repro_status_md",
            "release_diff_patch",
            "manuscript_figure_png",
            "manuscript_figure_svg",
        ],
        max_artifacts=25,
        summary_preferred=False,
        summary_limit=0,
        description="Generate final manuscript PDF and release artifacts.",
        module_name="writeup"
    ),
    "Principal Investigator": ContextViewSpec(
        role="Principal Investigator",
        read_scope=["*"],
        write_scope=[
            "implementation_plan_md",
            "implementation_plan_state_json",
            "hypothesis_trace_json",
            "project_snapshot_md",
        ],
        max_artifacts=30,
        summary_preferred=True,
        summary_limit=8,
        description="Orchestrate the project with overview access to artifacts and summaries.",
        module_name="oversight"
    ),
}


def get_context_view_spec(role: str) -> Optional[ContextViewSpec]:
    return ROLE_CONTEXT_SPECS.get(role)


def get_module_for_role(role: str) -> Optional[str]:
    spec = get_context_view_spec(role)
    if spec is None:
        return None
    return spec.module_name


def format_context_spec_for_prompt(spec: ContextViewSpec) -> str:
    read = ", ".join(spec.read_scope)
    write = ", ".join(spec.write_scope)
    summary_hint = "Summary-first: keep artifacts high-level." if spec.summary_preferred else "Full content permitted when needed."
    return (
        f"CONTEXT VIEW ({spec.role}): Read {len(spec.read_scope)} kinds ({read}); "
        f"Write {len(spec.write_scope)} kinds ({write}); "
        f"Limit context to {spec.max_artifacts} artifacts (Top-K) with {spec.summary_limit} summaries. {summary_hint} "
        f"{spec.description}"
    )


def active_role() -> str:
    return os.environ.get(ACTIVE_ROLE_ENV, "unknown")


def _context_log_path() -> Path:
    exp_dir = BaseTool.resolve_output_dir(None)
    return Path(exp_dir) / "context_usage.log"


def record_context_access(
    role: str,
    kind: Optional[str],
    path: Optional[str],
    access_type: str,
    artifact_id: Optional[str] = None,
    version: Optional[str] = None,
) -> None:
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "role": role,
        "access_type": access_type,
        "kind": kind,
        "path": path,
        "artifact_id": artifact_id,
        "version": version,
    }
    log_path = _context_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass  # best effort logging
    
    _update_role_status(role, access_type, kind, path, artifact_id, version)


def _update_role_status(
    role: str, 
    access_type: str, 
    kind: Optional[str], 
    path: Optional[str], 
    artifact_id: Optional[str],
    version: Optional[str]
) -> None:
    if not role or role == "unknown":
        return
    
    exp_dir = Path(BaseTool.resolve_output_dir(None))
    status_dir = exp_dir / "_status"
    try:
        status_dir.mkdir(parents=True, exist_ok=True)
        status_file = status_dir / f"status_{role}.json"
        
        # Simple atomic-ish update: read, update, write
        # In high concurrency this might race, but for single-threaded agent steps it's okay.
        # We rely on filesystem lock or just hope for best in this prototype.
        
        data = {"role": role, "read": {}, "written": {}, "updated_at": datetime.utcnow().isoformat()}
        if status_file.exists():
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass 
        
        # We track unique artifacts by path or ID.
        # Key = path (since ID might be missing for some reads)
        key = path or artifact_id or "unknown"
        val = {
            "kind": kind,
            "id": artifact_id,
            "version": version,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if "read" in access_type:
            data.setdefault("read", {})[key] = val
        elif "write" in access_type:
            data.setdefault("written", {})[key] = val
            
        data["updated_at"] = datetime.utcnow().isoformat()
        
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
    except Exception:
        pass


def ensure_write_permission(kind: str, role: str) -> tuple[bool, Optional[str]]:
    spec = get_context_view_spec(role)
    if spec is None:
        return True, None
    if spec.allows_write(kind):
        return True, None
    message = (
        f"Role '{role}' may not write artifacts of kind '{kind}'. "
        f"Allowed kinds: {spec.write_scope}"
    )
    record_context_access(role, kind, None, "write_denied")
    return False, message


def truncate_paths_for_role(role: str, paths: List[str], kind: Optional[str] = None) -> List[str]:
    spec = get_context_view_spec(role)
    if spec is None:
        return paths
    allowed = spec.max_artifacts
    truncated = paths[:allowed]
    if len(paths) > allowed:
        record_context_access(role, kind, None, "read_truncated")
    for path in truncated:
        record_context_access(role, kind, path, "read")
    return truncated
