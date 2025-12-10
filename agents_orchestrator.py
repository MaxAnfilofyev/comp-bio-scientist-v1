# pyright: reportMissingImports=false
import argparse
import csv
import json
import os
import os.path as osp
import re
from pathlib import Path
from datetime import datetime, timedelta
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import ast
import difflib
from string import Formatter
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, cast

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
from ai_scientist.tools.reference_verification import ReferenceVerificationTool
from ai_scientist.tools.compute_model_metrics import ComputeModelMetricsTool

from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_biological_interpretation import interpret_biological_results
from ai_scientist.utils.notes import NOTE_NAMES, ensure_note_files, read_note_file, write_note_file, append_run_note
from ai_scientist.utils.pathing import resolve_output_path
from ai_scientist.utils.transport_index import index_transport_runs, resolve_transport_sim
from ai_scientist.utils.health import log_missing_or_corrupt
from ai_scientist.utils import manifest as manifest_utils

# Cached idea for hypothesis trace bootstrapping
_ACTIVE_IDEA: Optional[Dict[str, Any]] = None
# --- Canonical Artifact Types (VI-01) ---
# Each kind maps to a canonical subdirectory (relative to experiment_results) and a filename pattern.
# Patterns may use {placeholders} that must be provided via meta_json in reserve_typed_artifact.
ARTIFACT_TYPE_REGISTRY: Dict[str, Dict[str, str]] = {
    "lit_summary_main": {
        "rel_dir": "experiment_results",
        "pattern": "lit_summary.json",
        "description": "Primary literature summary.",
    },
    "lit_summary_csv": {
        "rel_dir": "experiment_results",
        "pattern": "lit_summary.csv",
        "description": "CSV-formatted literature summary.",
    },
    "lit_reference_verification_table": {
        "rel_dir": "experiment_results",
        "pattern": "lit_reference_verification.csv",
        "description": "Reference verification table (CSV).",
    },
    "lit_reference_verification_json": {
        "rel_dir": "experiment_results",
        "pattern": "lit_reference_verification.json",
        "description": "Reference verification table (JSON).",
    },
    "claim_graph_main": {
        "rel_dir": "experiment_results",
        "pattern": "claim_graph.json",
        "description": "Claim graph JSON.",
    },
    "graph_pickle": {
        "rel_dir": "experiment_results/graphs",
        "pattern": "{graph_id}.gpickle",
        "description": "Pickled graph for simulations.",
    },
    "graph_topology_json": {
        "rel_dir": "experiment_results/graphs",
        "pattern": "{graph_id}_topology.json",
        "description": "Graph topology summary JSON.",
    },
    "parameter_set": {
        "rel_dir": "experiment_results/parameters",
        "pattern": "{name}_params.json",
        "description": "Parameter set definition for simulations or models.",
    },
    "biological_model_solution": {
        "rel_dir": "experiment_results/models",
        "pattern": "{model_key}_solution.json",
        "description": "Solution from built-in biological model.",
    },
    "transport_manifest": {
        "rel_dir": "experiment_results/simulations/transport_runs",
        "pattern": "manifest.json",
        "description": "Transport run manifest index.",
    },
    "transport_sim_json": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "{baseline}_sim.json",
        "description": "Simulation JSON for a transport run.",
    },
    "transport_sim_status": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "{baseline}_sim.status.json",
        "description": "Status JSON for a transport run.",
    },
    "transport_failure_matrix": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "{baseline}_sim_failure_matrix.npy",
        "description": "Failure matrix array.",
    },
    "transport_time_vector": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "{baseline}_sim_time_vector.npy",
        "description": "Time vector array.",
    },
    "transport_nodes_order": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "nodes_order_{baseline}_sim.txt",
        "description": "Node ordering text file.",
    },
    "transport_per_compartment": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "per_compartment.npz",
        "description": "Per-compartment export (binary/continuous/time).",
    },
    "transport_node_index_map": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "node_index_map.json",
        "description": "Node index map for per-compartment outputs.",
    },
    "transport_topology_summary": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "topology_summary.json",
        "description": "Topology summary for per-compartment outputs.",
    },
    "sensitivity_sweep_table": {
        "rel_dir": "experiment_results/simulations/sensitivity_sweep",
        "pattern": "sweep__{label}.csv",
        "description": "Sensitivity sweep CSV output.",
    },
    "intervention_table": {
        "rel_dir": "experiment_results/simulations/interventions",
        "pattern": "intervention__{label}.csv",
        "description": "Intervention tester CSV output.",
    },
    "plot_intermediate": {
        "rel_dir": "experiment_results/figures",
        "pattern": "{slug}.png",
        "description": "Intermediate plot or diagnostic figure.",
    },
    "manuscript_figure_png": {
        "rel_dir": "experiment_results/figures_for_manuscript",
        "pattern": "fig_{figure_id}.png",
        "description": "PNG figure for manuscript.",
    },
    "manuscript_figure_svg": {
        "rel_dir": "experiment_results/figures_for_manuscript",
        "pattern": "fig_{figure_id}.svg",
        "description": "SVG figure for manuscript.",
    },
    "phase_portrait": {
        "rel_dir": "experiment_results/figures",
        "pattern": "phase_portrait_{label}.png",
        "description": "Phase portrait plot.",
    },
    "energetic_landscape": {
        "rel_dir": "experiment_results/figures",
        "pattern": "energy_landscape_{label}.png",
        "description": "Energetic landscape plot.",
    },
    "ac_sweep": {
        "rel_dir": "experiment_results/simulations/ac_sweep",
        "pattern": "ac_sweep__{run_id}.csv",
        "description": "AC sweep results CSV.",
    },
    "figures_readme": {
        "rel_dir": "experiment_results/figures_for_manuscript",
        "pattern": "README.md",
        "description": "Figures README/index.",
    },
    "interpretation_json": {
        "rel_dir": "experiment_results",
        "pattern": "interpretation.json",
        "description": "Structured interpretation output.",
    },
    "interpretation_md": {
        "rel_dir": "experiment_results",
        "pattern": "interpretation.md",
        "description": "Markdown interpretation output.",
    },
    "model_spec_yaml": {
        "rel_dir": "experiment_results/models",
        "pattern": "{model_key}_spec.yaml",
        "description": "Model specification (YAML/JSON-compatible).",
    },
    "parameter_source_table": {
        "rel_dir": "experiment_results/parameters",
        "pattern": "{model_key}_param_sources.csv",
        "description": "Parameter provenance ledger.",
    },
    "verification_note": {
        "rel_dir": "experiment_results",
        "pattern": "{artifact_id}_verification.md",
        "description": "Proof-of-work verification note.",
    },
    "writeup_pdf": {
        "rel_dir": "experiment_results",
        "pattern": "manuscript.pdf",
        "description": "Final manuscript PDF.",
    },
    "manuscript_pdf": {
        "rel_dir": "experiment_results",
        "pattern": "manuscript.pdf",
        "description": "Final manuscript PDF.",
    },
    "hypothesis_trace_json": {
        "rel_dir": "experiment_results",
        "pattern": "hypothesis_trace.json",
        "description": "Traceability map: hypothesis -> experiments -> sim runs/figures.",
    },
    "model_metrics_json": {
        "rel_dir": "experiment_results/models",
        "pattern": "{model_key}_metrics.json",
        "description": "Model-level metrics and bifurcation proxies.",
    },
    "sweep_metrics_csv": {
        "rel_dir": "experiment_results/simulations",
        "pattern": "{label}_metrics.csv",
        "description": "Aggregated metrics for parameter sweeps or batches.",
    },
    "provenance_summary_md": {
        "rel_dir": "experiment_results",
        "pattern": "provenance_summary.md",
        "description": "Manuscript-ready provenance summary (literature, models, sims, stats).",
    },
}

def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    regex_parts: List[str] = []
    formatter = Formatter()
    for literal, field, _format_spec, _conv in formatter.parse(pattern):
        if literal:
            regex_parts.append(re.escape(literal))
        if field:
            # allow alphanumerics, dash, underscore, dot in placeholder substitutions
            regex_parts.append(r"(?P<%s>[A-Za-z0-9._-]+)" % field)
    regex = "".join(regex_parts)
    return re.compile(rf"^{regex}$")

def _format_artifact_path(kind: str, meta: Dict[str, Any]) -> Tuple[str, str]:
    """
    Resolve (rel_dir, name) for an artifact kind using the registry.
    """
    entry = ARTIFACT_TYPE_REGISTRY.get(kind)
    if not entry:
        raise KeyError(f"Unknown artifact kind '{kind}'")
    rel_dir_template = entry["rel_dir"]
    pattern_template = entry["pattern"]
    try:
        rel_dir = rel_dir_template.format(**meta)
    except KeyError as exc:
        raise KeyError(f"Missing placeholder '{exc.args[0]}' for rel_dir of kind '{kind}'") from exc
    try:
        name = pattern_template.format(**meta)
    except KeyError as exc:
        raise KeyError(f"Missing placeholder '{exc.args[0]}' for pattern of kind '{kind}'") from exc
    if "/" in name or name.strip() == "":
        raise ValueError("Pattern must resolve to a filename without directories.")
    regex = _pattern_to_regex(pattern_template)
    if not regex.fullmatch(name):
        raise ValueError(f"Resolved name '{name}' does not match pattern '{pattern_template}'.")
    return rel_dir, name


def _artifact_kind_catalog() -> str:
    parts = []
    for kind in sorted(ARTIFACT_TYPE_REGISTRY.keys()):
        entry = ARTIFACT_TYPE_REGISTRY[kind]
        parts.append(f"{kind}: {entry['rel_dir']}/{entry['pattern']}")
    return "; ".join(parts)

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
    p.add_argument("--skip_lit_gate", action="store_true", help="Allow modeling/sim tools to bypass literature readiness gate.")
    p.add_argument(
        "--enforce_param_provenance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require parameter provenance to be complete before running models (default: True).",
    )
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


def _bootstrap_hypothesis_trace(idea: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize a hypothesis_trace.json skeleton from the idea JSON.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    trace_path = exp_dir / "hypothesis_trace.json"
    if trace_path.exists():
        try:
            return json.loads(trace_path.read_text())
        except Exception:
            pass

    hypotheses = []
    idea_name = idea.get("Name") or idea.get("Title") or "HYP"
    hyp_id = "H1"
    hyp_text = idea.get("Short Hypothesis") or idea.get("Abstract") or ""

    experiments = []
    for idx, exp in enumerate(idea.get("Experiments", []) or []):
        exp_id = f"E{idx+1}"
        experiments.append(
            {
                "id": exp_id,
                "description": exp if isinstance(exp, str) else exp.get("description", str(exp)),
                "sim_runs": [],
                "figures": [],
                "metrics": [],
            }
        )

    hypotheses.append(
        {
            "id": hyp_id,
            "name": idea_name,
            "text": hyp_text,
            "experiments": experiments,
            "status": "inconclusive",
        }
    )

    skeleton = {"hypotheses": hypotheses}
    trace_path.write_text(json.dumps(skeleton, indent=2))
    return skeleton


def _load_hypothesis_trace() -> Dict[str, Any]:
    """
    Load hypothesis_trace.json, bootstrapping from the active idea when missing.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    trace_path = exp_dir / "hypothesis_trace.json"
    if trace_path.exists():
        try:
            return json.loads(trace_path.read_text())
        except Exception:
            pass
    return _bootstrap_hypothesis_trace(_ACTIVE_IDEA or {})


def _write_hypothesis_trace(trace: Dict[str, Any]) -> str:
    exp_dir = BaseTool.resolve_output_dir(None)
    trace_path = exp_dir / "hypothesis_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2))
    return str(trace_path)


def _ensure_hypothesis_entry(trace: Dict[str, Any], hypothesis_id: str) -> Dict[str, Any]:
    hypotheses = trace.setdefault("hypotheses", [])
    for hyp in hypotheses:
        if hyp.get("id") == hypothesis_id:
            return hyp
    hyp = {"id": hypothesis_id, "text": "", "experiments": [], "status": "inconclusive"}
    hypotheses.append(hyp)
    return hyp


def _ensure_experiment_entry(hypothesis: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
    exps = hypothesis.setdefault("experiments", [])
    for exp in exps:
        if exp.get("id") == experiment_id:
            return exp
    exp = {"id": experiment_id, "description": "", "sim_runs": [], "figures": [], "metrics": []}
    exps.append(exp)
    return exp


def _update_hypothesis_trace_with_sim(
    hypothesis_id: str,
    experiment_id: str,
    sim_entry: Dict[str, Any],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    trace = _load_hypothesis_trace()
    hyp = _ensure_hypothesis_entry(trace, hypothesis_id)
    exp = _ensure_experiment_entry(hyp, experiment_id)
    sim_runs = exp.setdefault("sim_runs", [])
    if sim_entry and sim_entry not in sim_runs:
        sim_runs.append(sim_entry)
    if metrics:
        metric_set = set(exp.get("metrics", []))
        metric_set.update(metrics)
        exp["metrics"] = sorted(metric_set)
    _write_hypothesis_trace(trace)
    return trace


def _update_hypothesis_trace_with_figures(
    hypothesis_id: str,
    experiment_id: str,
    figures: List[str],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    trace = _load_hypothesis_trace()
    hyp = _ensure_hypothesis_entry(trace, hypothesis_id)
    exp = _ensure_experiment_entry(hyp, experiment_id)
    fig_list = exp.setdefault("figures", [])
    for fig in figures:
        if fig and fig not in fig_list:
            fig_list.append(fig)
    if metrics:
        metric_set = set(exp.get("metrics", []))
        metric_set.update(metrics)
        exp["metrics"] = sorted(metric_set)
    _write_hypothesis_trace(trace)
    return trace


def _model_metadata_from_key(model_key: str) -> Dict[str, Any]:
    """
    Lightweight metadata for built-in biological models to seed spec/ledger files.
    """
    description_map = {
        "cooperation_evolution": "Replicator dynamics for a Prisoner’s Dilemma cooperation model.",
        "predator_prey": "Lotka-Volterra predator-prey ODEs with four rate parameters.",
        "epidemiology_sir": "Classic SIR epidemiological ODE model.",
    }
    equations_map = {
        "cooperation_evolution": [
            "dx_coop/dt = x_coop*(w_coop - avg_w)",
            "dx_defect/dt = x_defect*(w_defect - avg_w)",
            "w_coop = x_coop*(benefit-cost) + x_defect*benefit",
            "w_defect = x_coop*0 + x_defect*benefit",
        ],
        "predator_prey": [
            "dprey/dt = alpha*prey - beta*prey*predator",
            "dpredator/dt = -gamma*predator + delta*prey*predator",
        ],
        "epidemiology_sir": [
            "dS/dt = -beta*S*I",
            "dI/dt = beta*S*I - gamma*I",
            "dR/dt = gamma*I",
        ],
    }
    param_units_map = {
        "cooperation_evolution": {
            "benefit": "utility",
            "cost": "utility",
            "mutation_rate": "probability",
        },
        "predator_prey": {
            "alpha": "1/time",
            "beta": "1/(time*population)",
            "gamma": "1/time",
            "delta": "1/time",
        },
        "epidemiology_sir": {
            "beta": "1/(time*population)",
            "gamma": "1/time",
        },
    }

    params: Dict[str, Any] = {}
    initial_conditions: Dict[str, Any] = {}
    try:
        from ai_scientist.perform_biological_modeling import create_sample_models

        models = create_sample_models()
        model = models.get(model_key)
        if model:
            params = getattr(model, "parameters", {}) or {}
            initial_conditions = getattr(model, "initial_conditions", {}) or {}
    except Exception:
        # Best-effort; fall back to defaults below.
        pass

    states = list(initial_conditions.keys()) if initial_conditions else []
    if model_key == "cooperation_evolution":
        states = states or ["strategy_0", "strategy_1"]
    elif model_key == "epidemiology_sir":
        states = states or ["susceptible", "infected", "recovered"]
    elif model_key == "predator_prey":
        states = states or ["prey", "predator"]

    return {
        "description": description_map.get(model_key, f"Model {model_key} specification."),
        "equations": equations_map.get(model_key, []),
        "params": params,
        "param_units": param_units_map.get(model_key, {}),
        "initial_conditions": initial_conditions,
        "states": states,
    }


def _ensure_model_spec_and_params(model_key: str) -> Dict[str, Any]:
    """
    Ensure model specification YAML (JSON-compatible) and parameter source ledger exist.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    spec_path, _, _ = resolve_output_path(
        subdir="models",
        name=f"{model_key}_spec.yaml",
        run_root=exp_dir,
        allow_quarantine=False,
        unique=False,
    )
    param_path, _, _ = resolve_output_path(
        subdir="parameters",
        name=f"{model_key}_param_sources.csv",
        run_root=exp_dir,
        allow_quarantine=False,
        unique=False,
    )

    meta = _model_metadata_from_key(model_key)
    created_spec = False
    created_params = False

    if not spec_path.exists():
        spec_content = {
            "model_key": model_key,
            "description": meta.get("description", ""),
            "state_variables": meta.get("states", []),
            "equations": meta.get("equations", []),
            "parameters": meta.get("params", {}),
            "initial_conditions": meta.get("initial_conditions", {}),
            "notes": "Auto-generated template; replace with full equations, units, and citation links for each term.",
        }
        spec_path.write_text(json.dumps(spec_content, indent=2))
        created_spec = True

    existing_param_names: set[str] = set()
    if param_path.exists():
        try:
            with param_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("param_name") or "").strip()
                    if name:
                        existing_param_names.add(name)
        except Exception:
            existing_param_names = set()

    if not param_path.exists():
        fieldnames = [
            "param_name",
            "value",
            "units",
            "source_type",
            "lit_claim_id",
            "reference_id",
            "notes",
        ]
        param_path.parent.mkdir(parents=True, exist_ok=True)
        with param_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            existing_param_names = set()
        created_params = True

    # Append any missing parameters not yet logged
    params = meta.get("params", {}) or {}
    units_map = meta.get("param_units", {}) or {}
    missing_names = [n for n in params.keys() if n not in existing_param_names]
    if missing_names:
        fieldnames = [
            "param_name",
            "value",
            "units",
            "source_type",
            "lit_claim_id",
            "reference_id",
            "notes",
        ]
        with param_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # If file was corrupted/missing header, ensure it's written
            if param_path.stat().st_size == 0:
                writer.writeheader()
            for name in missing_names:
                writer.writerow(
                    {
                        "param_name": name,
                        "value": params.get(name, ""),
                        "units": units_map.get(name, "dimensionless"),
                        "source_type": "free_hyperparameter",
                        "lit_claim_id": "",
                        "reference_id": "",
                        "notes": "Auto-generated; update with provenance and set source_type to lit_value/dimensionless_scaling/fit_to_data as appropriate.",
                    }
                )
        created_params = created_params or bool(missing_names)

    return {
        "spec_path": str(spec_path),
        "param_path": str(param_path),
        "created_spec": created_spec,
        "created_params": created_params,
    }


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
    """
    Backward-compatible manifest loader. Prefers sharded manifest via manifest_utils, and
    falls back to the legacy file_manifest.json if needed.
    """
    try:
        entries = manifest_utils.load_entries(base_folder=BaseTool.resolve_output_dir(None))
        manifest_map = {e["path"]: e for e in entries if isinstance(e, dict) and e.get("path")}
        if manifest_map:
            return manifest_map
    except Exception:
        pass

    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, encoding="utf-8") as f:
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
def _scan_and_auto_update_manifest(exp_dir: Path, skip: bool = False) -> List[str]:
    """
    Background Watcher: Scans experiment_results for orphaned files (files not in manifest).
    Adds them with inferred types and 'auto_watcher' annotation.
    Returns list of added filenames.
    """
    if (
        skip
        or os.environ.get("AISC_SKIP_WATCHER", "").strip().lower()
        in {"1", "true", "yes"}
    ):
        # Explicit opt-out to avoid long manifest scans during troubleshooting.
        return []
    added_files = []

    manifest_root = exp_dir / "manifest"

    # Preload manifest entries so we only write new files and avoid slow rewrites.
    known_paths: set[str] = set()
    try:
        existing_entries = manifest_utils.load_entries(base_folder=exp_dir)
        for entry in existing_entries:
            raw_path = entry.get("path")
            if not raw_path:
                continue
            p = Path(raw_path)
            known_paths.add(str(p))
            try:
                known_paths.add(str(p.resolve()))
            except Exception:
                pass
            if not p.is_absolute():
                try:
                    known_paths.add(str((exp_dir.parent / p).resolve()))
                except Exception:
                    pass
    except Exception:
        known_paths = set()

    for root, _, files in os.walk(exp_dir):
        if manifest_root in Path(root).parents or Path(root) == manifest_root:
            continue
        for name in files:
            if name == "file_manifest.json":
                continue
            if Path(root) == manifest_root:
                continue

            full_path = Path(root) / name
            path_str = str(full_path)
            resolved_str = None
            try:
                resolved_str = str(full_path.resolve())
            except Exception:
                resolved_str = None

            # Skip anything already indexed (by stored or resolved path).
            if path_str in known_paths or (resolved_str and resolved_str in known_paths):
                continue

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
            
            # Create Entry (manifest v2 lean schema)
            try:
                size_bytes = full_path.stat().st_size
            except Exception:
                size_bytes = None
            entry = {
                "name": name,
                "path": path_str,
                "kind": etype,
                "created_by": "auto_watcher",
                "status": "ok",
                "size_bytes": size_bytes,
                "created_at": datetime.now().isoformat(),
            }
            res = manifest_utils.append_or_update(entry, base_folder=exp_dir)
            if not res.get("error"):
                added_files.append(name)
                known_paths.add(path_str)
                if resolved_str:
                    known_paths.add(resolved_str)

    return added_files


def _append_manifest_entry(name: str, metadata_json: Optional[str] = None, allow_missing: bool = False):
    exp_dir = BaseTool.resolve_output_dir(None)

    try:
        target_path = BaseTool.resolve_input_path(name, allow_dir=True)
    except FileNotFoundError:
        if not allow_missing:
            return {"error": f"Referenced file not found: {name}. Use reserve_typed_artifact/reserve_output + append_manifest after creation, or set allow_missing=True if intentional."}
        target_path = BaseTool.resolve_output_dir(None) / name

    meta: Dict[str, Any] = {}
    if metadata_json:
        if len(metadata_json) > 400:
            return {"error": "metadata_json too long; keep kind/created_by/status short", "raw_len": len(metadata_json)}
        try:
            parsed_meta = json.loads(metadata_json)
            if isinstance(parsed_meta, dict):
                meta = parsed_meta
            else:
                return {"error": "metadata_json must be a JSON object", "raw": metadata_json}
        except Exception as exc:
            return {"error": f"Invalid metadata_json: {exc}", "raw": metadata_json}

    path_str = str(target_path)
    name_only = os.path.basename(name or path_str)
    try:
        size_bytes = target_path.stat().st_size
    except Exception:
        size_bytes = None
    entry: Dict[str, Any] = {
        "name": name_only,
        "path": path_str,
        "kind": meta.get("kind") or meta.get("type"),
        "created_by": meta.get("created_by") or meta.get("source") or meta.get("actor"),
        "status": meta.get("status") or "ok",
        "size_bytes": size_bytes,
        "created_at": meta.get("created_at") or datetime.now().isoformat(),
    }
    res = manifest_utils.append_or_update(entry, base_folder=exp_dir)
    if res.get("error"):
        return res
    return {
        "manifest_index": res.get("manifest_index"),
        "shard": res.get("shard"),
        "n_entries": res.get("count"),
        "deduped": res.get("deduped", False),
    }


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
    # Delegate to shared indexer so transport run discovery is consistent across tools.
    run_root = root.parent.parent if root.name == "transport_runs" else root
    idx = index_transport_runs(base_dir=run_root)
    entries_dict = cast(Dict[str, Dict[str, Any]], idx.get("entries", {}) if isinstance(idx, dict) else {}) or {}
    # Attach actor/updated_at for backward compatibility with older manifest consumers.
    entries: List[Dict[str, Any]] = []
    for entry in entries_dict.values():
        if isinstance(entry, dict):
            entry.setdefault("actor", "system-scan")
            entry.setdefault("updated_at", datetime.now().isoformat())
            entries.append(entry)
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
def inspect_manifest(
    base_folder: Optional[str] = None,
    role: Optional[str] = None,
    path_glob: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 200,
    summary_only: bool = True,
    include_samples: int = 3,
):
    """
    Sharded manifest reader (defaults to summary-only). Filters by optional role, glob, and ISO8601 since timestamp.
    - limit: max entries returned when summary_only=False (capped at 2000).
    - include_samples: how many sample entries to return in summary mode.
    - base_folder: optional override run root; defaults to the active run.
    """
    exp_dir = Path(base_folder) if base_folder else BaseTool.resolve_output_dir(None)
    return manifest_utils.inspect_manifest(
        base_folder=exp_dir,
        role=role,
        path_glob=path_glob,
        since=since,
        limit=limit,
        summary_only=summary_only,
        include_samples=include_samples,
    )


@function_tool
def inspect_recent_manifest_entries(limit: int = 20, since_minutes: int = 1440, role: Optional[str] = None):
    """
    Forensic Tool: Check the manifest for updated entries.
    Crucial for recovering work after a sub-agent timeout or crash.
    - limit: Max number of files to return (default 20, capped to 2000).
    - since_minutes: Only return files updated in the last N minutes (default 1440 = 24 hours).
    - role: Optional filter (e.g. 'Modeler', 'Analyst'); matches metadata.source/actor.
    Returns the files in reverse chronological order (newest first), using the sharded manifest.
    """
    cutoff = datetime.now() - timedelta(minutes=since_minutes)
    data = manifest_utils.inspect_manifest(
        base_folder=BaseTool.resolve_output_dir(None),
        role=role,
        since=cutoff.isoformat(),
        limit=limit,
        summary_only=False,
    )
    entries = data.get("entries", [])
    if not entries:
        msg = f"No manifest updates found in the last {since_minutes} minutes"
        if role:
            msg += f" for role '{role}'"
        return msg + "."
    output = [f"Last {len(entries)} manifest updates (since {since_minutes} min ago):"]
    for entry in entries:
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


@function_tool
def append_run_note_tool(category: str, text: str, actor: str = "system"):
    """
    Append a short run note/reflection to experiment_results/run_notes.md (keeps manifest lean).
    """
    return append_run_note(category=category, text=text, actor=actor)
        
    return "Invalid action. Use 'add' or 'read'."

@function_tool
def check_project_state(base_folder: str) -> str:
    """
    Reads the project state to see what artifacts exist.
    UPDATED: Automatically scans for orphaned files and updates the manifest.
    Set env AISC_SKIP_WATCHER=1 or pass skip_watcher=True to skip the manifest scan.
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
        # Default to skipping the watcher for speed; can be overridden via env or caller args.
        orphans = _scan_and_auto_update_manifest(
            Path(exp_results),
            skip=os.environ.get("AISC_SKIP_WATCHER", "").strip().lower()
            not in {"0", "false", "no"},
        )
    
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
    idx = index_transport_runs(base_dir=root.parent.parent if root.name == "transport_runs" else root)
    entries_dict = cast(Dict[str, Dict[str, Any]], idx.get("entries", {}) if isinstance(idx, dict) else {})
    entries = list(entries_dict.values())
    manifest = {
        "schema_version": 1,
        "runs": entries,
        "updated_at": datetime.now().isoformat(),
        "index_path": idx.get("index_path") if isinstance(idx, dict) else None,
    }
    if write:
        err = _atomic_write_json(_transport_manifest_path(), manifest)
        if err:
            return {"error": err, "manifest_path": str(_transport_manifest_path())}
    return {"manifest_path": str(_transport_manifest_path()), "runs": entries}


@function_tool
def resolve_transport_run(baseline: str, transport: str, seed: str):
    """
    Resolve paths for a transport run using the shared index (refreshes if stale).
    """
    res = resolve_transport_sim(baseline=baseline, transport=float(transport), seed=int(seed))
    return res


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
    trace = _load_hypothesis_trace()
    hyp = _ensure_hypothesis_entry(trace, hypothesis_id)
    if status:
        hyp["status"] = status
    exp = _ensure_experiment_entry(hyp, experiment_id)
    if sim_runs:
        for sim in sim_runs:
            try:
                if sim and sim not in exp.setdefault("sim_runs", []):
                    exp["sim_runs"].append(sim)
            except Exception:
                continue
    if figures:
        for fig in figures:
            if fig and fig not in exp.setdefault("figures", []):
                exp["figures"].append(fig)
    if metrics:
        metric_set = set(exp.get("metrics", []))
        metric_set.update(metrics)
        exp["metrics"] = sorted(metric_set)
    path = _write_hypothesis_trace(trace)
    _append_manifest_entry(
        name=path,
        metadata_json=json.dumps(
            {"kind": "hypothesis_trace_json", "created_by": os.environ.get("AISC_ACTIVE_ROLE", "pi"), "status": "ok"}
        ),
        allow_missing=False,
    )
    return {"path": path, "hypotheses": trace.get("hypotheses", [])}
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
):
    """Searches for literature and creates a lit_summary."""
    if not queries and not seed_paths:
        return "Error: You provided no 'queries' or 'seed_paths'. Please provide specific keywords or paper IDs."
        
    result = LitDataAssemblyTool().use_tool(
        queries=queries,
        seed_paths=seed_paths,
        max_results=max_results,
        use_semantic_scholar=use_semantic_scholar,
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
    return result

@function_tool
def validate_lit_summary(path: str):
    """Validates the structure of the literature summary."""
    return LitSummaryValidatorTool().use_tool(path=path)


def _resolve_lit_summary_path(path: Optional[str]) -> Path:
    """
    Resolve the lit summary path, preferring JSON then CSV under the active run.
    """
    if path:
        return BaseTool.resolve_input_path(path)
    exp_dir = BaseTool.resolve_output_dir(None)
    for candidate in (exp_dir / "lit_summary.json", exp_dir / "lit_summary.csv"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("lit_summary.json/csv not found under experiment_results.")


def _resolve_verification_path(path: Optional[str]) -> Path:
    """
    Resolve the reference verification output path.
    """
    if path:
        return BaseTool.resolve_input_path(path)
    exp_dir = BaseTool.resolve_output_dir(None)
    for candidate in (exp_dir / "lit_reference_verification.json", exp_dir / "lit_reference_verification.csv"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("lit_reference_verification.json/csv not found under experiment_results.")


def _load_verification_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError("lit_reference_verification.json must be a list or object.")
    if path.suffix.lower() == ".csv":
        with path.open() as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    raise ValueError("Unsupported verification file type; use .json or .csv.")


def _verification_row_confirmed(row: Dict[str, Any], score_threshold: float) -> bool:
    status = str(row.get("status") or "").strip().lower()
    if status in {"confirmed", "found", "verified", "ok"}:
        return True
    if status in {"not found", "missing", "unclear"}:
        return False

    found = row.get("found")
    if isinstance(found, str):
        if found.strip().lower() in {"true", "1", "yes", "y"}:
            return True
        if found.strip().lower() in {"false", "0", "no", "n"}:
            return False
    if isinstance(found, bool):
        return found

    score_raw = row.get("match_score")
    if score_raw is None:
        return False
    try:
        score = float(score_raw)
        return score >= score_threshold
    except Exception:
        return False


def _record_lit_gate_in_provenance(status_line: str):
    """
    Write/update a Lit gate line under the Literature section of provenance_summary.md.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    out_path = exp_dir / "provenance_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gate_line = f"- Lit gate: {status_line}"

    if not out_path.exists():
        content = "# Provenance Summary\n\n## Literature Sources\n"
        content += f"{gate_line}\n"
        out_path.write_text(content)
        return str(out_path)

    lines = out_path.read_text().splitlines()
    insert_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("## literature"):
            insert_idx = idx + 1
            while insert_idx < len(lines) and lines[insert_idx].strip() == "":
                insert_idx += 1
            break

    if insert_idx is None:
        lines.extend(["", "## Literature Sources", gate_line])
    else:
        replaced = False
        for j in range(insert_idx, len(lines)):
            if lines[j].startswith("## "):
                break
            if "Lit gate:" in lines[j]:
                lines[j] = gate_line
                replaced = True
                break
        if not replaced:
            lines.insert(insert_idx, gate_line)
    out_path.write_text("\n".join(lines) + "\n")
    return str(out_path)


def _record_model_provenance_in_provenance(model_key: str, status_line: str):
    """
    Write/update a Model provenance line under the Model Definitions section of provenance_summary.md.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    out_path = exp_dir / "provenance_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gate_line = f"- {model_key}: {status_line}"

    if not out_path.exists():
        content = "# Provenance Summary\n\n## Literature Sources\n- Missing or not generated.\n\n## Model Definitions\n"
        content += f"{gate_line}\n"
        content += "\n## Simulation Protocols\n- Missing or not generated.\n\n## Statistical Analyses\n- Missing or not generated.\n"
        out_path.write_text(content)
        return str(out_path)

    lines = out_path.read_text().splitlines()
    insert_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("## model definitions"):
            insert_idx = idx + 1
            while insert_idx < len(lines) and lines[insert_idx].strip() == "":
                insert_idx += 1
            break

    if insert_idx is None:
        lines.extend(["", "## Model Definitions", gate_line])
    else:
        replaced = False
        for j in range(insert_idx, len(lines)):
            if lines[j].startswith("## "):
                break
            if f"{model_key}:" in lines[j]:
                lines[j] = gate_line
                replaced = True
                break
        if not replaced:
            lines.insert(insert_idx, gate_line)
    out_path.write_text("\n".join(lines) + "\n")
    return str(out_path)


def _log_lit_gate_decision(status: str, confirmed_pct: float, n_unverified: int, thresholds: Dict[str, Any], reasons: List[str]):
    """
    Persist the gate outcome to project_knowledge and provenance summary.
    """
    summary = (
        f"Status={status.upper()}, confirmed={confirmed_pct:.1f}%, "
        f"unverified={n_unverified}, thresholds: confirmed>={thresholds['confirmed_threshold']*100:.1f}%, "
        f"max_unverified<={thresholds['max_unverified']}"
    )
    if reasons:
        summary += f"; reasons: {', '.join(reasons)}"
    try:
        cast(Any, manage_project_knowledge)(
            action="add",
            category="decision",
            observation="Literature gate evaluation before modeling/simulation.",
            solution=summary,
        )
    except Exception:
        pass
    try:
        _record_lit_gate_in_provenance(f"{status.upper()} (confirmed={confirmed_pct:.1f}%, max_unverified={thresholds['max_unverified']})")
    except Exception:
        pass


def _coerce_float(value: Optional[float | str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _coerce_int(value: Optional[int | str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _load_spec_content(spec_path: Path) -> Dict[str, Any]:
    try:
        with spec_path.open() as f:
            return json.load(f)
    except Exception:
        text = spec_path.read_text()
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except Exception:
            raise ValueError(f"Failed to parse spec file: {spec_path}")


def _evaluate_model_provenance(model_key: str, allow_free: bool = False) -> Dict[str, Any]:
    exp_dir = BaseTool.resolve_output_dir(None)
    spec_path, _, _ = resolve_output_path(
        subdir="models",
        name=f"{model_key}_spec.yaml",
        run_root=exp_dir,
        allow_quarantine=False,
        unique=False,
    )
    param_path, _, _ = resolve_output_path(
        subdir="parameters",
        name=f"{model_key}_param_sources.csv",
        run_root=exp_dir,
        allow_quarantine=False,
        unique=False,
    )
    if not spec_path.exists() or not param_path.exists():
        raise FileNotFoundError(f"Model spec/params missing for {model_key}; expected {spec_path} and {param_path}")

    spec = _load_spec_content(spec_path)
    params_declared = set((spec.get("parameters") or {}).keys())

    rows: Dict[str, Dict[str, Any]] = {}
    with param_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("param_name") or "").strip()
            if name:
                rows[name] = row

    missing = sorted(list(params_declared - set(rows.keys())))
    unsourced = sorted(
        [
            name
            for name, row in rows.items()
            if (row.get("source_type") or "").strip().lower() == "free_hyperparameter"
        ]
    )

    counts: Dict[str, int] = {}
    for row in rows.values():
        stype = (row.get("source_type") or "unknown").strip().lower() or "unknown"
        counts[stype] = counts.get(stype, 0) + 1

    status = "ready"
    if missing or (unsourced and not allow_free):
        status = "not_ready"

    summary = {
        "status": status,
        "missing_params": missing,
        "unsourced_params": unsourced,
        "counts_by_source_type": counts,
        "spec_path": str(spec_path),
        "param_path": str(param_path),
    }
    return summary


@function_tool
def check_model_provenance(model_key: str, allow_free_hyperparameters: bool = False):
    """
    Validate that a model's spec and parameter ledger are complete and sourced.
    """
    result = _evaluate_model_provenance(model_key=model_key, allow_free=allow_free_hyperparameters)
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
        _record_model_provenance_in_provenance(model_key, summary_line)
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


def _evaluate_lit_ready(
    lit_path: Optional[str],
    verification_path: Optional[str],
    confirmed_threshold: Optional[float],
    max_unverified: Optional[int],
) -> Dict[str, Any]:
    thresholds = {
        "confirmed_threshold": _coerce_float(
            confirmed_threshold if confirmed_threshold is not None else os.environ.get("AISC_LIT_GATE_CONFIRMED_THRESHOLD", 0.7),
            0.7,
        ),
        "max_unverified": _coerce_int(
            max_unverified if max_unverified is not None else os.environ.get("AISC_LIT_GATE_MAX_UNVERIFIED", 3),
            3,
        ),
    }
    reasons: List[str] = []
    result: Dict[str, Any] = {"status": "not_ready", "thresholds": thresholds, "reasons": reasons}

    try:
        lit_summary_path = _resolve_lit_summary_path(lit_path)
    except Exception as exc:
        reasons.append(f"Missing lit summary: {exc}")
        result["error"] = str(exc)
        return result

    validator = LitSummaryValidatorTool().use_tool(path=str(lit_summary_path))
    missing_fields = validator.get("missing_fields") or []
    n_records = validator.get("n_records", 0)
    result.update({"lit_summary_path": str(lit_summary_path), "validator": validator})
    if n_records == 0:
        reasons.append("Lit summary contains no records.")
    if missing_fields:
        reasons.append(f"Missing required fields: {', '.join(missing_fields)}")

    try:
        verification_resolved = _resolve_verification_path(verification_path)
    except Exception as exc:
        reasons.append(f"Missing verification outputs: {exc}")
        result["error"] = str(exc)
        return result

    rows = _load_verification_rows(verification_resolved)
    n_total = len(rows)
    n_confirmed = sum(1 for row in rows if _verification_row_confirmed(row, thresholds["confirmed_threshold"]))
    n_unverified = n_total - n_confirmed
    confirmed_pct = (n_confirmed / n_total * 100.0) if n_total else 0.0

    result.update(
        {
            "verification_path": str(verification_resolved),
            "n_references": n_total,
            "n_confirmed": n_confirmed,
            "n_unverified": n_unverified,
            "confirmed_pct": confirmed_pct,
        }
    )

    if n_total == 0:
        reasons.append("Reference verification table is empty.")
    if confirmed_pct < thresholds["confirmed_threshold"] * 100.0:
        reasons.append(
            f"Confirmed percentage {confirmed_pct:.1f}% below threshold {thresholds['confirmed_threshold']*100:.1f}%."
        )
    if n_unverified > thresholds["max_unverified"]:
        reasons.append(f"Unverified references {n_unverified} exceed limit {thresholds['max_unverified']}.")

    result["status"] = "ready" if not reasons else "not_ready"
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
    result = _evaluate_lit_ready(
        lit_path=lit_path,
        verification_path=verification_path,
        confirmed_threshold=confirmed_threshold,
        max_unverified=max_unverified,
    )
    _log_lit_gate_decision(
        status=result.get("status", "not_ready"),
        confirmed_pct=result.get("confirmed_pct", 0.0),
        n_unverified=result.get("n_unverified", 0),
        thresholds=result.get("thresholds", {"confirmed_threshold": confirmed_threshold, "max_unverified": max_unverified}),
        reasons=result.get("reasons", []),
    )
    return result


def _should_skip_lit_gate(skip_flag: bool = False) -> bool:
    env_skip = os.environ.get("AISC_SKIP_LIT_GATE", "").strip().lower() in {"1", "true", "yes", "on"}
    return skip_flag or env_skip


def _ensure_lit_gate_ready(skip_gate: bool = False):
    if _should_skip_lit_gate(skip_gate):
        return
    gate = _evaluate_lit_ready(
        lit_path=None,
        verification_path=None,
        confirmed_threshold=None,
        max_unverified=None,
    )
    if gate.get("status") == "ready":
        return
    reasons = gate.get("reasons", [])
    raise RuntimeError(f"Literature gate not satisfied: {', '.join(reasons) or 'see check_lit_ready for details.'}")


@function_tool
def verify_references(
    lit_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_results: int = 5,
    score_threshold: float = 0.65,
):
    """
    Verify lit_summary entries via Semantic Scholar; writes lit_reference_verification.csv/json.
    """
    res = ReferenceVerificationTool().use_tool(
        lit_path=lit_path,
        output_dir=output_dir,
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
    _ensure_lit_gate_ready(skip_gate=skip_lit_gate)
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
        output_dir=output_dir,
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


def _collect_provenance_sections() -> Dict[str, Any]:
    """
    Gather manifest-derived provenance snippets for reviewer/publisher to summarize.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    sections: Dict[str, Any] = {
        "literature": [],
        "models": [],
        "simulations": [],
        "stats": [],
    }
    try:
        manifest = manifest_utils.inspect_manifest(base_folder=exp_dir, summary_only=False, limit=5000).get("entries", [])
    except Exception:
        manifest = []
    for entry in manifest:
        name = entry.get("name", "")
        path = entry.get("path", "")
        kind = (entry.get("kind") or entry.get("type") or "").lower()
        if "lit_summary" in name or "lit_summary" in kind or "lit_reference_verification" in name:
            sections["literature"].append(path or name)
        if "model_spec" in kind or name.endswith("_spec.yaml"):
            sections["models"].append(path or name)
        if "model_metrics" in kind or name.endswith("_metrics.json"):
            sections["models"].append(path or name)
        if "parameters" in path and "param_sources" in name:
            sections["models"].append(path or name)
        if "sweep" in path or "transport_runs" in path or "sim.json" in name or "intervention" in name:
            sections["simulations"].append(path or name)
        if "metrics" in name or "stats" in kind:
            sections["stats"].append(path or name)
    return sections


def _render_provenance_markdown(sections: Dict[str, Any]) -> str:
    def fmt_section(title: str, items: List[str]) -> str:
        if not items:
            return f"## {title}\n- Missing or not generated.\n"
        lines = "\n".join(f"- {i}" for i in sorted(set(items)))
        return f"## {title}\n{lines}\n"

    parts = [
        "# Provenance Summary\n",
        fmt_section("Literature Sources", sections.get("literature", [])),
        fmt_section("Model Definitions", sections.get("models", [])),
        fmt_section("Simulation Protocols", sections.get("simulations", [])),
        fmt_section("Statistical Analyses", sections.get("stats", [])),
    ]
    return "\n".join(parts)


@function_tool
def generate_provenance_summary():
    """
    Aggregate provenance from manifest and write experiment_results/provenance_summary.md.
    """
    sections = _collect_provenance_sections()
    content = _render_provenance_markdown(sections)
    exp_dir = BaseTool.resolve_output_dir(None)
    out_path = exp_dir / "provenance_summary.md"
    out_path.write_text(content)
    _append_manifest_entry(
        name=str(out_path),
        metadata_json=json.dumps({"kind": "provenance_summary_md", "created_by": os.environ.get("AISC_ACTIVE_ROLE", "reviewer"), "status": "ok"}),
        allow_missing=False,
    )
    return {"path": str(out_path), "sections": sections}

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
    hypothesis_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    compute_metrics: bool = False,
    enforce_param_provenance: Optional[bool] = None,
    skip_lit_gate: bool = False,
):
    """Run a built-in biological ODE/replicator model and save JSON results."""
    _ensure_lit_gate_ready(skip_gate=skip_lit_gate)
    ledger = _ensure_model_spec_and_params(model_key)
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
    _ensure_lit_gate_ready(skip_gate=skip_lit_gate)
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
    _ensure_lit_gate_ready(skip_gate=skip_lit_gate)
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
def list_artifacts_by_kind(kind: str, limit: int = 100):
    """
    List artifacts from manifest v2 filtered by kind.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    entries = manifest_utils.load_entries(base_folder=exp_dir, limit=None)
    filtered = [e for e in entries if e.get("kind") == kind]
    return {"kind": kind, "paths": [e.get("path") for e in filtered[:limit]], "total": len(filtered)}


def _reserve_typed_artifact_impl(kind: str, meta_json: Optional[str], unique: bool) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if meta_json:
        try:
            parsed = json.loads(meta_json)
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid meta_json: {exc}", "kind": kind}
        if not isinstance(parsed, dict):
            return {"error": "meta_json must decode to an object/dict.", "kind": kind}
        meta = parsed
    try:
        rel_dir, name = _format_artifact_path(kind, meta)
    except Exception as exc:
        return {"error": str(exc), "kind": kind}
    try:
        target, quarantined, note = resolve_output_path(
            subdir=rel_dir, name=name, allow_quarantine=True, unique=unique
        )
    except Exception as exc:
        return {"error": f"Failed to reserve path: {exc}", "kind": kind}
    result: Dict[str, Any] = {
        "reserved_path": str(target),
        "kind": kind,
        "name": name,
        "rel_dir": rel_dir,
        "quarantined": quarantined,
    }
    if note:
        result["note"] = note
    return result


@function_tool
def reserve_typed_artifact(kind: str, meta_json: Optional[str] = None, unique: bool = True):
    """
    Reserve a canonical artifact path using the artifact type registry (VI-01).
    Provide meta_json to fill any {placeholders} in rel_dir/pattern. Errors on unknown kinds.
    """
    return _reserve_typed_artifact_impl(kind=kind, meta_json=meta_json, unique=unique)


@function_tool
def reserve_and_register_artifact(kind: str, meta_json: Optional[str] = None, status: str = "pending", unique: bool = True):
    """
    Reserve a canonical artifact path using the registry and immediately register it in manifest v2 with provided status.
    """
    meta: Dict[str, Any] = {}
    if meta_json:
        try:
            parsed = json.loads(meta_json)
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid meta_json: {exc}", "kind": kind}
        if not isinstance(parsed, dict):
            return {"error": "meta_json must decode to an object/dict.", "kind": kind}
        meta = parsed

    reserve = _reserve_typed_artifact_impl(kind=kind, meta_json=meta_json, unique=unique)
    if reserve.get("error"):
        return reserve
    path = reserve.get("reserved_path")
    if not path:
        return {"error": "failed_to_reserve", "kind": kind}
    entry = {
        "path": path,
        "name": reserve.get("name") or os.path.basename(path),
        "kind": kind,
        "created_by": meta.get("created_by") or meta.get("actor") or os.environ.get("AISC_ACTIVE_ROLE") or "unknown",
        "status": status,
    }
    manifest_res = manifest_utils.append_or_update(entry, base_folder=BaseTool.resolve_output_dir(None))
    if manifest_res.get("error"):
        reserve["manifest_error"] = manifest_res["error"]
    else:
        manifest_idx = manifest_res.get("manifest_index")
        if manifest_idx:
            reserve["manifest_index"] = manifest_idx
    return reserve


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


@function_tool
def append_manifest(name: str, metadata_json: Optional[str] = None, allow_missing: bool = False):
    """
    Append an entry to the run's sharded manifest (experiment_results/manifest/...).
    Pass metadata as a JSON string (e.g., '{"type":"figure","source":"analyst"}').
    Creates the manifest file if missing.
    """
    return _append_manifest_entry(name=name, metadata_json=metadata_json, allow_missing=allow_missing)


@function_tool
def read_manifest_entry(path_or_name: str):
    """
    Read a single manifest entry by path key or filename (basename).
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    entry = manifest_utils.find_manifest_entry(path_or_name, base_folder=exp_dir)
    if entry:
        return {"entry": entry}
    return {"error": "Not found", "path_or_name": path_or_name}


@function_tool
def check_manifest():
    """
    Validate manifest entries: report missing files, entries lacking type, and duplicate basenames.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    entries = manifest_utils.load_entries(base_folder=exp_dir)
    if not entries:
        return {"error": "Manifest empty or missing", "path": str(exp_dir / 'manifest')}

    missing = []
    missing_type = []
    by_name: Dict[str, list[str]] = {}
    duplicates_by_path: Dict[str, int] = {}
    for entry in entries:
        path = entry.get("path", "")
        try:
            exists = Path(path).exists()
        except Exception:
            exists = False
        if not exists:
            missing.append(path)
        if not entry.get("kind"):
            missing_type.append(path)
        name = entry.get("name") or os.path.basename(path or "")
        by_name.setdefault(name, []).append(path)
        duplicates_by_path[path] = duplicates_by_path.get(path, 0) + 1

    duplicates = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    duplicate_paths = [p for p, count in duplicates_by_path.items() if count > 1]
    health_entries: List[Dict[str, Any]] = []
    if missing:
        health_entries.append({"missing_files": missing})
    if missing_type:
        health_entries.append({"missing_type": missing_type})
    if duplicates:
        health_entries.append({"duplicate_names": duplicates})
    if duplicate_paths:
        health_entries.append({"duplicate_paths": duplicate_paths})
    if health_entries:
        log_missing_or_corrupt(health_entries)
    return {
        "manifest_index": str(BaseTool.resolve_output_dir(None) / "manifest" / "manifest_index.json"),
        "n_entries": len(entries),
        "missing_files": missing,
        "missing_type": missing_type,
        "duplicate_names": duplicates,
        "duplicate_paths": duplicate_paths,
    }


@function_tool
def read_manifest():
    """Read the run's manifest with a capped entry list; use inspect_manifest for filtered views."""
    exp_dir = BaseTool.resolve_output_dir(None)
    data = manifest_utils.inspect_manifest(base_folder=exp_dir, summary_only=False, limit=500)
    return {
        "manifest_index": data.get("manifest_index"),
        "entries": data.get("entries", []),
        "summary": data.get("summary", {}),
        "note": "Entries capped at 500; use inspect_manifest for filtered views.",
    }


@function_tool
def check_manifest_unique_paths():
    """
    Validate that manifest paths are unique (COUNT(DISTINCT path) == COUNT(*)).
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    result = manifest_utils.unique_path_check(base_folder=exp_dir)
    ok = result["total"] == result["distinct_paths"]
    result["ok"] = ok
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
def get_artifact_index(max_entries: int = 2000):
    """
    Build a lightweight index of artifacts under experiment_results and include manifest entries if present.
    """
    root = BaseTool.resolve_output_dir(None)
    manifest = manifest_utils.inspect_manifest(
        base_folder=root,
        summary_only=False,
        limit=min(500, max_entries),
    )
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
    artifact_catalog = _artifact_kind_catalog()
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
        "FILE IO POLICY: Every persistent artifact must be reserved via 'reserve_typed_artifact(kind=..., meta_json=...)' using the registry below; do NOT invent filenames or bypass the registry. "
        f"Known kinds: {artifact_catalog}. "
        "Preferred flow: 'reserve_and_register_artifact' -> write -> (optional) update status via append_manifest. "
        "Use 'reserve_output' only for PI/Coder scratch logs; other roles must stay within typed helpers. When writing text, pass the reserved path into write_text_artifact instead of freehand names. "
        "Outputs are anchored to experiment_results; if a directory is unavailable, writes are auto-rerouted to experiment_results/_unrouted with a manifest note. "
        "NEVER log reflections or notes to the manifest—use append_run_note or manage_project_knowledge instead."
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
            "3. Use 'reserve_typed_artifact(kind=\"lit_summary_main\")' for lit summaries and 'reserve_typed_artifact(kind=\"claim_graph_main\")' for claim graphs; do not invent filenames.\n"
            "4. Immediately run 'verify_references' on lit_summary to produce lit_reference_verification.csv/json. Treat this as REQUIRED provenance.\n"
            "5. Reject readiness if more than 20% of references are missing (found==False) or any match_score < 0.5; report FAILURE with counts.\n"
            "6. If verification repeatedly fails for a venue/source, log a reflection via manage_project_knowledge with the specific venue.\n"
            "7. If you create or deeply analyze artifacts not yet in the manifest, log them with 'append_manifest' (include kind + created_by + status).\n"
            "8. CRITICAL: If no papers are found, report FAILURE. Do not invent 'TBD' citations.\n"
            "9. Log reflections to run_notes via 'append_run_note_tool' or manage_project_knowledge; never to manifest.\n"
            f"10. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            list_artifacts,
            list_artifacts_by_kind,
            read_artifact,
            reserve_typed_artifact,
            reserve_and_register_artifact,
            append_manifest,
            read_manifest,
            read_manifest_entry,
            check_manifest,
            check_manifest_unique_paths,
            assemble_lit_data,
            validate_lit_summary,
            verify_references,
            search_semantic_scholar,
            update_claim_graph,
            manage_project_knowledge,
            append_run_note_tool,
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
            "3b. Before first sim of a given model_key, generate model_spec_yaml and parameter_source_table (one row per parameter with source_type and lit/claim links). Update the ledger if you change parameters; runs missing rows are a hard failure.\n"
            "3c. Update hypothesis_trace.json after each sim/ensemble: record hypothesis_id/experiment_id, sim run identifiers, and metrics produced.\n"
            "3d. After sweeps or transport batches, call 'compute_model_metrics' to emit *_metrics.csv and/or {model_key}_metrics.json; rerun if metrics are missing when figures/text depend on them.\n"
            "4. Ensure parameter sweeps cover the range specified in the hypothesis.\n"
            "5. Save raw outputs to experiment_results/.\n"
            "5b. Reserve every persistent artifact via 'reserve_typed_artifact' (transport_* kinds for sims, sensitivity_sweep_table/intervention_table for sweeps, verification_note for proof-of-work); do NOT invent filenames or call reserve_output for data assets.\n"
            "6. Always produce arrays for each (baseline, transport, seed): prefer export_arrays during sim; otherwise immediately run 'sim_postprocess' on the produced sim.json so failure_matrix.npy/time_vector.npy/nodes_order_*.txt exist before marking the run complete. Every run must also emit per_compartment.npz + node_index_map.json + topology_summary.json (binary_states, continuous_states/time); validate with validate_per_compartment_outputs before marking status=complete.\n"
            "7. Use the transport run manifest (read_transport_manifest / update_transport_manifest); consult it before reruns and update it after completing or failing a run. Do not mark status=complete unless arrays + sim.json + sim.status.json all exist; otherwise mark partial and note missing files.\n"
            "7b. Resolve baselines via resolve_baseline_path before running batches; only pass graph baselines (.npy/.npz/.graphml/.gpickle/.gml), never sim.json.\n"
            "7c. Process one baseline per call; if run_recipe.json exists under experiment_results/simulations/transport_runs, load it for template/output roots instead of embedding long paths. Append ensemble CSV incrementally and write per-baseline status to pi_notes/user_inbox.\n"
            "8. Before calling 'append_manifest', ask if appending new info to the artifact's record in the manifest adds new value (new file or materially new analysis/description). Log only when yes, with name + kind + created_by + status.\n"
            "9. If you encounter simulation failures or parameter issues, log them to Project Knowledge via 'manage_project_knowledge'.\n"
            f"10. {proof_of_work_instruction}\n"
            "11. Log reflections to run_notes via 'append_run_note_tool' or manage_project_knowledge; never to manifest.\n"
            f"12. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            get_artifact_index,
            list_artifacts,
            list_artifacts_by_kind,
            read_artifact,
            summarize_artifact,
            reserve_typed_artifact,
            reserve_and_register_artifact,
            append_manifest,
            read_manifest,
            read_manifest_entry,
            check_manifest,
            check_manifest_unique_paths,
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
            update_hypothesis_trace,
            compute_model_metrics,
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
            "3e. Update hypothesis_trace.json with figure filenames under the correct hypothesis/experiment after plotting.\n"
            "3f. Prefer metrics artifacts (sweep_metrics_csv/model_metrics_json) over raw CSVs when plotting; if missing, ask Modeler to run compute_model_metrics.\n"
            "3g. Reserve figure and verification outputs via 'reserve_typed_artifact' (plot_intermediate/manuscript_figure_png/manuscript_figure_svg/verification_note); do NOT invent filenames or call reserve_output for figures.\n"
            "4. Validate models vs lit via 'run_validation_compare' and use 'run_biological_stats' for significance/enrichment.\n"
            "5. Before calling 'append_manifest', ask if the artifact adds new value (new figure/analysis). Log only when yes, with name + kind + created_by + status.\n"
            "6. Check Project Knowledge for visualization standards (e.g., colormaps) before starting.\n"
            "7. When plots are ready, confirm provenance_summary.md exists or ask Reviewer to generate it.\n"
            f"7. {proof_of_work_instruction}\n"
            "8. Log reflections to run_notes via 'append_run_note_tool' or manage_project_knowledge; never to manifest.\n"
            f"9. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            get_artifact_index,
            list_artifacts,
            list_artifacts_by_kind,
            read_artifact,
            summarize_artifact,
            reserve_typed_artifact,
            reserve_and_register_artifact,
            append_manifest,
            read_manifest,
            read_manifest_entry,
            check_manifest,
            check_manifest_unique_paths,
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
            compute_model_metrics,
            update_hypothesis_trace,
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
            "2b. Read lit_reference_verification.csv/json; if any reference has found==False or match_score below the reported threshold, mark the draft as unsupported until fixed.\n"
            "2c. Verify that every simulation/model parameter appearing in figures/tables has a row in parameter_source_table with a declared source_type and (when lit_value) lit_claim_id/reference_id.\n"
            "2d. Check hypothesis_trace.json: any hypothesis marked 'supported' must list sim_runs and figures that exist on disk; flag gaps.\n"
            "2e. Ensure metrics artifacts exist for referenced sweeps/models (sweep_metrics_csv or model_metrics_json). If missing, flag and request compute_model_metrics.\n"
            "2f. Generate provenance_summary.md via 'generate_provenance_summary'; if major inputs (lit_summary, model_spec, sims) are missing, flag the section and request fixes.\n"
            "3. Check consistency: Does Figure 3 actually support the claim in paragraph 2?\n"
            "4. If gaps exist, report them clearly to the PI.\n"
            "5. Only report 'NO GAPS' if the PDF validates completely.\n"
            "6. If you create or materially analyze artifacts, log them with 'append_manifest' (name + kind + created_by + status) only when it adds value.\n"
            "7. VERIFY PROOF OF WORK: Reject any result artifact that lacks a corresponding `_verification.md` or `_log.md` file explaining how it was derived.\n"
            "8. Reserve any review artifacts/notes with 'reserve_typed_artifact' (e.g., verification_note) instead of inventing filenames.\n"
            "9. Log reflections to run_notes via 'append_run_note_tool' or manage_project_knowledge; never to manifest.\n"
            f"10. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            get_artifact_index,
            list_artifacts,
            list_artifacts_by_kind,
            read_artifact,
            summarize_artifact,
            reserve_typed_artifact,
            reserve_and_register_artifact,
            append_manifest,
            read_manifest,
            read_manifest_entry,
            check_manifest,
            check_manifest_unique_paths,
            read_manuscript,
            check_claim_graph,
            run_biological_stats,
            verify_references,
            compute_model_metrics,
            update_hypothesis_trace,
            generate_provenance_summary,
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
            "4. Reserve interpretation outputs via 'reserve_typed_artifact' (interpretation_json or interpretation_md) and use the reserved path with 'write_text_artifact'.\n"
            "5. Before calling 'append_manifest', ask if the artifact adds new value (new interpretation or substantial edit). Log only when yes, with name + kind + created_by + status.\n"
            "6. Log reflections to run_notes via 'append_run_note_tool' or manage_project_knowledge; never to manifest.\n"
            f"7. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            get_artifact_index,
            list_artifacts,
            list_artifacts_by_kind,
            read_artifact,
            summarize_artifact,
            reserve_typed_artifact,
            reserve_and_register_artifact,
            append_manifest,
            read_manifest,
            read_manifest_entry,
            check_manifest,
            check_manifest_unique_paths,
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
            "2. If you add tools/helpers, document them briefly and log via 'append_manifest' (name + kind + created_by + status).\n"
            "3. Prefer small, dependency-light snippets; avoid large libraries or network access.\n"
            "4. If you need existing artifacts, list them with 'list_artifacts' or read via 'read_artifact' (use summary_only for large files).\n"
            "5. Log code patterns or library constraints to Project Knowledge.\n"
            "6. Reserve any persisted helper outputs via 'reserve_typed_artifact' (e.g., verification_note) instead of inventing filenames.\n"
            "7. Log reflections to run_notes via 'append_run_note_tool' or manage_project_knowledge; never to manifest.\n"
            f"8. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            list_artifacts,
            list_artifacts_by_kind,
            read_artifact,
            reserve_output,
            reserve_typed_artifact,
            reserve_and_register_artifact,
            append_manifest,
            read_manifest,
            check_manifest_unique_paths,
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
            "3. Reserve outputs (figures README, manuscript PDF) via 'reserve_typed_artifact' before writing; do not invent filenames.\n"
            "4. Ensure compile success. Debug LaTeX errors autonomously.\n"
            "5. Log reflections to run_notes via 'append_run_note_tool' or manage_project_knowledge; never to manifest.\n"
            f"6. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            list_artifacts,
            list_artifacts_by_kind,
            read_artifact,
            summarize_artifact,
            reserve_output,
            reserve_typed_artifact,
            reserve_and_register_artifact,
            append_manifest,
            read_manifest,
            check_manifest_unique_paths,
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
            "3b. Maintain hypothesis_trace.json: when drafting the plan, ensure every idea experiment is mapped to a hypothesis/experiment id (H*, E*) in hypothesis_trace.json (skeleton allowed). Update as new experiments/figures/sim runs become planned.\n"
            "4. DELEGATE: Handoff to specialized agents based on missing artifacts. **MANDATORY: When calling a sub-agent, lookup the exact file paths first (via inspect_manifest or list_artifacts) and pass the EXACT PATH in the prompt. Do not ask them to 'find the file'.**\n"
            "   - Before any modeling/simulation, run 'check_lit_ready' (defaults: confirmed refs >=70%, <=3 unverified). If it returns not_ready, stop and fix lit/references or pass --skip_lit_gate explicitly.\n"
            "   - Before running built-in models, ensure 'check_model_provenance' passes (no missing params or free_hyperparameter rows). If enforcement is intentionally disabled, log the failure pattern first.\n"
            "   - Missing Lit Review -> Archivist\n"
            "   - Missing Data -> Modeler\n"
            "   - Missing Plots -> Analyst\n"
            "   - Theoretical Interpretation -> Interpreter\n"
            "   - Draft Exists -> Reviewer\n"
            "   - Validated & Ready -> Publisher\n"
            "5. ASYNC FEEDBACK: Call `check_user_inbox` frequently (e.g., between tasks) to see if the user has steered the project.\n"
            "6. HANDLE FAILURES: If a sub-agent reports error or max turns, call 'inspect_manifest(summary_only=False, role=..., limit=50)' to see what they accomplished before crashing. If artifacts exist, instruct the next run to continue from there rather than restarting.\n"
            "7. END OF RUN: Write a concise summary and next actions to 'pi_notes.md' using 'write_pi_notes' so it persists across resumes.\n"
            "8. TERMINATE: Stop only when Reviewer confirms 'NO GAPS' and PDF is generated.\n"
            "9. Keep reflections/notes in run_notes via append_run_note_tool or project_knowledge; never store notes in manifest."
        ),
        model=model,
        tools=[
            check_project_state,
            log_strategic_pivot,
            inspect_manifest,
            inspect_recent_manifest_entries,
            get_run_paths,
            resolve_path,
            list_artifacts,
            list_artifacts_by_kind,
            read_artifact,
            head_artifact,
            summarize_artifact,
            reserve_output,
            reserve_typed_artifact,
            reserve_and_register_artifact,
            append_manifest,
            read_manifest,
            check_status,
            check_manifest_unique_paths,
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
            update_hypothesis_trace,
            generate_provenance_summary,
            # New interactive tools
            wait_for_human_review,
            check_user_inbox,
            append_run_note_tool,
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
    os.environ["AISC_SKIP_LIT_GATE"] = str(args.skip_lit_gate).lower()
    os.environ["AISC_ENFORCE_PARAM_PROVENANCE"] = str(args.enforce_param_provenance).lower()
    
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
    global _ACTIVE_IDEA
    _ACTIVE_IDEA = idea

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
    # Initialize hypothesis trace skeleton for this run
    try:
        _bootstrap_hypothesis_trace(idea)
    except Exception as exc:
        print(f"⚠️ Failed to bootstrap hypothesis_trace.json: {exc}")

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
