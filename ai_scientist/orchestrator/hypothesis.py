# pyright: reportMissingImports=false
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.lit_validator import LitSummaryValidatorTool
from ai_scientist.utils import manifest as manifest_utils
from ai_scientist.utils.pathing import resolve_output_path
from ai_scientist.orchestrator.manifest_service import manage_project_knowledge

# Module-level state to replace global _ACTIVE_IDEA
_ACTIVE_IDEA: Optional[Dict[str, Any]] = None

# --------------------------------------------------------------------------------
# Public API stubs (to be filled incrementally)
# --------------------------------------------------------------------------------

def set_active_idea(idea: Dict[str, Any]) -> None:
    """Set the active idea for hypothesis trace bootstrapping."""
    global _ACTIVE_IDEA
    _ACTIVE_IDEA = idea

def bootstrap_hypothesis_trace(idea: Dict[str, Any]) -> Dict[str, Any]:
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


def load_hypothesis_trace() -> Dict[str, Any]:
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
    return bootstrap_hypothesis_trace(_ACTIVE_IDEA or {})


def write_hypothesis_trace(trace: Dict[str, Any]) -> str:
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
    trace = load_hypothesis_trace()
    hyp = _ensure_hypothesis_entry(trace, hypothesis_id)
    exp = _ensure_experiment_entry(hyp, experiment_id)
    sim_runs = exp.setdefault("sim_runs", [])
    if sim_entry and sim_entry not in sim_runs:
        sim_runs.append(sim_entry)
    if metrics:
        metric_set = set(exp.get("metrics", []))
        metric_set.update(metrics)
        exp["metrics"] = sorted(metric_set)
    write_hypothesis_trace(trace)
    return trace


def _update_hypothesis_trace_with_figures(
    hypothesis_id: str,
    experiment_id: str,
    figures: List[str],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    trace = load_hypothesis_trace()
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
    write_hypothesis_trace(trace)
    return trace


def update_hypothesis_trace_impl(
    hypothesis_id: str,
    experiment_id: str,
    sim_runs: Optional[List[Dict[str, Any]]] = None,
    figures: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Updates hypothesis trace; matches tool wrapper signature."""
    trace = load_hypothesis_trace()
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
    write_hypothesis_trace(trace)
    return {"path": str(trace), "hypotheses": trace.get("hypotheses", [])}

def derive_idea_from_manuscript(manuscript_path: str, title_override: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Derive an idea/seed context from an existing manuscript file."""
    from ai_scientist.orchestrator.manuscript_processor import _derive_idea_from_manuscript as derive_impl

    return derive_impl(manuscript_path, title_override)


def persist_manuscript_seed(manuscript_context: Dict[str, Any], idea: Dict[str, Any]) -> Dict[str, str]:
    """Persist manuscript-derived seeds in the experiment_results folder."""
    from ai_scientist.orchestrator.manuscript_processor import _persist_manuscript_seed as persist_impl

    return persist_impl(manuscript_context, idea)


def resolve_lit_summary_path(path: Optional[str]) -> Path:
    if path:
        return BaseTool.resolve_input_path(path)
    exp_dir = BaseTool.resolve_output_dir(None)
    for candidate in (exp_dir / "lit_summary.json", exp_dir / "lit_summary.csv"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("lit_summary.json/csv not found under experiment_results.")


def resolve_verification_path(path: Optional[str]) -> Path:
    if path:
        return BaseTool.resolve_input_path(path)
    exp_dir = BaseTool.resolve_output_dir(None)
    for candidate in (
        exp_dir / "lit_reference_verification.json",
        exp_dir / "lit_reference_verification.csv",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("lit_reference_verification.json/csv not found under experiment_results.")


def _load_verification_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return cast(List[Dict[str, Any]], data)
        if isinstance(data, dict):
            return [cast(Dict[str, Any], data)]
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
        value = found.strip().lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n"}:
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


def evaluate_lit_ready(
    lit_path: Optional[str] = None,
    verification_path: Optional[str] = None,
    confirmed_threshold: Optional[float] = None,
    max_unverified: Optional[int] = None,
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
        lit_summary_path = resolve_lit_summary_path(lit_path)
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
        verification_resolved = resolve_verification_path(verification_path)
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


def log_lit_gate_decision(
    status: str,
    confirmed_pct: float,
    n_unverified: int,
    thresholds: Dict[str, Any],
    reasons: List[str],
) -> None:
    summary = (
        f"Status={status.upper()}, confirmed={confirmed_pct:.1f}%, "
        f"unverified={n_unverified}, thresholds: confirmed>={thresholds['confirmed_threshold']*100:.1f}%, "
        f"max_unverified<={thresholds['max_unverified']}"
    )
    if reasons:
        summary += f"; reasons: {', '.join(reasons)}"
    try:
        manage_project_knowledge(
            action="add",
            category="decision",
            observation="Literature gate evaluation before modeling/simulation.",
            solution=summary,
        )
    except Exception:
        pass
    try:
        record_lit_gate_in_provenance(
            f"{status.upper()} (confirmed={confirmed_pct:.1f}%, max_unverified={thresholds['max_unverified']})"
        )
    except Exception:
        pass


def record_lit_gate_in_provenance(status_line: str) -> str:
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


def should_skip_lit_gate(skip_flag: bool = False) -> bool:
    env_skip = os.environ.get("AISC_SKIP_LIT_GATE", "").strip().lower() in {"1", "true", "yes", "on"}
    return skip_flag or env_skip


def ensure_lit_gate_ready(skip_gate: bool = False):
    if should_skip_lit_gate(skip_gate):
        return
    gate = evaluate_lit_ready(
        lit_path=None,
        verification_path=None,
        confirmed_threshold=None,
        max_unverified=None,
    )
    if gate.get("status") == "ready":
        return
    reasons = gate.get("reasons", [])
    raise RuntimeError(
        f"Literature gate not satisfied: {', '.join(reasons) or 'see check_lit_ready for details.'}"
    )


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


def _model_metadata_from_key(model_key: str) -> Dict[str, Any]:
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


def ensure_model_spec_and_params(model_key: str) -> Dict[str, Any]:
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


def evaluate_model_provenance(model_key: str, allow_free: bool = False) -> Dict[str, Any]:
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
        raise FileNotFoundError(
            f"Model spec/params missing for {model_key}; expected {spec_path} and {param_path}"
        )

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


def record_model_provenance_in_provenance(model_key: str, status_line: str) -> str:
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


def resolve_claim_graph_path() -> Path:
    base = os.environ.get("AISC_BASE_FOLDER", "")
    if base:
        return Path(base) / "claim_graph.json"
    exp_dir = BaseTool.resolve_output_dir(None)
    base_dir = exp_dir.parent if exp_dir.name == "experiment_results" else exp_dir
    return base_dir / "claim_graph.json"


def _load_claim_graph(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"claim_graph.json not found: {path}")
    try:
        return cast(List[Dict[str, Any]], json.loads(path.read_text()))
    except Exception as exc:
        raise ValueError(f"Failed to read claim graph: {exc}")


def _load_hypothesis_trace_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"hypothesis_trace.json not found: {path}")
    try:
        return cast(Dict[str, Any], json.loads(path.read_text()))
    except Exception as exc:
        raise ValueError(f"Failed to read hypothesis trace: {exc}")


def _gather_support_from_trace(trace: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    supports: Dict[str, Dict[str, Any]] = {}
    for hyp in trace.get("hypotheses", []) or []:
        hyp_id = hyp.get("id") or hyp.get("name") or ""
        for exp in hyp.get("experiments", []) or []:
            exp_id = exp.get("id") or ""
            key = f"{hyp_id}:{exp_id}"
            sim_runs = exp.get("sim_runs") or []
            metrics = exp.get("metrics") or []
            figures = exp.get("figures") or []
            supports[key] = {
                "sim_runs": sim_runs,
                "metrics": metrics,
                "figures": figures,
            }
    return supports


def evaluate_claim_consistency() -> Dict[str, Any]:
    exp_dir = BaseTool.resolve_output_dir(None)
    claim_path = exp_dir / "claim_graph.json"
    trace_path = exp_dir / "hypothesis_trace.json"
    claims = _load_claim_graph(claim_path)
    trace = _load_hypothesis_trace_file(trace_path)
    support_map = _gather_support_from_trace(trace)

    results: List[Dict[str, Any]] = []
    missing_count = 0
    weak_count = 0
    for claim in claims:
        cid = claim.get("claim_id") or "unknown"
        status = (claim.get("status") or "").strip().lower()
        support = claim.get("support") or []
        has_trace_support = any(
            (v.get("sim_runs") or v.get("metrics"))
            for v in support_map.values()
        )
        if status == "unlinked" or (not support and not has_trace_support):
            support_status = "missing"
            missing_count += 1
        elif support or has_trace_support:
            support_status = "ok" if status not in {"partial", "weak"} else "weak"
            if support_status == "weak":
                weak_count += 1
        else:
            support_status = "weak"
            weak_count += 1
        results.append(
            {
                "claim_id": cid,
                "claim_text": claim.get("claim_text", ""),
                "status": status,
                "support": support,
                "support_status": support_status,
            }
        )

    overall = "ready_for_publication"
    if missing_count > 0:
        overall = "not_ready_for_publication"
    elif weak_count > 0:
        overall = "review_needed"

    return {
        "claims": results,
        "overall_status": overall,
        "n_missing": missing_count,
        "n_weak": weak_count,
        "claim_graph_path": str(claim_path),
        "hypothesis_trace_path": str(trace_path),
    }


def record_claim_consistency_in_provenance(status_line: str) -> str:
    exp_dir = BaseTool.resolve_output_dir(None)
    out_path = exp_dir / "provenance_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gate_line = f"- Claim consistency: {status_line}"

    if not out_path.exists():
        content = (
            "# Provenance Summary\n\n"
            "## Literature Sources\n- Missing or not generated.\n\n"
            "## Model Definitions\n- Missing or not generated.\n\n"
            "## Simulation Protocols\n- Missing or not generated.\n\n"
            "## Statistical Analyses\n- Missing or not generated.\n"
        )
        out_path.write_text(content + "\n")

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
            if "Claim consistency:" in lines[j]:
                lines[j] = gate_line
                replaced = True
                break
        if not replaced:
            lines.insert(insert_idx, gate_line)
    out_path.write_text("\n".join(lines) + "\n")
    return str(out_path)


def collect_provenance_sections() -> Dict[str, Any]:
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
        if "sweep" in path or "transport_runs" in path or "sim.json" in name or "intervention" in path:
            sections["simulations"].append(path or name)
        if "metrics" in name or "stats" in kind:
            sections["stats"].append(path or name)
    return sections


def render_provenance_markdown(sections: Dict[str, Any]) -> str:
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


def generate_provenance_summary_impl() -> Dict[str, Any]:
    sections = collect_provenance_sections()
    content = render_provenance_markdown(sections)
    exp_dir = BaseTool.resolve_output_dir(None)
    out_path = exp_dir / "provenance_summary.md"
    out_path.write_text(content)
    entry = {
        "name": out_path.name,
        "path": str(out_path),
        "kind": "provenance_summary_md",
        "created_by": os.environ.get("AISC_ACTIVE_ROLE", "reviewer"),
        "status": "ok",
    }
    try:
        manifest_utils.append_or_update(entry, base_folder=BaseTool.resolve_output_dir(None))
    except Exception:
        pass
    return {"path": str(out_path), "sections": sections}

# Utility functions
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
