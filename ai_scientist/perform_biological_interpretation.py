import json
import os
import os.path as osp
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from ai_scientist.llm import (
    get_response_from_llm,
    extract_json_between_markers,
    create_client,
)
from ai_scientist.treesearch.utils.config import load_cfg, Config
from ai_scientist.perform_icbinb_writeup import (
    load_idea_text,
    load_exp_summaries,
    filter_experiment_summaries,
)


def _biology_cfg_to_dict(cfg: Config) -> Dict[str, Any]:
    """Serialize cfg.biology into a plain dict suitable for prompting."""
    biology = getattr(cfg, "biology", None)
    if biology is None:
        return {}

    out: Dict[str, Any] = {
        "research_type": getattr(biology, "research_type", None),
        "domain": getattr(biology, "domain", None),
        "phases": list(getattr(biology, "phases", []) or []),
        "eval_focus": getattr(biology, "eval_focus", None),
    }

    modeling = getattr(biology, "modeling", None)
    if modeling is not None:
        out["modeling"] = {
            "framework": getattr(modeling, "framework", None),
            "time_horizon": getattr(modeling, "time_horizon", None),
            "num_time_points": getattr(modeling, "num_time_points", None),
        }

    targets = getattr(biology, "targets", None)
    if targets is not None:
        out["targets"] = {
            "primary_quantity": getattr(targets, "primary_quantity", None),
            "secondary_quantities": list(
                getattr(targets, "secondary_quantities", []) or []
            ),
        }

    # Remove keys that are entirely None/empty to reduce noise
    cleaned: Dict[str, Any] = {}
    for k, v in out.items():
        if v is None:
            continue
        if isinstance(v, (list, dict)) and not v:
            continue
        cleaned[k] = v
    return cleaned


def _interpretation_to_markdown(interpretation: Dict[str, Any]) -> str:
    """Render the structured interpretation JSON into a human-readable markdown report."""
    lines: list[str] = []

    title = interpretation.get("title") or "Mathematical–Biological Interpretation"
    lines.append(f"# {title}")
    lines.append("")

    math_res = interpretation.get("mathematical_results") or {}
    bio_int = interpretation.get("biological_interpretation") or {}
    assumptions = interpretation.get("assumptions_and_limitations") or {}
    predictions = interpretation.get("predictions") or {}

    # Mathematical results
    lines.append("## Mathematical Results")
    equilibria = math_res.get("equilibria_or_ESS")
    if equilibria:
        lines.append("**Equilibria / ESS:**")
        lines.append(equilibria)
        lines.append("")
    stability = math_res.get("stability")
    if stability:
        lines.append("**Stability:**")
        lines.append(stability)
        lines.append("")
    regimes = math_res.get("parameter_regimes") or []
    if regimes:
        lines.append("**Parameter regimes:**")
        for r in regimes:
            param = r.get("parameter", "parameter")
            thr = r.get("threshold", "N/A")
            behavior = r.get("behavior", "")
            lines.append(f"- When {param} ≈ {thr}: {behavior}")
        lines.append("")
    key_obs = math_res.get("key_observables") or []
    if key_obs:
        lines.append("**Key observables:**")
        for o in key_obs:
            lines.append(f"- {o}")
        lines.append("")

    # Biological interpretation
    lines.append("## Biological Interpretation")
    mech = bio_int.get("mechanistic_explanation")
    if mech:
        lines.append("**Mechanistic explanation:**")
        lines.append(mech)
        lines.append("")
    conds = bio_int.get("conditions_for_behavior") or []
    if conds:
        lines.append("**Conditions for qualitative behaviors (e.g., cooperation, extinction):**")
        for c in conds:
            lines.append(f"- {c}")
        lines.append("")
    bio_preds = bio_int.get("predictions") or []
    if bio_preds:
        lines.append("**Biological predictions:**")
        for p in bio_preds:
            lines.append(f"- {p}")
        lines.append("")

    # Assumptions & limitations
    if assumptions:
        lines.append("## Assumptions and Limitations")
        for k, v in assumptions.items():
            if not v:
                continue
            if isinstance(v, list):
                lines.append(f"**{k.replace('_', ' ').title()}:**")
                for item in v:
                    lines.append(f"- {item}")
            else:
                lines.append(f"**{k.replace('_', ' ').title()}:** {v}")
        lines.append("")

    # Experimental predictions / validation
    if predictions:
        lines.append("## Experimental Validation and Predictions")
        exps = predictions.get("experimental_setups") or []
        metrics = predictions.get("measurement_targets") or []
        if exps:
            lines.append("**Suggested experimental setups:**")
            for e in exps:
                lines.append(f"- {e}")
            lines.append("")
        if metrics:
            lines.append("**Suggested readouts / measurements:**")
            for m in metrics:
                lines.append(f"- {m}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _call_interpretation_llm(
    cfg: Config,
    idea_text: str,
    summaries: Dict[str, Any],
    base_folder: str,
    model: str = "gpt-5.1-2025-11-13",
) -> Optional[Dict[str, Any]]:
    """Call an LLM to synthesize mathematical and biological interpretation in JSON form."""
    biology_dict = _biology_cfg_to_dict(cfg)

    system_message = (
        "You are a theoretical and computational biologist. "
        "Given a theoretical computational biology project configuration, idea, "
        "and summaries of in silico experiments, extract:\n"
        "1) Clear mathematical results (equilibria/ESS, stability properties, "
        "   parameter thresholds/regimes, key observables), and\n"
        "2) Their biological interpretation (mechanistic explanations, conditions "
        "   for behaviors like cooperation, extinction, pattern formation, etc.),\n"
        "3) Assumptions/limitations that affect interpretation, and\n"
        "4) Concrete, testable biological predictions.\n\n"
        "Respond STRICTLY as a single JSON object with the following top-level keys:\n"
        "- \"title\" (string, short descriptive title)\n"
        "- \"mathematical_results\" (object)\n"
        "- \"biological_interpretation\" (object)\n"
        "- \"assumptions_and_limitations\" (object)\n"
        "- \"predictions\" (object)\n\n"
        "Do NOT include any text outside the JSON. Do NOT hallucinate quantities "
        "that are not supported by the summaries."
    )

    user_prompt = f"""
Biology configuration:
```json
{json.dumps(biology_dict, indent=2)}
```

Idea (markdown):
```markdown
{idea_text}
```

Best experiment summaries (JSON):
```json
{json.dumps(summaries, indent=2)}
```
"""

    client, client_model = create_client(model)
    try:
        text, _ = get_response_from_llm(
            prompt=user_prompt,
            client=client,
            model=client_model,
            system_message=system_message,
            msg_history=[],
            print_debug=False,
        )
    except Exception:
        print("EXCEPTION in interpretation LLM call:")
        print(traceback.format_exc())
        return None

    # Try to extract JSON
    data = extract_json_between_markers(text)
    if data is not None:
        return data

    # Fallback: attempt direct json.loads
    try:
        return json.loads(text)
    except Exception:
        print("Failed to parse interpretation JSON from LLM response.")
        debug_path = osp.join(base_folder, "interpretation_raw_llm_output.txt")
        try:
            with open(debug_path, "w") as f:
                f.write(text)
        except Exception:
            pass
        return None


def interpret_biological_results(
    base_folder: str,
    config_path: str,
    model: str = "gpt-5.1-2025-11-13",
) -> bool:
    """
    Run a mathematical-biological interpretation phase for a given experiment.

    This:
    - Loads the per-run config (including `biology` block),
    - Short-circuits for non-theoretical projects,
    - Loads idea text and experiment summaries,
    - Calls an LLM to synthesize a structured interpretation, and
    - Writes `interpretation.json` and `interpretation.md` into `base_folder`.

    Returns:
        True if interpretation artifacts were successfully generated, else False.
    """
    try:
        cfg = load_cfg(Path(config_path))
    except Exception:
        print(f"Failed to load config from: {config_path}")
        print(traceback.format_exc())
        return False

    biology = getattr(cfg, "biology", None)
    if biology is None or getattr(biology, "research_type", None) != "theoretical":
        # Interpretation phase is only defined for theoretical computational biology
        print("Skipping biological interpretation (not a theoretical biology project).")
        return False

    try:
        idea_text = load_idea_text(base_folder)
    except Exception:
        print("Error loading idea text for interpretation:")
        print(traceback.format_exc())
        idea_text = ""

    try:
        exp_summaries = load_exp_summaries(base_folder)
        # Reuse the 'writeup' filtering to get best nodes and their analyses
        filtered_summaries = filter_experiment_summaries(
            exp_summaries, step_name="writeup"
        )
    except Exception:
        print("Error loading experiment summaries for interpretation:")
        print(traceback.format_exc())
        filtered_summaries = {}

    interpretation = _call_interpretation_llm(
        cfg=cfg,
        idea_text=idea_text,
        summaries=filtered_summaries,
        base_folder=base_folder,
        model=model,
    )
    if interpretation is None:
        print("Biological interpretation LLM call failed.")
        return False

    # Write JSON artifact
    json_path = osp.join(base_folder, "interpretation.json")
    try:
        with open(json_path, "w") as f:
            json.dump(interpretation, f, indent=2)
    except Exception:
        print("Error writing interpretation.json:")
        print(traceback.format_exc())
        return False

    # Write markdown narrative
    md_path = osp.join(base_folder, "interpretation.md")
    try:
        md_text = _interpretation_to_markdown(interpretation)
        with open(md_path, "w") as f:
            f.write(md_text)
    except Exception:
        print("Error writing interpretation.md:")
        print(traceback.format_exc())
        return False

    print(f"Biological interpretation written to:\n- {json_path}\n- {md_path}")
    return True
