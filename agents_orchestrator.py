"""
Agents-SDK orchestrator with a PI agent that delegates to role agents (tools/handoffs)
until the Holistic Reviewer reports no gaps.
"""

import argparse
import json
import os
import os.path as osp
from datetime import datetime
import asyncio
import inspect
from typing import Any, Dict, List, Optional
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from agents import Agent, Runner, function_tool, ModelSettings

# Wrap existing tools with function_tool adapters
from ai_scientist.tools.lit_data_assembly import LitDataAssemblyTool
from ai_scientist.tools.lit_validator import LitSummaryValidatorTool
from ai_scientist.tools.graph_builder import BuildGraphsTool
from ai_scientist.tools.compartmental_sim import RunCompartmentalSimTool
from ai_scientist.tools.sensitivity_sweep import RunSensitivitySweepTool
from ai_scientist.tools.validation_compare import RunValidationCompareTool
from ai_scientist.tools.intervention_tester import RunInterventionTesterTool
from ai_scientist.tools.manuscript_reader import ManuscriptReaderTool
from ai_scientist.tools.biological_plotting import RunBiologicalPlottingTool
from ai_scientist.tools.claim_graph import ClaimGraphTool
from ai_scientist.tools.claim_graph_checker import ClaimGraphCheckTool
from ai_scientist.perform_biological_interpretation import interpret_biological_results
from ai_scientist.perform_writeup import perform_writeup


def parse_args():
    p = argparse.ArgumentParser(description="Agents orchestrator with PI and role agents.")
    p.add_argument("--load_idea", required=True, help="Path to idea JSON (single idea object).")
    p.add_argument("--idea_idx", type=int, default=0, help="Index if the idea file contains a list of ideas.")
    p.add_argument("--model", default="gpt-4o-mini", help="LLM model id.")
    p.add_argument("--max_cycles", type=int, default=25, help="Max PI cycles (or stop earlier if reviewer says 'no gaps' and PDF exists).")
    p.add_argument(
        "--input",
        default="Begin planning and execution for this project. Identify next actions and delegate.",
        help="Initial input message to the PI agent.",
    )
    return p.parse_args()


# Tool adapters with explicit schemas
def _fill_output_dir(output_dir: Optional[str]) -> str:
    return output_dir or os.environ.get("AISC_EXP_RESULTS", "experiment_results")


@function_tool
def assemble_lit_data(
    output_dir: Optional[str] = None,
    queries: Optional[List[str]] = None,
    seed_paths: Optional[List[str]] = None,
    max_results: int = 25,
    use_semantic_scholar: bool = True,
):
    return LitDataAssemblyTool().use_tool(
        output_dir=_fill_output_dir(output_dir),
        queries=queries,
        seed_paths=seed_paths,
        max_results=max_results,
        use_semantic_scholar=use_semantic_scholar,
    )


@function_tool
def validate_lit_summary(path: str):
    return LitSummaryValidatorTool().use_tool(path=path)


@function_tool
def build_graphs(n_nodes: int = 100, output_dir: Optional[str] = None, seed: int = 0):
    return BuildGraphsTool().use_tool(n_nodes=n_nodes, output_dir=_fill_output_dir(output_dir), seed=seed)


@function_tool
def run_comp_sim(
    graph_path: str,
    output_dir: Optional[str] = None,
    steps: int = 200,
    dt: float = 0.1,
    transport_rate: float = 0.05,
    demand_scale: float = 0.5,
    mitophagy_rate: float = 0.02,
    noise_std: float = 0.0,
    seed: int = 0,
):
    return RunCompartmentalSimTool().use_tool(
        graph_path=graph_path,
        output_dir=_fill_output_dir(output_dir),
        steps=steps,
        dt=dt,
        transport_rate=transport_rate,
        demand_scale=demand_scale,
        mitophagy_rate=mitophagy_rate,
        noise_std=noise_std,
        seed=seed,
    )


@function_tool
def run_sensitivity_sweep(
    graph_path: str,
    output_dir: Optional[str] = None,
    transport_vals: Optional[List[float]] = None,
    demand_vals: Optional[List[float]] = None,
    steps: int = 150,
    dt: float = 0.1,
):
    return RunSensitivitySweepTool().use_tool(
        graph_path=graph_path,
        output_dir=_fill_output_dir(output_dir),
        transport_vals=transport_vals,
        demand_vals=demand_vals,
        steps=steps,
        dt=dt,
    )


@function_tool
def run_validation_compare(lit_path: str, sim_path: str):
    return RunValidationCompareTool().use_tool(lit_path=lit_path, sim_path=sim_path)


@function_tool
def run_intervention_tests(
    graph_path: str,
    output_dir: Optional[str] = None,
    transport_vals: Optional[List[float]] = None,
    demand_vals: Optional[List[float]] = None,
    baseline_transport: float = 0.05,
    baseline_demand: float = 0.5,
):
    return RunInterventionTesterTool().use_tool(
        graph_path=graph_path,
        output_dir=_fill_output_dir(output_dir),
        transport_vals=transport_vals,
        demand_vals=demand_vals,
        baseline_transport=baseline_transport,
        baseline_demand=baseline_demand,
    )


@function_tool
def run_biological_plotting(solution_path: str, output_dir: Optional[str] = None, make_phase_portrait: bool = True):
    return RunBiologicalPlottingTool().use_tool(
        solution_path=solution_path,
        output_dir=_fill_output_dir(output_dir),
        make_phase_portrait=make_phase_portrait,
    )


@function_tool
def run_interpretation(base_folder: Optional[str] = None, config_path: Optional[str] = None, model: str = "gpt-4o-mini"):
    base_folder = base_folder or os.environ.get("AISC_BASE_FOLDER", "")
    config_path = config_path or os.environ.get("AISC_CONFIG_PATH", "")
    ok = interpret_biological_results(base_folder=base_folder, config_path=config_path, model=model)
    return {"success": ok}


@function_tool
def run_writeup(
    base_folder: Optional[str] = None,
    model_writeup_small: str = "gpt-4o-mini",
    model_writeup: str = "gpt-4o",
    page_limit: int = 6,
):
    base_folder = base_folder or os.environ.get("AISC_BASE_FOLDER", "")
    ok = perform_writeup(
        base_folder=base_folder,
        small_model=model_writeup_small,
        big_model=model_writeup,
        page_limit=page_limit,
        citations_text=None,
    )
    return {"success": ok}


@function_tool
def read_manuscript(path: str):
    return ManuscriptReaderTool().use_tool(path=path)


@function_tool
def update_claim_graph(
    path: str,
    claim_id: str,
    claim_text: str,
    parent_id: Optional[str] = None,
    support: Optional[List[str]] = None,
    status: str = "unlinked",
    notes: str = "",
):
    return ClaimGraphTool().use_tool(
        path=path,
        claim_id=claim_id,
        claim_text=claim_text,
        parent_id=parent_id,
        support=support,
        status=status,
        notes=notes,
    )


@function_tool
def check_claim_graph(path: str):
    return ClaimGraphCheckTool().use_tool(path=path)


def build_agents(model: str, idea: Dict[str, Any]):
    idea_title = idea.get("Title", "the project")
    idea_name = idea.get("Name", "")
    idea_hyp = idea.get("Short Hypothesis", "")
    idea_abs = idea.get("Abstract", "")
    idea_exps = idea.get("Experiments", [])
    idea_risks = idea.get("Risk Factors and Limitations", [])

    compact_exps = "; ".join(idea_exps) if isinstance(idea_exps, list) else str(idea_exps)
    compact_risks = "; ".join(idea_risks) if isinstance(idea_risks, list) else str(idea_risks)

    # Sub-agents
    archivist = Agent(
        name="Archivist",
        instructions=(
            f"You are the senior literature curator for '{idea_title}'. Gather and normalize literature on the project's topic. "
            "Use lit tools. Save lit_summary.csv/json with fields: region, axon_length, branch_order, "
            "node_degree, transport_rate, mitophagy_rate, atp_diffusion_time, calcium_energy_cost. "
            "Return structured status (status/artifacts/notes) and write it to experiment_results/status_archivist.json (or append to tool_summary.txt). "
            f"Context: {idea_abs}"
        ),
        model=model,
        tools=[assemble_lit_data, validate_lit_summary],
    )
    modeler = Agent(
        name="Modeler",
        instructions=(
            f"You are the senior computational modeler for '{idea_title}'. Build canonical graphs and run compartmental simulations/sweeps/"
            "interventions relevant to the hypothesis. Use deterministic seeds. Save graphs to graphs/, outputs to "
            "experiment_results/. "
            "Use modeling/stats utilities when helpful (perform_biological_modeling.py, perform_biological_stats.py). "
            "If plots are needed, coordinate with Analyst/Publisher to target the blank_theoretical_biology_latex structure (Abstract, Intro, Model/Methods, Results with figures, Discussion, References) at ai_scientist/blank_theoretical_biology_latex/. "
            "Return structured status (status/artifacts/notes) and write it to experiment_results/status_modeler.json (or append to tool_summary.txt). "
            f"Context: {idea_abs}"
        ),
        model=model,
        tools=[build_graphs, run_comp_sim, run_sensitivity_sweep, run_intervention_tests],
    )
    analyst = Agent(
        name="Analyst",
        instructions=(
            f"You are the senior scientific visualization/analysis expert for '{idea_title}', producing publication-quality plots (SVG/PNG) for venues like PLOS Computational Biology. "
            "Compare simulations vs lit_summary, compute correlations/effect sizes, produce clear, well-labeled figures. Save artifacts to experiment_results/. "
            "Use plot aggregation if helpful (perform_plotting.py) and stats utilities (perform_biological_stats.py). "
            "Coordinate with Publisher to align figures with the blank_theoretical_biology_latex sections (Abstract, Intro, Model/Methods, Results with figures, Discussion, References) at ai_scientist/blank_theoretical_biology_latex/. "
            "Write concise, informative captions when producing plots. Place final figures in figures/ when using plot aggregation; otherwise keep plots in experiment_results/."
            "Return structured status (status/artifacts/notes) and write it to experiment_results/status_analyst.json (or append to tool_summary.txt). "
            f"Context: {idea_abs}"
        ),
        model=model,
        tools=[run_validation_compare, run_biological_plotting],
    )
    reviewer = Agent(
        name="HolisticReviewer",
        instructions=(
            f"You are the senior holistic reviewer for '{idea_title}'. Review artifacts (lit_summary, sims, sweeps, plots, text). "
            "Trace claims/theses to supporting evidence: either generated analyses (figures/tables/formulae) or cited prior work. "
            "Check novelty/significance framing, Methods reproducibility (parameters, code availability), correct inline citations/bibliography, and structural completeness (Abstract, Intro, Methods, Results, Discussion). "
            "Ensure alignment with the target template (blank_theoretical_biology_latex at ai_scientist/blank_theoretical_biology_latex/), including well-placed figures/tables and natbib citations. "
            "If other templates are used (blank_bioinformatics_latex or blank_icbinb_latex), ensure structure/citations match. "
            "In Discussion, ensure coverage of: main finding/mechanism, relation to literature, translational relevance, limitations/future directions. "
            "Verify outputs follow conventions (figures in figures/ if aggregated, data/artifacts in experiment_results/), and that natbib citation style from the template dirs is respected. "
            "Use the manuscript reader tool to inspect current and previous drafts (PDF or text) to spot dropped claims or missing citations that should be reinstated."
            "Flag contradictions, missing support, missing/incorrect citations, or structural gaps in a 'gaps' list; respond 'no gaps' only when clear. "
            "Return structured status (status/artifacts/notes) and write it to experiment_results/status_reviewer.json (or append to tool_summary.txt). "
            f"Experiments plan: {compact_exps} Risks: {compact_risks}"
        ),
        model=model,
        tools=[read_manuscript, update_claim_graph, check_claim_graph],
    )
    publisher = Agent(
        name="Publisher",
        instructions=(
            f"You are the senior production editor for '{idea_title}'. When artifacts are ready, run interpretation/writeup for theoretical biology. "
            "Ensure formatting, clarity, and completeness. Target the blank_theoretical_biology_latex structure (Abstract, Intro, Model/Methods, Results with figures, Discussion, References) located at ai_scientist/blank_theoretical_biology_latex/. "
            "Use provided LaTeX deps (natbib, fancyhdr), plot aggregation (perform_plotting.py) if helpful, and consider other templates if specified (blank_bioinformatics_latex, blank_icbinb_latex). "
            "Ensure all prior-knowledge claims are cited inline and the References list is complete (natbib citation style from the template dirs). "
            "Ensure figures are numbered with clear captions; include Analyst captions or refine them as needed. "
            "Note code/data paths and parameter dumps for reproducibility. Save PDFs to the run root; keep data/artifacts in experiment_results/ and figures in figures/ when aggregated. Use the experiments plan and risks for alignment. "
            "Return structured status (status/artifacts/notes) and write it to experiment_results/status_publisher.json (or append to tool_summary.txt). "
            f"Plan: {compact_exps} Risks: {compact_risks}"
        ),
        model=model,
        tools=[run_interpretation, run_writeup],
    )

    # PI orchestrator with handoffs
    pi = Agent(
        name="PI",
        instructions=(
            f"You are the senior PI for '{idea_title}'. Hypothesis: {idea_hyp}. "
            f"Abstract: {idea_abs} Experiments: {compact_exps} Risks: {compact_risks}. "
            "Plan and delegate tasks to close gaps. Always delegate via a tool call each turn (no free-text-only turns). Start by calling the archivist tool to assemble/validate literature, then call other specialists as needed. Ensure claims are backed by analyses or cited prior work and citations are correct. Keep everyone aligned to the target template (blank_theoretical_biology_latex at ai_scientist/blank_theoretical_biology_latex/) with figures/tables and natbib citations. "
            "If other templates are required (blank_bioinformatics_latex, blank_icbinb_latex), direct Publisher accordingly. "
            "Enforce output conventions: figures in figures/ when using plot aggregation, data/artifacts in experiment_results/, PDFs at run root, natbib citation style. Ensure reproducibility notes (code/data paths, params) are captured. "
            "For each delegation, pass a clear task and work_dir (e.g., experiment_results/), require structured status back, and ensure the agent writes status_*.json (or appends to tool_summary.txt). "
            "Call sub-agents as tools. Once the reviewer reports no gaps, call the Publisher to run interpretation/writeup (target blank_theoretical_biology_latex sections: Abstract, Intro, Model/Methods, Results with figures, Discussion, References) and confirm a PDF exists before stopping."
        ),
        model=model,
        tools=[
            archivist.as_tool(tool_name="archivist", tool_description="Literature assembly/validation."),
            modeler.as_tool(tool_name="modeler", tool_description="Graphs, simulations, sweeps."),
            analyst.as_tool(tool_name="analyst", tool_description="Validation compare, plotting."),
            reviewer.as_tool(tool_name="reviewer", tool_description="Holistic gap check (uses manuscript reader, claim graph)."),
            publisher.as_tool(tool_name="publisher", tool_description="Interpretation/writeup."),
        ],
        model_settings=ModelSettings(tool_choice="required"),
    )
    return pi


def main():
    args = parse_args()

    # Load .env early to ensure OPENAI_API_KEY (and others) are set, overriding any lower-priority defaults.
    if load_dotenv is not None:
        load_dotenv(dotenv_path=".env", override=True)

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

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_folder = f"experiments/{date}_{idea['Name']}_agents"
    os.makedirs(base_folder, exist_ok=True)
    exp_results = osp.join(base_folder, "experiment_results")
    os.makedirs(exp_results, exist_ok=True)

    # Save idea
    with open(osp.join(base_folder, "idea.json"), "w") as f:
        json.dump(idea, f, indent=2)

    # Set env defaults for tools
    os.environ["AISC_EXP_RESULTS"] = exp_results
    os.environ["AISC_BASE_FOLDER"] = base_folder
    os.environ["AISC_CONFIG_PATH"] = osp.join(base_folder, "bfts_config.yaml")

    pi = build_agents(model=args.model, idea=idea)

    context: Dict[str, Any] = {
        "base_folder": base_folder,
        "exp_results": exp_results,
        "idea": idea,
        "config_path": osp.join(base_folder, "bfts_config.yaml"),  # optional
    }

    print(f"Running agents orchestrator for {idea['Name']} into {base_folder}")
    result = Runner.run(pi, input=args.input, context=context, max_turns=args.max_cycles)
    # If Runner.run returned a coroutine (some versions are async), drive it.
    if inspect.iscoroutine(result):
        result = asyncio.run(result)

    # Persist the final runner output so we have a trace even if agents/tools produced no files.
    try:
        with open(osp.join(base_folder, "runner_output.txt"), "w") as f:
            f.write(str(result))
    except Exception:
        pass
    print("Done.")


if __name__ == "__main__":
    main()
