# pyright: reportMissingImports=false
import argparse
import json
import os
import os.path as osp
from datetime import datetime
import asyncio
from typing import Any, Dict, List, Optional

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
    # Helper to default to environment variable if not passed by agent
    return output_dir or os.environ.get("AISC_EXP_RESULTS", "experiment_results")

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
):
    """Runs a compartmental simulation and saves CSV data."""
    return RunCompartmentalSimTool().use_tool(
        graph_path=graph_path or "", # Handle optional
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
def run_biological_plotting(solution_path: str, output_dir: Optional[str] = None, make_phase_portrait: bool = True):
    """Generates plots from simulation data."""
    return RunBiologicalPlottingTool().use_tool(
        solution_path=solution_path,
        output_dir=_fill_output_dir(output_dir),
        make_phase_portrait=make_phase_portrait,
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
    return BuildGraphsTool().use_tool(n_nodes=n_nodes, output_dir=_fill_output_dir(output_dir), seed=seed)

@function_tool
def run_biological_model(
    model_key: str = "cooperation_evolution",
    time_end: float = 20.0,
    num_points: int = 200,
    output_dir: Optional[str] = None,
):
    """Run a built-in biological ODE/replicator model and save JSON results."""
    return RunBiologicalModelTool().use_tool(
        model_key=model_key,
        time_end=time_end,
        num_points=num_points,
        output_dir=_fill_output_dir(output_dir),
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
    """Sweep transport_rate and demand_scale over a graph and log frac_failed."""
    return RunSensitivitySweepTool().use_tool(
        graph_path=graph_path,
        output_dir=_fill_output_dir(output_dir),
        transport_vals=transport_vals,
        demand_vals=demand_vals,
        steps=steps,
        dt=dt,
    )

@function_tool
def run_intervention_tests(
    graph_path: str,
    output_dir: Optional[str] = None,
    transport_vals: Optional[List[float]] = None,
    demand_vals: Optional[List[float]] = None,
    baseline_transport: float = 0.05,
    baseline_demand: float = 0.5,
):
    """Test parameter interventions vs a baseline and report delta frac_failed."""
    return RunInterventionTesterTool().use_tool(
        graph_path=graph_path,
        output_dir=_fill_output_dir(output_dir),
        transport_vals=transport_vals,
        demand_vals=demand_vals,
        baseline_transport=baseline_transport,
        baseline_demand=baseline_demand,
    )

@function_tool
def run_validation_compare(lit_path: str, sim_path: str):
    """Correlate lit_summary metrics with simulation frac_failed."""
    return RunValidationCompareTool().use_tool(lit_path=lit_path, sim_path=sim_path)

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
    return {
        "success": interpret_biological_results(
            base_folder=base_folder or os.environ.get("AISC_BASE_FOLDER", ""),
            config_path=config_path or os.environ.get("AISC_CONFIG_PATH", ""),
        )
    }

# --- Agent Definitions ---

def build_team(model: str, idea: Dict[str, Any], dirs: Dict[str, str]):
    """
    Constructs the agents with strict context partitioning.
    """
    common_settings = ModelSettings(tool_choice="auto")
    
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
            "1. Use 'assemble_lit_data' or 'search_semantic_scholar' to gather papers.\n"
            "2. Maintain a claim graph via 'update_claim_graph' when mapping evidence.\n"
            "3. Output 'lit_summary.json' to experiment_results/.\n"
            "4. CRITICAL: If no papers are found, report FAILURE. Do not invent 'TBD' citations."
        ),
        model=model,
        tools=[assemble_lit_data, validate_lit_summary, search_semantic_scholar, update_claim_graph],
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
            "1. You do NOT care about LaTeX or writing styles. Focus on DATA.\n"
            "2. Build graphs ('build_graphs'), run baselines ('run_biological_model') or custom sims ('run_comp_sim').\n"
            "3. Explore parameter space using 'run_sensitivity_sweep' and 'run_intervention_tests'.\n"
            "4. Ensure parameter sweeps cover the range specified in the hypothesis.\n"
            "5. Save raw outputs to experiment_results/."
        ),
        model=model,
        tools=[build_graphs, run_biological_model, run_comp_sim, run_sensitivity_sweep, run_intervention_tests], 
        model_settings=common_settings
    )

    # 3. The Analyst (Scope: Visualization & Validation)
    analyst = Agent(
        name="Analyst",
        instructions=(
            "Role: Scientific Visualization Expert.\n"
            "Goal: Convert simulation data into PLOS-quality figures.\n"
            "Directives:\n"
            "1. Read data from experiment_results/.\n"
            "2. Assert that the data supports the hypothesis BEFORE plotting. If data contradicts hypothesis, report this back immediately.\n"
            "3. Generate PNG/SVG files using 'run_biological_plotting'.\n"
            "4. Validate models vs lit via 'run_validation_compare' and use 'run_biological_stats' for significance/enrichment."
        ),
        model=model,
        tools=[run_biological_plotting, run_validation_compare, run_biological_stats],
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
            "1. Read the manuscript draft using 'read_manuscript'.\n"
            "2. Check claim support using 'check_claim_graph' and sanity-check stats with 'run_biological_stats' if needed.\n"
            "3. Check consistency: Does Figure 3 actually support the claim in paragraph 2?\n"
            "4. If gaps exist, report them clearly to the PI.\n"
            "5. Only report 'NO GAPS' if the PDF validates completely."
        ),
        model=model,
        tools=[read_manuscript, check_claim_graph, run_biological_stats],
        model_settings=common_settings
    )

    # 5. The Interpreter (Scope: Theoretical Interpretation)
    interpreter = Agent(
        name="Interpreter",
        instructions=(
            "Role: Mathematical–Biological Interpreter.\n"
            "Goal: Produce interpretation.json/md for theoretical biology projects.\n"
            "Directives:\n"
            "1. Call 'interpret_biology' only when biology.research_type == theoretical.\n"
            "2. Use experiment summaries and idea text; do NOT hallucinate unsupported claims.\n"
            "3. If interpretation fails, report the error clearly."
        ),
        model=model,
        tools=[interpret_biology],
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
        tools=[run_writeup_task],
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
            archivist.as_tool(tool_name="archivist", tool_description="Search literature."),
            modeler.as_tool(tool_name="modeler", tool_description="Run simulations."),
            analyst.as_tool(tool_name="analyst", tool_description="Create figures."),
            interpreter.as_tool(tool_name="interpreter", tool_description="Generate theoretical interpretation."),
            publisher.as_tool(tool_name="publisher", tool_description="Write and compile text."),
            reviewer.as_tool(tool_name="reviewer", tool_description="Critique the draft."),
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
