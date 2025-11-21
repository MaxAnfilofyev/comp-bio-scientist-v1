"""
Tool-driven runner: lets the LLM choose and call predefined tools to execute a theoretical
computational biology plan without engaging the full tree-search.

Workflow:
- Loads an idea from JSON.
- Creates an experiment directory.
- Exposes a registry of tools (lit assembly, validation, graph build, sims, sweeps, interventions).
- Loops for a fixed number of steps: LLM decides which tool to call next and with what args,
  receives the tool output, and can iterate or finish.
"""

import argparse
import json
import os
import os.path as osp
from datetime import datetime
from typing import Any, Dict, List

from ai_scientist.llm import create_client
from ai_scientist.treesearch.bfts_utils import idea_to_markdown
from ai_scientist.tools.lit_data_assembly import LitDataAssemblyTool
from ai_scientist.tools.lit_validator import LitSummaryValidatorTool
from ai_scientist.tools.graph_builder import BuildGraphsTool
from ai_scientist.tools.compartmental_sim import RunCompartmentalSimTool
from ai_scientist.tools.sensitivity_sweep import RunSensitivitySweepTool
from ai_scientist.tools.validation_compare import RunValidationCompareTool
from ai_scientist.tools.intervention_tester import RunInterventionTesterTool
from ai_scientist.tools.biological_model import RunBiologicalModelTool
from ai_scientist.tools.biological_plotting import RunBiologicalPlottingTool
from ai_scientist.tools.biological_stats import RunBiologicalStatsTool


def parse_args():
    p = argparse.ArgumentParser(description="Tool-driven runner for theoretical comp-bio experiments")
    p.add_argument("--load_ideas", required=True, help="Path to pregenerated ideas JSON")
    p.add_argument("--idea_idx", type=int, default=0, help="Index of idea to use")
    p.add_argument("--attempt_id", type=int, default=0, help="Attempt id for naming output directory")
    p.add_argument("--model", type=str, default="gpt-5o-mini", help="LLM model identifier")
    p.add_argument("--max_steps", type=int, default=8, help="Max tool-calling iterations")
    return p.parse_args()


def build_tool_registry(exp_dir: str):
    """
    Instantiate tools and return a mapping name -> instance with sensible defaults.
    """
    return {
        "AssembleLitData": LitDataAssemblyTool(),
        "ValidateLitSummary": LitSummaryValidatorTool(),
        "BuildGraphs": BuildGraphsTool(),
        "RunCompartmentalSimulation": RunCompartmentalSimTool(),
        "RunSensitivitySweep": RunSensitivitySweepTool(),
        "RunValidationCompare": RunValidationCompareTool(),
        "RunInterventionTester": RunInterventionTesterTool(),
        "RunBiologicalModel": RunBiologicalModelTool(),
        "RunBiologicalPlotting": RunBiologicalPlottingTool(),
        "RunBiologicalStats": RunBiologicalStatsTool(),
    }


def tool_descriptions(tool_registry: Dict[str, Any]) -> str:
    lines = []
    for name, tool in tool_registry.items():
        desc = getattr(tool, "description", "")
        params = getattr(tool, "parameters", [])
        params_str = "; ".join(f"{p['name']} ({p['type']})" for p in params)
        lines.append(f"- {name}: {desc} | Params: {params_str}")
    return "\n".join(lines)


def main():
    args = parse_args()
    with open(args.load_ideas) as f:
        ideas = json.load(f)
    idea = ideas[args.idea_idx]

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    idea_dir = f"experiments/{date}_{idea['Name']}_attempt_{args.attempt_id}"
    os.makedirs(idea_dir, exist_ok=True)
    exp_results = osp.join(idea_dir, "experiment_results")
    os.makedirs(exp_results, exist_ok=True)

    # Save idea artifacts
    idea_path_json = osp.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(idea, f, indent=2)
    idea_path_md = osp.join(idea_dir, "idea.md")
    idea_to_markdown(idea, idea_path_md, code_str=idea.get("Code", ""))

    client, model_name = create_client(args.model)

    tools = build_tool_registry(exp_dir=exp_results)
    tool_desc = tool_descriptions(tools)

    system_message = (
        "You are a computational biology agent. You can call tools to execute your experimental plan. "
        "Return JSON with keys: action (UseTool|Finish), tool (when UseTool), args (dict), and a short justification. "
        "Available tools:\n" + tool_desc
    )

    history = []
    tool_logs: List[Dict[str, Any]] = []
    summary_notes: List[str] = []

    for step in range(args.max_steps):
        prompt = {
            "idea": idea,
            "notes": summary_notes[-3:],  # send last few notes
            "tool_logs_tail": tool_logs[-2:],
            "instruction": "Decide next action to advance the experiments. Use tools when helpful; finish when done. Reply in JSON.",
        }
        response, history = client.chat.completions.create(
            model=model_name,
            messages=history
            + [
                {"role": "system", "content": system_message},
                {"role": "user", "content": json.dumps(prompt, indent=2)},
            ],
        ), history

        try:
            content = response.choices[0].message.content
            action_json = json.loads(content)
        except Exception as e:
            summary_notes.append(f"Failed to parse agent response at step {step}: {e}")
            break

        action = action_json.get("action")
        if action == "Finish":
            summary_notes.append(f"Agent finished: {action_json.get('justification','')}")
            break

        if action == "UseTool":
            tool_name = action_json.get("tool")
            args_dict = action_json.get("args", {}) or {}
            if tool_name not in tools:
                summary_notes.append(f"Unknown tool {tool_name}")
                continue

            # Fill in default output_dir if missing
            if "output_dir" in tools[tool_name].parameters[0].get("name", ""):
                args_dict.setdefault("output_dir", exp_results)

            try:
                result = tools[tool_name].use_tool(**args_dict)
                log_entry = {"step": step, "tool": tool_name, "args": args_dict, "result": result}
                tool_logs.append(log_entry)
                summary_notes.append(f"{tool_name} ok: {result}")
            except Exception as e:
                log_entry = {"step": step, "tool": tool_name, "args": args_dict, "error": str(e)}
                tool_logs.append(log_entry)
                summary_notes.append(f"{tool_name} failed: {e}")
        else:
            summary_notes.append(f"Invalid action: {action}")

    # Save tool logs
    with open(osp.join(idea_dir, "tool_runs.json"), "w") as f:
        json.dump(tool_logs, f, indent=2)
    with open(osp.join(idea_dir, "tool_summary.txt"), "w") as f:
        f.write("\n".join(summary_notes))

    print("Tool run complete.")
    print(f"- Tool logs: {osp.join(idea_dir, 'tool_runs.json')}")
    print(f"- Summary:   {osp.join(idea_dir, 'tool_summary.txt')}")


if __name__ == "__main__":
    main()
