import argparse
import json
import os.path as osp
import traceback
from typing import Any, Dict, List

import sys

# sys.path.append(osp.join(osp.dirname(__file__), ".."))
from ai_scientist.llm import (
    AVAILABLE_LLMS,
    create_client,
    get_response_from_llm,
    AIScientistAction,
)

from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool
from ai_scientist.tools.lit_data_assembly import LitDataAssemblyTool
from ai_scientist.tools.base_tool import BaseTool

# Define structured output format for GPT models
STRUCTURED_OUTPUT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ai_scientist_response",
        "schema": AIScientistAction.model_json_schema(),
        "strict": True,
    },
}

# Create tool instances
semantic_scholar_tool = SemanticScholarSearchTool()
lit_data_tool = LitDataAssemblyTool()

# Define tools at the top of the file
tools = [
    semantic_scholar_tool,
    lit_data_tool,
    {
        "name": "FinalizeIdea",
        "description": """Finalize your idea by providing the idea details.

The IDEA JSON should include the following fields:
- "Name": A short descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A catchy and informative title for the proposal.
- "Short Hypothesis": A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.
- "Related Work": A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.
- "Abstract": An abstract that summarizes the proposal in conference format (approximately 250 words).
- "Experiments": A list of experiments that would be conducted to validate the proposal. Ensure these are simple and feasible. Be specific in exactly how you would test the hypothesis, and detail precise algorithmic changes. Include the evaluation metrics you would use.
- "Risk Factors and Limitations": A list of potential risks and limitations of the proposal.""",
    },
]

# Create a tools dictionary for easy lookup
tools_dict = {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}

# Create a string with the tool descriptions
tool_descriptions = "\n\n".join(
    (
        f"- **{tool.name}**: {tool.description}"
        if isinstance(tool, BaseTool)
        else f"- **{tool['name']}**: {tool['description']}"
    )
    for tool in tools
)

# Extract tool names for the prompt
tool_names = [
    f'"{tool.name}"' if isinstance(tool, BaseTool) else f'"{tool["name"]}"'
    for tool in tools
]
tool_names_str = ", ".join(tool_names)

system_prompt = f"""You are an experienced computational biology researcher whose goal is to propose high-impact research ideas that could plausibly become strong papers in top computational biology, systems neuroscience, or quantitative biology venues.

Constraints and capabilities:

- You **must** work purely from:
  - Published literature and data accessible via Semantic Scholar (through the provided tools),
  - Publicly available datasets referenced in those works,
  - Your own **mathematical / computational modeling, simulations, and re-analyses**.
- You **do not** have access to:
  - Wet-lab facilities,
  - New animal or human experiments,
  - Proprietary clinical datasets,
  - Any data source beyond what can be inferred from published work accessed via Semantic Scholar.
- Any empirical component must be limited to **re-analysis, meta-analysis, or in silico experiments** built on published results.

Your task is to propose **novel, elegant, and computationally grounded** research ideas that:

- Start from a simple, sharp question, observation, or hypothesis about a biological system (e.g., disease mechanisms, cellular energetics, signaling networks, population dynamics).
- Leverage **existing publications and data** plus new modeling/analysis to challenge assumptions or reveal non-obvious structure (e.g., bifurcations, tipping points, phase transitions, scaling laws, or constraints).
- Clearly explain how the proposal **differs from and extends** existing literature.
- Are feasible for a  small, well-run **computational lab** with standard resources (CPUs, typical storage, no special hardware).

You have access to the following tools:

{tool_descriptions}

RESPOND WITH A JSON OBJECT WITH THE FOLLOWING STRUCTURE:
```json
{{
  "action": "SearchSemanticScholar" | "FinalizeIdea",
  "arguments": {{
    "query": "<search query for SearchSemanticScholar>"
  }} | {{
    "idea": {{
      "Name": "<short descriptor>",
      "Title": "<catchy and informative title>",
      "Short Hypothesis": "<concise statement>",
      "Related Work": "<brief discussion of related work>",
      "Abstract": "<conference format abstract>",
      "Experiments": "<list of experiments>",
      "Risk Factors and Limitations": "<list of risks and limitations>"
    }}
  }}
}}
```

IMPORTANT: Always respond with valid JSON that matches this exact structure. Do not include any explanatory text outside the JSON."""

# Define the initial idea generation prompt
idea_generation_prompt = """{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Begin by generating a **new** high-level computational biology research proposal that:
- Differs meaningfully from what you previously proposed,
- Can be executed with **literature-derived information + modeling/simulation only**,
- Centers on a clear, elegant hypothesis about a biological system (e.g., neuronal energetics, disease tipping points, signaling networks, evolutionary dynamics),
- And explicitly leverages **published data or reported summary statistics** together with new modeling.
"""

# Define the reflection prompt
idea_reflection_prompt = """Round {current_round}/{num_reflections}.

In your thoughts, carefully evaluate the **computational biology** proposal you just created:

- Is the core hypothesis clear, sharp, and testable **using only published data + new modeling/simulation**?
- Is the idea genuinely **novel** relative to the literature you have seen (via Semantic Scholar)?
- Is it feasible for a small computational lab (no wet-lab, no proprietary datasets)?
- Are the proposed “Experiments” strictly computational (re-analysis, meta-analysis, simulations, model fitting, parameter sweeps, bifurcation analysis, etc.)?
- Are the idea name, title, and abstract concise and coherent?
- Is the IDEA JSON correctly formatted?

If there are weaknesses (e.g., unclear novelty, missing data sources, unrealistic modeling requirements), note them explicitly and plan how to fix them in the next attempt.
Do not make things unnecessarily complicated; prefer **simple, elegant models and questions** with high explanatory power.

If you have new information from tools (e.g., Semantic Scholar search results), incorporate it into your reflection and refine the proposal accordingly.

Results from your last action (if any):

{last_tool_results}
"""


def generate_temp_free_idea(
    idea_fname: str,
    client: Any,
    model: str,
    workshop_description: str,
    max_num_generations: int = 20,
    num_reflections: int = 5,
    reload_ideas: bool = True,
) -> List[Dict]:
    idea_str_archive = []
    # load ideas from file
    if reload_ideas and osp.exists(idea_fname):
        with open(idea_fname, "r") as f:
            idea_str_content = json.load(f)
            for idea in idea_str_content:
                idea_str_archive.append(json.dumps(idea))
            print(f"Loaded {len(idea_str_archive)} ideas from {idea_fname}")
    else:
        print(f"No ideas found in {idea_fname}. Starting from scratch.")

    for gen_idx in range(max_num_generations):
        print()
        print(f"Generating proposal {gen_idx + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            last_tool_results = ""
            idea_finalized = False
            msg_history = []

            for reflection_round in range(num_reflections):
                if reflection_round == 0:
                    # Use the initial idea generation prompt
                    prompt_text = idea_generation_prompt.format(
                        workshop_description=workshop_description,
                        prev_ideas_string=prev_ideas_string,
                    )
                else:
                    # Use the reflection prompt, including tool results if any
                    prompt_text = idea_reflection_prompt.format(
                        current_round=reflection_round + 1,
                        num_reflections=num_reflections,
                        last_tool_results=last_tool_results or "No new results.",
                    )

                response_text, msg_history = get_response_from_llm(
                    prompt=prompt_text,
                    client=client,
                    model=model,
                    system_message=system_prompt,
                    msg_history=msg_history,
                    response_format=STRUCTURED_OUTPUT_FORMAT
                    if "gpt" in model
                    else None,
                )

                # Parse the structured JSON response
                response_json = None
                try:
                    # Parse the JSON response
                    response_json = json.loads(response_text.strip())

                    # Validate the response structure
                    action = response_json["action"]
                    arguments = response_json["arguments"]
                    print(f"Action: {action}")
                    print(f"Arguments: {arguments}")

                    # Process the action and arguments
                    if action in tools_dict:
                        # It's a tool we have defined
                        tool = tools_dict[action]
                        # Use the tool with arguments
                        try:
                            result = tool.use_tool(**arguments)
                            last_tool_results = result
                        except Exception as e:
                            last_tool_results = f"Error using tool {action}: {str(e)}"
                    elif action == "FinalizeIdea":
                        # Get the idea from arguments
                        idea = arguments.get("idea")
                        if not idea:
                            raise ValueError("Missing 'idea' in arguments.")

                        # Append the idea to the archive
                        idea_str_archive.append(json.dumps(idea))
                        print(f"Proposal finalized: {idea}")
                        idea_finalized = True
                        break
                    else:
                        print(f"Invalid action: {action}")
                        print(f"Available actions are: {tool_names_str}")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON response: {e}")
                    print(f"Response text was:\n{response_text}")
                    break
                except KeyError as e:
                    print(f"Missing required field in response: {e}")
                    print(f"Response JSON was: {response_json}")
                    break
                except Exception as e:
                    print(f"Failed to process LLM response: {e}")
                    traceback.print_exc()
                    break  # Exit the loop if parsing fails

            if idea_finalized:
                continue  # Move to the next idea

        except Exception:
            print("Failed to generate proposal:")
            traceback.print_exc()
            continue

    # Save ideas
    ideas = [json.loads(idea_str) for idea_str in idea_str_archive]

    with open(idea_fname, "w") as f:
        json.dump(ideas, f, indent=4)
    print(f"Stored {len(ideas)} ideas in {idea_fname}")
    return ideas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI scientist proposals - template free"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--max-num-generations",
        type=int,
        default=1,
        help="Maximum number of proposal generations.",
    )
    parser.add_argument(
        "--workshop-file",
        type=str,
        default="ideas/i_cant_believe_its_not_better.md",
        help="Path to the workshop description file.",
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=5,
        help="Number of reflection rounds per proposal.",
    )
    args = parser.parse_args()

    # Create the LLM client
    client, client_model = create_client(args.model)

    if osp.exists(args.workshop_file):
        with open(args.workshop_file, "r") as f:
            workshop_description = f.read()
    else:
        # Check relative to top-level if not found
        top_level_path = osp.join(osp.dirname(__file__), "../..", args.workshop_file)
        if osp.exists(top_level_path):
            with open(top_level_path, "r") as f:
                workshop_description = f.read()
        else:
            print(f"Could not find workshop file: {args.workshop_file}")
            sys.exit(1)

    print(f"Using workshop description from {args.workshop_file} for idea generation.")
    print(f"Workshop description:\n{workshop_description}")

    # Create output filename by replacing .md extension with .json
    idea_fname = args.workshop_file.replace(".md", ".json")
    print("Starting idea generation for", idea_fname)
    ideas = generate_temp_free_idea(
        idea_fname=idea_fname,
        client=client,
        model=client_model,
        workshop_description=workshop_description,
        max_num_generations=args.max_num_generations,
        num_reflections=args.num_reflections,
    )
    print(f"{args.workshop_file} generated {len(ideas)} ideas.")
