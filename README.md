Fully autonomous scientific research systems are becoming increasingly capable, with AI playing a pivotal role in transforming how scientific discoveries are made.
We are excited to introduce The AI Scientist-v2, a generalized end-to-end agentic system that has generated the first workshop paper written entirely by AI and accepted through peer review.

This system autonomously generates hypotheses, runs experiments, analyzes data, and writes scientific manuscripts. Unlike [its predecessor (AI Scientist-v1)](https://github.com/SakanaAI/AI-Scientist), the AI Scientist-v2 removes reliance on human-authored templates, generalizes across Machine Learning (ML) domains, and employs a progressive agentic tree search, guided by an experiment manager agent.

> **Note:**
> The AI Scientist-v2 doesn’t necessarily produce better papers than v1, especially when a strong starting template is available. v1 follows well-defined templates, leading to high success rates, while v2 takes a broader, more exploratory approach with lower success rates. v1 works best for tasks with clear objectives and a solid foundation, whereas v2 is designed for open-ended scientific exploration.

## Table of Contents

1.  [Requirements](#requirements)
    *   [Installation](#installation)
    *   [Supported Models and API Keys](#supported-models-and-api-keys)
2.  [Generate Research Ideas](#generate-research-ideas)
3.  [Run AI Scientist-v2 Paper Generation Experiments](#run-ai-scientist-v2-paper-generation-experiments)
4.  [Citing The AI Scientist-v2](#citing-the-ai-scientist-v2)
5.  [Frequently Asked Questions](#frequently-asked-questions)
6.  [Acknowledgement](#acknowledgement)

## Requirements

This code is designed to run on Linux with NVIDIA GPUs using CUDA and PyTorch.

### Installation

```bash
# Create a new conda environment
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Install PyTorch with CUDA support (adjust pytorch-cuda version for your setup)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PDF and LaTeX tools
conda install anaconda::poppler
conda install conda-forge::chktex

# Install Python package requirements
pip install -r requirements.txt

# If you upgrade NetworkX to >=3.5, gpickle helpers fall back to plain pickle
# loading (NetworkX removed read_gpickle). No extra action is required.
# If matplotlib emits cache-permissions warnings on macOS, set MPLCONFIGDIR to a
# writable folder (e.g., export MPLCONFIGDIR=$HOME/.config/matplotlib).
```

Installation usually takes no more than one hour.

### Persistent PI/User Notes

During runs, agents write progress summaries to `pi_notes.md` and user feedback to `user_inbox.md`. These files live canonically under `experiment_results/`; the orchestrator maintains root-level symlinks or copies for convenience. Avoid writing them manually—use the orchestrator tools so paths stay consistent across resumes.

### Typed Artifacts & Health Checks

Outputs are routed through typed helpers so agents cannot invent paths. Use `reserve_typed_artifact` / `reserve_and_register_artifact` (preferred) plus `list_artifacts_by_kind` to stay within the canonical registry (figures, sims, lit summaries, parameter sets, etc.). Notes and reflections belong in `run_notes.md` or `project_knowledge.md`, never in the manifest. A manifest health checker is available via:

```bash
python -m ai_scientist.lab_tools.check_run_health --base-folder <run_root>
```

It validates registry naming, one-row-per-path manifest entries, missing files, and uncatalogued artifacts (excluding scratch dirs).

### Output Pathing & Health Reports

All file writes now flow through sanitized helpers (`reserve_output`, `write_text_artifact`, resolver-backed tool wrappers) rather than manual joins. Paths reject traversal, auto-unique on collisions, and fall back to `experiment_results/_unrouted/` if the primary location is unavailable, with a note recorded. Simulation, plotting, graph-building, and lit-assembly tools use the same layer. `check_manifest` also emits a health report under `experiment_results/_health/verification_missing_report_post_run.json` when it detects missing/duplicate entries so runs degrade gracefully instead of halting.

### Supported Models and API Keys

#### OpenAI Models

By default, the system uses the `OPENAI_API_KEY` environment variable for OpenAI models.

#### Gemini Models

By default, the system uses the `GEMINI_API_KEY` environment variable for Gemini models through OpenAI API.


#### Semantic Scholar API (Literature Search)

Our code can optionally use a Semantic Scholar API Key (`S2_API_KEY`) for higher throughput during literature search [if you have one](https://www.semanticscholar.org/product/api). This is used during both the ideation and paper writing stages. The system should work without it, though you might encounter rate limits or reduced novelty checking during ideation. If you experience issues with Semantic Scholar, you can skip the citation phase during paper generation.

#### Setting API Keys

Ensure you provide the necessary API keys as environment variables for the models you intend to use. For example:
```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export S2_API_KEY="YOUR_S2_KEY_HERE"

```
You can also place these in a local `.env` file (loaded automatically if `python-dotenv` is installed), for example:
```
OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE
```

## Generate Research Ideas & Run Pipelines

Before running the full AI Scientist-v2 experiment pipeline, you first use the `ai_scientist/perform_ideation_temp_free.py` script to generate potential research ideas. This script uses an LLM to brainstorm and refine ideas based on a high-level topic description you provide, interacting with tools like Semantic Scholar to check for novelty.

1.  **Prepare a Topic Description:** Create a Markdown file (e.g., `my_research_topic.md`) describing the research area or theme you want the AI to explore. This file should contain sections like `Title`, `Keywords`, `TL;DR`, and `Abstract` to define the scope of the research. Refer to the example file `ai_scientist/ideas/i_cant_believe_its_not_better.md` for the expected structure and content format. Place your file in a location accessible by the script (e.g., the `ai_scientist/ideas/` directory).

2.  **Run the Ideation Script:** Execute the script from the main project directory, pointing it to your topic description file and specifying the desired LLM.

    ```bash
    python ai_scientist/perform_ideation_temp_free.py \
     --workshop-file "ai_scientist/ideas/my_research_topic.md" \
     --model gpt-5o-mini \
     --max-num-generations 20 \
     --num-reflections 5
    ```
    *   `--workshop-file`: Path to your topic description Markdown file.
    *   `--model`: The LLM to use for generating ideas (ensure you have the corresponding API key set).
    *   `--max-num-generations`: How many distinct research ideas to attempt generating.
    *   `--num-reflections`: How many refinement steps the LLM should perform for each idea.

3.  **Output:** The script will generate a JSON file named after your input Markdown file (e.g., `ai_scientist/ideas/my_research_topic.json`). This file will contain a list of structured research ideas, including hypotheses, proposed experiments, and related work analysis.

4.  **Proceed to Experiments:** Once you have the generated JSON file containing research ideas, you can proceed to the next section to run the experiments.

This ideation step guides the AI Scientist towards specific areas of interest and produces concrete research directions to be tested in the main experimental pipeline.


## Tool-Driven Orchestration (Agents)

For a tool-driven, multi-agent workflow (PI + Archivist/Modeler/Analyst/Interpreter/Reviewer/Publisher), use `agents_orchestrator.py`:

* Delegates tasks via tools/handoffs until the reviewer reports no gaps and a PDF exists.
* Uses idea context (Title/Hypothesis/Abstract/Experiments/Risks) and targets the `blank_theoretical_biology_latex` template by default.
* Enforces output conventions (artifacts in `experiment_results/`, figures in `figures/` when aggregated, PDFs at run root) and structured status files.
* Tool highlights: Archivist (`AssembleLitData`, `ValidateLitSummary`, `SearchSemanticScholar`, `VerifyReferences`, `CheckLitReady`, `UpdateClaimGraph`), Modeler (`BuildGraphs`, `RunBiologicalModel`, `RunCompartmentalSimulation`, `RunSensitivitySweep`, `RunInterventionTester`), Analyst (`RunBiologicalPlotting`, `RunValidationCompare`, `RunBiologicalStats`), Interpreter (`interpret_biological_results` wrapper for theoretical runs), Reviewer (`ReadManuscript`, `CheckClaimGraph`, `RunBiologicalStats`), Coder (`coder_create_python`, `run_ruff`, `run_pyright` for quick lint/type checks), plus shared helpers (`get_run_paths`, `resolve_path`, `list_artifacts`, `read_artifact` w/ summary mode, `read_npy_artifact` with summary-first caps + slice support for `.npy` loads, `reserve_output` with sanitized/auto-unique/quarantine pathing, `write_text_artifact` + conveniences, `append_manifest`/`read_manifest`/`read_manifest_entry`/`check_manifest`, `check_status`). Graph-based tools expect a file path (not a directory) and accept `.gpickle`, `.graphml`, `.gml`, `.npz`, or `.npy` via the shared loader. Manifest is path-keyed; use `read_manifest_entry`/`check_manifest` to inspect before logging new artifacts.
* Simulation standardization: all sim runs must emit `per_compartment.npz` (`binary_states`, `continuous_states`, `time`) plus `node_index_map.json` and `topology_summary.json` (schema v1.0) with an ordering checksum of the morphology IDs; `validate_per_compartment_outputs` enforces shape/time alignment and checksum agreement before a run is marked complete or used for plotting. If arrays are missing from legacy sim.json files, use `repair_sim_outputs` (tool or CLI `python ai_scientist/perform_repair_sim_outputs.py --manifest <path>`) to bulk run `sim_postprocess`, validate per-compartment outputs, and update the manifest/tool_summary safely (lock-aware, idempotent).
* Manifest shape: `path` is the key, `name` is just the basename (no directories), annotations hold the descriptive fields, and legacy `metadata` only carries `{"type": ...}` for compatibility—avoid duplicating annotation content there.
* Robust PI visibility: when sub-agents (Modeler/Analyst/etc.) hit `max_turns` or return sparse text, the orchestrator now surfaces their final message plus a summary of tool calls (`tools_called: ...`) so the PI sees what actually ran.
* Awareness of plot aggregation (`perform_plotting.py`), modeling/stats utilities (`perform_biological_modeling.py`, `perform_biological_stats.py`), interpretation (`perform_biological_interpretation.py`), manuscript reader tool, and alternative templates (`blank_bioinformatics_latex`, `blank_icbinb_latex`).
* Traceability & provenance:
  * Hypothesis→Experiment→Artifact map is written to `experiment_results/hypothesis_trace.json` (agents can update via `update_hypothesis_trace`; sim/plot helpers accept optional hypothesis/experiment ids).
  * Metrics: `compute_model_metrics` aggregates sweeps/model outputs into `experiment_results/simulations/{label}_metrics.csv` and `experiment_results/models/{model_key}_metrics.json`, so figures/text can reference named metrics (`critical_transport_est`, `mean_frac_failed`, etc.) instead of raw CSVs.
  * Manuscript-ready provenance summary: `generate_provenance_summary` compiles literature/model/spec/sim/stats artifacts into `experiment_results/provenance_summary.md`; Reviewer regenerates it if missing.
* Literature gate (publishable runs): `check_lit_ready` enforces that lit_summary coverage and reference verification meet minimum thresholds (defaults: confirmed ≥ 70%, at most 3 unverified). Modeling/sim tools enforce the gate unless `--skip_lit_gate` is passed or `AISC_SKIP_LIT_GATE=true`. Thresholds can be tuned via `AISC_LIT_GATE_CONFIRMED_THRESHOLD` and `AISC_LIT_GATE_MAX_UNVERIFIED`. Gate outcomes are logged to `project_knowledge.md` and reflected in `provenance_summary.md` under Literature.

Typical invocation:
```bash
python agents_orchestrator.py \
  --load_idea ai_scientist/ideas/my_research_topic.json \
  --idea_idx 0 \
  --model gpt-4o-mini \
  --max_cycles 25 \
  --base_folder experiments/20251121_1801_axonal_arbor_percolation  # optional: restart from existing folder
```

### Computational biology (theoretical modeling) quick start

Local checks for code changes (run from repo root):
```bash
ruff check .
pyright
```

Once the initial experimental stage is complete, you will find a timestamped log folder inside the `experiments/` directory. Navigate to `experiments/"timestamp_ideaname"/logs/0-run/` within that folder to find the tree visualization file `unified_tree_viz.html`.
After all experiment stages are complete, the writeup stage begins. The writeup stage typically takes about 20 to 30 minutes in total. Once it finishes, you should see `timestamp_ideaname.pdf` in the `timestamp_ideaname` folder.
For this example run, all stages typically finish within several hours.

## Citing The AI Scientist-v2

If you use **The AI Scientist-v2** in your research, please cite our work as follows:

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2025}
}
```

## Frequently Asked Questions

**Why wasn't a PDF or a review generated for my experiment?**

The AI Scientist-v2 completes experiments with a success rate that depends on the chosen foundation model, and the complexity of the idea. Higher success rates are generally observed when using powerful models like Claude 3.5 Sonnet for the experimentation phase.

**What is the estimated cost per experiment?**

The ideation step cost depends on the LLM used and the number of generations/reflections, but is generally low (a few dollars). For the main experiment pipeline, using Claude 3.5 Sonnet for the experimentation phase typically costs around $15–$20 per run. The subsequent writing phase adds approximately $5 when using the default models specified in the example command. Using GPT-4o for `model_citation` is recommended as it can help reduce writing costs.

**How do I run The AI Scientist-v2 for different subject fields?**

First, perform the [Generate Research Ideas](#generate-research-ideas) step. Create a new Markdown file describing your desired subject field or topic, following the structure of the example `ai_scientist/ideas/i_cant_believe_its_not_better.md`. Run the `perform_ideation_temp_free.py` script with this file to generate a corresponding JSON idea file. Then, proceed to the [Run AI Scientist-v2 Paper Generation Experiments](#run-ai-scientist-v2-paper-generation-experiments) step, using this JSON file with the `launch_scientist_bfts.py` script via the `--load_ideas` argument.

**What should I do if I have problems accessing the Semantic Scholar API?**

The Semantic Scholar API is used to assess the novelty of generated ideas and to gather citations during the paper write-up phase. If you don't have an API key, encounter rate limits, you may be able to skip these phases.
