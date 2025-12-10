# pyright: reportMissingImports=false
import json
import os
import os.path as osp
from datetime import datetime
import asyncio
from typing import Any, Dict, Optional, TYPE_CHECKING

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

from agents import Runner

# --- Underlying Tool Imports ---
from ai_scientist.utils.notes import read_note_file

from ai_scientist.orchestrator.artifacts import (
    _reserve_typed_artifact_impl as _reserve_typed_artifact_impl_internal,
)

from ai_scientist.orchestrator.context import (
    parse_args,
    _bootstrap_note_links,
    _ensure_transport_readme,
    _report_capabilities,
)

from ai_scientist.orchestrator.transport import generate_run_recipe

from ai_scientist.orchestrator.hypothesis import (
    set_active_idea,
    bootstrap_hypothesis_trace,
)

from ai_scientist.orchestrator.manuscript_processor import (
    _derive_idea_from_manuscript,
    _persist_manuscript_seed,
)

from ai_scientist.orchestrator.agents import build_team

# Cached idea for hypothesis trace bootstrapping
_ACTIVE_IDEA: Optional[Dict[str, Any]] = None
# --- Canonical Artifact Types (VI-01)
# Each kind maps to a canonical subdirectory (relative to experiment_results) and a filename pattern.
# Patterns may use {placeholders} that must be provided via meta_json in reserve_typed_artifact.

# Backward-compatible alias for internal reserve helper
_reserve_typed_artifact_impl = _reserve_typed_artifact_impl_internal

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
    os.environ["AISC_ENFORCE_CLAIM_CONSISTENCY"] = str(args.enforce_claim_consistency).lower()
    
    # Load Idea
    manuscript_context: Optional[Dict[str, Any]] = None
    if args.load_manuscript:
        if args.idea_idx not in (0, None):
            print("⚠️  Ignoring --idea_idx because --load_manuscript was provided.")
        idea, manuscript_context = _derive_idea_from_manuscript(args.load_manuscript, args.manuscript_title)
        print(f"📝 Loaded manuscript seed from {args.load_manuscript}")
    else:
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

    if manuscript_context:
        persisted = _persist_manuscript_seed(manuscript_context, idea)
        manuscript_context.update(persisted)

    # Ensure standard transport_runs README exists for this run
    _ensure_transport_readme(base_folder)
    # Generate per-run run_recipe.json from morphologies/templates
    generate_run_recipe(base_folder)
    # Ensure canonical PI/user inbox files live under experiment_results with root-level symlinks
    _bootstrap_note_links()
    # Initialize hypothesis trace skeleton for this run
    try:
        set_active_idea(idea)
        bootstrap_hypothesis_trace(idea)
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

    # 4. Inject Manuscript Context if provided
    if manuscript_context:
        preview = manuscript_context.get("preview", {})
        preview_text = ""
        if isinstance(preview, dict):
            preview_text = preview.get("content") or ""
            note = preview.get("note")
        else:
            preview_text = str(preview)
            note = None
        if not preview_text:
            preview_text = manuscript_context.get("raw_text", "")[:2000]
        seed_path = manuscript_context.get("seed_idea_path", "experiment_results/seed_idea_from_manuscript.json")
        text_path = manuscript_context.get("manuscript_text_path", "experiment_results/manuscript_input.txt")
        header = (
            f"\n\n--- MANUSCRIPT SEED ({manuscript_context.get('path', '')}) ---\n"
            f"Title: {manuscript_context.get('title', idea.get('Title', 'Project'))}\n"
            f"Seed idea saved to: {seed_path}\n"
            f"Manuscript text cached at: {text_path}"
        )
        if note:
            header += f"\n{note}"
        initial_prompt_parts.append(f"{header}\n\n{preview_text}")

    # 5. Inject implementation plan if present (avoid extra tool calls on resume)
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
