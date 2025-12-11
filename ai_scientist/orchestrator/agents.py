# pyright: reportMissingImports=false
import os
from typing import Any, Dict, List

try:
    from agents.types import RunResult
except ImportError:
    class RunResult:  # minimal stub
        def __init__(self, output=None, error=None, status=None):
            self.output = output
            self.error = error
            self.status = status

from agents import Agent, ModelSettings

from ai_scientist.orchestrator.artifacts import _artifact_kind_catalog
from ai_scientist.orchestrator.context_specs import (
    format_context_spec_for_prompt,
    get_context_view_spec,
    get_module_for_role,
)
from ai_scientist.orchestrator.tool_wrappers import (
    append_manifest,
    append_run_note_tool,
    assemble_lit_data,
    build_graphs,
    check_claim_graph,
    check_manifest,
    check_manifest_unique_paths,
    check_project_state,
    check_status,
    check_user_inbox,
    coder_create_python,
    compute_model_metrics,
    generate_provenance_summary,
    graph_diagnostics,
    get_artifact_index,
    get_run_paths,
    head_artifact,
    inspect_manifest,
    inspect_recent_manifest_entries,
    interpret_biology,
    list_artifacts,
    list_artifacts_by_kind,
    log_strategic_pivot,
    manage_project_knowledge,
    mirror_artifacts,
    promote_artifact_to_canonical,
    check_dependency_staleness,
    generate_project_snapshot,
    read_artifact,
    read_manifest,
    read_manifest_entry,
    read_manuscript,
    read_npy_artifact,
    read_note,
    read_transport_manifest,
    reserve_and_register_artifact,
    reserve_output,
    reserve_typed_artifact,
    resolve_baseline_path,
    resolve_path,
    resolve_sim_path,
    run_biological_model,
    run_biological_plotting,
    run_biological_stats,
    run_comp_sim,
    run_intervention_tests,
    run_pyright,
    run_ruff,
    run_sensitivity_sweep,
    run_transport_batch,
    run_validation_compare,
    run_writeup_task,
    search_semantic_scholar,
    scan_transport_manifest,
    sim_postprocess,
    summarize_artifact,
    ensure_module_summary,
    update_claim_graph,
    update_hypothesis_trace,
    update_transport_manifest,
    validate_lit_summary,
    validate_per_compartment_outputs,
    verify_references,
    wait_for_human_review,
    write_figures_readme,
    write_interpretation_text,
    write_pi_notes,
    write_text_artifact,
    format_list_field,
    create_lit_summary_artifact,
    create_claim_graph_artifact,
    list_lit_summaries,
    list_claim_graphs,
    read_archivist_artifact,
    create_transport_artifact,
    create_sensitivity_table_artifact,
    create_intervention_table_artifact,
    create_verification_note_artifact,
    list_model_specs,
    get_latest_model_spec,
    list_experiment_results,
    get_latest_metrics,
    read_model_spec,
    create_model_spec_artifact,
    read_experiment_config,
    read_metrics,
    create_plot_artifact,
    publish_figure_to_manuscript_gallery,
    list_available_runs_for_plotting,
    get_metrics_for_plotting,
    create_review_note_artifact,
    check_parameter_sources_for_manuscript,
    check_metrics_for_referenced_models,
    check_hypothesis_trace_consistency,
    check_proof_of_work_for_results,
    get_lit_reference_verification,
    check_references_completeness,
)
from ai_scientist.orchestrator.lit_tools import (
    create_lit_review_artifact,
    create_lit_bibliography_artifact,
    create_lit_coverage_artifact,
    create_lit_integration_memo_artifact,
)


def _make_agent(name: str, instructions: str, tools: List[Any], model: str, settings: ModelSettings) -> Agent:
    return Agent(name=name, instructions=instructions, model=model, tools=tools, model_settings=settings)


async def extract_run_output(run_result: RunResult) -> str:
    parts: List[str] = []

    def get_attr(obj: Any, attr: str) -> Any:
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


def build_team(model: str, idea: Dict[str, Any], dirs: Dict[str, str]) -> Agent:
    artifact_catalog = _artifact_kind_catalog()
    common_settings = ModelSettings(tool_choice="auto")
    role_max_turns = 40
    title = idea.get('Title', 'Project')
    abstract = idea.get('Abstract', '')
    hypothesis = idea.get('Short Hypothesis', 'None')
    related_work = idea.get('Related Work', 'None provided.')

    experiments_plan = format_list_field(idea.get('Experiments', []))
    risk_factors = format_list_field(idea.get('Risk Factors and Limitations', []))

    path_context = (
        f"SYSTEM CONTEXT: Run Root='{dirs['base']}', Exp Results='{dirs['results']}'. "
        f"Figures='{os.path.join(dirs['base'], 'figures')}'. "
        "Use these paths directly; do NOT call get_run_paths. "
        "Assume provided input paths exist; only list_artifacts if path is missing."
    )
    path_guardrails = (
        "FILE IO POLICY: Every persistent artifact must be reserved via 'reserve_typed_artifact(kind=..., meta_json=...)' using the registry below; do NOT invent filenames or bypass the registry. "
        f"Known kinds: {artifact_catalog}. "
        "Refer to docs/artifact_metadata_requirements.md for the required metadata fields every artifact must carry. "
        "Preferred flow: 'reserve_and_register_artifact' -> write -> (optional) update status via append_manifest. "
        "Use 'reserve_output' only for PI/Coder scratch logs; other roles must stay within typed helpers. When writing text, pass the reserved path into write_text_artifact instead of freehand names. "
        "Outputs are anchored to experiment_results; if a directory is unavailable, writes are auto-rerouted to experiment_results/_unrouted with a manifest note. "
        "NEVER log reflections or notes to the manifest—use append_run_note or manage_project_knowledge instead. "
        "Prefer 'summarize_artifact' to collect condensed views and call 'ensure_module_summary' for the relevant module before requesting full content."
    )
    metadata_reminder = (
        "METADATA REMINDER: When calling 'reserve_typed_artifact' or 'reserve_and_register_artifact', pass 'meta_json' including "
        "`id`, `type`, `version`, `parent_version`, `status`, `module`, `summary`, `content`, and `metadata` (see docs/artifact_metadata_requirements.md)."
    )
    def _context_spec_intro(role_name: str) -> str:
        spec = get_context_view_spec(role_name)
        if spec is None:
            raise ValueError(f"Missing context view spec for role: {role_name}")
        return format_context_spec_for_prompt(spec)

    def _summary_advisory(role_name: str) -> str:
        module = get_module_for_role(role_name)
        base_text = (
            "SUMMARY STRATEGY: Use 'summarize_artifact' to gather condensed context first and prefer the resulting summary."
        )
        if module:
            base_text += (
                f" Before ingesting raw content for {role_name}, verify the latest module memo via "
                f"'ensure_module_summary(module=\"{module}\")'. Proceed with full content only when the spec explicitly needs it."
            )
        return base_text
    reflection_instruction = (
        "SELF-REFLECTION: When finished (or if stuck), ask: 'What missing tool or knowledge would have made this trivial?' "
        "If you have a concrete, new insight, log it via manage_project_knowledge(action='add', category='reflection', "
        "observation='<your specific friction>', solution='<your specific fix>', actor='<your role name>'). "
        "Do NOT log boilerplate or repeated reflections; skip logging if nothing new. Use your actual role name (e.g., 'PI', 'Modeler')."
    )

    proof_of_work_instruction = (
        "PROOF OF WORK: For every significant result (data or figure), you must write a corresponding "
        "`_verification.md` file. This file must list: 1) Input files used, 2) Key parameters/filters applied, "
        "3) Explicit validation checks (e.g., 'Checked x > 0: Pass'). Do not output the artifact without this proof."
    )

    archivist = _make_agent(
        name="Archivist",
        instructions=(
            f"You are an expert Literature Curator.\n"
            f"Goal: Verify novelty of '{title}' and map claims to citations.\n"
            f"Context: {abstract}\n"
            f"Related Work to Consider: {related_work}\n"
            f"{path_context}\n{path_guardrails}\n{metadata_reminder}\n{_context_spec_intro('Archivist')}\n{_summary_advisory('Archivist')}\n"
            "Directives:\n"
            "1. Use 'assemble_lit_data' or 'search_semantic_scholar' to gather papers.\n"
            "2. Maintain a claim graph via 'update_claim_graph' when mapping evidence.\n"
            "3. Use 'create_lit_summary_artifact(module=\"lit\")' and 'create_claim_graph_artifact(module=\"lit\")' to create new artifacts. Use 'create_lit_review_artifact', 'create_lit_bibliography_artifact', 'create_lit_coverage_artifact', 'create_lit_integration_memo_artifact' to persist respective outputs. Do NOT use generic reserve calls.\n"
            "4. Immediately run 'verify_references' on lit_summary to produce lit_reference_verification.csv/json. Treat this as REQUIRED provenance.\n"
            "5. Reject readiness if more than 20% of references are missing (found==False) or any match_score < 0.5; report FAILURE with counts.\n"
            "6. If verification repeatedly fails for a venue/source, log a reflection via manage_project_knowledge with the specific venue.\n"
            "7. CRITICAL: If no papers are found, report FAILURE. Do not invent 'TBD' citations.\n"
            "8. Log reflections to run_notes via 'append_run_note_tool' or manage_project_knowledge; never to manifest.\n"
            f"9. {reflection_instruction}"
        ),
        tools=[
            assemble_lit_data,
            validate_lit_summary,
            verify_references,
            search_semantic_scholar,
            update_claim_graph,
            manage_project_knowledge,
            append_run_note_tool,
            create_lit_summary_artifact,
            create_claim_graph_artifact,
            create_lit_review_artifact,
            create_lit_bibliography_artifact,
            create_lit_coverage_artifact,
            create_lit_integration_memo_artifact,
            list_lit_summaries,
            list_claim_graphs,
            read_archivist_artifact,
        ],
        model=model,
        settings=common_settings,
    )

    modeler = _make_agent(
        name="Modeler",
        instructions=(
            f"You are an expert Computational Biologist.\n"
            f"Goal: Execute simulations for '{title}'.\n"
            f"Hypothesis: {hypothesis}\n"
            f"Experimental Plan:\n{experiments_plan}\n"
            f"{path_context}\n{path_guardrails}\n{metadata_reminder}\n{_context_spec_intro('Modeler')}\n{_summary_advisory('Modeler')}\n"
            "Directives:\n"
            "1. You do NOT care about LaTeX or writing styles. Focus on DATA.\n"
            "2. Build graphs ('build_graphs'), run baselines ('run_biological_model') or custom sims ('run_comp_sim').\n"
            "3. Explore parameter space using 'run_sensitivity_sweep' and 'run_intervention_tests'.\n"
            "3b. Before first sim of a given model_key, call 'create_model_spec_artifact' to register the model specification. Update the ledger if you change parameters; runs missing rows are a hard failure.\n"
            "3c. Update hypothesis_trace.json after each sim/ensemble: record hypothesis_id/experiment_id, sim run identifiers, and metrics produced.\n"
            "3d. After sweeps or transport batches, call 'compute_model_metrics' to emit metrics. You can use 'get_latest_metrics' to verify they exist; rerun if metrics are missing when figures/text depend on them.\n"
            "4. Ensure parameter sweeps cover the range specified in the hypothesis.\n"
            "5. Save raw outputs to experiment_results/.\n"
            "5b. Use specialized helpers ('create_transport_artifact', 'create_sensitivity_table_artifact', 'create_intervention_table_artifact', 'create_verification_note_artifact') to reserve artifacts. Do NOT use generic reserve tools.\n"
            "6. Always produce arrays for each (baseline, transport, seed): prefer export_arrays during sim; otherwise immediately run 'sim_postprocess' on the produced sim.json so failure_matrix.npy/time_vector.npy/nodes_order_*.txt exist before marking the run complete. Every run must also emit per_compartment.npz + node_index_map.json + topology_summary.json (binary_states, continuous_states/time); validate with validate_per_compartment_outputs before marking status=complete.\n"
            "7. Use the transport run manifest (read_transport_manifest / update_transport_manifest); consult it before reruns and update it after completing or failing a run. Do not mark status=complete unless arrays + sim.json + sim.status.json all exist; otherwise mark partial and note missing files.\n"
            "7b. Resolve baselines via resolve_baseline_path before running batches; only pass graph baselines (.npy/.npz/.graphml/.gpickle/.gml), never sim.json.\n"
            "7c. Process one baseline per call; if run_recipe.json exists under experiment_results/simulations/transport_runs, load it for template/output roots instead of embedding long paths. Append ensemble CSV incrementally and write per-baseline status to pi_notes/user_inbox.\n"
            "8. If you encounter simulation failures or parameter issues, log them to Project Knowledge via 'manage_project_knowledge'.\n"
            f"9. {proof_of_work_instruction}\n"
            "10. Log reflections to run_notes via 'append_run_note_tool' or manage_project_knowledge; never to manifest.\n"
            f"11. {reflection_instruction}"
        ),
        tools=[
            list_model_specs,
            get_latest_model_spec,
            list_experiment_results,
            get_latest_metrics,
            create_transport_artifact,
            create_sensitivity_table_artifact,
            create_intervention_table_artifact,
            create_verification_note_artifact,
            create_model_spec_artifact,
            read_model_spec,
            read_experiment_config,
            read_metrics,
            read_artifact,
            build_graphs,
            run_biological_model,
            run_comp_sim,
            sim_postprocess,
            run_sensitivity_sweep,
            run_intervention_tests,
            run_transport_batch,
            read_transport_manifest,
            resolve_baseline_path,
            resolve_sim_path,
            update_transport_manifest,
            update_hypothesis_trace,
            compute_model_metrics,
            read_npy_artifact,
            validate_per_compartment_outputs,
            manage_project_knowledge,
            write_text_artifact,
        ],
        model=model,
        settings=common_settings,
    )

    analyst = _make_agent(
        name="Scientific Visualization Expert",
        instructions=(
            "You are an expert Scientific Visualization Expert.\n"
            "Goal: Convert simulation data into PLOS-quality figures.\n"
            f"{path_context}\n{path_guardrails}\n{metadata_reminder}\n{_context_spec_intro('Analyst')}\n{_summary_advisory('Analyst')}\n"
            "Directives:\n"
            "1. Read data from provided input paths. Do NOT list files to find them; assume the path is correct.\n"
            "2. Assert that the data supports the hypothesis BEFORE plotting. If data contradicts hypothesis, report this back immediately.\n"
            "3. Generate PNG/SVG files using 'run_biological_plotting'. Use 'sim_postprocess' if you need failure_matrix/time_vector/node order from sim.json before plotting.\n"
            "3b. Use 'list_available_runs_for_plotting' to see what completed runs are available. Resolve sim.json via resolve_sim_path if needed.\n"
            "3c. When a figure is final, call 'publish_figure_to_manuscript_gallery(artifact_id=...)' to mirror it to the manuscript folder. Do NOT manually copy/move files.\n"
            "3d. Before computing cluster/finite-size metrics, run validate_per_compartment_outputs on the sim folder; if per_compartment artifacts are missing or invalid, report and request rerun instead of plotting placeholders.\n"
            "3e. Update hypothesis_trace.json with figure filenames under the correct hypothesis/experiment after plotting.\n"
            "3f. Use 'get_metrics_for_plotting' to find pre-computed metrics. Do NOT run compute_model_metrics yourself.\n"
            "3g. Call 'create_plot_artifact' for any new figure (plot_intermediate/manuscript_figure_png/svg). Do not call reserve_typed_artifact directly for plots.\n"
            "4. Validate models vs lit via 'run_validation_compare' and use 'run_biological_stats' for significance/enrichment.\n"
            "5. When calling 'create_plot_artifact', provide a 'change_summary' if you are updating an existing artifact or creating a significant new version. Explain WHAT changed and WHY.\n"
            "6. Check Project Knowledge for visualization standards (e.g., colormaps) before starting.\n"
            "7. When plots are ready, confirm provenance_summary.md exists or ask Reviewer to generate it.\n"
            f"7. {proof_of_work_instruction}\n"
            "8. Log reflections to run_notes via 'append_run_note_tool' or manage_project_knowledge; never to manifest.\n"
            f"9. {reflection_instruction}"
        ),
        tools=[
            list_available_runs_for_plotting,
            get_metrics_for_plotting,
            create_plot_artifact,
            publish_figure_to_manuscript_gallery,
            create_verification_note_artifact,
            read_artifact,
            summarize_artifact,
            read_transport_manifest,
            resolve_sim_path,
            update_transport_manifest,
            update_hypothesis_trace,
            run_biological_plotting,
            run_validation_compare,
            run_biological_stats,
            sim_postprocess,
            update_transport_manifest, # Duplicate? read_transport_manifest is there. update is needed for plot status/figure linkage? User said: "update_transport_manifest (only for plot status / figure linkage)" - wait, user said "Keep directly: update_transport_manifest". I will keep it.
            write_figures_readme,
            write_text_artifact,
            graph_diagnostics,
            read_npy_artifact,
            validate_per_compartment_outputs,
            manage_project_knowledge,
        ],
        model=model,
        settings=common_settings,
    )

    reviewer = _make_agent(
        name="Reviewer",
        instructions=(
            "You are an expert Holistic Reviewer.\n"
            "Goal: Identify logical gaps and structural flaws.\n"
            f"Risk Factors & Limitations to Check:\n{risk_factors}\n"
            f"{path_context}\n{path_guardrails}\n{metadata_reminder}\n{_context_spec_intro('Reviewer')}\n{_summary_advisory('Reviewer')}\n"
            "Directives:\n"
            "1. Read the manuscript draft using 'read_manuscript'.\n"
            "2. Check claim support using 'check_claim_graph' and sanity-check stats with 'run_biological_stats' if needed.\n"
            "2b. Use 'check_references_completeness' and 'get_lit_reference_verification' to audit references. If completeness check fails, mark draft unsupported.\n"
            "2c. Run 'check_parameter_sources_for_manuscript' to validate parameter sourcing. Flag any reported issues.\n"
            "2d. Run 'check_hypothesis_trace_consistency'. Flag any gaps in supported hypotheses.\n"
            "2e. Run 'check_metrics_for_referenced_models'. If metrics are missing, flag and request compute_model_metrics from the Modeler.\n"
            f"2f. Generate provenance_summary.md via 'generate_provenance_summary'; if major inputs (lit_summary, model_spec, sims) are missing, flag the section and request fixes.\n"
            "3. Check consistency: Does Figure 3 actually support the claim in paragraph 2?\n"
            "4. If gaps exist, report them clearly to the PI.\n"
            "5. Only report 'NO GAPS' if the PDF validates completely.\n"
            "6. If you create or materially analyze artifacts, log them via 'create_review_note_artifact' usage, not manifest lists.\n"
            "7. VERIFY PROOF OF WORK: Run 'check_proof_of_work_for_results' to audit verification note coverage. Reject results if coverage is poor.\n"
            "8. Use 'create_review_note_artifact' to create 'verification_note' or 'review_report'. Do NOT use 'reserve_typed_artifact' directly.\n"
            "9. Log reflections to run_notes via 'append_run_note_tool' or manage_project_knowledge; never to manifest.\n"
            f"10. {reflection_instruction}"
        ),
        tools=[
            read_artifact,
            summarize_artifact,
            create_review_note_artifact,
            check_parameter_sources_for_manuscript,
            check_metrics_for_referenced_models,
            check_hypothesis_trace_consistency,
            check_proof_of_work_for_results,
            get_lit_reference_verification,
            check_references_completeness,
            read_manuscript,
            check_claim_graph,
            run_biological_stats,
            verify_references,
            generate_provenance_summary,
            write_text_artifact,
            manage_project_knowledge,
        ],
        model=model,
        settings=common_settings,
    )

    interpreter = _make_agent(
        name="Theoretical Biological Interpreter",
        instructions=(
            "You are an expert Theoretical Biological Interpreter.\n"
            "Goal: Produce interpretation.json/md for theoretical biology projects.\n"
            f"{path_context}\n{path_guardrails}\n{metadata_reminder}\n{_context_spec_intro('Interpreter')}\n{_summary_advisory('Interpreter')}\n"
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

    coder = _make_agent(
        name="Coder",
        instructions=(
            "You are an expert Utility Engineer.\n"
            "Goal: Write or update lightweight Python helpers/tools confined to this run folder.\n"
            f"{path_context}\n{path_guardrails}\n{metadata_reminder}\n{_context_spec_intro('Coder')}\n{_summary_advisory('Coder')}\n"
            "Directives:\n"
            "1. Use 'coder_create_python' to create/update files under the run root; do NOT write outside AISC_BASE_FOLDER.\n"
            "2. If you add tools/helpers, document them briefly and log via 'append_manifest' (name + kind + created_by + status). Include 'change_summary' if updating.\n"
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

    publisher = _make_agent(
        name="Publisher",
        instructions=(
            "You are an expert Production Editor.\n"
            "Goal: Compile final PDF.\n"
            f"{path_context}\n{path_guardrails}\n{metadata_reminder}\n{_context_spec_intro('Publisher')}\n{_summary_advisory('Publisher')}\n"
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

    pi = Agent(
        name="Principal Investigator",
        instructions=(
            f"You are an expert Principal Investigator for project: {title}.\n"
            f"Hypothesis: {hypothesis}\n"
            f"{_context_spec_intro('Principal Investigator')}\n"
            f"{metadata_reminder}\n"
            "Responsibilities:\n"
            "0. Agents are stateless tools with a hard ~40-turn budget (including their tool calls). Do NOT send 'prepare' or 'wait until X' tasks. Delegate small, end-to-end units with concrete paths; if a job is large, split it into multiple invocations (e.g., one per batch of sims/plots) and ask the agent to persist outputs plus a brief status note to user_inbox.md/pi_notes.md before returning. You may spawn multiple parallel calls to the same role if that speeds work, as long as each request is end-to-end and self-contained. If you already know the relevant file paths or artifact names, include them in the prompt to save turn budget.\n"
            "1. STATE CHECK: First, read the provided PI_notes, user_inbox, and prior 'check_project_state' runs that appears in your system/user message. Only call 'read_note' or 'check_project_state' if you need a fresh snapshot beyond what is already provided.\n"
            "2. REVIEW KNOWLEDGE: Check 'manage_project_knowledge' for constraints or decisions before delegating.\n"
            "3. MITIGATE ITERATIVE GAP: Before complex phases (e.g., large simulations, drafting full sections), write an `implementation_plan.md` using `write_text_artifact` (default path: experiment_results/implementation_plan.md). Update the plan when priorities or completion status change—do not carry a stale plan forward. If `--human_in_the_loop` is active, call `wait_for_human_review` on this plan before proceeding.\n"
            "3b. Maintain hypothesis_trace.json: when drafting the plan, ensure every idea experiment is mapped to a hypothesis/experiment id (H*, E*) in hypothesis_trace.json (skeleton allowed). Update as new experiments/figures/sim runs become planned.\n"
            "4. DELEGATE: Handoff to specialized agents based on missing artifacts. **MANDATORY: When calling a sub-agent, lookup the exact file paths first (via inspect_manifest or list_artifacts) and pass the EXACT PATH in the prompt. Do not ask them to 'find the file'.**\n"
            "   - Before delegating to a module, call 'ensure_module_summary' with the module name from the context spec (e.g., 'modeling', 'analysis', 'writeup'). If it reports 'missing' or 'stale', wait for the integration memo to materialize or re-run the summarizer before proceeding.\n"
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
            "7. PROMOTION & END OF RUN: Review major artifacts. If they are final/valid, call 'promote_artifact_to_canonical(name=..., kind=..., notes=...)' to lock them. Check for stale dependencies via 'check_dependency_staleness' before promoting.\n"
            "8. TERMINATE: Stop only when Reviewer confirms 'NO GAPS' and PDF is generated. Before stopping, call 'generate_project_snapshot' to create a human-readable summary.\n"
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
            promote_artifact_to_canonical,
            check_dependency_staleness,
            generate_project_snapshot,
            manage_project_knowledge,
            scan_transport_manifest,
            read_transport_manifest,
            resolve_baseline_path,
            resolve_sim_path,
            update_transport_manifest,
            mirror_artifacts,
            write_text_artifact,
            update_hypothesis_trace,
            generate_provenance_summary,
            wait_for_human_review,
            check_user_inbox,
            append_run_note_tool,
            ensure_module_summary,
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
