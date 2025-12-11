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

try:
    from agents import Agent, ModelSettings
except ImportError:
    Agent = Any  # type: ignore
    ModelSettings = Any  # type: ignore

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
    check_manifest_unique_paths,
    check_project_state,
    check_status,
    check_user_inbox,
    coder_create_python,
    compute_model_metrics,
    generate_provenance_summary,
    graph_diagnostics,
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
    get_lit_recommendations,
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
from ai_scientist.orchestrator.interpretation_tools import (
    create_interpretation_json_artifact,
    create_interpretation_md_artifact,
)
from ai_scientist.orchestrator.publisher_tools import (
    create_release_manifest_artifact,
    create_code_release_archive_artifact,
    create_env_manifest_artifact,
    create_release_diff_patch_artifact,
    create_release_repro_status_artifact,
    create_repro_methods_artifact,
    create_repro_protocol_artifact,
    create_manuscript_figure_artifact,
    create_manuscript_figure_svg_artifact,
)
from ai_scientist.orchestrator.tool_wrappers import (
    check_lit_ready,
    check_model_provenance,
)


# TOOL WRITER PERMISSIONS (to prevent cargo-culting instructions across agents):
# - hypothesis_trace writers: {Modeler, PI} only
# - run_notes writers (via append_run_note_tool): {Archivist, Modeler} only
# - all other agents: manage_project_knowledge only for reflections



def _make_agent(name: str, instructions: str, tools: List[Any], model: str, settings: Any) -> Any:
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


def build_team(model: str, idea: Dict[str, Any], dirs: Dict[str, str]) -> Any:
    artifact_catalog = _artifact_kind_catalog()
    common_settings = ModelSettings(tool_choice="auto")
    role_max_turns = 40
    title = idea.get('Title', 'Project')
    abstract = idea.get('Abstract', '')
    hypothesis = idea.get('Short Hypothesis', 'None')
    related_work = idea.get('Related Work', 'None provided.')

    experiments_plan = format_list_field(idea.get('Experiments', []))
    risk_factors = format_list_field(idea.get('Risk Factors and Limitations', []))

    def _agent_has_reserve_tools(role_name: str) -> bool:
        spec = get_context_view_spec(role_name)
        return spec is not None and "*" in spec.write_scope

    def _get_path_context(role_name: str, dirs: Dict[str, str]) -> str:
        spec = get_context_view_spec(role_name)
        if not spec: 
            return ""
        if "*" in spec.read_scope or "*" in spec.write_scope:
            return (
                f"SYSTEM CONTEXT: Run Root='{dirs['base']}', Exp Results='{dirs['results']}'. "
                f"Figures='{os.path.join(dirs['base'], 'figures')}'. "
                "Use these paths directly; do NOT call get_run_paths. "
                "Assume provided input paths exist; only list_artifacts if path is missing."
            )
        return (
            "SYSTEM CONTEXT: Your specialized tools handle file paths automatically. "
            "Focus on your domain tasks; path management is abstracted away."
        )

    def _get_file_io_policy(role_name: str) -> str:
        spec = get_context_view_spec(role_name)
        if not spec: 
            return ""
        
        has_reserve_tools = "*" in spec.write_scope
        
        if has_reserve_tools:
            return (
                "FILE IO POLICY: Every persistent artifact must be reserved via 'reserve_typed_artifact(kind=..., meta_json=...)' using the registry below; do NOT invent filenames or bypass the registry. "
                f"Known kinds: {artifact_catalog}. "
                "Refer to docs/artifact_metadata_requirements.md for the required metadata fields every artifact must carry. "
                "Preferred flow: 'reserve_and_register_artifact' -> write -> (optional) update status via append_manifest. "
                "Use 'reserve_output' only for PI/Coder scratch logs; other roles must stay within typed helpers. When writing text, pass the reserved path into write_text_artifact instead of freehand names. "
                "Outputs are anchored to experiment_results; if a directory is unavailable, writes are auto-rerouted to experiment_results/_unrouted with a manifest note. "
                "NEVER log reflections or notes to the manifest—use append_run_note or manage_project_knowledge instead. "
                "Prefer 'summarize_artifact' to collect condensed views and call 'ensure_module_summary' for the relevant module before requesting full content."
            )

        # Simplified policy
        kinds_str = ", ".join(spec.write_scope)
        return (
            "FILE IO POLICY: Use specialized artifact creation tools provided to you. "
            "Do NOT attempt to reserve artifacts directly—use the create_* helpers specific to your role. "
            "These tools handle path reservation and manifest registration automatically. "
            f"You are authorized to create: {kinds_str}."
        )

    def _get_metadata_reminder(role_name: str) -> str:
        if _agent_has_reserve_tools(role_name):
            return (
                "METADATA REMINDER: When calling 'reserve_typed_artifact' or 'reserve_and_register_artifact', pass 'meta_json' including "
                "`id`, `type`, `version`, `parent_version`, `status`, `module`, `summary`, `content`, and `metadata` (see docs/artifact_metadata_requirements.md)."
            )
        return ""
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

    # Shared instruction components (DRY principle)
    common_efficiency_note = (
        "EFFICIENCY: You have ~40 tool calls before timeout. "
        "Batch operations when possible. Use summarize_artifact before read_artifact for large files."
    )

    common_error_recovery = (
        "ERROR RECOVERY:\n"
        "- If tool fails: Report error details to PI, do NOT retry silently\n"
        "- If data missing: Use list_artifacts to confirm, then request from PI\n"
        "- If validation fails: Log to manage_project_knowledge with failure pattern"
    )

    archivist = _make_agent(
        name="Archivist",
        instructions=(
            f"You are an expert Literature Curator.\n"
            f"Goal: Verify novelty of '{title}' and map claims to citations.\n\n"
            f"TL;DR: Search papers → Create lit_summary → Verify references → Build claim graph → Check quality gates\n\n"
            f"## CONTEXT\n"
            f"{abstract}\n"
            f"Related Work to Consider: {related_work}\n"
            f"{_get_path_context('Archivist', dirs)}\n{_get_file_io_policy('Archivist')}\n{_get_metadata_reminder('Archivist')}\n{_context_spec_intro('Archivist')}\n{_summary_advisory('Archivist')}\n\n"
            "## CORE WORKFLOW\n"
            "1. Use 'assemble_lit_data' or 'search_semantic_scholar' to gather papers\n"
            "2. Use 'get_lit_recommendations' to discover highly relevant papers based on findings\n"
            "3. Maintain a claim graph via 'update_claim_graph' when mapping evidence\n"
            "3. Use specialized artifact creators (do NOT use generic reserve calls):\n"
            "   - 'create_lit_summary_artifact(module=\"lit\")'\n"
            "   - 'create_claim_graph_artifact(module=\"lit\")'\n"
            "   - 'create_lit_review_artifact', 'create_lit_bibliography_artifact'\n"
            "   - 'create_lit_coverage_artifact', 'create_lit_integration_memo_artifact'\n\n"
            "## QUALITY GATES (REQUIRED)\n"
            "4. Immediately run 'verify_references' on lit_summary to produce lit_reference_verification.csv/json\n"
            "5. Reject readiness if:\n"
            "   - More than 20% of references are missing (found==False)\n"
            "   - Any match_score < 0.5\n"
            "   - Report FAILURE with specific counts\n"
            "6. If verification repeatedly fails for a venue/source, log via manage_project_knowledge with the specific venue\n"
            "7. CRITICAL: If no papers are found, report FAILURE. Do not invent 'TBD' citations\n\n"
            "## SUCCESS CRITERIA\n"
            "✓ lit_summary.json created with verified references\n"
            "✓ Reference verification shows ≥80% found, all match_score ≥0.5\n"
            "✓ claim_graph.json maps all major claims to citations\n"
            "✓ No placeholder or 'TBD' citations\n\n"
            f"{common_efficiency_note}\n\n"
            f"{common_error_recovery}\n\n"
            "8. Log reflections via 'append_run_note_tool' or manage_project_knowledge; never to manifest\n"
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
            get_lit_recommendations,
        ],
        model=model,
        settings=common_settings,
    )

    modeler = _make_agent(
        name="Modeler",
        instructions=(
            f"You are an expert Computational Biologist.\\n"
            f"Goal: Execute simulations for '{title}'.\\n\\n"
            f"TL;DR: Register model spec → Run sims/sweeps → Postprocess → Compute metrics → Update hypothesis_trace\\n\\n"
            f"## FOCUS\\n"
            f"You do NOT care about LaTeX or writing styles. Focus on DATA.\\n\\n"
            f"## HYPOTHESIS & PLAN\\n"
            f"{hypothesis}\\n"
            f"Experimental Plan:\\n{experiments_plan}\\n"
            f"{_get_path_context('Modeler', dirs)}\n{_get_file_io_policy('Modeler')}\n{_get_metadata_reminder('Modeler')}\n{_context_spec_intro('Modeler')}\n{_summary_advisory('Modeler')}\n\n"
            "## REQUIRED SEQUENCE (for each model)\\n"
            "BEFORE first sim → create_model_spec_artifact(model_key, params)\\n"
            "DURING sim       → run_comp_sim / run_transport_batch / run_sensitivity_sweep / run_intervention_tests\\n"
            "AFTER sim        → sim_postprocess (if arrays missing) → validate_per_compartment_outputs\\n"
            "AFTER batch      → compute_model_metrics → update_hypothesis_trace(experiment_id, run_id)\\n\\n"
            "## CORE WORKFLOW\\n"
            "1. Build graphs: 'build_graphs'\\n"
            "2. Run baselines: 'run_biological_model' or custom sims: 'run_comp_sim'\\n"
            "3. Explore parameter space: 'run_sensitivity_sweep' and 'run_intervention_tests'\\n"
            "4. Ensure parameter sweeps cover the range specified in the hypothesis\\n\\n"
            "## ARTIFACT MANAGEMENT\\n"
            "5. Save raw outputs to experiment_results/\\n"
            "6. Use specialized helpers (do NOT use generic reserve tools):\\n"
            "   - 'create_transport_artifact'\\n"
            "   - 'create_sensitivity_table_artifact'\\n"
            "   - 'create_intervention_table_artifact'\\n"
            "   - 'create_verification_note_artifact'\\n\\n"
            "## OUTPUT REQUIREMENTS\\n"
            "7. Every run must produce:\\n"
            "   - Arrays: failure_matrix.npy, time_vector.npy, nodes_order_*.txt\\n"
            "   - Per-compartment: per_compartment.npz + node_index_map.json + topology_summary.json\\n"
            "   - Metadata: sim.json + sim.status.json\\n"
            "   - Run 'sim_postprocess' if arrays missing, 'validate_per_compartment_outputs' before marking complete\\n\\n"
            "## TRANSPORT RUN MANIFEST\\n"
            "8. Use read_transport_manifest / update_transport_manifest:\\n"
            "   - Consult before reruns\\n"
            "   - Update after completing or failing a run\\n"
            "   - Do NOT mark status=complete unless all files exist (arrays + sim.json + sim.status.json)\\n"
            "   - Otherwise mark 'partial' and note missing files\\n"
            "9. Resolve baselines via 'resolve_baseline_path' before batches\\n"
            "   - Only pass graph baselines (.npy/.npz/.graphml/.gpickle/.gml), never sim.json\\n"
            "10. Process one baseline per call; load run_recipe.json if it exists under experiment_results/simulations/transport_runs\\n\\n"
            "## SUCCESS CRITERIA\\n"
            "✓ Every model has model_spec_artifact registered\\n"
            "✓ Every completed run has metrics computed (compute_model_metrics)\\n"
            "✓ Every run has verification note with input files listed\\n"
            "✓ hypothesis_trace.json updated after each experiment\\n"
            "✓ All required output files present before marking complete\\n\\n"
            f"{common_efficiency_note}\\n\\n"
            f"{common_error_recovery}\\n\\n"
            f"{proof_of_work_instruction}\\n\\n"
            "11. Log reflections via 'append_run_note_tool' or manage_project_knowledge; never to manifest\\n"
            f"12. {reflection_instruction}"
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
            "Goal: Convert simulation data into PLOS-quality figures.\n\n"
            "TL;DR: Read sim data → Validate hypothesis support → Plot → Publish to manuscript gallery\n\n"
            f"{_get_path_context('Analyst', dirs)}\n{_get_file_io_policy('Analyst')}\n{_get_metadata_reminder('Analyst')}\n{_context_spec_intro('Analyst')}\n{_summary_advisory('Analyst')}\n\n"
            "## CORE WORKFLOW\n"
            "1. Read data from provided input paths (do NOT list files; assume path is correct)\n"
            "2. CRITICAL: Assert data supports hypothesis BEFORE plotting\n"
            "   ✓ GOOD: 'Simulation shows 80% failure rate, supporting H1 prediction of >50%'\n"
            "   ✗ BAD: 'Data collected, proceeding to plot'\n"
            "   If data contradicts hypothesis, report immediately with specifics\n"
            "3. Use 'list_available_runs_for_plotting' to see completed runs\n"
            "4. Generate PNG/SVG using 'run_biological_plotting'\n"
            "   - Use 'sim_postprocess' if you need failure_matrix/time_vector/node order from sim.json\n"
            "   - Resolve sim.json via 'resolve_sim_path' if needed\n"
            "5. Before computing cluster/finite-size metrics, run 'validate_per_compartment_outputs'\n"
            "   - If artifacts missing/invalid, report and request rerun (do NOT plot placeholders)\n"
            "6. Use 'get_metrics_for_plotting' for pre-computed metrics (do NOT run compute_model_metrics yourself)\n\n"
            "## ARTIFACT MANAGEMENT\n"
            "7. Call 'create_plot_artifact' for any new figure (plot_intermediate/manuscript_figure_png/svg)\n"
            "   - Provide 'change_summary' if updating existing artifact\n"
            "   - Do NOT call reserve_typed_artifact directly for plots\n"
            "8. When figure is final, call 'publish_figure_to_manuscript_gallery(artifact_id=...)'\n"
            "   - Do NOT manually copy/move files\n\n"
            "## VALIDATION\n"
            "9. Validate models vs lit via 'run_validation_compare'\n"
            "10. Use 'run_biological_stats' for significance/enrichment\n"
            "11. Check Project Knowledge for visualization standards (e.g., colormaps) before starting\n"
            "12. When plots ready, confirm provenance_summary.md exists or ask Reviewer to generate it\n\n"
            "## SUCCESS CRITERIA\n"
            "✓ Data-hypothesis alignment verified before plotting\n"
            "✓ All figures have verification notes\n"
            "✓ Final figures published to manuscript gallery\n"
            "✓ provenance_summary.md exists\n\n"
            f"{common_efficiency_note}\n\n"
            f"{common_error_recovery}\n\n"
            f"{proof_of_work_instruction}\\n\\n"
            "13. Log reflections via manage_project_knowledge; never to manifest\n"
            f"14. {reflection_instruction}"
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
            "You are an expert Holistic Reviewer.\\n"
            "Goal: Identify logical gaps and structural flaws.\\n\\n"
            "TL;DR: Read manuscript → Audit references → Check claim support → Verify proof of work → Report gaps or approve\\n\\n"
            f"## RISK FACTORS TO CHECK\\n{risk_factors}\\n"
            f"{_get_path_context('Reviewer', dirs)}\n{_get_file_io_policy('Reviewer')}\n{_get_metadata_reminder('Reviewer')}\n{_context_spec_intro('Reviewer')}\n{_summary_advisory('Reviewer')}\n\n"
            "## CORE WORKFLOW\\n"
            "1. Read manuscript draft: 'read_manuscript'\\n"
            "2. Check claim support: 'check_claim_graph' and 'run_biological_stats' if needed\\n\\n"
            "## REFERENCE AUDIT\\n"
            "3. Use 'check_references_completeness' and 'get_lit_reference_verification'\\n"
            "   - If completeness check fails, mark draft unsupported\\n"
            "4. Run 'check_parameter_sources_for_manuscript' to validate parameter sourcing\\n"
            "   - Flag any reported issues\\n\\n"
            "## CONSISTENCY CHECKS\\n"
            "5. Run 'check_hypothesis_trace_consistency'\\n"
            "   - Flag any gaps in supported hypotheses\\n"
            "6. Run 'check_metrics_for_referenced_models'\\n"
            "   - If metrics missing, flag and request compute_model_metrics from Modeler\\n"
            "7. Generate 'provenance_summary.md' via 'generate_provenance_summary'\\n"
            "   - If major inputs missing (lit_summary, model_spec, sims), flag and request fixes\\n"
            "8. Check consistency: Does Figure 3 actually support the claim in paragraph 2?\\n\\n"
            "## PROOF OF WORK VERIFICATION\\n"
            "9. CRITICAL: Run 'check_proof_of_work_for_results' to audit verification note coverage\\n"
            "   - Reject results if coverage is poor\\n\\n"
            "## REPORTING\\n"
            "10. If gaps exist, report them clearly to PI with specifics\\n"
            "11. Use 'create_review_note_artifact' to create 'verification_note' or 'review_report'\\n"
            "    - Do NOT use 'reserve_typed_artifact' directly\\n\\n"
            "## 'NO GAPS' CRITERIA (all must be true)\\n"
            "✓ All references verified and complete (≥80% found)\\n"
            "✓ All claims supported by claim_graph\\n"
            "✓ All parameters sourced (no free hyperparameters)\\n"
            "✓ All referenced models have metrics\\n"
            "✓ hypothesis_trace.json is consistent\\n"
            "✓ Proof of work exists for all results\\n"
            "✓ provenance_summary.md complete\\n"
            "✓ Figures support their claims\\n"
            "Only report 'NO GAPS' if ALL criteria pass\\n\\n"
            f"{common_efficiency_note}\\n\\n"
            f"{common_error_recovery}\\n\\n"
            "12. Log reflections via manage_project_knowledge; never to manifest\\n"
            f"13. {reflection_instruction}"
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
            "Goal: Produce interpretation.json/md for theoretical biology projects.\n\n"
            "TL;DR: Check research type → Interpret biology → Reserve artifact → Write interpretation\n\n"
            f"{_get_path_context('Interpreter', dirs)}\n{_get_file_io_policy('Interpreter')}\n{_get_metadata_reminder('Interpreter')}\n{_context_spec_intro('Interpreter')}\n{_summary_advisory('Interpreter')}\n\n"
            "## WORKFLOW\n"
            "1. CRITICAL: Call 'interpret_biology' ONLY when biology.research_type == theoretical\n"
            "2. Use experiment summaries and idea text (do NOT hallucinate unsupported claims)\n"
            "3. Create interpretation artifacts via specialized tools:\n"
            "   - 'create_interpretation_json_artifact()' for JSON format\n"
            "   - 'create_interpretation_md_artifact()' for Markdown format\n"
            "   - Use reserved path with 'write_text_artifact' or 'write_interpretation_text'\n"
            "4. These tools handle path reservation and manifest registration automatically\n\n"
            "## SUCCESS CRITERIA\n"
            "✓ Only called for theoretical research type\n"
            "✓ Interpretation grounded in experiment data\n"
            "✓ No hallucinated claims\n\n"
            f"{common_efficiency_note}\n\n"
            f"{common_error_recovery}\n\n"
            "5. Log reflections via manage_project_knowledge; never to manifest\n"
            f"6. {reflection_instruction}"
        ),
        tools=[
            read_artifact,
            summarize_artifact,
            create_interpretation_json_artifact,
            create_interpretation_md_artifact,
            write_text_artifact,
            write_interpretation_text,
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
            "Goal: Write or update lightweight Python helpers/tools confined to this run folder.\n\n"
            "TL;DR: Create Python helpers → Document → Log to manifest → Lint check\n\n"
            f"{_get_path_context('Coder', dirs)}\n{_get_file_io_policy('Coder')}\n{_get_metadata_reminder('Coder')}\n{_context_spec_intro('Coder')}\n{_summary_advisory('Coder')}\n\n"
            "## CORE WORKFLOW\n"
            "1. Use 'coder_create_python' to create/update files under run root\n"
            "   - Do NOT write outside AISC_BASE_FOLDER\n"
            "2. Document tools/helpers briefly\n"
            "3. Log via 'append_manifest': name + kind + created_by + status\n"
            "   - Include 'change_summary' if updating\n\n"
            "## BEST PRACTICES\n"
            "4. Prefer small, dependency-light snippets\n"
            "5. Avoid large libraries or network access\n"
            "6. If you need existing artifacts:\n"
            "   - Use 'list_artifacts' to find them\n"
            "   - Use 'read_artifact' with summary_only for large files\n"
            "7. Reserve persisted outputs via 'reserve_typed_artifact' (e.g., verification_note)\n"
            "8. Log code patterns or library constraints to Project Knowledge\n\n"
            "## SUCCESS CRITERIA\n"
            "✓ Code confined to run folder\n"
            "✓ Dependencies minimal\n"
            "✓ Tools documented\n"
            "✓ Lint checks pass (run_pyright, run_ruff)\n\n"
            f"{common_efficiency_note}\n\n"
            f"{common_error_recovery}\n\n"
            "9. Log reflections via manage_project_knowledge; never to manifest\n"
            f"10. {reflection_instruction}"
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
            "You are an expert Production Editor.\\n"
            "Goal: Compile final PDF.\\n\\n"
            "TL;DR: Integrate lit_summary + figures \u2192 Compile LaTeX \u2192 Debug errors \u2192 Produce PDF\\n\\n"
            f"{_get_path_context('Publisher', dirs)}\n{_get_file_io_policy('Publisher')}\n{_get_metadata_reminder('Publisher')}\n{_context_spec_intro('Publisher')}\n{_summary_advisory('Publisher')}\n\n"
            "## CORE WORKFLOW\\n"
            "1. Target the 'ai_scientist/templates/blank_theoretical_biology_latex' template\\n"
            "2. Integrate 'lit_summary.json' and figures into the text\\n"
            "3. Reserve outputs via 'reserve_typed_artifact' before writing:\\n"
            "   - figures README\\n"
            "   - manuscript PDF\\n"
            "   - Do NOT invent filenames\\n\\n"
            "## LATEX COMPILATION\\n"
            "4. Ensure compile success\\n"
            "5. Debug LaTeX errors autonomously:\\n"
            "   - Missing packages \u2192 Check template requirements\\n"
            "   - Undefined references \u2192 Verify citations exist in lit_summary\\n"
            "   - Figure not found \u2192 Check figure paths in manuscript gallery\\n"
            "   - Syntax errors \u2192 Fix and recompile\\n\\n"
            "## SUCCESS CRITERIA\\n"
            "\u2713 PDF compiles without errors\\n"
            "\u2713 All figures integrated\\n"
            "\u2713 All citations resolved\\n"
            "\u2713 Template formatting preserved\\n\\n"
            f"{common_efficiency_note}\\n\\n"
            f"{common_error_recovery}\\n\\n"
            "6. Log reflections via manage_project_knowledge; never to manifest\\n"
            f"7. {reflection_instruction}"
        ),
        tools=[
            get_run_paths,
            resolve_path,
            list_artifacts,
            list_artifacts_by_kind,
            read_artifact,
            summarize_artifact,
            create_release_manifest_artifact,
            create_code_release_archive_artifact,
            create_env_manifest_artifact,
            create_release_diff_patch_artifact,
            create_release_repro_status_artifact,
            create_repro_methods_artifact,
            create_repro_protocol_artifact,
            create_manuscript_figure_artifact,
            create_manuscript_figure_svg_artifact,
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

    pi = Agent(  # type: ignore
        name="Principal Investigator",
        instructions=(
            f"You are an expert Principal Investigator for project: {title}.\\n"
            f"Hypothesis: {hypothesis}\\n\\n"
            f"TL;DR: Check state \u2192 Plan \u2192 Delegate to specialists \u2192 Monitor progress \u2192 Promote artifacts \u2192 Generate snapshot\\n\\n"
            f"{_get_path_context('Principal Investigator', dirs)}\\n"
            f"{_get_file_io_policy('Principal Investigator')}\\n"
            f"{_get_metadata_reminder('Principal Investigator')}\\n"
            f"{_context_spec_intro('Principal Investigator')}\\n\\n"
            "## AGENT DELEGATION PRINCIPLES\\n"
            "Agents are stateless tools with ~40-turn budget. Do NOT send 'prepare' or 'wait until X' tasks.\\n"
            "- Delegate small, end-to-end units with concrete paths\\n"
            "- If job is large, split into multiple invocations (e.g., one per batch)\\n"
            "- Ask agents to persist outputs + status note to user_inbox.md/pi_notes.md\\n"
            "- You may spawn parallel calls if each is end-to-end and self-contained\\n"
            "- If you know file paths/artifact names, include them to save turn budget\\n\\n"
            "## WORKFLOW\\n\\n"
            "### 1. STATE CHECK\\n"
            "Read provided PI_notes, user_inbox, and prior check_project_state runs in your message.\\n"
            "Only call 'read_note' or 'check_project_state' if you need a fresh snapshot.\\n\\n"
            "### 2. REVIEW KNOWLEDGE\\n"
            "Check 'manage_project_knowledge' for constraints or decisions before delegating.\\n\\n"
            "### 3. STRUCTURED PLANNING\\n"
            "Maintain implementation plan:\\n"
            "- Use 'get_or_create_implementation_plan' to obtain implementation_plan_md\\n"
            "- After major decisions, call 'update_implementation_plan_from_state'\\n"
            "- Never leave plan stale at end of run\\n"
            "- If --human_in_the_loop active, call 'wait_for_human_review' before proceeding\\n"
            "- Maintain hypothesis_trace.json: map every experiment to H*/E* ids\\n\\n"
            "**IMPORTANT - Plan Merge Behavior**:\\n"
            "When you call 'update_implementation_plan_from_state', you are providing a *delta* or partial update.\\n"
            "The system will MERGE your new experiments/tasks/decisions into the existing plan.\\n"
            "- Existing experiments/tasks are PRESERVED unless you explicitly update them (same ID)\\n"
            "- Do NOT omit experiments/tasks you still care about—they won't be deleted\\n"
            "- To explicitly drop an experiment/task, note it in the `decisions` list\\n"
            "  (e.g., '2025-12-11: Drop E4 due to irrelevance')\\n\\n"
            "### 4. DELEGATE TO SPECIALISTS\\n"
            "**MANDATORY**: Lookup exact file paths first (inspect_manifest/list_artifacts) and pass EXACT PATH.\\n"
            "Do NOT ask agents to 'find the file'.\\n\\n"
            "**Pre-delegation checks**:\\n"
            "- Call 'ensure_module_summary' before delegating to a module\\n"
            "- Before modeling: run 'check_lit_ready' (≥70% confirmed refs, ≤3 unverified)\\n"
            "- Before built-in models: ensure 'check_model_provenance' passes\\n\\n"
            "**Delegation routing**:\\n"
            "- Missing Lit Review \u2192 Archivist\\n"
            "- Missing Data \u2192 Modeler\\n"
            "- Missing Plots \u2192 Analyst\\n"
            "- Theoretical Interpretation \u2192 Interpreter\\n"
            "- Draft Exists \u2192 Reviewer\\n"
            "- Validated & Ready \u2192 Publisher\\n\\n"
            "### 5. ASYNC FEEDBACK\\n"
            "Call 'check_user_inbox' frequently (between tasks) for user steering.\\n\\n"
            "### 6. HANDLE FAILURES\\n"
            "If sub-agent reports error/max turns:\\n"
            "- Call 'inspect_manifest(summary_only=False, role=..., limit=50)'\\n"
            "- See what they accomplished before crashing\\n"
            "- If artifacts exist, instruct next run to continue (not restart)\\n\\n"
            "### 7. PROMOTION & END OF RUN\\n"
            "Review major artifacts:\\n"
            "- If final/valid, call 'promote_artifact_to_canonical'\\n"
            "- Check 'check_dependency_staleness' before promoting\\n\\n"
            "### 8. TERMINATION\\n"
            "Stop ONLY when:\\n"
            "- Reviewer confirms 'NO GAPS'\\n"
            "- PDF is generated\\n"
            "Before stopping, call 'generate_project_snapshot'.\\n\\n"
            "## STATUS PERSISTENCE (CRITICAL)\\n"
            "Chat messages are ephemeral and NOT read by user or future calls.\\n"
            "To persist information, you MUST write to files via tools.\\n\\n"
            "After ANY nontrivial reasoning/planning, you must:\\n"
            "1) Update plan via 'update_implementation_plan_from_state', AND\\n"
            "2) Call 'log_status_to_user_inbox' with status summary\\n\\n"
            "If you skip these, your work is LOST.\\n\\n"
            "## SUCCESS CRITERIA\\n"
            "\u2713 Implementation plan maintained and up-to-date\\n"
            "\u2713 All delegations include exact file paths\\n"
            "\u2713 Status logged to user_inbox after each phase\\n"
            "\u2713 Artifacts promoted when final\\n"
            "\u2713 Project snapshot generated before termination\\n\\n"
            "9. Keep reflections in run_notes via append_run_note_tool or project_knowledge; never in manifest"
        ),
        model=model,
        tools=[
            # Import planning tools inline to avoid circular dependency
            *([getattr(__import__('ai_scientist.orchestrator.tool_wrappers', fromlist=['get_or_create_implementation_plan']), 'get_or_create_implementation_plan'),
               getattr(__import__('ai_scientist.orchestrator.tool_wrappers', fromlist=['update_implementation_plan_from_state']), 'update_implementation_plan_from_state'),
               getattr(__import__('ai_scientist.orchestrator.tool_wrappers', fromlist=['log_status_to_user_inbox']), 'log_status_to_user_inbox')]),
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
            check_lit_ready,
            check_model_provenance,
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
