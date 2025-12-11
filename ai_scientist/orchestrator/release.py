# pyright: reportMissingImports=false
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.biological_plotting import RunBiologicalPlottingTool
from ai_scientist.tools.compute_model_metrics import ComputeModelMetricsTool
from ai_scientist.utils import manifest as manifest_utils

from ai_scientist.orchestrator.artifacts import _reserve_typed_artifact_impl
from ai_scientist.orchestrator.hypothesis import generate_provenance_summary_impl, load_hypothesis_trace
from ai_scientist.orchestrator.manifest_service import _append_manifest_entry, manage_project_knowledge


# Helper utilities

def _safe_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_repo_root() -> Path:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
        if res.stdout.strip():
            return Path(res.stdout.strip()).resolve()
    except Exception:
        pass
    return Path.cwd().resolve()


def _collect_manifest_artifacts(exp_dir: Path) -> set[Path]:
    paths: set[Path] = set()
    try:
        entries = manifest_utils.load_entries(base_folder=exp_dir, limit=None)
    except Exception:
        entries = []
    for entry in entries:
        p_str = entry.get("path") or entry.get("name")
        if not p_str:
            continue
        try:
            p = BaseTool.resolve_input_path(str(p_str), allow_dir=True)
        except FileNotFoundError:
            continue
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        if resolved.is_file():
            paths.add(resolved)
    return paths


def _collect_paths_from_trace(trace: Dict[str, Any]) -> set[Path]:
    collected: set[Path] = set()

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
        elif isinstance(obj, str):
            try:
                p = BaseTool.resolve_input_path(obj, allow_dir=True)
            except FileNotFoundError:
                return
            try:
                resolved = p.resolve()
            except Exception:
                resolved = p
            if resolved.is_file():
                collected.add(resolved)

    _walk(trace)
    return collected


def _write_env_manifest(path: Path, tag: str) -> Dict[str, Any]:
    repo_root = _detect_repo_root()
    req_path = repo_root / "requirements.txt"
    env_yml_path = repo_root / "environment.yml"
    freeze_output: Optional[str] = None
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
        freeze_output = res.stdout.strip()
    except Exception:
        freeze_output = None

    manifest: Dict[str, Any] = {
        "tag": tag,
        "created_at": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "os_uname": list(platform.uname()),
    }
    if req_path.exists():
        manifest["requirements_txt"] = req_path.read_text()
    if env_yml_path.exists():
        manifest["environment_yml"] = env_yml_path.read_text()
    if freeze_output:
        manifest["pip_freeze"] = freeze_output.splitlines()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _create_code_archive(repo_root: Path, archive_path: Path, diff_path: Optional[Path]) -> None:
    skip_dirs = {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
        "experiment_results",
        "experiments",
        "figures",
    }
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(repo_root):
            rel_root = Path(root).relative_to(repo_root)
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            if "releases" in rel_root.parts and "experiment_results" in rel_root.parts:
                continue
            for name in files:
                p = Path(root) / name
                if any(part in skip_dirs for part in p.parts):
                    continue
                if not p.is_file():
                    continue
                arcname = p.relative_to(repo_root)
                zf.write(p, arcname)
        if diff_path and diff_path.exists():
            zf.write(diff_path, Path("release_meta") / diff_path.name)


def _gather_git_state(repo_root: Path, release_root: Path) -> Dict[str, Any]:
    git_info: Dict[str, Any] = {"commit": None, "dirty": False}
    status_lines: List[str] = []
    try:
        commit_res = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_res.stdout.strip():
            git_info["commit"] = commit_res.stdout.strip()
        status_res = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if status_res.stdout.strip():
            git_info["dirty"] = True
            status_lines = status_res.stdout.strip().splitlines()
            git_info["status_summary"] = status_lines[:50]
            diff_chunks: List[str] = []
            cached_res = subprocess.run(
                ["git", "-C", str(repo_root), "diff", "--cached"],
                capture_output=True,
                text=True,
                check=False,
            )
            worktree_res = subprocess.run(
                ["git", "-C", str(repo_root), "diff"],
                capture_output=True,
                text=True,
                check=False,
            )
            for res in (cached_res, worktree_res):
                if res.stdout:
                    diff_chunks.append(res.stdout)
            if status_lines:
                diff_chunks.append("# Untracked/dirty summary\n" + "\n".join(status_lines))
            diff_text = "\n\n".join(diff_chunks).strip()
            if diff_text:
                diff_path = release_root / "diff.patch"
                diff_path.write_text(diff_text)
                git_info["diff_path"] = str(diff_path)
        else:
            git_info["dirty"] = False
    except Exception as exc:
        git_info["error"] = str(exc)
    return git_info


def _copy_release_sources(
    sources: List[Path],
    release_root: Path,
    repo_root: Path,
    artifacts_root: Path,
    include_large: bool,
    large_threshold: int,
    hard_limit: int,
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    entries: List[Dict[str, Any]] = []
    missing: List[str] = []
    skipped: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    base_folder = BaseTool.resolve_output_dir(None).parent
    for src in sources:
        try:
            resolved = src.resolve()
        except Exception:
            resolved = src
        if resolved in seen:
            continue
        if release_root in resolved.parents or resolved == release_root:
            continue
        if not resolved.exists():
            missing.append(str(src))
            continue
        if resolved.is_dir():
            continue
        try:
            size_bytes = resolved.stat().st_size
        except Exception:
            size_bytes = None
        if size_bytes is not None and size_bytes > hard_limit:
            skipped.append({"path": str(resolved), "reason": "over_hard_limit", "size_bytes": size_bytes})
            continue
        if size_bytes is not None and size_bytes > large_threshold and not include_large:
            skipped.append({"path": str(resolved), "reason": "too_large", "size_bytes": size_bytes})
            continue
        try:
            rel = resolved.relative_to(repo_root)
        except ValueError:
            try:
                rel = resolved.relative_to(base_folder)
            except Exception:
                rel = Path(resolved.name)
        dest = artifacts_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, dest)
        checksum = _safe_sha256(dest)
        entries.append(
            {
                "path": str(dest.relative_to(release_root)),
                "source": str(resolved),
                "size_bytes": size_bytes,
                "checksum": checksum,
            }
        )
        seen.add(resolved)
    return entries, missing, skipped


def _relative_to_release(path: Path, release_root: Path) -> str:
    try:
        return str(path.relative_to(release_root))
    except ValueError:
        return path.name


def _load_release_manifest(release_root: Path, tag: str) -> Dict[str, Any]:
    manifest_path = release_root / "release_manifest.json"
    if not manifest_path.exists():
        reserve = _reserve_typed_artifact_impl("release_manifest", json.dumps({"tag": tag}), unique=False)
        alt_path = reserve.get("reserved_path")
        if alt_path:
            manifest_path = Path(alt_path)
    if not manifest_path.exists():
        return {"error": "missing_release_manifest", "path": str(manifest_path)}
    try:
        data = json.loads(manifest_path.read_text())
    except Exception as exc:
        return {"error": f"invalid_release_manifest: {exc}", "path": str(manifest_path)}
    data["__path"] = str(manifest_path)
    return data


def _verify_release_files(manifest: Dict[str, Any], release_root: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    missing: List[str] = []
    mismatched: List[Dict[str, Any]] = []
    files = manifest.get("files", []) or []
    for entry in files:
        rel = entry.get("path") or entry.get("source")
        if not rel:
            continue
        p = Path(rel)
        target = p if p.is_absolute() else release_root / p
        if not target.exists():
            missing.append(str(target))
            continue
        expected = entry.get("checksum")
        try:
            actual = _safe_sha256(target)
        except Exception as exc:
            mismatched.append({"path": str(target), "expected": expected, "error": str(exc)})
            continue
        if expected and actual != expected:
            mismatched.append({"path": str(target), "expected": expected, "actual": actual})
    return missing, mismatched


def _read_env_manifest(env_path: Path) -> Dict[str, Any]:
    if not env_path.exists():
        return {}
    try:
        return json.loads(env_path.read_text())
    except Exception:
        return {}


def _select_first(paths: List[Path], suffix: str) -> Optional[Path]:
    for p in paths:
        if p.suffix.lower() == suffix.lower():
            return p
    return None


def _build_figure_mapping_table(files: List[Dict[str, Any]], release_root: Path) -> str:
    rows = ["| Figure artifact | Tool/command | Source path |", "| --- | --- | --- |"]
    for entry in files:
        rel = entry.get("path")
        if not rel:
            continue
        p = Path(rel)
        if p.suffix.lower() not in {".png", ".pdf", ".svg"}:
            continue
        tool = "run_biological_plotting"
        name = p.name
        if "sweep" in name or "intervention" in name:
            tool = "run_sensitivity_sweep"
        if "transport" in name:
            tool = "run_compartmental_sim"
        rows.append(f"| {name} | {tool} | {rel} |")
    if len(rows) == 2:
        rows.append("| (no figures found in release manifest) | - | - |")
    return "\n".join(rows)


def _word_limit(text: str, max_words: int = 400) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# Public API

def freeze_release(tag: str, description: str = "", include_large_artifacts: bool = False) -> Dict[str, Any]:
    if not tag or not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
        return {"error": "invalid_tag", "tag": tag}

    exp_dir = BaseTool.resolve_output_dir(None)
    release_root = exp_dir / "releases" / tag
    release_root.mkdir(parents=True, exist_ok=True)
    artifacts_root = release_root / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    prov_res = generate_provenance_summary_impl()
    prov_path = Path(prov_res.get("path", "")) if isinstance(prov_res, dict) else exp_dir / "provenance_summary.md"

    base_root = Path(os.environ.get("AISC_BASE_FOLDER", "") or ".").resolve()
    key_sources = [
        base_root / "project_knowledge.md",
        exp_dir / "hypothesis_trace.json",
        BaseTool.resolve_input_path("claim_graph.json", must_exist=False),
        prov_path,
    ]
    manifest_sources = _collect_manifest_artifacts(exp_dir)
    trace = load_hypothesis_trace()
    hypothesis_sources = _collect_paths_from_trace(trace)

    all_sources: List[Path] = []
    for p in key_sources:
        if p not in all_sources:
            all_sources.append(p)
    for src in sorted(manifest_sources | hypothesis_sources):
        all_sources.append(src)

    repo_root = _detect_repo_root()
    large_threshold = int(os.environ.get("AISC_RELEASE_LARGE_THRESHOLD_MB", "500")) * 1024 * 1024
    hard_limit = int(os.environ.get("AISC_RELEASE_HARD_LIMIT_MB", "2048")) * 1024 * 1024

    copied_entries, missing, skipped = _copy_release_sources(
        all_sources,
        release_root,
        repo_root,
        artifacts_root,
        include_large_artifacts,
        large_threshold,
        hard_limit,
    )

    git_state = _gather_git_state(repo_root, release_root)

    env_reserve = _reserve_typed_artifact_impl("env_manifest", json.dumps({"tag": tag}), unique=False)
    env_path = Path(env_reserve.get("reserved_path") or (release_root / "env_manifest.json"))
    if not str(env_path).startswith(str(release_root)):
        env_path = release_root / env_path.name
    _write_env_manifest(env_path, tag)
    env_checksum = _safe_sha256(env_path)
    _append_manifest_entry(
        name=str(env_path),
        metadata_json=json.dumps({"kind": "env_manifest", "created_by": os.environ.get("AISC_ACTIVE_ROLE", "publisher"), "status": "ok"}),
        allow_missing=False,
    )

    code_reserve = _reserve_typed_artifact_impl("code_release_archive", json.dumps({"tag": tag}), unique=False)
    code_archive_path = Path(code_reserve.get("reserved_path") or (release_root / "code_release.zip"))
    if not str(code_archive_path).startswith(str(release_root)):
        code_archive_path = release_root / code_archive_path.name
    _create_code_archive(repo_root, code_archive_path, Path(git_state.get("diff_path", "")) if git_state.get("diff_path") else None)
    code_checksum = _safe_sha256(code_archive_path)
    _append_manifest_entry(
        name=str(code_archive_path),
        metadata_json=json.dumps({"kind": "code_release_archive", "created_by": os.environ.get("AISC_ACTIVE_ROLE", "publisher"), "status": "ok"}),
        allow_missing=False,
    )

    if git_state.get("diff_path"):
        diff_path = Path(git_state["diff_path"])
        _append_manifest_entry(
            name=str(diff_path),
            metadata_json=json.dumps({"kind": "release_diff_patch", "created_by": os.environ.get("AISC_ACTIVE_ROLE", "publisher"), "status": "ok"}),
            allow_missing=False,
        )
        if diff_path.exists():
            copied_entries.append(
                {
                    "path": _relative_to_release(diff_path, release_root),
                    "source": str(diff_path),
                    "size_bytes": diff_path.stat().st_size,
                    "checksum": _safe_sha256(diff_path),
                }
            )

    copied_entries.append(
        {
            "path": _relative_to_release(code_archive_path, release_root),
            "source": str(repo_root),
            "size_bytes": code_archive_path.stat().st_size,
            "checksum": code_checksum,
            "kind": "code_release_archive",
        }
    )
    copied_entries.append(
        {
            "path": _relative_to_release(env_path, release_root),
            "source": str(env_path),
            "size_bytes": env_path.stat().st_size,
            "checksum": env_checksum,
            "kind": "env_manifest",
        }
    )

    prov_release_path = None
    if prov_path.exists():
        for entry in copied_entries:
            if Path(entry.get("source", "")).resolve() == prov_path.resolve():
                prov_release_path = entry.get("path")
                break

    release_manifest_data: Dict[str, Any] = {
        "tag": tag,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "release_dir": str(release_root),
        "git": git_state,
        "provenance_summary_path": prov_release_path or (str(prov_path) if prov_path.exists() else None),
        "code_archive_path": _relative_to_release(code_archive_path, release_root),
        "env_manifest_path": _relative_to_release(env_path, release_root),
        "include_large_artifacts": include_large_artifacts,
        "skipped": {"missing": missing, "too_large": skipped},
        "files": copied_entries,
    }

    release_manifest_reserve = _reserve_typed_artifact_impl("release_manifest", json.dumps({"tag": tag}), unique=False)
    release_manifest_path = Path(release_manifest_reserve.get("reserved_path") or (release_root / "release_manifest.json"))
    if not str(release_manifest_path).startswith(str(release_root)):
        release_manifest_path = release_root / release_manifest_path.name
    release_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    release_manifest_path.write_text(json.dumps(release_manifest_data, indent=2))
    release_manifest_checksum = _safe_sha256(release_manifest_path)
    _append_manifest_entry(
        name=str(release_manifest_path),
        metadata_json=json.dumps({"kind": "release_manifest", "created_by": os.environ.get("AISC_ACTIVE_ROLE", "publisher"), "status": "ok"}),
        allow_missing=False,
    )

    return {
        "release_dir": str(release_root),
        "code_archive": str(code_archive_path),
        "env_manifest": str(env_path),
        "release_manifest": str(release_manifest_path),
        "release_manifest_checksum": release_manifest_checksum,
        "missing": missing,
        "skipped": skipped,
        "git": git_state,
    }


def check_release_reproducibility(tag: str, quick: bool = True) -> Dict[str, Any]:
    if not tag or not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
        return {"error": "invalid_tag", "tag": tag}

    exp_dir = BaseTool.resolve_output_dir(None)
    release_root = exp_dir / "releases" / tag
    if not release_root.exists():
        return {"error": "release_not_found", "path": str(release_root)}

    manifest = _load_release_manifest(release_root, tag)
    if manifest.get("error"):
        return manifest

    missing, mismatched = _verify_release_files(manifest, release_root)

    quick_steps: List[Dict[str, Any]] = []
    quick_errors: List[str] = []
    env_path = release_root / manifest.get("env_manifest_path", "env_manifest.json")
    env_hash = _safe_sha256(env_path) if env_path.exists() else None

    files = manifest.get("files", []) or []
    resolved_files: List[Path] = []
    for entry in files:
        rel = entry.get("path")
        if not rel:
            continue
        p = Path(rel)
        resolved_files.append(p if p.is_absolute() else release_root / p)

    if quick:
        csv_candidate = _select_first(resolved_files, ".csv")
        if csv_candidate and csv_candidate.exists():
            try:
                res = ComputeModelMetricsTool().use_tool(input_path=str(csv_candidate))
                quick_steps.append({"step": "compute_model_metrics", "path": str(csv_candidate), "result": res})
            except Exception as exc:
                quick_errors.append(f"compute_model_metrics:{exc}")
        solution_candidate = None
        for p in resolved_files:
            if p.name.endswith("_solution.json") and p.exists():
                solution_candidate = p
                break
        if solution_candidate:
            try:
                res = RunBiologicalPlottingTool().use_tool(
                    solution_path=str(solution_candidate),
                    output_dir=str(release_root / "repro_figures"),
                    make_phase_portrait=False,
                    make_combined_svg=True,
                )
                quick_steps.append({"step": "run_biological_plotting", "path": str(solution_candidate), "result": res})
            except Exception as exc:
                quick_errors.append(f"run_biological_plotting:{exc}")

    status = "ok"
    reasons: List[str] = []
    if missing or mismatched:
        status = "partial"
        if mismatched:
            reasons.append("checksum_mismatch")
        if missing:
            reasons.append("missing_files")
    if quick_errors:
        status = "partial" if status == "ok" else status
        reasons.extend(quick_errors)

    summary_lines = [
        f"# Release Repro Status: {tag}",
        f"- status: {status}",
        f"- tag: {tag}",
        f"- git_commit: {manifest.get('git', {}).get('commit')}",
        f"- env_checksum: {env_hash}",
        f"- checked_at: {datetime.now().isoformat()}",
        f"- missing_files: {len(missing)}",
        f"- checksum_mismatches: {len(mismatched)}",
        f"- quick_errors: {quick_errors}",
    ]
    report_content = "\n".join(summary_lines)
    report_reserve = _reserve_typed_artifact_impl("release_repro_status_md", json.dumps({"tag": tag}), unique=False)
    report_path = Path(report_reserve.get("reserved_path") or (release_root / "repro_status.md"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content)
    _append_manifest_entry(
        name=str(report_path),
        metadata_json=json.dumps({"kind": "release_repro_status_md", "created_by": os.environ.get("AISC_ACTIVE_ROLE", "publisher"), "status": status}),
        allow_missing=False,
    )

    if status != "ok":
        suggestion = "Missing or corrupted release files; consider re-running freeze_release or adding the absent artifacts to the release registry."
        try:
            manage_project_knowledge(
                action="add",
                category="constraint",
                observation=f"Release {tag} repro check failed: {reasons or (missing or mismatched)}",
                solution=suggestion,
                actor=os.environ.get("AISC_ACTIVE_ROLE", "publisher"),
            )
        except Exception:
            pass

    return {
        "status": status,
        "missing": missing,
        "mismatched": mismatched,
        "quick_steps": quick_steps,
        "quick_errors": quick_errors,
        "repro_report": str(report_path),
        "env_checksum": env_hash,
        "release_manifest": manifest.get("__path"),
        "reasons": reasons,
    }


def generate_reproduction_section(tag: str, style: str = "methods_and_supp") -> Dict[str, Any]:
    if not tag or not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
        return {"error": "invalid_tag", "tag": tag}

    exp_dir = BaseTool.resolve_output_dir(None)
    release_root = exp_dir / "releases" / tag
    if not release_root.exists():
        return {"error": "release_not_found", "path": str(release_root)}

    manifest = _load_release_manifest(release_root, tag)
    if manifest.get("error"):
        return manifest

    env_rel = manifest.get("env_manifest_path") or "env_manifest.json"
    env_path = release_root / env_rel if not Path(env_rel).is_absolute() else Path(env_rel)
    env_data = _read_env_manifest(env_path)
    env_hash = _safe_sha256(env_path) if env_path.exists() else None
    python_version = str(env_data.get("python_version") or "").split()[0] or "3.10"
    git_commit = manifest.get("git", {}).get("commit", "unknown")
    doi = manifest.get("doi") or manifest.get("zenodo_doi")

    files = manifest.get("files", []) or []
    figure_table = _build_figure_mapping_table(files, release_root)

    methods_lines = [
        f"Code and data for this study are bundled as release `{tag}` (git commit `{git_commit}`) under `experiment_results/releases/{tag}`.",
        f"Environment details (Python {python_version}) are captured in `env_manifest.json` (checksum {env_hash or 'n/a'}), and the full code snapshot is in `code_release.zip`.",
    ]
    if doi:
        methods_lines.append(f"The archived bundle is available at {doi}.")
    methods_lines.append(
        "To reproduce results, install the recorded environment, unpack the code archive, and run the provided tooling for simulations (e.g., `run_compartmental_sim`, `run_sensitivity_sweep`) and analysis (`compute_model_metrics`, `run_biological_plotting`) against the artifacts listed in `release_manifest.json`."
    )
    methods_text = _word_limit("\n\n".join(methods_lines), 400)

    env_major_minor = ".".join(python_version.split(".")[:2]) if python_version else "3.10"
    commands = [
        f"conda create -n {tag} python={env_major_minor}",
        f"conda activate {tag}",
        "unzip code_release.zip -d code_release",
        "cd code_release",
        "pip install -r requirements.txt",
        'export AISC_BASE_FOLDER="$(pwd)/.."',
        'export AISC_EXP_RESULTS="$AISC_BASE_FOLDER/experiment_results"',
        '# re-run plotting/statistics against release artifacts',
        f'python ai_scientist/tools/biological_stats.py --task adjust_pvalues --input "$AISC_EXP_RESULTS/releases/{tag}/artifacts/<stats_table>.csv"',
        'python perform_plotting.py --base_folder "$AISC_BASE_FOLDER"',
        '# regenerate manuscript (optional)',
        'python agents_orchestrator.py --load_manuscript ai_scientist/ideas/manuscript_v3.md --resume --base_folder "$AISC_BASE_FOLDER" --model gpt-5o-mini --max_cycles 5 --skip_lit_gate  # or --load_idea <idea>.json',
    ]
    commands_block = "\n".join(commands)

    supp_parts = [
        f"# Reproduction Protocol for {tag}",
        "## Setup",
        "```bash",
        commands_block,
        "```",
        "## Figure/tool mapping",
        figure_table,
        "## Notes",
        "- Use only artifacts present in `release_manifest.json`; avoid referencing external paths.",
        "- Swap `<idea>.json`/`<stats_table>.csv` for the files provided in the release bundle.",
    ]
    supp_text = "\n\n".join(supp_parts)

    return {
        "methods_section_md": methods_text,
        "supplementary_protocol_md": supp_text,
        "subdir": str(Path("releases") / tag),
        "env_checksum": env_hash,
        "git_commit": git_commit,
        "doi": doi,
        "figure_table": figure_table,
    }


def safe_sha256(path: Path) -> str:
    """Public wrapper around the internal SHA256 helper."""
    return _safe_sha256(path)


def load_release_manifest(release_root: Path, tag: str) -> Dict[str, Any]:
    return _load_release_manifest(release_root, tag)
