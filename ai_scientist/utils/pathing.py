import os
import uuid
from pathlib import Path
from typing import Dict, Tuple, Optional

# ...

def _resolve_output_root(run_root: Optional[Path] = None) -> Path:
    """
    Resolve a base output directory without importing BaseTool to avoid circular imports.
    Prefers (in order): provided run_root, AISC_EXP_RESULTS, AISC_BASE_FOLDER/experiment_results, then relative experiment_results.
    """
    if run_root:
        return Path(run_root)
    env_dir = os.environ.get("AISC_EXP_RESULTS", "")
    if env_dir:
        return Path(env_dir)
    base = os.environ.get("AISC_BASE_FOLDER", "")
    if base:
        return Path(base) / "experiment_results"
    return Path("experiment_results")


def resolve_output_path(
    subdir: Optional[str],
    name: str,
    *,
    run_root: Optional[Path] = None,
    create: bool = True,
    allow_quarantine: bool = True,
    unique: bool = True,
) -> Tuple[Path, bool, Optional[str]]:
    """
    Resolve an output path anchored to experiment_results, rejecting traversal and auto-creating parents.
    Returns (path, quarantined, note).
    """
    base = _resolve_output_root(run_root)
    # Ensure we anchor to experiment_results even if only base folder was provided.
    env_base = os.environ.get("AISC_BASE_FOLDER", "")
    if base.name != "experiment_results" and env_base and "experiment_results" not in str(base):
        base = Path(env_base) / "experiment_results"

    subdir_rel = _sanitize_relative(Path(subdir or "."))
    name_rel = _sanitize_relative(Path(name))
    target_dir = base / subdir_rel
    target_path = target_dir / name_rel

    quarantined = False
    note = None

    try:
        if create:
            target_dir.mkdir(parents=True, exist_ok=True)
        if unique:
            target_path = _make_unique(target_path)
    except Exception as exc:
        if not allow_quarantine:
            raise
        quarantined = True
        note = f"Primary path unavailable ({exc}); routed to _unrouted."
        target_dir = base / "_unrouted" / subdir_rel
        if create:
            target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / name_rel
        if unique:
            target_path = _make_unique(target_path)

    return target_path, quarantined, note
