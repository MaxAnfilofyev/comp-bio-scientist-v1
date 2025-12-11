import os
from pathlib import Path
from typing import Tuple, Optional

# ...

def _sanitize_relative(path: Path) -> Path:
    """
    Ensure a path is relative and does not contain traversal components.
    Raises ValueError if the path attempts to escape the base directory.
    """
    # Resolve to remove .. and . components
    try:
        # Convert to string and check for absolute paths
        path_str = str(path)
        if os.path.isabs(path_str):
            raise ValueError(f"Path must be relative, got absolute path: {path}")
        
        # Normalize the path to remove .. and .
        normalized = Path(os.path.normpath(path_str))
        
        # Check if any part starts with .. (path traversal attempt)
        parts = normalized.parts
        if any(part == ".." for part in parts):
            raise ValueError(f"Path traversal not allowed: {path}")
        
        return normalized
    except Exception as exc:
        raise ValueError(f"Invalid path: {path}") from exc


def _make_unique(path: Path) -> Path:
    """
    If path exists, append a counter to make it unique.
    Returns a path that doesn't exist yet.
    """
    if not path.exists():
        return path
    
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1
        # Safety limit to prevent infinite loops
        if counter > 10000:
            raise ValueError(f"Could not create unique path for {path}")


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
