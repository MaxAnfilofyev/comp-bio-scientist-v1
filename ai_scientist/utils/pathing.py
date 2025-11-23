import os
import uuid
from pathlib import Path
from typing import Dict, Tuple

from ai_scientist.tools.base_tool import BaseTool


class ResolvedPath(Dict[str, object]):
    """Dictionary style return for resolved output paths."""


def _sanitize_relative(path: Path) -> Path:
    cleaned_parts = []
    for part in path.parts:
        if part in {"", ".", "/"}:
            continue
        if part == "..":
            raise ValueError("Path traversal ('..') is not allowed for outputs.")
        cleaned_parts.append(part)
    return Path(*cleaned_parts)


def _make_unique(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for _ in range(32):
        candidate = parent / f"{stem}__{uuid.uuid4().hex[:8]}{suffix}"
        if not candidate.exists():
            return candidate
    return parent / f"{stem}__{uuid.uuid4().hex}{suffix}"


def resolve_output_path(
    subdir: str | None,
    name: str,
    *,
    run_root: Path | None = None,
    create: bool = True,
    allow_quarantine: bool = True,
    unique: bool = True,
) -> Tuple[Path, bool, str | None]:
    """
    Resolve an output path anchored to experiment_results, rejecting traversal and auto-creating parents.
    Returns (path, quarantined, note).
    """
    base = run_root or BaseTool.resolve_output_dir(None)
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
