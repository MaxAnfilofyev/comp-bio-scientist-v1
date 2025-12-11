import os
import shutil
from pathlib import Path
from typing import Union

from ai_scientist.utils.health import log_missing_or_corrupt
from ai_scientist.utils.pathing import resolve_output_path


def normalize_sim_dir(sim_dir: Union[str, Path]) -> Path:
    """
    Normalize a sim directory to a canonical path anchored via resolve_output_path.
    If a duplicate run_root prefix is detected (e.g., experiment_results/.../experiment_results/...),
    strip the extra prefix.
    """
    base_root = resolve_output_path(subdir=None, name="", run_root=None, allow_quarantine=False, unique=False)[0]
    sim_dir_path = Path(sim_dir)
    # Collapse any duplicated experiment_results prefix regardless of absolute/relative path shape
    duplicate_token = f"{os.sep}experiment_results{os.sep}experiment_results"
    path_str = str(sim_dir_path)
    while duplicate_token in path_str:
        path_str = path_str.replace(duplicate_token, f"{os.sep}experiment_results")
    sim_dir_path = Path(path_str)
    # Re-anchor to base_root to ensure we are under experiment_results
    if not sim_dir_path.is_absolute():
        sim_dir_path = base_root / sim_dir_path
    sim_dir_path.mkdir(parents=True, exist_ok=True)
    return sim_dir_path


def safe_move_into_sim_dir(src: Path, sim_dir: Path) -> Path:
    """
    Move an artifact into the canonical sim_dir safely.
    If already under sim_dir, returns src.
    Uses atomic rename when possible; falls back to copy+fsync+remove.
    On failure, logs to health and returns the original path.
    """
    src = Path(src)
    sim_dir = Path(sim_dir)
    try:
        sim_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive
        log_missing_or_corrupt([{"move_error": f"mkdir failed for {sim_dir}: {exc}"}], filename="repair_move_failed.json")
        return src

    try:
        src.relative_to(sim_dir)
        return src
    except Exception:
        pass

    dest = sim_dir / src.name
    try:
        src.replace(dest)
        return dest
    except Exception:
        try:
            # Cross-device fallback
            shutil.copy2(src, dest)
            dest_fd = os.open(dest, os.O_RDONLY)
            try:
                os.fsync(dest_fd)
            finally:
                os.close(dest_fd)
            src.unlink()
            return dest
        except Exception as exc:  # pragma: no cover - defensive
            log_missing_or_corrupt(
                [{"move_error": f"failed to move {src} -> {dest}: {exc}"}],
                filename="repair_move_failed.json",
            )
            return src
