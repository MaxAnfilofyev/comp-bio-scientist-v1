import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:
    from ai_scientist.tools.base_tool import BaseTool
    from ai_scientist.utils.pathing import resolve_output_path
except Exception:
    BaseTool = None
    resolve_output_path = None

NOTE_NAMES = {"pi_notes.md", "user_inbox.md", "run_notes.md"}


def _run_root(explicit: Optional[Path] = None) -> Path:
    if explicit:
        return Path(explicit)
    base = os.environ.get("AISC_BASE_FOLDER", "")
    return Path(base) if base else Path(".")


def _resolve_output_dir(default: str = "experiment_results") -> Path:
    if resolve_output_path:
        try:
            resolved, _, _ = resolve_output_path(subdir=None, name="", allow_quarantine=False, unique=False)
            return resolved if resolved.name == default else resolved
        except Exception:
            pass
    return BaseTool.resolve_output_dir(None, default=default) if BaseTool else Path(default)


def _normalize_name(name: str) -> str:
    name = name if name.endswith(".md") else f"{name}.md"
    if name not in NOTE_NAMES:
        raise ValueError(f"Unsupported note name '{name}'. Expected one of {sorted(NOTE_NAMES)}.")
    return name


def _atomic_write(path: Path, content: str) -> None:
    tmp_dir = path.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=tmp_dir, delete=False, prefix=path.name, suffix=".tmp") as fh:
        fh.write(content)
        tmp_name = fh.name
    os.replace(tmp_name, path)


def _read_text_with_backup(path: Path) -> Tuple[str, Optional[Path]]:
    try:
        return path.read_text(encoding="utf-8"), None
    except UnicodeDecodeError:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        return path.read_text(encoding="utf-8", errors="replace"), backup


def _ensure_shadow_link(shadow: Path, canonical: Path) -> None:
    if shadow.is_symlink():
        try:
            if shadow.resolve() == canonical:
                return
        except OSError:
            pass
        shadow.unlink(missing_ok=True)
    if shadow.exists():
        shadow.unlink()

    def _try_symlink() -> bool:
        try:
            shadow.symlink_to(canonical)
            return True
        except OSError:
            shadow.unlink(missing_ok=True)
            return False

    if _try_symlink():
        return

    shadow.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(canonical, shadow)


def ensure_note_files(name: str, run_root: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Ensure the canonical note and its shadow (symlink/copy) exist.
    Returns (canonical_path, shadow_path).
    """
    normalized = _normalize_name(name)
    root = _run_root(run_root).resolve()
    canonical = _resolve_output_dir().joinpath(normalized).resolve()
    shadow = root / normalized

    canonical.parent.mkdir(parents=True, exist_ok=True)
    shadow.parent.mkdir(parents=True, exist_ok=True)

    if shadow.exists() and not shadow.is_symlink():
        shadow_text, _ = _read_text_with_backup(shadow)
        if canonical.exists():
            canonical_text, _ = _read_text_with_backup(canonical)
            if shadow_text and shadow_text != canonical_text:
                merged = canonical_text
                if merged and not merged.endswith("\n"):
                    merged += "\n"
                merged += shadow_text
                _atomic_write(canonical, merged)
        else:
            _atomic_write(canonical, shadow_text)
        shadow.unlink(missing_ok=True)

    if not canonical.exists():
        _atomic_write(canonical, "")

    _ensure_shadow_link(shadow, canonical)
    return canonical, shadow


def read_note_file(name: str = "pi_notes.md", run_root: Optional[Path] = None) -> dict:
    try:
        normalized = _normalize_name(name)
    except ValueError as exc:
        return {"error": str(exc), "path": ""}
    canonical, _ = ensure_note_files(normalized, run_root)
    try:
        content, backup = _read_text_with_backup(canonical)
        result = {"path": str(canonical), "content": content}
        if backup:
            result["warning"] = f"Recovered from backup {backup}"
        return result
    except Exception as exc:
        return {"path": str(canonical), "error": str(exc)}


def write_note_file(content: str, name: str = "pi_notes.md", append: bool = False, run_root: Optional[Path] = None) -> dict:
    try:
        normalized = _normalize_name(name)
    except ValueError as exc:
        return {"error": str(exc), "path": ""}
    canonical, shadow = ensure_note_files(normalized, run_root)
    try:
        warning: Optional[str] = None
        if append and canonical.exists():
            existing, backup = _read_text_with_backup(canonical)
            if backup:
                warning = f"Recovered from backup {backup}"
            if existing and content:
                if not existing.endswith("\n"):
                    existing += "\n"
                content = existing + content
            else:
                content = existing + content
        _atomic_write(canonical, content)
        _ensure_shadow_link(shadow, canonical)
        result = {"path": str(canonical)}
        if append and warning:
            result["warning"] = warning
        return result
    except Exception as exc:
        pending = canonical.parent / f"pending_{normalized}"
        try:
            _atomic_write(pending, content)
            return {"path": str(pending), "warning": f"write failed for canonical {canonical}: {exc}"}
        except Exception as pending_exc:
            return {"error": f"failed to write note: {pending_exc}", "path": str(canonical)}


def append_run_note(category: str, text: str, actor: str = "system", run_root: Optional[Path] = None) -> dict:
    """
    Append a run note to experiment_results/run_notes.md. Keeps manifest free of reflections.
    """
    try:
        normalized = _normalize_name("run_notes.md")
    except ValueError as exc:
        return {"error": str(exc), "path": ""}
    canonical, shadow = ensure_note_files(normalized, run_root)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"### [{ts}] {actor} :: {category}\n"
    body = text.strip() + ("\n" if text.strip() else "")
    payload = header + body + "\n"
    try:
        with open(canonical, "a", encoding="utf-8") as f:
            f.write(payload)
        _ensure_shadow_link(shadow, canonical)
        return {"path": str(canonical), "note_len": len(payload)}
    except Exception as exc:
        return {"error": str(exc), "path": str(canonical)}
