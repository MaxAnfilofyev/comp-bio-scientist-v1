import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

class BaseTool(ABC):
    """
    An abstract base class for defining custom tools.

    Attributes:
    -----------
    - name (str): The name of the tool.
    - description (str): A short description of what the tool does.
    - parameters (list): A list of parameters that the tool requires, each parameter should be a dictionary with 'name', 'type', and 'description' key/value pairs.

    Usage:
    ------
    To use this class, you should subclass it and provide an implementation for the `use_tool` abstract method.
    """

    def __init__(self, name: str, description: str, parameters: List[Dict[str, Any]]):
        self.name = name
        self.description = description
        self.parameters = parameters

    @staticmethod
    def resolve_output_dir(output_dir: Optional[str], default: str = "experiment_results") -> Path:
        """
        Resolve an output directory, preferring the orchestrator-provided env var when present.
        Falls back to the caller-provided path, then a project-relative default.
        """
        env_dir = os.environ.get("AISC_EXP_RESULTS", "")
        target = output_dir or env_dir or default
        p = Path(target)

        if p.is_absolute():
            return p

        exp_path = Path(env_dir) if env_dir else None
        base = os.environ.get("AISC_BASE_FOLDER", "")

        if exp_path:
            # If the caller already pointed at the exp path (or inside it), respect it.
            if p == exp_path or str(p).startswith(str(exp_path)):
                return p
            # If the caller used an experiment_results-relative prefix, lift it under the base folder.
            if p.parts and p.parts[0] == "experiment_results":
                if base:
                    return Path(base) / p
                return p
            # Default: anchor relative paths under the experiment_results directory.
            return exp_path / p

        if base:
            base_path = Path(base)
            target_str = str(p)
            if not target_str.startswith(str(base_path)) and not target_str.startswith("experiments/"):
                return base_path / p

        return p

    @staticmethod
    def resolve_input_path(
        path_str: str,
        *,
        must_exist: bool = True,
        allow_dir: bool = False,
        default_subdir: str = "experiment_results",
    ) -> Path:
        """
        Resolve an input path against common run directories.
        Search order:
        1. As provided (absolute or relative).
        2. AISC_EXP_RESULTS / <name>
        3. AISC_BASE_FOLDER / default_subdir / <name>
        4. AISC_BASE_FOLDER / <name>
        If must_exist is True, will raise FileNotFoundError on miss.
        """
        if not path_str:
            raise FileNotFoundError("Empty path provided")

        candidates: List[Path] = []
        p = Path(path_str)

        # Guard against double-default-subdir paths like experiment_results/experiment_results/<file>.
        if default_subdir:
            parts = list(p.parts)
            dedup_parts: List[str] = []
            for part in parts:
                if dedup_parts and part == default_subdir and dedup_parts[-1] == default_subdir:
                    continue
                dedup_parts.append(part)
            dedup_path = Path(*dedup_parts)
            if dedup_path != p:
                candidates.append(dedup_path)

        candidates.append(p)

        env_dir = os.environ.get("AISC_EXP_RESULTS", "")
        if env_dir:
            if p.is_absolute():
                candidates.append(Path(env_dir) / p.name)
            else:
                candidates.append(Path(env_dir) / p)
                candidates.append(Path(env_dir) / p.name)

        base = os.environ.get("AISC_BASE_FOLDER", "")
        if base:
            if p.is_absolute():
                candidates.append(Path(base) / default_subdir / p.name)
                candidates.append(Path(base) / p.name)
            else:
                # If caller passed a path already starting with experiment_results, preserve it relative to base
                if str(p).startswith("experiment_results"):
                    candidates.append(Path(base) / p)
                candidates.append(Path(base) / default_subdir / p)
                candidates.append(Path(base) / p)
                candidates.append(Path(base) / default_subdir / p.name)
                candidates.append(Path(base) / p.name)

        for cand in candidates:
            if cand.exists():
                if not allow_dir and cand.is_dir():
                    continue
                return cand

        if must_exist:
            raise FileNotFoundError(f"path not found (searched {len(candidates)} locations): {path_str}")

        # If not required to exist, return the first candidate anchored to base if relative
        chosen = candidates[0]
        if not chosen.is_absolute() and base:
            chosen = Path(base) / chosen
        return chosen

    @abstractmethod
    def use_tool(self, **kwargs) -> Any:
        """Abstract method that should be implemented by subclasses to define the functionality of the tool."""
        pass
