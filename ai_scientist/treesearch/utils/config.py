"""configuration and setup utils"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, cast, Literal, Optional

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from . import tree_export
from . import copytree, preproc_data, serialize

shutup.mute_warnings()
logging.basicConfig(
    level="WARNING", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("ai-scientist")
logger.setLevel(logging.WARNING)


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class ThinkingConfig:
    type: str
    budget_tokens: Optional[int] = None


@dataclass
class StageConfig:
    model: str
    temp: float
    thinking: ThinkingConfig
    betas: str
    max_tokens: Optional[int] = None


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int


@dataclass
class DebugConfig:
    stage4: bool


@dataclass
class AgentConfig:
    steps: int
    stages: dict[str, int]
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool

    code: StageConfig
    feedback: StageConfig
    vlm_feedback: StageConfig

    search: SearchConfig
    num_workers: int
    type: str
    multi_seed_eval: dict[str, int]

    summary: Optional[StageConfig] = None
    select_node: Optional[StageConfig] = None

@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool


@dataclass
class ExperimentConfig:
    num_syn_datasets: int


@dataclass
class BiologyModelingConfig:
    framework: str = "unspecified"
    time_horizon: float = 100.0
    num_time_points: int = 1000


@dataclass
class BiologyTargetsConfig:
    primary_quantity: str = "unspecified"
    secondary_quantities: list[str] = field(default_factory=list)


@dataclass
class BiologyReproducibilityConfig:
    require_random_seed_logging: bool = True
    require_environment_description: bool = True
    require_dataset_accessions: bool = True
    require_code_archive: bool = True


@dataclass
class BiologyCitationPolicyConfig:
    use_semantic_scholar: bool = True
    min_references: int = 10
    require_dataset_citations: bool = True
    require_model_citations: bool = True
    require_software_citations: bool = False


@dataclass
class BiologyConfig:
    research_type: str = "applied"   # "applied" (bioinformatics) or "theoretical" (mathematical modeling)
    domain: str = "unspecified"      # e.g., "evolutionary_dynamics", "protein_structure"
    phases: list[str] = field(
        default_factory=lambda: [
            "conceptualization",
            "modeling",
            "simulation",
            "analysis",
            "interpretation",
            "communication",
        ]
    )
    modeling: BiologyModelingConfig = field(default_factory=BiologyModelingConfig)
    targets: BiologyTargetsConfig = field(default_factory=BiologyTargetsConfig)
    eval_focus: str = "unspecified"
    reproducibility: BiologyReproducibilityConfig = field(default_factory=BiologyReproducibilityConfig)
    citation_policy: BiologyCitationPolicyConfig = field(default_factory=BiologyCitationPolicyConfig)


@dataclass
class Config(Hashable):
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    generate_report: bool
    report: StageConfig
    agent: AgentConfig
    experiment: ExperimentConfig
    debug: DebugConfig

    biology: Optional[BiologyConfig] = None


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if (current_index := int(p.name.split("-")[0])) > max_index:
                max_index = current_index
        except ValueError:
            pass
    print("max_index: ", max_index)
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=False
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    ind = max(_get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir))
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.exp_name = f"{ind}-{cfg.exp_name}"

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    if cfg.agent.type not in ["parallel", "sequential"]:
        raise ValueError("agent.type must be either 'parallel' or 'sequential'")

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    # Add biological configuration section if available
    biology_cfg = getattr(cfg, "biology", None)
    if biology_cfg is not None:
        lines: list[str] = []
        if getattr(biology_cfg, "research_type", None):
            lines.append(f"Research type: {biology_cfg.research_type}")
        if getattr(biology_cfg, "domain", None):
            lines.append(f"Domain: {biology_cfg.domain}")
        if getattr(biology_cfg, "phases", None):
            phases_str = ", ".join(biology_cfg.phases)
            lines.append(f"Phases: {phases_str}")
        if getattr(biology_cfg, "modeling", None) is not None:
            if getattr(biology_cfg.modeling, "framework", None):
                lines.append(f"Modeling framework: {biology_cfg.modeling.framework}")
            if getattr(biology_cfg.modeling, "time_horizon", None) is not None:
                lines.append(f"Time horizon: {biology_cfg.modeling.time_horizon}")
            if getattr(biology_cfg.modeling, "num_time_points", None) is not None:
                lines.append(
                    f"Number of time points: {biology_cfg.modeling.num_time_points}"
                )
        if getattr(biology_cfg, "targets", None) is not None:
            if getattr(biology_cfg.targets, "primary_quantity", None):
                lines.append(
                    f"Primary quantity of interest: {biology_cfg.targets.primary_quantity}"
                )
            if getattr(biology_cfg.targets, "secondary_quantities", None):
                if biology_cfg.targets.secondary_quantities:
                    sec = ", ".join(biology_cfg.targets.secondary_quantities)
                    lines.append(f"Secondary quantities: {sec}")
        if getattr(biology_cfg, "eval_focus", None):
            lines.append(f"Evaluation focus: {biology_cfg.eval_focus}")

        # Reproducibility standards
        repro = getattr(biology_cfg, "reproducibility", None)
        if repro is not None:
            repro_items = []
            if getattr(repro, "require_random_seed_logging", None):
                repro_items.append("log random seeds")
            if getattr(repro, "require_environment_description", None):
                repro_items.append("describe software/hardware environment")
            if getattr(repro, "require_dataset_accessions", None):
                repro_items.append("record dataset accessions/versions")
            if getattr(repro, "require_code_archive", None):
                repro_items.append("archive code used for experiments")
            if repro_items:
                lines.append("Reproducibility standards: " + "; ".join(repro_items))

        # Citation policy
        cite_pol = getattr(biology_cfg, "citation_policy", None)
        if cite_pol is not None:
            policy_bits = []
            if getattr(cite_pol, "use_semantic_scholar", None):
                policy_bits.append("use Semantic Scholar for citation search")
            if getattr(cite_pol, "min_references", None):
                policy_bits.append(f"aim for ≥{cite_pol.min_references} references")
            if getattr(cite_pol, "require_dataset_citations", None):
                policy_bits.append("cite datasets explicitly")
            if getattr(cite_pol, "require_model_citations", None):
                policy_bits.append("cite key models/algorithms")
            if getattr(cite_pol, "require_software_citations", None):
                policy_bits.append("cite critical software tools")
            if policy_bits:
                lines.append("Citation policy: " + "; ".join(policy_bits))

        if lines:
            task_desc["Biology configuration"] = "\n".join(lines)

    print(task_desc)
    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal, stage_name: str = None):
    if stage_name is None:
        stage_name = "NoStageRun"
    save_dir = cfg.log_dir / stage_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    try:
        serialize.dump_json(journal, save_dir / "journal.json")
    except Exception as e:
        print(f"Error saving journal: {e}")
        raise
    # save config
    try:
        OmegaConf.save(config=cfg, f=save_dir / "config.yaml")
    except Exception as e:
        print(f"Error saving config: {e}")
        raise
    # create the tree + code visualization
    try:
        tree_export.generate(cfg, journal, save_dir / "tree_plot.html")
    except Exception as e:
        print(f"Error generating tree: {e}")
        raise
    # save the best found solution
    try:
        best_node = journal.get_best_node(only_good=False, cfg=cfg)
        if best_node is not None:
            for existing_file in save_dir.glob("best_solution_*.py"):
                existing_file.unlink()
            # Create new best solution file
            filename = f"best_solution_{best_node.id}.py"
            with open(save_dir / filename, "w") as f:
                f.write(best_node.code)
            # save best_node.id to a text file
            with open(save_dir / "best_node_id.txt", "w") as f:
                f.write(str(best_node.id))
        else:
            print("No best node found yet")
    except Exception as e:
        print(f"Error saving best solution: {e}")
