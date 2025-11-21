import os
import os.path as osp
import shutil
import yaml


def idea_to_markdown(data: dict, output_path: str, load_code: str) -> None:
    """
    Convert a dictionary into a markdown file.

    Args:
        data: Dictionary containing the data to convert
        output_path: Path where the markdown file will be saved
        load_code: Path to a code file to include in the markdown
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in data.items():
            # Convert key to title format and make it a header
            header = key.replace("_", " ").title()
            f.write(f"## {header}\n\n")

            # Handle different value types
            if isinstance(value, (list, tuple)):
                for item in value:
                    f.write(f"- {item}\n")
                f.write("\n")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.write(f"### {sub_key}\n")
                    f.write(f"{sub_value}\n\n")
            else:
                f.write(f"{value}\n\n")

        # Add the code to the markdown file
        if load_code:
            # Assert that the code file exists before trying to open it
            assert os.path.exists(load_code), f"Code path at {load_code} must exist if using the 'load_code' flag. This is an optional code prompt that you may choose to include; if not, please do not set 'load_code'."
            f.write(f"## Code To Potentially Use\n\n")
            f.write(f"Use the following code as context for your experiments:\n\n")
            with open(load_code, "r") as code_file:
                code = code_file.read()
                f.write(f"```python\n{code}\n```\n\n")


def _deep_update(target: dict, updates: dict) -> dict:
    """Recursively update nested dictionaries."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            target[k] = _deep_update(target.get(k, {}), v)
        else:
            target[k] = v
    return target


def edit_bfts_config_file(
    config_path: str,
    idea_dir: str,
    idea_path: str,
    biology_overrides: dict | None = None,
) -> str:
    """
    Edit the bfts_config.yaml file to point to the idea-specific files and optionally
    override biological configuration fields.

    Args:
        config_path: Path to the base bfts_config.yaml file
        idea_dir: Directory where the idea files are located
        idea_path: Path to the idea.json file (or markdown description)
        biology_overrides: Optional nested dict of overrides for the `biology` section

    Returns:
        Path to the edited bfts_config.yaml file for this idea
    """
    run_config_path = osp.join(idea_dir, "bfts_config.yaml")
    shutil.copy(config_path, run_config_path)
    with open(run_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["desc_file"] = idea_path
    config["workspace_dir"] = idea_dir

    # make an empty data directory
    data_dir = osp.join(idea_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    config["data_dir"] = data_dir

    # make an empty log directory
    log_dir = osp.join(idea_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    config["log_dir"] = log_dir

    # Optionally override biology configuration
    if biology_overrides is not None:
        if "biology" not in config or not isinstance(config["biology"], dict):
            config["biology"] = {}
        config["biology"] = _deep_update(config["biology"], biology_overrides)

    with open(run_config_path, "w") as f:
        yaml.dump(config, f)
    return run_config_path
