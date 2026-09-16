"""
src/utils/config.py

WHY?
    Hard-coding paths, test sizes, and hyperparameters directly in your notebook means
    every experiment requires editing code. Pulling those values out into a YAML file
    means you can change an experiment by editing a config, not by hunting through cells --
    and it means your experiment settings are captured in a diffable, reviewable file.

WHAT?
    `load_config()` reads a YAML file into a plain nested dict, with a clear error if the
    file is missing.

HOW?
    from src.utils.config import load_config
    config = load_config("configs/config.yaml")

    df = load_data(config["data"]["path"])
    X_train, X_test, y_train, y_test = split_data(
        df, target=config["data"]["target"], **config["split"]
    )
"""
from pathlib import Path
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load a YAML configuration file into a dict.

    Parameters
    ----------
    path : str, default "configs/config.yaml"

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError
        If no file exists at `path`.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No config file found at '{path}'.")

    with open(file_path) as f:
        config = yaml.safe_load(f)
    return config
