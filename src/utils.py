"""
General utility functions used across the MEMS Scientific Machine
Learning project.

This module contains only lightweight project utilities such as:

- YAML configuration loading,
- directory creation,
- reproducible random-number generator creation.

Scientific modelling logic belongs in the dedicated source modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml_config(
    file_path: str | Path,
) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.

    ValueError
        If the YAML file is empty.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            f"Configuration file is empty: {path}"
        )

    return config


def ensure_directory(
    directory_path: str | Path,
) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    directory_path : str or pathlib.Path
        Directory to create.

    Returns
    -------
    pathlib.Path
        Path object representing the directory.
    """

    path = Path(directory_path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def create_random_generator(
    seed: int | None = None,
) -> np.random.Generator:
    """
    Create a NumPy random-number generator.

    Parameters
    ----------
    seed : int or None, optional
        Random seed.

    Returns
    -------
    numpy.random.Generator
        NumPy random-number generator.
    """

    return np.random.default_rng(seed)


def project_root() -> Path:
    """
    Return the root directory of the project.

    The function assumes the source file is located inside:

        project_root/src/utils.py

    Returns
    -------
    pathlib.Path
        Project root directory.
    """

    return Path(__file__).resolve().parent.parent


def config_path(
    filename: str,
) -> Path:
    """
    Return the full path to a file in the configs directory.

    Parameters
    ----------
    filename : str
        Configuration filename.

    Returns
    -------
    pathlib.Path
        Full configuration-file path.
    """

    return project_root() / "configs" / filename


def data_path(
    *parts: str,
) -> Path:
    """
    Construct a path inside the data directory.

    Examples
    --------
    data_path("processed", "features.csv")

    Parameters
    ----------
    *parts : str
        Subdirectories or filename.

    Returns
    -------
    pathlib.Path
        Constructed path.
    """

    return project_root() / "data" / Path(*parts)


def results_path(
    *parts: str,
) -> Path:
    """
    Construct a path inside the results directory.

    Parameters
    ----------
    *parts : str
        Subdirectories or filename.

    Returns
    -------
    pathlib.Path
        Constructed path.
    """

    return project_root() / "results" / Path(*parts)


def figures_path(
    *parts: str,
) -> Path:
    """
    Construct a path inside the figures directory.

    Parameters
    ----------
    *parts : str
        Subdirectories or filename.

    Returns
    -------
    pathlib.Path
        Constructed path.
    """

    return project_root() / "figures" / Path(*parts)