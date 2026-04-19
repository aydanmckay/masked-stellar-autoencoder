"""
Expand ${VAR} / ~ in YAML file paths so configs work on HPC (Narval, etc.).
"""

from __future__ import annotations

import os
from typing import Any, Dict


def expand_path(p: Any) -> Any:
    if not isinstance(p, str):
        return p
    return os.path.expandvars(os.path.expanduser(p))


def expand_config_paths(config: Dict[str, Any]) -> None:
    """In-place: resolve environment variables in known filesystem path fields."""
    data = config.get("data")
    if isinstance(data, dict):
        for k in ("datafile", "ft_datafile"):
            if k in data:
                data[k] = expand_path(data[k])

    model = config.get("model")
    if isinstance(model, dict) and "saved_weights" in model:
        model["saved_weights"] = expand_path(model["saved_weights"])

    saving = config.get("saving")
    if isinstance(saving, dict):
        for k in ("model_str", "log_file"):
            if k in saving:
                saving[k] = expand_path(saving[k])

    training = config.get("training")
    if isinstance(training, dict) and "presaved" in training:
        v = training["presaved"]
        if v is None or v == "":
            training["presaved"] = None
        elif isinstance(v, str):
            s = v.strip()
            training["presaved"] = expand_path(s) if s else None
        # else: leave non-string values untouched (unusual)

    ft = config.get("finetuning")
    if isinstance(ft, dict) and "ensemble_path" in ft:
        ft["ensemble_path"] = expand_path(ft["ensemble_path"])


def ft_checkpoint_paths(config: Dict[str, Any], seed: int) -> tuple[str, str]:
    """
    Fine-tune artifact paths. When ensemble mode is on, append _seed{seed} before the extension
    so parallel/array jobs do not clobber checkpoints.
    """
    ft_save = config["saving"]["model_str"]
    ft_log = config["saving"]["log_file"]
    if not config["finetuning"].get("ensemble"):
        return ft_save, ft_log

    def suffixed(path: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}_seed{int(seed)}{ext}"

    return suffixed(ft_save), suffixed(ft_log)
