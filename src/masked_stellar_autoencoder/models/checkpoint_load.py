"""Trusted local checkpoint loading (full pickle; not for untrusted files)."""

import inspect
from typing import Any

import torch

PathLike = str | Any


def torch_load_trusted(
    fpath: PathLike,
    map_location: Any | None = None,
    weights_only: bool = True,
) -> Any:
    """
    Load a checkpoint dict from disk.

    By default, uses ``weights_only=True`` for security against malicious
    checkpoints. If loading older checkpoints from this repo that contain
    NumPy arrays, you must explicitly pass ``weights_only=False`` and ensure
    the checkpoint is trusted.
    """
    kwargs: dict[str, Any] = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = weights_only
    return torch.load(fpath, **kwargs)
