"""Trusted local checkpoint loading (full pickle; not for untrusted files)."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Union

import torch

PathLike = Union[str, Any]


def torch_load_trusted(
    fpath: PathLike,
    map_location: Optional[Any] = None,
) -> Any:
    """
    Load a checkpoint dict from disk.

    PyTorch 2.6+ defaults ``torch.load(..., weights_only=True)``, which rejects
    checkpoints that contain NumPy arrays or other non-weight objects. Training
    checkpoints from this repo are trusted; use ``weights_only=False`` when the
    API supports it.
    """
    kwargs: Dict[str, Any] = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(fpath, **kwargs)
