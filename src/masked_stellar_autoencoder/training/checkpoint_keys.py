"""
Normalize checkpoint dict keys across pretrain vs fine-tune saves (no torch import).
"""

from typing import Any


def autoencoder_state_dict(ckpt: dict[str, Any]) -> dict[str, Any]:
    """
    Fine-tune saves use ``autoencoder_state_dict``; pretrain saves use ``model_state_dict``.
    """
    if "autoencoder_state_dict" in ckpt:
        return ckpt["autoencoder_state_dict"]
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    raise KeyError(
        "Checkpoint must contain 'autoencoder_state_dict' (fine-tune) or "
        "'model_state_dict' (pretrain) for the TabResnet autoencoder weights."
    )


def prediction_head_state_dict(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Required for eval; absent on pretrain-only checkpoints."""
    if "prediction_head_state_dict" in ckpt:
        return ckpt["prediction_head_state_dict"]
    raise KeyError(
        "Checkpoint has no 'prediction_head_state_dict'. "
        "eval_ensemble.py expects a fine-tuned checkpoint, not pretrain-only weights."
    )
