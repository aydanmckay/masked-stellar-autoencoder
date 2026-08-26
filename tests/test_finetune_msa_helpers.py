from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("torch")

from masked_stellar_autoencoder.training.finetune_msa import _print_parallax_consistency


def test_print_parallax_consistency_legacy(capsys):
    pack = {
        "astrometry_input_policy": "legacy_raw",
        "parallax_target_space": "linear_mas",
        "featurescaler": MagicMock(center_=[0.0, 1.0], scale_=[1.0, 2.0]),
    }
    scalers = [
        MagicMock(),
        MagicMock(mean_=np.array([[0.5]]), scale_=np.array([[2.0]])),
    ]

    _print_parallax_consistency(pack, ["G", "PARALLAX"], scalers)

    assert "Consistency Params for Parallax" in capsys.readouterr().out


def test_print_parallax_consistency_skips_non_legacy(capsys):
    pack = {
        "astrometry_input_policy": "snr",
        "parallax_target_space": "log10_mas",
        "featurescaler": MagicMock(center_=[0.0], scale_=[1.0]),
    }

    _print_parallax_consistency(pack, ["PARALLAX"], [MagicMock()])

    captured = capsys.readouterr().out
    assert "Skipping parallax feature/label consistency check" in captured
