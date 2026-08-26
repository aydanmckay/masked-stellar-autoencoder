from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from masked_stellar_autoencoder.training.finetune_data import (
    _augment_below_feh,
    _filter_metal_poor,
    _scale_paired_label_blocks,
    _split_data,
    prepare_finetune_arrays,
)


def test_prepare_finetune_arrays_invalid_label_scaler():
    config = {
        "data": {
            "ft_datafile": "dummy.fits",
            "feature_cols": ["f1"],
            "classes": ["teff", "fe_h"],
            "error_cols": ["e_teff", "e_fe_h"],
            "recon_cols": ["r1"],
        },
        "finetuning": {},
        "preprocessing": {"label_scaler": "invalid_scaler_type"},
    }

    mock_df = pd.DataFrame(
        {
            "teff": [5000.0] * 20,
            "fe_h": [0.0] * 20,
            "f1": [1.0] * 20,
            "e_teff": [10.0] * 20,
            "e_fe_h": [0.1] * 20,
        }
    )

    with patch(
        "masked_stellar_autoencoder.training.finetune_data.Table.read"
    ) as mock_read:
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = mock_df
        mock_read.return_value = mock_table

        with pytest.raises(
            ValueError,
            match="preprocessing.label_scaler must be 'standard', 'robust', or 'power', got 'invalid_scaler_type'",
        ):
            prepare_finetune_arrays(config)


def _sample_frames(n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.DataFrame(
        {
            "teff": np.linspace(4000, 6000, n),
            "fe_h": np.linspace(-2.5, 0.5, n),
            "e_fe_h": np.full(n, 0.1),
            "f1": np.ones(n),
        }
    )
    errordata = pd.DataFrame({"e_teff": np.full(n, 50.0), "e_fe_h": data["e_fe_h"]})
    return data, errordata


def test_filter_metal_poor_drops_rows_by_thresholds():
    data, errordata = _sample_frames()
    data.loc[0, "e_fe_h"] = np.nan
    data.loc[1, "teff"] = 3500.0
    data.loc[2, "e_fe_h"] = 0.5

    mp = {"require_finite_e_fe_h": True, "max_e_fe_h": 0.2, "min_teff": 4000.0}
    out_data, out_err = _filter_metal_poor(data, errordata, mp)

    assert len(out_data) == len(data) - 3
    assert len(out_err) == len(out_data)
    assert out_data["e_fe_h"].notna().all()
    assert (out_data["e_fe_h"] <= 0.2).all()
    assert (out_data["teff"] >= 4000.0).all()


def test_split_data_falls_back_when_stratify_fails(capsys):
    data, errordata = _sample_frames(n=8)
    mp = {"stratify_feh": True, "feh_stratify_bins": [-np.inf, 0.0, np.inf]}

    with patch(
        "masked_stellar_autoencoder.training.finetune_data.train_test_split",
        side_effect=[
            ValueError("too few samples per class"),
            (
                data.iloc[:5].to_numpy(),
                data.iloc[5:].to_numpy(),
                errordata.iloc[:5].to_numpy(),
                errordata.iloc[5:].to_numpy(),
            ),
            (
                data.iloc[5:7].to_numpy(),
                data.iloc[7:].to_numpy(),
                errordata.iloc[5:7].to_numpy(),
                errordata.iloc[7:].to_numpy(),
            ),
        ],
    ):
        splits = _split_data(data, errordata, mp)

    assert len(splits) == 6
    captured = capsys.readouterr()
    assert "stratified split failed" in captured.out


def test_augment_below_feh_duplicates_metal_poor_rows():
    trainset = np.array([[5000.0, -2.5], [5200.0, -0.5], [5400.0, 0.0]])
    etrainset = np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
    mp = {"augment_below_feh": -1.0, "augment_fraction": 0.5}

    out_train, out_err = _augment_below_feh(trainset, etrainset, mp, feh_col=1, seed=0)

    assert out_train.shape[0] == trainset.shape[0] + 1
    assert out_err.shape[0] == out_train.shape[0]
    assert np.all(out_train[-1, 1] < -1.0)


def test_scale_paired_label_blocks_standard_scaler():
    target_train = np.array([[5000.0, 50.0, -1.0, 0.1]], dtype=np.float64)
    target_valid = np.array([[5100.0, 55.0, -0.5, 0.1]], dtype=np.float64)

    labelled, e_labelled, vlabelled, e_vlabelled, scalers = _scale_paired_label_blocks(
        target_train, target_valid, num_classes=4, scaler_cls=StandardScaler
    )

    assert labelled.shape == (1, 2)
    assert e_labelled.shape == (1, 2)
    assert vlabelled.shape == (1, 2)
    assert len(scalers) == 2


def test_augment_below_feh_noop_when_no_metal_poor_rows():
    trainset = np.array([[5000.0, 0.0], [5200.0, 0.5]])
    etrainset = np.array([[0.1, 0.1], [0.1, 0.1]])
    mp = {"augment_below_feh": -1.0, "augment_fraction": 0.5}

    out_train, out_err = _augment_below_feh(trainset, etrainset, mp, feh_col=1, seed=0)

    np.testing.assert_array_equal(out_train, trainset)
    np.testing.assert_array_equal(out_err, etrainset)
