from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from masked_stellar_autoencoder.training.finetune_data import prepare_finetune_arrays


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
