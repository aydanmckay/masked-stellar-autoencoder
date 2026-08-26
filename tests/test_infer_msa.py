from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from masked_stellar_autoencoder.training.infer_msa import (
    infer_catalogue,
    load_inference_data,
    save_predictions,
)


class _EncoderOnly(torch.nn.Module):
    def __init__(self, latent_dim: int = 4):
        super().__init__()
        self.encoder = torch.nn.Linear(2, latent_dim)

    def forward(self, x):
        return self.encoder(x)


class _QuantileHead(torch.nn.Module):
    def forward(self, z):
        batch_size, n_labels = z.shape[0], 2
        out = torch.zeros(batch_size, n_labels, 3)
        out[:, :, 0] = z[:, :n_labels] - 1.0
        out[:, :, 1] = z[:, :n_labels]
        out[:, :, 2] = z[:, :n_labels] + 1.0
        return out


def test_infer_catalogue_shapes():
    model = _EncoderOnly(latent_dim=4)
    head = _QuantileHead()
    x = np.arange(12, dtype=np.float32).reshape(6, 2)

    embeddings, quantiles = infer_catalogue(
        model, head, x, torch.device("cpu"), batch_size=2
    )

    assert embeddings.shape == (6, 4)
    assert quantiles.shape == (6, 2, 3)


def test_load_inference_data_from_test_pack():
    pack = {"testset": np.ones((3, 2), dtype=np.float64)}
    source_ids, x_scaled = load_inference_data(None, pack, MagicMock(), ["a", "b"])

    assert len(source_ids) == 3
    np.testing.assert_array_equal(x_scaled, pack["testset"])


def test_load_inference_data_from_hdf5():
    cols = ["f1", "f2"]
    raw = np.array(
        [(1.0, 2.0), (3.0, 4.0)],
        dtype=[("f1", "f8"), ("f2", "f8")],
    )
    mock_dset = MagicMock()
    mock_dset.dtype.names = ("f1", "f2")
    mock_dset.__getitem__.side_effect = lambda name: raw[name]

    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.__contains__.return_value = False
    mock_file.keys.return_value = ["data"]
    mock_file.__getitem__.return_value = mock_dset
    mock_file.get.return_value = np.array([10, 20])

    scaler = MagicMock()
    scaler.transform.return_value = np.array([[0.0, 0.0], [1.0, 1.0]])

    with patch(
        "masked_stellar_autoencoder.training.infer_msa.h5py.File",
        return_value=mock_file,
    ):
        source_ids, x_scaled = load_inference_data("fake.h5", {}, scaler, cols)

    assert list(source_ids) == [10, 20]
    np.testing.assert_array_equal(x_scaled, [[0.0, 0.0], [1.0, 1.0]])
    scaler.transform.assert_called_once()


def test_save_predictions_csv(tmp_path):
    out = tmp_path / "out.csv"
    source_ids = np.array([1, 2])
    label_names = ["teff", "fe_h"]
    phys_q = np.array(
        [
            [[4000.0, 5000.0, 6000.0], [-1.0, 0.0, 1.0]],
            [[4100.0, 5100.0, 6100.0], [-0.5, 0.5, 1.5]],
        ]
    )
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])

    save_predictions(str(out), source_ids, label_names, phys_q, embeddings)

    text = out.read_text()
    assert "teff_med" in text
    assert "fe_h_lower" in text
    assert "embedding_0" in text
