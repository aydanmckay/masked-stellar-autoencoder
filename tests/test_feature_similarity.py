import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist

from masked_stellar_autoencoder.models.model import FeatureSimilarity


def test_feature_similarity_l2_basic():
    """Test FeatureSimilarity computes negative L2 distance correctly on simple inputs."""
    features = torch.tensor([[0.0, 0.0], [3.0, 4.0]])

    fs = FeatureSimilarity(similarity_type="l2")
    out = fs(features)

    # Pairwise L2 distance between [0, 0] and [3, 4] is 5.
    expected_dist = torch.tensor([[0.0, 5.0], [5.0, 0.0]])

    # FeatureSimilarity returns negative cdist
    expected_out = -expected_dist

    assert torch.allclose(out, expected_out)


def test_feature_similarity_l2_matches_scipy():
    """Test FeatureSimilarity matches scipy cdist output."""
    bs, feat_dim = 16, 32
    features_np = np.random.randn(bs, feat_dim)
    features = torch.tensor(features_np, dtype=torch.float32)

    fs = FeatureSimilarity(similarity_type="l2")
    out = fs(features)

    # compute scipy distance
    scipy_dist = cdist(features_np, features_np, metric="euclidean")
    expected_out = -torch.tensor(scipy_dist, dtype=torch.float32)

    assert torch.allclose(out, expected_out, atol=1e-5)


def test_feature_similarity_invalid_type():
    """Test FeatureSimilarity raises ValueError on invalid similarity type."""
    fs = FeatureSimilarity(similarity_type="invalid")
    features = torch.randn(4, 10)

    with pytest.raises(ValueError, match="invalid"):
        fs(features)
