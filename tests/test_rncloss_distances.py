import pytest
import os
import sys

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _repo)

torch = pytest.importorskip("torch")

from models.model import LabelDifference, FeatureSimilarity

def test_label_difference_valid_type():
    ld = LabelDifference(distance_type='l1')
    labels = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    res = ld(labels)
    assert res.shape == (2, 2)
    # L1 distance: |1-3| + |2-4| = 2 + 2 = 4
    assert torch.isclose(res[0, 1], torch.tensor(4.0))

def test_label_difference_invalid_type():
    ld = LabelDifference(distance_type='invalid_type')
    labels = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match='invalid_type'):
        ld(labels)

def test_feature_similarity_valid_type():
    fs = FeatureSimilarity(similarity_type='l2')
    features = torch.tensor([[1.0, 2.0], [4.0, 6.0]])
    res = fs(features)
    assert res.shape == (2, 2)
    # Negative L2 distance: -sqrt((1-4)^2 + (2-6)^2) = -sqrt(9 + 16) = -5
    assert torch.isclose(res[0, 1], torch.tensor(-5.0))

def test_feature_similarity_invalid_type():
    fs = FeatureSimilarity(similarity_type='invalid_type')
    features = torch.tensor([[1.0, 2.0], [4.0, 6.0]])
    with pytest.raises(ValueError, match='invalid_type'):
        fs(features)
