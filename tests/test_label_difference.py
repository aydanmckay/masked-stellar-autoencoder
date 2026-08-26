import pytest
import torch

from masked_stellar_autoencoder.models.model import LabelDifference


def test_label_difference_l1():
    # Setup dummy labels: batch size of 3, label dimension of 2
    labels = torch.tensor([[1.0, 2.0], [3.0, 0.0], [-1.0, 5.0]])

    # Instantiate the module
    module = LabelDifference(distance_type="l1")

    # Run forward pass
    output = module(labels)

    # Expected shapes
    # output: [3, 3]
    assert output.shape == (3, 3)

    # Manually calculate L1 distances
    # Pair (0, 0): |1-1| + |2-2| = 0
    # Pair (0, 1): |1-3| + |2-0| = 2 + 2 = 4
    # Pair (0, 2): |1-(-1)| + |2-5| = 2 + 3 = 5
    # Pair (1, 0): 4
    # Pair (1, 1): 0
    # Pair (1, 2): |3-(-1)| + |0-5| = 4 + 5 = 9
    # Pair (2, 0): 5
    # Pair (2, 1): 9
    # Pair (2, 2): 0

    expected = torch.tensor([[0.0, 4.0, 5.0], [4.0, 0.0, 9.0], [5.0, 9.0, 0.0]])

    assert torch.allclose(output, expected)


def test_label_difference_invalid_type():
    module = LabelDifference(distance_type="l2")
    labels = torch.randn(3, 2)

    with pytest.raises(ValueError, match="l2"):
        module(labels)
