import torch
from creativity_measure.distances.base import EuclideanDistance


def test_known_distances():
    X = torch.tensor([[0., 0.], [3., 4.]])
    refs = torch.tensor([[0., 0.]])
    D = EuclideanDistance()
    out = D.pairwise(X, refs)
    assert out.shape == (2, 1)
    assert torch.allclose(out[:, 0], torch.tensor([0., 5.]))


def test_output_shape():
    X = torch.randn(3, 7)
    refs = torch.randn(4, 7)
    D = EuclideanDistance()
    out = D.pairwise(X, refs)
    assert out.shape == (3, 4)
