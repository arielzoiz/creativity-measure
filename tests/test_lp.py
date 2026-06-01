import torch
from creativity_measure.distances.lp import LpDistance


def test_known_distances():
    X = torch.tensor([[0., 0.], [3., 4.]])
    refs = torch.tensor([[0., 0.]])
    D = LpDistance(2.0)
    out = D.pairwise(X, refs)
    assert out.shape == (2, 1)
    assert torch.allclose(out[:, 0], torch.tensor([0., 5.]))


def test_output_shape():
    X = torch.randn(3, 7)
    refs = torch.randn(4, 7)
    D = LpDistance(2.0)
    out = D.pairwise(X, refs)
    assert out.shape == (3, 4)


def test_p_variants():
    X = torch.tensor([[0., 0.], [3., 4.]])
    refs = torch.tensor([[0., 0.]])
    l1 = LpDistance(1.0).pairwise(X, refs)
    linf = LpDistance(float("inf")).pairwise(X, refs)
    assert torch.allclose(l1[:, 0], torch.tensor([0., 7.]))    # |3| + |4|
    assert torch.allclose(linf[:, 0], torch.tensor([0., 4.]))  # max(|3|, |4|)
