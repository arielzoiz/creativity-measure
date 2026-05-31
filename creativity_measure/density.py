# creativity_measure/density.py
import torch


class Density:
    """
    A probability distribution p, represented by evaluable log-densities.

    Generalizes the notebook's module-level log_pX / log_p_Y into an object.

    Args:
        log_p_X: callable (x: (..., d)) -> (...)        log p_X(x)
        log_p_Y: callable (y: (..., d), gamma) -> (...) log p_{Y_gamma}(y),
                 the marginal of Y = gamma*X + sqrt(gamma)*W, W ~ N(0, I).
                 For a GMM this is analytic (see demo). Used by both IEM
                 distances (Hessian for local, score for global).
        sample:  optional callable (n: int) -> (n, d). Convenience for drawing
                 reference points x' ~ p; the library never calls it internally
                 inside the distances -- x_refs is always passed in.
        d:       optional int, the dimensionality (else inferred on first use).
    """

    def __init__(self, log_p_X, log_p_Y, sample=None, d=None):
        self._log_p_X = log_p_X
        self._log_p_Y = log_p_Y
        self._sample = sample
        self.d = d

    def log_p_X(self, x):
        return self._log_p_X(x)

    def log_p_Y(self, y, gamma):
        return self._log_p_Y(y, gamma)

    def sample(self, n, seed=None):
        if self._sample is None:
            raise RuntimeError("This Density has no sampler; pass x_refs explicitly.")
        if seed is not None:
            torch.manual_seed(seed)
        return self._sample(n)
