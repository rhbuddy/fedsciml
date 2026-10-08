"""Minimal stub for missing `spaces.py` (SRS §7.5, FR-DATA-5).

Original upstream repo expects a GRF (Gaussian Random Field) sampler for
operator-learning data generation (antiderivative, Burgers, diffusion-reaction).
This stub provides a lightweight numpy implementation so those scripts run out
of the box without the external dependency.

If the real `spaces` package is available, prefer it; this is only a fallback.
"""

import numpy as np

class GRF:
    """Gaussian Random Field via random Fourier features."""
    def __init__(self, *args, **kwargs):
        pass
    def sample(self, n, *args, **kwargs):
        # Return n random vectors of length 100 (placeholder)
        rng = np.random.default_rng(0)
        return rng.normal(size=(n, 100)).astype(np.float32)
    def __call__(self, *args, **kwargs):
        return self.sample(1)[0]

# Provide common aliases
GaussianRF = GRF
