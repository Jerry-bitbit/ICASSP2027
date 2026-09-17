"""Ground-truth-free adapter to the paper's unmodified controller."""
import numpy as np
from methods.ectv import certify_candidate, tv_energy_float64


def control(f, v, beta):
    """Only observation, shared proposal and a scalar budget are accepted."""
    if f.dtype != np.float32 or v.dtype != np.float32:
        raise ValueError("Endpoints must be float32")
    if f.shape != v.shape or f.ndim != 3 or f.shape[-1] != 3:
        raise ValueError("Endpoints must have matching HWC RGB shapes")
    if not all(np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1 for x in (f, v)):
        raise ValueError("Endpoints must be finite and clipped to [0,1]")
    if not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0,1]")
    result = certify_candidate(
        f, v, budget_override=float(beta) * (tv_energy_float64(f) - 1e-8),
        bisection_steps=28, feasibility_tolerance=1e-12, vectorial=True)
    return result.image, {"eta": result.eta,
                          "accepted_proposal": result.accepted_without_projection}


def blend(f, v, eta):
    # Exact endpoints, especially eta=1, avoid a float32 round trip through f.
    if eta == 0:
        return f.copy()
    if eta == 1:
        return v.copy()
    return np.asarray(f + float(eta) * (v - f), dtype=np.float32)
