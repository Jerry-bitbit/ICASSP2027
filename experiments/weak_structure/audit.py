"""Independent audit of arrays loaded from disk; never reconstructs from eta."""
import numpy as np

EPS = 1e-8
TOL = 1e-12


def tv_reference(array):
    # Channel-wise accumulation differs from the production vectorized TV.
    a = np.asarray(array, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] != 3 or not np.isfinite(a).all():
        raise ValueError("TV audit requires finite HWC RGB")
    energy = np.full(a.shape[:2], EPS * EPS, dtype=np.float64)
    for c in range(3):
        energy[:, :-1] += np.square(a[:, 1:, c] - a[:, :-1, c])
        energy[:-1, :] += np.square(a[1:, :, c] - a[:-1, :, c])
    return float(np.sqrt(energy).mean(dtype=np.float64))


def audit_output(f, u, beta):
    initial, output = tv_reference(f), tv_reference(u)
    budget = float(beta) * (initial - EPS)
    residual = max(0.0, initial - output) - budget
    return {"tv": output, "budget": budget, "budget_residual": residual,
            "budget_violation": max(0.0, residual), "audit_pass": residual <= TOL}
