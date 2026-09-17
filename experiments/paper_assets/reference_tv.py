"""New 2026-09-15 audit implementation; never used to generate historical results."""
import math
import numpy as np

EPS = 1e-8
TAU = 1e-12

def tv_reference(array):
    """Channel-wise accumulation, with float64 conversion before arithmetic."""
    a = np.asarray(array, dtype=np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3 or min(a.shape) == 0 or not np.isfinite(a).all():
        raise ValueError("Expected a nonempty finite HWC or HW array")
    energy = np.full(a.shape[:2], EPS * EPS, dtype=np.float64)
    for c in range(a.shape[2]):
        energy[:, :-1] += np.square(a[:, 1:, c] - a[:, :-1, c])
        energy[:-1, :] += np.square(a[1:, :, c] - a[:-1, :, c])
    return float(np.sum(np.sqrt(energy), dtype=np.float64) / (a.shape[0] * a.shape[1]))

def tv_loop(array):
    """Scalar loops and compensated final sum, for tiny independent test cases."""
    a = np.asarray(array, dtype=np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    h, w, channels = a.shape
    values = []
    for i in range(h):
        for j in range(w):
            terms = [EPS * EPS]
            for c in range(channels):
                if i + 1 < h:
                    terms.append((float(a[i + 1, j, c]) - float(a[i, j, c])) ** 2)
                if j + 1 < w:
                    terms.append((float(a[i, j + 1, c]) - float(a[i, j, c])) ** 2)
            values.append(math.sqrt(math.fsum(terms)))
    return math.fsum(values) / (h * w)

def output_audit(f, v, u, beta):
    vf, vv, vu = map(tv_reference, (f, v, u))
    budget = float(beta) * (vf - EPS)  # Deliberately no clamp.
    removed = max(0.0, vf - vu)
    residual = removed - budget
    return dict(V_f=vf, V_v=vv, V_u=vu, B=budget, R=removed,
                signed_residual=residual, positive_residual=max(0., residual),
                accepted_under_tau=bool(residual <= TAU), finite_status='finite')
