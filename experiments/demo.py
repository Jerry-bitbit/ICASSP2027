"""Run the controller on a synthetic observation without datasets or weights."""
import numpy as np
from methods.ectv import certify_candidate, tv_energy_float64
from experiments.run_controller_comparison import tv_audit


def main():
    rng = np.random.default_rng(0)
    f = rng.random((32, 32, 3), dtype=np.float32)
    v = np.full_like(f, 0.5)
    beta = 0.6
    budget = beta * (tv_energy_float64(f) - 1e-8)
    result = certify_candidate(f, v, budget_override=budget, bisection_steps=28)
    residual = max(0.0, tv_audit(f) - tv_audit(result.image)) - budget
    assert result.image.dtype == np.float32
    assert residual <= 1e-12
    print(f"beta={beta}, eta={result.eta:.9f}, budget={budget:.9f}")
    print(f"Independent float64 audit: residual={residual:.3e} <= 1e-12")


if __name__ == "__main__":
    main()
