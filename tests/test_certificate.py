import unittest

import numpy as np

from methods.ectv import certify_candidate, tv_energy_float64


class CertificateTests(unittest.TestCase):
    def test_explicit_budget_and_float64_audit(self) -> None:
        rng = np.random.default_rng(7)
        noisy = rng.random((31, 29, 3), dtype=np.float32)
        proposal = np.full_like(noisy, 0.5)
        budget = 0.2 * (tv_energy_float64(noisy) - 1e-8)
        result = certify_candidate(
            noisy,
            proposal,
            budget_override=budget,
            bisection_steps=28,
            feasibility_tolerance=1e-12,
        )
        removed = max(0.0, tv_energy_float64(noisy) - tv_energy_float64(result.image))
        self.assertLessEqual(removed, budget + 1e-12)
        self.assertGreater(result.eta, 0.0)
        self.assertLess(result.eta, 1.0)

    def test_feasible_proposal_passes_through(self) -> None:
        rng = np.random.default_rng(9)
        noisy = rng.random((17, 19, 3), dtype=np.float32)
        proposal = np.clip(noisy + 0.05 * rng.standard_normal(noisy.shape), 0.0, 1.0).astype(np.float32)
        result = certify_candidate(noisy, proposal, budget_override=tv_energy_float64(noisy))
        self.assertEqual(result.eta, 1.0)
        np.testing.assert_array_equal(result.image, proposal)


if __name__ == "__main__":
    unittest.main()
