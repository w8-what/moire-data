import unittest

import numpy as np

from moire.extract_features import extract_Tcoh


class ExtractTcohTests(unittest.TestCase):
    def setUp(self):
        self.T = np.r_[np.arange(0.05, 1.05, 0.05), np.arange(1.5, 12.5, 0.5)]

    def test_finds_quadratic_to_linear_crossover(self):
        transition = 5.0
        rho_T2 = 100.0 + 8.0 * self.T**2
        rho_after = 100.0 + 8.0 * transition**2 + 15.0 * (self.T - transition)
        smooth = np.where(self.T <= transition, rho_T2, rho_after)

        rng = np.random.default_rng(12)
        sigma = np.full_like(self.T, 3.0)
        rho = smooth + rng.normal(0.0, sigma)
        candidates = extract_Tcoh(
            self.T, {"rho": rho, "rho_smoothed": smooth, "local_noise": sigma, "nu": 0.75}
        )

        self.assertTrue(candidates)
        self.assertEqual(set(candidates[0]), {"T", "nu", "type", "confidence"})
        self.assertEqual(candidates[0]["type"], "Tcoh")
        self.assertLessEqual(transition, candidates[0]["T"])
        self.assertLess(candidates[0]["T"], 7.5)
        self.assertGreater(candidates[0]["confidence"], 0.25)

    def test_returns_no_candidate_for_pure_T2_curve(self):
        rho = 100.0 + 8.0 * self.T**2
        sigma = np.full_like(self.T, 2.0)
        candidates = extract_Tcoh(
            self.T, {"rho": rho, "rho_smoothed": rho, "local_noise": sigma, "nu": 0.75}
        )

        self.assertEqual(candidates, [])

    def test_rejects_line_without_low_temperature_T2_regime(self):
        rho = 100.0 + 20.0 * self.T
        sigma = np.full_like(self.T, 3.0)
        candidates = extract_Tcoh(
            self.T, {"rho": rho, "rho_smoothed": rho, "local_noise": sigma, "nu": 0.75}
        )

        self.assertEqual(candidates, [])

    def test_uses_extraction_range(self):
        transition = 5.0
        rho_T2 = 80.0 + 6.0 * self.T**2
        rho_after = 80.0 + 6.0 * transition**2 + 10.0 * (self.T - transition)
        rho = np.where(self.T <= transition, rho_T2, rho_after)
        sigma = np.full_like(self.T, 2.0)
        linecut = {
            "rho": rho,
            "rho_smoothed": rho,
            "local_noise": sigma,
            "nu": 1.2,
            "behaviors": [{"type": "extraction_range", "T_lower": 1.5, "T_upper": 10.0}],
        }

        candidates = extract_Tcoh(self.T, linecut)

        self.assertTrue(candidates)
        self.assertGreaterEqual(candidates[0]["T"], transition)


if __name__ == "__main__":
    unittest.main()
