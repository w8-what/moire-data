import unittest
from unittest.mock import patch

import numpy as np

from moire.extract_features import (
    extract_Tcoh,
    extract_Tcoh_best_fits,
    extract_Tcoh_direct_fits,
    extract_Tcoh_new,
)


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


class ExtractTcohNewTests(unittest.TestCase):
    def test_weights_quadratic_support_by_temperature_span(self):
        temperatures = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.5, 2.5, 3.5])
        fit = {
            "n": [2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
            "n_sigma": np.full(8, 0.05),
        }
        linecut = {
            "behaviors": [
                {"type": "extraction_range", "T_lower": 0.1, "T_upper": 3.5}
            ]
        }

        with patch("moire.extract_features.extract_local_fits", return_value=fit):
            result = extract_Tcoh_new(temperatures, linecut, min_pts=5, min_T=0.3)

        # Temperature weighting stops at 1.5 K with the current 50% threshold.
        # A point-count mean would let the dense low-T points extend to 3.5 K.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["T_upper"], 1.5)

    def test_does_not_extend_past_parent_extraction_range(self):
        temperatures = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.5, 2.5])
        fit = {
            "n": np.full(7, 2.0),
            "n_sigma": np.full(7, 0.05),
        }
        linecut = {
            "behaviors": [
                {"type": "extraction_range", "T_lower": 0.1, "T_upper": 0.5}
            ]
        }

        with patch("moire.extract_features.extract_local_fits", return_value=fit):
            result = extract_Tcoh_new(temperatures, linecut, min_pts=5, min_T=0.3)

        self.assertEqual(result[0]["T_upper"], 0.5)


class ExtractTcohBestFitsTests(unittest.TestCase):
    def setUp(self):
        self.T = np.r_[np.arange(0.05, 1.05, 0.05), np.arange(1.5, 12.5, 0.5)]
        self.transition = 5.0
        self.quadratic = 100.0 + 8.0 * self.T**2
        after = 100.0 + 8.0 * self.transition**2 + 15.0 * (
            self.T - self.transition
        )
        self.smooth = np.where(self.T <= self.transition, self.quadratic, after)
        self.sigma = np.full_like(self.T, 3.0)
        rng = np.random.default_rng(12)
        self.rho = self.smooth + rng.normal(0.0, self.sigma)

    def linecut(self, smooth=None):
        return {
            "rho": self.rho,
            "rho_smoothed": self.smooth if smooth is None else smooth,
            "local_noise": self.sigma,
            "nu": 0.75,
        }

    def test_returns_five_independently_ranked_candidates(self):
        candidates = extract_Tcoh_best_fits(self.T, self.linecut())

        self.assertEqual(len(candidates), 5)
        self.assertEqual(
            [candidate["reduced_chi2"] for candidate in candidates],
            sorted(candidate["reduced_chi2"] for candidate in candidates),
        )
        self.assertTrue(all(self.transition <= candidate["T"] < 7.5 for candidate in candidates))
        self.assertTrue(
            all(
                {"fit_T_lower", "fit_T_upper", "n", "n_sigma", "n_probability"}
                <= candidate.keys()
                for candidate in candidates
            )
        )

    def test_respects_requested_candidate_limit(self):
        candidates = extract_Tcoh_best_fits(
            self.T, self.linecut(), max_candidates=2
        )

        self.assertEqual(len(candidates), 2)

    def test_rejects_an_isolated_deviation(self):
        smooth = self.quadratic.copy()
        smooth[np.argmin(abs(self.T - 6.0))] *= 1.2
        linecut = self.linecut(smooth=smooth)
        rng = np.random.default_rng(4)
        linecut["rho"] = self.quadratic + rng.normal(0.0, self.sigma)

        candidates = extract_Tcoh_best_fits(self.T, linecut)

        self.assertEqual(candidates, [])

    def test_stays_inside_parent_extraction_range(self):
        linecut = self.linecut()
        linecut["behaviors"] = [
            {"type": "extraction_range", "T_lower": 1.5, "T_upper": 4.5}
        ]

        candidates = extract_Tcoh_best_fits(self.T, linecut)

        self.assertEqual(candidates, [])


class ExtractTcohDirectFitsTests(unittest.TestCase):
    def setUp(self):
        self.T = np.arange(0.1, 3.1, 0.1)
        self.sigma = np.full_like(self.T, 0.5)

    def linecut(self, rho, smooth=None, upper=3.0):
        return {
            "rho": rho,
            "rho_smoothed": rho if smooth is None else smooth,
            "local_noise": self.sigma,
            "nu": 1.05,
            "behaviors": [
                {"type": "extraction_range", "T_lower": 0.1, "T_upper": upper}
            ],
        }

    def test_finds_low_temperature_direct_departure(self):
        transition = 0.6
        quadratic = 10.0 + 8.0 * self.T**2
        after = 10.0 + 8.0 * transition**2 + 1.0 * (self.T - transition)
        smooth = np.where(self.T <= transition, quadratic, after)

        candidates = extract_Tcoh_direct_fits(
            self.T, self.linecut(smooth)
        )

        self.assertTrue(candidates)
        # The first point beyond the fit reaches 10% deviation at 0.8 K.
        self.assertAlmostEqual(candidates[0]["T"], 0.8)
        self.assertLessEqual(len(candidates), 5)
        self.assertTrue(
            all(candidate["reduced_chi2"] <= 3.0 for candidate in candidates)
        )
        self.assertEqual(
            [candidate["reduced_chi2"] for candidate in candidates],
            sorted(candidate["reduced_chi2"] for candidate in candidates),
        )

    def test_returns_no_candidate_for_exact_quadratic_curve(self):
        rho = 10.0 + 8.0 * self.T**2

        candidates = extract_Tcoh_direct_fits(self.T, self.linecut(rho))

        self.assertEqual(candidates, [])

    def test_keeps_departure_inside_extraction_range(self):
        transition = 0.6
        quadratic = 10.0 + 8.0 * self.T**2
        after = 10.0 + 8.0 * transition**2 + 1.0 * (self.T - transition)
        rho = np.where(self.T <= transition, quadratic, after)

        candidates = extract_Tcoh_direct_fits(
            self.T, self.linecut(rho, upper=0.6)
        )

        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
