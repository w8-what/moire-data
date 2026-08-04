import json
from pathlib import Path
import unittest

import numpy as np

from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth
from moire.phase_diagram import (
    PhasePoint,
    extract_tcoh_linecut,
    extract_tneel_candidates,
    extract_tprime_linecut,
    select_primary_physical_path,
)


class PhaseDiagramExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temperature = np.r_[np.linspace(0.05, 0.8, 15), np.linspace(0.9, 5.0, 35)]
        self.sigma = np.full(self.temperature.shape, 0.15)

    def test_tcoh_requires_a_genuine_t_squared_regime(self):
        for exponent in (1.6, 1.8, 2.2, 2.4):
            with self.subTest(exponent=exponent):
                resistance = 30.0 + 4.0 * self.temperature**exponent
                point = extract_tcoh_linecut(
                    self.temperature, resistance, resistance, self.sigma, field=-1, nu=1
                )
                self.assertIsNone(point)

        quadratic = 30.0 + 4.0 * self.temperature**2
        censored = extract_tcoh_linecut(
            self.temperature, quadratic, quadratic, self.sigma, field=-1, nu=1
        )
        self.assertIsNotNone(censored)
        self.assertTrue(censored.censored)
        self.assertAlmostEqual(censored.temperature, self.temperature[-1], places=12)

    def test_tcoh_finds_persistent_departure_from_t_squared(self):
        temperature = self.temperature
        departure = np.where(temperature > 1.5, 8.0 * np.maximum(temperature - 1.5, 0) ** 1.5, 0)
        resistance = 30.0 + 4.0 * temperature**2 + departure
        point = extract_tcoh_linecut(
            temperature, resistance, resistance, self.sigma, field=-1, nu=0.85
        )
        self.assertIsNotNone(point)
        self.assertEqual(point.transition, "Tcoh")
        self.assertLess(abs(point.exponent - 2.0), 0.08)
        self.assertGreater(point.temperature, 1.5)
        self.assertLess(point.temperature, 2.6)

    def test_tcoh_rejects_noisy_single_nonquadratic_powers(self):
        rng = np.random.default_rng(20260803)
        for exponent in (1.6, 1.8, 2.2, 2.4):
            false_positives = 0
            noiseless = 30.0 + 4.0 * self.temperature**exponent
            for _ in range(10):
                observed = noiseless + rng.normal(0.0, self.sigma)
                point = extract_tcoh_linecut(
                    self.temperature, observed, noiseless, self.sigma, field=-1, nu=1
                )
                false_positives += point is not None
            with self.subTest(exponent=exponent):
                self.assertEqual(false_positives, 0)

    def test_tcoh_rejects_realistically_smoothed_noisy_powers(self):
        rng = np.random.default_rng(20260805)
        for noise in (0.5, 1.0):
            sigma = np.full(self.temperature.shape, noise)
            for exponent in (1.6, 1.8, 2.2, 2.4):
                false_positives = 0
                noiseless = 30.0 + 4.0 * self.temperature**exponent
                for _ in range(8):
                    observed = noiseless + rng.normal(0.0, sigma)
                    smoothed = adaptive_multiscale_smooth(
                        self.temperature, observed, sigma, z_threshold=2.0
                    )
                    point = extract_tcoh_linecut(
                        self.temperature, observed, smoothed, sigma, field=-1, nu=1
                    )
                    false_positives += point is not None
                with self.subTest(noise=noise, exponent=exponent):
                    self.assertEqual(false_positives, 0)

    def test_crossovers_are_field_blind_and_resistance_scale_invariant(self):
        grids = (
            self.temperature,
            np.linspace(0.05, 5.0, 70),
            np.r_[np.geomspace(0.05, 0.9, 40), np.linspace(1.0, 5.0, 55)],
        )
        tcoh_grid_centers = []
        tprime_grid_centers = []
        for temperature in grids:
            sigma = np.full(temperature.shape, 0.15)
            tcoh_departure = np.where(
                temperature > 1.5, 8.0 * np.maximum(temperature - 1.5, 0.0) ** 1.5, 0.0
            )
            tcoh_resistance = 30.0 + 4.0 * temperature**2 + tcoh_departure
            tprime_departure = np.where(temperature < 1.6, 10.0 * (1.6 - temperature) ** 2, 0.0)
            tprime_resistance = 20.0 + 5.0 * temperature + tprime_departure
            tcoh_values = []
            tprime_values = []
            for scale, field in ((0.2, -74.0), (1.0, -1.0), (7.0, -999.0)):
                tcoh = extract_tcoh_linecut(
                    temperature,
                    scale * tcoh_resistance,
                    scale * tcoh_resistance,
                    scale * sigma,
                    field=field,
                    nu=0.85,
                )
                tprime = extract_tprime_linecut(
                    temperature,
                    scale * tprime_resistance,
                    scale * tprime_resistance,
                    scale * sigma,
                    field=field,
                    nu=0.9,
                )
                self.assertIsNotNone(tcoh)
                self.assertIsNotNone(tprime)
                tcoh_values.append(tcoh.temperature)
                tprime_values.append(tprime.temperature)
            self.assertLess(np.ptp(tcoh_values), 1e-8)
            self.assertLess(np.ptp(tprime_values), 1e-8)
            tcoh_grid_centers.append(tcoh_values[1])
            tprime_grid_centers.append(tprime_values[1])
        self.assertLess(np.ptp(tcoh_grid_centers), 0.02)
        self.assertLess(np.ptp(tprime_grid_centers), 0.02)

    def test_tprime_requires_and_finds_high_temperature_linear_baseline(self):
        temperature = self.temperature
        departure = np.where(temperature < 1.6, 10.0 * (1.6 - temperature) ** 2, 0)
        resistance = 20.0 + 5.0 * temperature + departure
        point = extract_tprime_linecut(
            temperature, resistance, resistance, self.sigma, field=-1, nu=0.9
        )
        self.assertIsNotNone(point)
        self.assertEqual(point.transition, "Tprime")
        self.assertLess(abs(point.exponent - 1.0), 0.08)
        self.assertGreater(point.temperature, 0.8)
        self.assertLess(point.temperature, 1.4)

        quadratic = 20.0 + 5.0 * temperature**2
        self.assertIsNone(
            extract_tprime_linecut(temperature, quadratic, quadratic, self.sigma, field=-1, nu=0.9)
        )

    def test_tneel_is_a_significant_local_resistance_minimum(self):
        resistance = 10.0 + 4.0 * (self.temperature - 1.2) ** 2
        points = extract_tneel_candidates(
            self.temperature, resistance, resistance, self.sigma, field=-1, nu=1
        )
        self.assertEqual(len(points), 1)
        self.assertLess(abs(points[0].temperature - 1.2), 0.15)

    def test_tneel_path_prefers_global_continuity_over_local_score_ties(self):
        fillings = np.linspace(0.9, 1.1, 21)
        candidates = []
        for index, nu in enumerate(fillings):
            primary_temperature = 2.0 + 20.0 * (nu - 1.0) ** 2
            candidates.append(
                PhasePoint(
                    field=-1,
                    nu=nu,
                    transition="Tneel",
                    temperature=primary_temperature,
                    uncertainty=0.1,
                    confidence=0.91,
                    model="synthetic primary",
                )
            )
            if 6 <= index <= 14:
                candidates.append(
                    PhasePoint(
                        field=-1,
                        nu=nu,
                        transition="Tneel",
                        temperature=7.0 if index % 2 else 0.7,
                        uncertainty=0.1,
                        confidence=0.92,
                        model="locally stronger distractor",
                    )
                )
        path = select_primary_physical_path(candidates, fillings)
        self.assertEqual(len(path), len(fillings))
        self.assertTrue(all(point.model == "synthetic primary" for point in path))

    def test_tprime_rejects_noisy_single_curved_powers(self):
        rng = np.random.default_rng(20260804)
        for exponent in (1.2, 1.4):
            false_positives = 0
            noiseless = 30.0 + 4.0 * self.temperature**exponent
            for _ in range(10):
                observed = noiseless + rng.normal(0.0, self.sigma)
                point = extract_tprime_linecut(
                    self.temperature, observed, noiseless, self.sigma, field=-1, nu=0.9
                )
                false_positives += point is not None
            with self.subTest(exponent=exponent):
                self.assertEqual(false_positives, 0)


class PublishedReferenceTests(unittest.TestCase):
    def test_official_source_data_topology_is_frozen(self):
        path = Path(__file__).parents[1] / "reference_data" / "wse2_fig3_reference.json"
        reference = json.loads(path.read_text())
        expected_counts = {
            "103": {"Tcoh": 12, "Tneel": 19, "Tprime": 7},
            "99": {"Tcoh": 12, "Tneel": 21, "Tprime": 7},
            "96": {"Tcoh": 11, "Tneel": 11, "Tprime": 7},
            "87": {"Tcoh": 10, "Tneel": 0, "Tprime": 0},
        }
        for field, counts in expected_counts.items():
            self.assertEqual(
                {name: len(points) for name, points in reference["fields"][field].items()}, counts
            )


if __name__ == "__main__":
    unittest.main()
