import unittest

import numpy as np

from moire.extract_power_law import extract_local_fits


class ExtractPowerLawTests(unittest.TestCase):
    def test_default_bounds_allow_exponents_above_four(self):
        temperatures = np.linspace(1.0, 3.0, 20)
        linecut = {
            "rho": 12.0 + 0.02 * temperatures**6,
            "local_noise": np.ones_like(temperatures),
            "behaviors": [
                {
                    "type": "extraction_range",
                    "T_lower": temperatures[0],
                    "T_upper": temperatures[-1],
                }
            ],
        }

        fits = extract_local_fits(temperatures, linecut)
        fitted_exponents = np.asarray(fits["n"], float)

        self.assertTrue(np.all(np.isfinite(fitted_exponents)))
        self.assertTrue(np.allclose(fitted_exponents, 6.0, rtol=1e-4, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
