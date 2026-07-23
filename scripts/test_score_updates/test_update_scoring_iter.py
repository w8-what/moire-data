import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from moire.update_scoring import update_scores, update_scores_iter


def example_linecuts():
    temperatures = [0.0, 1.0, 2.0]
    return [
        {
            "T": temperatures,
            "features": [{"T": 1.0, "type": "upturn", "confidence": confidence}],
        }
        for confidence in (0.2, 0.8, 0.2)
    ]


class IterativeScoreTests(unittest.TestCase):
    def test_update_modes_diverge_after_the_first_round(self):
        anchored = example_linecuts()
        recursive = example_linecuts()

        update_scores_iter(anchored, 3, mode="original_anchor", n_hood=1)
        update_scores_iter(recursive, 3, mode="recursive", n_hood=1)

        anchored_middle = anchored[1]["features"][0]
        recursive_middle = recursive[1]["features"][0]
        self.assertAlmostEqual(anchored_middle["score_1"], 0.5)
        self.assertAlmostEqual(recursive_middle["score_1"], 0.5)
        self.assertAlmostEqual(anchored_middle["score_2"], 0.65)
        self.assertAlmostEqual(recursive_middle["score_2"], 0.5)
        self.assertAlmostEqual(anchored_middle["score_3"], 0.575)

    def test_backward_compatible_keys_are_preserved(self):
        linecuts = example_linecuts()
        update_scores(linecuts, n_hood=1)
        feature = linecuts[1]["features"][0]
        self.assertEqual(feature["support"], feature["support_1"])
        self.assertIn("score_1", feature)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            update_scores_iter(example_linecuts(), 1, mode="unknown")

    def test_sigmoid_gates_raw_support(self):
        linecuts = example_linecuts()
        update_scores_iter(
            linecuts,
            1,
            mode="original_anchor",
            n_hood=1,
            sigmoid_support=True,
            sigmoid_center=0.6,
            sigmoid_width=0.1,
        )
        middle = linecuts[1]["features"][0]
        raw_support = 0.2
        expected_multiplier = 1 / (1 + 2.718281828459045**4)
        self.assertAlmostEqual(
            middle["support_1"], raw_support * expected_multiplier, places=12
        )

    def test_tau_changes_temperature_attenuation(self):
        narrow = example_linecuts()
        broad = example_linecuts()
        narrow[0]["features"][0]["T"] = 0.0
        broad[0]["features"][0]["T"] = 0.0
        update_scores_iter(narrow, 1, n_hood=1, tau=0.5)
        update_scores_iter(broad, 1, n_hood=1, tau=5.0)
        self.assertLess(
            narrow[1]["features"][0]["support_1"],
            broad[1]["features"][0]["support_1"],
        )


if __name__ == "__main__":
    unittest.main()
