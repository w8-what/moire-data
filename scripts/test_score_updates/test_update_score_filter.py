import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from moire.update_scoring import update_score


def example_linecuts():
    return [
        {
            "T": [0.0, 1.0, 2.0],
            "features": [
                {"T": 1.0, "type": "upturn", "confidence": 0.8},
                {"T": 2.0, "type": "downturn", "confidence": 0.02},
            ],
        }
    ]


class FilteredScoreTests(unittest.TestCase):
    def test_defaults_create_fifteen_scores_on_surviving_features(self):
        linecuts = example_linecuts()

        result = update_score(linecuts)

        self.assertIs(result, linecuts)
        self.assertEqual(len(linecuts[0]["features_new"]), 1)
        survivor = linecuts[0]["features_new"][0]
        self.assertEqual(survivor["confidence"], 0.8)
        self.assertTrue(
            all(f"score_{iteration}" in survivor for iteration in range(1, 16))
        )

    def test_filter_removes_noise_without_changing_original_features(self):
        linecuts = example_linecuts()
        original_low_feature = linecuts[0]["features"][1]

        update_score(linecuts, num_iter=1, num_passes=1, filter=0.01)

        self.assertEqual(len(linecuts[0]["features"]), 2)
        self.assertNotIn("score_1", original_low_feature)
        self.assertEqual(
            [feature["confidence"] for feature in linecuts[0]["features_new"]],
            [0.8],
        )
        self.assertIsNot(
            linecuts[0]["features_new"][0],
            linecuts[0]["features"][0],
        )

    def test_invalid_pass_configuration_is_rejected(self):
        for arguments in (
            {"num_iter": 0},
            {"num_passes": 0},
            {"filter": -0.01},
            {"filter": 1.01},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    update_score(example_linecuts(), **arguments)


if __name__ == "__main__":
    unittest.main()
