from pathlib import Path
import sys
import unittest
import json
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare import mask, FEATURES, weights
from analyze import selective
from fitting import frozen_probabilities, fit_predict
import fitting


class Contracts(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            [{**{c: 1.0 for c in FEATURES}, "participant": 1, "label": 0}]
        )

    def test_accelerometer_loss_removes_both_derived_streams(self):
        m = mask(self.frame(), "acc_missing")
        self.assertTrue((m.filter(regex="^(body_acc|total_acc)").to_numpy() == 0).all())
        self.assertTrue((m.filter(regex="^body_gyro").to_numpy() == 1).all())
        self.assertEqual(m.acc_present.iloc[0], 0)

    def test_gyroscope_loss_retains_acceleration(self):
        m = mask(self.frame(), "gyro_missing")
        self.assertTrue((m.filter(regex="^body_gyro").to_numpy() == 0).all())
        self.assertTrue((m.filter(regex="^(body_acc|total_acc)").to_numpy() == 1).all())

    def test_absent_inputs_force_abstention_even_with_confident_model(self):
        m = mask(self.frame(), "both_missing")
        for i in range(6):
            m[f"p{i}"] = 1.0 if i == 0 else 0.0
        self.assertEqual(selective(m, 0.0)["accepted"], 0)

    def test_participant_training_weights_have_equal_totals(self):
        f = pd.DataFrame({"participant": [1, 2, 2, 2]})
        w = weights(f)
        self.assertAlmostEqual(w[0], sum(w[1:]))
        self.assertAlmostEqual(sum(w), 4.0)

    def test_threshold_boundary_is_inclusive(self):
        f = mask(self.frame(), "intact")
        for i in range(6):
            f[f"p{i}"] = 0.5 if i in [0, 1] else 0.0
        self.assertEqual(selective(f, 0.5)["accepted"], 1)
        self.assertEqual(selective(f, 0.50001)["accepted"], 0)

    def evidence(self):
        return {
            "active_features": [True] + [False] * (len(FEATURES) - 1),
            "scaler_mean": [0.0],
            "scaler_scale": [1.0],
            "coefficients": [[1000.0], [-1000.0], [0.0], [1.0], [2.0], [3.0]],
            "intercepts": [0.0] * 6,
        }

    def test_frozen_inference_ignores_evaluation_labels(self):
        a = self.frame()
        b = a.copy()
        b["label"] = 5
        np.testing.assert_array_equal(
            frozen_probabilities(self.evidence(), a),
            frozen_probabilities(self.evidence(), b),
        )

    def test_frozen_softmax_is_finite_for_extreme_logits(self):
        p = frozen_probabilities(self.evidence(), self.frame())
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertEqual(int(p.argmax()), 0)

    def test_changed_training_identity_rejects_frozen_model(self):
        spec = {"name": "contract", "kind": "logistic", "augmentation": False, "C": 1.0}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            fitting, "FITTING", Path(directory)
        ):
            Path(directory, "contract.json").write_text(
                json.dumps({"status": "completed", "training_fingerprint": "wrong"})
            )
            with self.assertRaises(AssertionError):
                fit_predict(spec, self.frame(), self.frame(), use_frozen=True)


if __name__ == "__main__":
    unittest.main()
