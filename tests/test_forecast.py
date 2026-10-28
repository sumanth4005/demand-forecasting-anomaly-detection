import unittest

import numpy as np
import pandas as pd

from src.forecast import load_data, make_features, time_split, flag_anomalies, train_and_compare


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.spikes = load_data(return_spikes=True)
        cls.df = make_features(cls.raw)

    def test_no_leakage_in_rolling_features(self):
        # roll_mean_7 on day t must equal the mean of the 7 days *before* t
        t = self.df.index[50]
        expected = self.raw.loc[t - pd.Timedelta(days=7): t - pd.Timedelta(days=1), "demand"].mean()
        self.assertAlmostEqual(self.df.loc[t, "roll_mean_7"], expected, places=6)

    def test_lag_feature(self):
        t = self.df.index[10]
        self.assertAlmostEqual(self.df.loc[t, "lag_7"], self.raw.loc[t - pd.Timedelta(days=7), "demand"])

    def test_time_split_is_ordered(self):
        train, test = time_split(self.df, 120)
        self.assertLess(train.index.max(), test.index.min())
        self.assertEqual(len(test), 120)

    def test_best_model_beats_baseline(self):
        train, test = time_split(self.df, 120)
        results, best, _, _ = train_and_compare(train, test)
        self.assertLess(results[best], results["Seasonal naive (lag_7)"])

    def test_anomaly_flags_are_boolean(self):
        train, test = time_split(self.df, 120)
        _, _, model, features = train_and_compare(train, test)
        scored = flag_anomalies(model, features, train, test)
        self.assertTrue(scored["anomaly"].dtype == bool)


if __name__ == "__main__":
    unittest.main()
