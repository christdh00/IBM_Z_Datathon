"""Run from project root: python -m unittest discover -s batch_fault_injection/tests -v"""
import unittest
import numpy as np
import pandas as pd
from batch_fault_injection.fault_injection import (
    FaultEvent, compute_reference_stats, generate_contamination_dataset, inject_batch,
)


class BatchBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 2600
        rng = np.random.default_rng(2026)
        t = np.arange(n)
        labels = np.zeros(n, dtype=int)
        labels[490:520] = 1
        labels[1820:1850] = 1
        cls.sensors = ("CMS air pressure", "Concentrazione ossigeno base", "Nitrogen pressure")
        cls.df = pd.DataFrame({
            "timestamp": pd.date_range("2025-04-01", periods=n, freq="min"),
            cls.sensors[0]: 12+0.05*np.sin(t/10)+rng.normal(0, 0.2, n),
            cls.sensors[1]: 21+0.04*np.cos(t/8)+rng.normal(0, 0.3, n),
            cls.sensors[2]: 45+0.06*np.sin(t/14)+rng.normal(0, 0.15, n),
            "Oxygen over threshold": rng.integers(0, 2, n),
            "PW_0.5h": labels,
        })
        cls.stats = compute_reference_stats(cls.df, cls.sensors)

    def test_levels_specials_and_preservation(self):
        for pct in (0, 0.5, 1, 2, 5, 10):
            with self.subTest(pct=pct):
                result, manifest, metrics = generate_contamination_dataset(
                    self.df, reference_stats=self.stats, sensor_cols=self.sensors,
                    target_percent=pct, special_share=0.075,
                    overlap_event_fraction=0.4, seed=2026+int(pct*100),
                )
                expected = round(len(self.df)*pct/100)
                self.assertEqual(int(result["synthetic_modified"].sum()), expected)
                self.assertEqual(metrics["modified_rows"], expected)
                self.assertEqual(metrics["special_modified_rows"], round(expected*0.075))
                self.assertTrue(result["PW_0.5h"].equals(self.df["PW_0.5h"]))
                self.assertTrue(result["timestamp"].equals(self.df["timestamp"]))
                self.assertTrue(result["Oxygen over threshold"].equals(self.df["Oxygen over threshold"]))
                self.assertFalse((result["synthetic_anomaly"].astype(bool) & (self.df["PW_0.5h"] == 1)).any())
                self.assertTrue((result["synthetic_modified"].to_numpy() == np.any(
                    ~np.isclose(result[list(self.sensors)].to_numpy(dtype=float),
                                self.df[list(self.sensors)].to_numpy(dtype=float),
                                equal_nan=True, rtol=0, atol=1e-12), axis=1)).all())
                if expected >= 10:
                    self.assertGreater(metrics["overlapping_rows"], 0)
                    self.assertTrue(len(manifest) > 0)

    def test_reproducible(self):
        kw = dict(reference_stats=self.stats, sensor_cols=self.sensors,
                  target_percent=2, overlap_event_fraction=0.5, seed=101)
        a, ma, sa = generate_contamination_dataset(self.df, **kw)
        b, mb, sb = generate_contamination_dataset(self.df, **kw)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(ma, mb)
        self.assertEqual(sa, sb)

    def test_all_ten_types(self):
        kinds = ("gaussian_noise", "spike", "short_burst", "random_corruption",
                 "dropout", "bias", "drift", "stuck_sensor", "increased_noise",
                 "intermittent_fault")
        starts = (110, 210, 310, 410, 610, 710, 810, 910, 1010, 1110)
        events = []
        for i, (kind, start) in enumerate(zip(kinds, starts)):
            duration = 1 if kind == "spike" else 7
            events.append(FaultEvent(f"F{i:03d}", self.sensors[i % 3], kind,
                                     start, duration, None if kind in ("dropout", "stuck_sensor") else 2.0,
                                     100+i))
        output, manifest = inject_batch(self.df, events,
                                        reference_stats=self.stats, sensor_cols=self.sensors)
        self.assertEqual(len(manifest), 10)
        self.assertEqual(output.loc[210, "synthetic_anomaly"], 1)
        self.assertEqual(output.loc[110, "synthetic_anomaly"], 0)
        self.assertEqual(output.loc[1110, "synthetic_anomaly"], 1)
        self.assertEqual(output.loc[1111, "synthetic_anomaly"], 0)

    def test_overlapping_faults_priority(self):
        sensor = self.sensors[0]
        events = [
            FaultEvent("F001", sensor, "bias", 120, 12, 2.0, 10),
            FaultEvent("F002", sensor, "increased_noise", 122, 7, 0.3, 11),
            FaultEvent("F003", sensor, "dropout", 125, 2, None, 12),
            FaultEvent("F004", self.sensors[1], "drift", 125, 4, 1.0, 13),
        ]
        output, events_meta = inject_batch(self.df, events, reference_stats=self.stats, sensor_cols=self.sensors)
        self.assertTrue(np.isnan(output.loc[125, sensor]))
        self.assertEqual(output.loc[125, "overlap_count"], 4)
        self.assertEqual(output.loc[125, "synthetic_anomaly"], 1)
        self.assertIn("F004", output.loc[125, "anomaly_id"])
        self.assertEqual(len(events_meta), 4)
        with self.assertRaises(ValueError):
            inject_batch(self.df, [
                FaultEvent("A", sensor, "stuck_sensor", 100, 20, None, 2),
                FaultEvent("B", sensor, "random_corruption", 105, 5, 2.0, 3),
            ], reference_stats=self.stats, sensor_cols=self.sensors)


if __name__ == "__main__":
    unittest.main()
