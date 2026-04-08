import unittest

import numpy as np

from multibeam.planner_utils import build_coverage_summary


class MetricsTestCase(unittest.TestCase):
    def test_unique_coverage_is_bounded_by_total_area(self):
        coverage_counts = np.array([[1, 2], [0, 1]])
        valid_mask = np.array([[True, True], [True, True]])

        summary = build_coverage_summary(
            coverage_counts=coverage_counts,
            valid_mask=valid_mask,
            cell_area=10.0,
            raw_coverage_area=50.0,
        )

        self.assertLessEqual(summary.unique_coverage_area, summary.total_area)
        self.assertGreaterEqual(summary.coverage_rate, 0.0)
        self.assertLessEqual(summary.coverage_rate, 1.0)
        self.assertGreaterEqual(summary.leakage_rate, 0.0)
        self.assertLessEqual(summary.leakage_rate, 1.0)

    def test_leakage_is_zero_when_all_valid_cells_are_covered(self):
        coverage_counts = np.ones((2, 3), dtype=int)
        valid_mask = np.ones((2, 3), dtype=bool)

        summary = build_coverage_summary(
            coverage_counts=coverage_counts,
            valid_mask=valid_mask,
            cell_area=5.0,
            raw_coverage_area=40.0,
        )

        self.assertEqual(summary.unique_coverage_area, summary.total_area)
        self.assertEqual(summary.leakage_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
