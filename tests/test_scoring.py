import unittest

import numpy as np

from multibeam.planner_utils import evaluate_candidate_line


class ScoringTestCase(unittest.TestCase):
    def test_high_gain_candidate_scores_better_than_overlap_heavy_candidate(self):
        xs = np.array([0.0, 10.0, 20.0, 30.0])
        ys = np.array([0.0, 10.0, 20.0, 30.0])
        grid_x, grid_y = np.meshgrid(xs, ys)
        valid_mask = np.ones_like(grid_x, dtype=bool)
        coverage_counts = np.zeros_like(grid_x, dtype=int)
        coverage_counts[1, 1] = 1
        coverage_counts[1, 2] = 1

        score_weights = {"gain": 1.0, "overlap": 1.4, "length": 0.01, "bend": 1.0}

        high_gain_line = np.array([[0.0, 0.0, 12.0], [0.0, 30.0, 12.0]])
        overlap_heavy_line = np.array([[10.0, 10.0, 16.0], [20.0, 10.0, 16.0]])

        high_gain_score, _ = evaluate_candidate_line(
            high_gain_line,
            coverage_counts,
            grid_x,
            grid_y,
            valid_mask,
            cell_area=25.0,
            score_weights=score_weights,
        )
        overlap_score, _ = evaluate_candidate_line(
            overlap_heavy_line,
            coverage_counts,
            grid_x,
            grid_y,
            valid_mask,
            cell_area=25.0,
            score_weights=score_weights,
        )

        self.assertGreaterEqual(
            high_gain_score.unique_gain_cells, overlap_score.unique_gain_cells
        )
        self.assertGreater(high_gain_score.score, overlap_score.score)


if __name__ == "__main__":
    unittest.main()
