"""
统一网格覆盖状态模块。

用途：
1. 直接复用 Coverage 阶段生成的统一网格，不再在分区内重建局部细网格。
2. 使用采样法对统一网格 cell 做覆盖去重统计，避免重复积分导致的覆盖面积高估。
3. 为规划过程提供可累计的覆盖状态，并为统计模块输出真实覆盖面积近似值。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multibeam.GridCell import (
    compute_axis_effective_lengths,
    compute_effective_cell_area_matrix,
)
from tool.Data import get_height, get_w_left, get_w_right


@dataclass
class PartitionCoverageSummary:
    partition_id: int
    cell_size: float
    planning_step: float
    microstep: float
    total_area: float
    covered_area: float
    uncovered_area: float
    coverage_rate: float
    miss_rate: float
    total_cells: int
    full_cells: int
    partial_cells: int
    uncovered_cells: int
    max_cells_guard_triggered: bool
    reference_point_x: float
    reference_point_y: float
    reference_depth: float
    reference_width: float


def infer_uniform_cell_size(xs: np.ndarray, ys: np.ndarray) -> float:
    """从统一网格坐标中推断网格边长。"""

    x_diff = np.diff(np.asarray(xs, dtype=float))
    y_diff = np.diff(np.asarray(ys, dtype=float))
    positive_steps = np.concatenate([x_diff[x_diff > 1e-12], y_diff[y_diff > 1e-12]])
    if len(positive_steps) == 0:
        raise ValueError("无法从 xs/ys 推断统一网格边长。")
    return float(np.min(positive_steps))


class PartitionCoverageStateGrid:
    """统一网格上的分区覆盖状态视图。"""

    SAMPLE_LAYOUTS = {
        4: np.array(
            [
                (-0.25, -0.25),
                (0.25, -0.25),
                (-0.25, 0.25),
                (0.25, 0.25),
            ],
            dtype=float,
        ),
        5: np.array(
            [
                (-0.25, -0.25),
                (0.25, -0.25),
                (0.0, 0.0),
                (-0.25, 0.25),
                (0.25, 0.25),
            ],
            dtype=float,
        ),
        9: np.array(
            [(dx, dy) for dy in (-1 / 3, 0.0, 1 / 3) for dx in (-1 / 3, 0.0, 1 / 3)],
            dtype=float,
        ),
    }

    def __init__(
        self,
        partition_id: int,
        xs: np.ndarray,
        ys: np.ndarray,
        cluster_matrix: np.ndarray,
        *,
        theta: float = 120.0,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        boundary_mask: np.ndarray | None = None,
        cell_effective_area: np.ndarray | None = None,
        cell_size: float | None = None,
        sampling_points: int = 9,
        full_threshold: float = 1.0,
        step_scale: float = 1.0,
        microstep_min: float = 10.0,
        microstep_max: float = 70.0,
        legacy_microstep: float = 35.0,
    ):
        if sampling_points not in self.SAMPLE_LAYOUTS:
            raise ValueError(f"不支持的采样模式: {sampling_points}")

        self.partition_id = int(partition_id)
        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.asarray(ys, dtype=float)
        self.cluster_matrix = np.asarray(cluster_matrix, dtype=int)
        self.theta = float(theta)
        self.x_min = float(x_min) if x_min is not None else float(np.min(self.xs))
        self.x_max = float(x_max) if x_max is not None else float(np.max(self.xs))
        self.y_min = float(y_min) if y_min is not None else float(np.min(self.ys))
        self.y_max = float(y_max) if y_max is not None else float(np.max(self.ys))

        self.sampling_points = int(sampling_points)
        self.sample_count = len(self.SAMPLE_LAYOUTS[self.sampling_points])
        self.full_threshold = float(full_threshold)
        self.step_scale = float(step_scale)
        self.microstep_min = float(microstep_min)
        self.microstep_max = float(microstep_max)
        self.legacy_microstep = float(legacy_microstep)

        self.cell_size = (
            float(cell_size)
            if cell_size is not None
            else infer_uniform_cell_size(xs, ys)
        )
        self.cell_area = self.cell_size**2
        self.max_cells_guard_triggered = False

        if boundary_mask is None or cell_effective_area is None:
            _, _, inferred_effective_area, _ = compute_effective_cell_area_matrix(
                self.xs,
                self.ys,
                self.cell_size,
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
            )
            inferred_boundary_mask = inferred_effective_area > 0
            self.boundary_mask = inferred_boundary_mask
            self.cell_domain_area = inferred_effective_area
        else:
            self.boundary_mask = np.asarray(boundary_mask, dtype=bool)
            self.cell_domain_area = np.asarray(cell_effective_area, dtype=float)

        self.domain_x_min = self.x_min
        self.domain_x_max = self.x_max
        self.domain_y_min = self.y_min
        self.domain_y_max = self.y_max
        self.domain_area = float((self.x_max - self.x_min) * (self.y_max - self.y_min))

        self.x_centers = self.xs.copy()
        self.y_centers = self.ys.copy()
        self.domain_x_effective_lengths = compute_axis_effective_lengths(
            self.x_centers, self.cell_size, self.domain_x_min, self.domain_x_max
        )
        self.domain_y_effective_lengths = compute_axis_effective_lengths(
            self.y_centers, self.cell_size, self.domain_y_min, self.domain_y_max
        )

        self.planning_step = self.step_scale * self.cell_size
        self.microstep = self._clip(
            self.cell_size / 2.0, self.microstep_min, self.microstep_max
        )

        self.reference_point_x = 0.0
        self.reference_point_y = 0.0
        self.reference_depth = 0.0
        self.reference_width = 0.0

        self.sample_x = None
        self.sample_y = None
        self.domain_sample_mask = None
        self.sample_partition_ids = None
        self.domain_sample_counts = None
        self.partition_sample_mask = None
        self.covered_sample_mask = None
        self.partition_sample_counts = None
        self.covered_sample_counts = None
        self.partition_area_ratio_matrix = None
        self.coverage_ratio_matrix = None
        self.state_code_matrix = None
        self.total_partition_area = 0.0
        self.total_cells = 0

        self._build()

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _compute_total_width(self, x: float, y: float, microstep: float) -> float:
        return float(
            get_w_left(x, y, self.theta, microstep=microstep)
            + get_w_right(x, y, self.theta, microstep=microstep)
        )

    def _select_reference_point(self) -> tuple[float, float, float]:
        rows, cols = np.where(
            (self.cluster_matrix == self.partition_id) & self.boundary_mask
        )
        if len(rows) == 0:
            raise ValueError(f"分区 {self.partition_id} 无有效统一网格点")

        candidate_x = self.xs[cols]
        candidate_y = self.ys[rows]
        depths = np.array(
            [get_height(float(x), float(y)) for x, y in zip(candidate_x, candidate_y)],
            dtype=float,
        )
        min_idx = int(np.argmin(depths))
        return (
            float(candidate_x[min_idx]),
            float(candidate_y[min_idx]),
            float(depths[min_idx]),
        )

    def _map_points_to_grid_indices(
        self, points_x: np.ndarray, points_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        col_idx = np.rint(
            (points_x - float(self.x_centers[0])) / self.cell_size
        ).astype(int)
        row_idx = np.rint(
            (points_y - float(self.y_centers[0])) / self.cell_size
        ).astype(int)
        col_idx = np.clip(col_idx, 0, len(self.x_centers) - 1)
        row_idx = np.clip(row_idx, 0, len(self.y_centers) - 1)
        return row_idx, col_idx

    def _build(self) -> None:
        ref_x, ref_y, ref_depth = self._select_reference_point()
        self.reference_point_x = ref_x
        self.reference_point_y = ref_y
        self.reference_depth = ref_depth
        self.reference_width = self._compute_total_width(ref_x, ref_y, self.microstep)

        grid_x, grid_y = np.meshgrid(self.x_centers, self.y_centers)
        offsets = self.SAMPLE_LAYOUTS[self.sampling_points] * self.cell_size
        self.sample_x = grid_x[:, :, None] + offsets[:, 0].reshape(1, 1, -1)
        self.sample_y = grid_y[:, :, None] + offsets[:, 1].reshape(1, 1, -1)

        self.domain_sample_mask = (
            (self.sample_x >= self.domain_x_min)
            & (self.sample_x <= self.domain_x_max)
            & (self.sample_y >= self.domain_y_min)
            & (self.sample_y <= self.domain_y_max)
        )

        sample_row_idx, sample_col_idx = self._map_points_to_grid_indices(
            self.sample_x, self.sample_y
        )
        self.sample_partition_ids = np.full_like(sample_row_idx, -1, dtype=int)
        valid_points = self.domain_sample_mask
        self.sample_partition_ids[valid_points] = self.cluster_matrix[
            sample_row_idx[valid_points], sample_col_idx[valid_points]
        ]

        self.domain_sample_counts = np.sum(self.domain_sample_mask, axis=2)
        self.partition_sample_mask = self.domain_sample_mask & (
            self.sample_partition_ids == self.partition_id
        )
        self.covered_sample_mask = np.zeros_like(self.partition_sample_mask, dtype=bool)
        self.partition_sample_counts = np.sum(self.partition_sample_mask, axis=2)

        self.partition_area_ratio_matrix = np.zeros_like(
            self.partition_sample_counts, dtype=float
        )
        valid_domain_cells = self.domain_sample_counts > 0
        self.partition_area_ratio_matrix[valid_domain_cells] = (
            self.partition_sample_counts[valid_domain_cells]
            / self.domain_sample_counts[valid_domain_cells]
        )

        self.total_cells = int(np.sum(self.partition_sample_counts > 0))
        self.total_partition_area = float(
            np.sum(self.partition_area_ratio_matrix * self.cell_domain_area)
        )

    def _candidate_index_range(
        self, centers: np.ndarray, low: float, high: float
    ) -> tuple[int, int]:
        start = int(np.searchsorted(centers, low, side="left"))
        end = int(np.searchsorted(centers, high, side="right")) - 1
        start = max(0, start)
        end = min(len(centers) - 1, end)
        return start, end

    @staticmethod
    def _points_within_segment_swath(
        points_x: np.ndarray,
        points_y: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> np.ndarray:
        x0, y0, w0 = float(start[0]), float(start[1]), max(float(start[2]), 0.0)
        x1, y1, w1 = float(end[0]), float(end[1]), max(float(end[2]), 0.0)

        dx = x1 - x0
        dy = y1 - y0
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq <= 1e-12:
            radius = max(w0, w1) / 2.0
            return (points_x - x0) ** 2 + (points_y - y0) ** 2 <= radius**2

        t = ((points_x - x0) * dx + (points_y - y0) * dy) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        proj_x = x0 + t * dx
        proj_y = y0 + t * dy
        widths = w0 + (w1 - w0) * t
        radii = np.maximum(widths / 2.0, 0.0)
        dist_sq = (points_x - proj_x) ** 2 + (points_y - proj_y) ** 2
        return dist_sq <= radii**2

    def update_segment(self, start: np.ndarray, end: np.ndarray) -> None:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        radius = max(float(start[2]), float(end[2])) / 2.0
        if radius <= 0:
            return

        x_low = min(float(start[0]), float(end[0])) - radius
        x_high = max(float(start[0]), float(end[0])) + radius
        y_low = min(float(start[1]), float(end[1])) - radius
        y_high = max(float(start[1]), float(end[1])) + radius

        row_start, row_end = self._candidate_index_range(self.y_centers, y_low, y_high)
        col_start, col_end = self._candidate_index_range(self.x_centers, x_low, x_high)

        if row_end < row_start or col_end < col_start:
            return

        for i in range(row_start, row_end + 1):
            for j in range(col_start, col_end + 1):
                if self.partition_sample_counts[i, j] == 0:
                    continue

                pending_mask = self.partition_sample_mask[i, j] & (
                    ~self.covered_sample_mask[i, j]
                )
                if not np.any(pending_mask):
                    continue

                sample_indices = np.where(pending_mask)[0]
                covered_now = self._points_within_segment_swath(
                    self.sample_x[i, j, sample_indices],
                    self.sample_y[i, j, sample_indices],
                    start,
                    end,
                )
                if np.any(covered_now):
                    self.covered_sample_mask[i, j, sample_indices[covered_now]] = True

    def update_polyline(self, line: np.ndarray | list[list[float]]) -> None:
        line_arr = np.asarray(line, dtype=float)
        if len(line_arr) < 2:
            return
        for start, end in zip(line_arr[:-1], line_arr[1:]):
            self.update_segment(start, end)

    def _estimate_swath_gain(self, start: np.ndarray, end: np.ndarray) -> dict:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        radius = max(float(start[2]), float(end[2])) / 2.0
        if radius <= 0:
            return {
                "candidate_cells": 0,
                "nonfull_cells_touched": 0,
                "new_samples_in_nonfull_cells": 0,
            }

        valid_mask = self._compute_state_matrices()
        x_low = min(float(start[0]), float(end[0])) - radius
        x_high = max(float(start[0]), float(end[0])) + radius
        y_low = min(float(start[1]), float(end[1])) - radius
        y_high = max(float(start[1]), float(end[1])) + radius

        row_start, row_end = self._candidate_index_range(self.y_centers, y_low, y_high)
        col_start, col_end = self._candidate_index_range(self.x_centers, x_low, x_high)
        if row_end < row_start or col_end < col_start:
            return {
                "candidate_cells": 0,
                "nonfull_cells_touched": 0,
                "new_samples_in_nonfull_cells": 0,
            }

        candidate_cells = 0
        nonfull_cells_touched = 0
        new_samples_in_nonfull_cells = 0

        for i in range(row_start, row_end + 1):
            for j in range(col_start, col_end + 1):
                if not valid_mask[i, j]:
                    continue

                sample_indices = np.where(self.partition_sample_mask[i, j])[0]
                if len(sample_indices) == 0:
                    continue

                covered_now = self._points_within_segment_swath(
                    self.sample_x[i, j, sample_indices],
                    self.sample_y[i, j, sample_indices],
                    start,
                    end,
                )
                if not np.any(covered_now):
                    continue

                candidate_cells += 1
                current_ratio = float(self.coverage_ratio_matrix[i, j])
                if current_ratio >= self.full_threshold:
                    continue

                nonfull_cells_touched += 1
                candidate_sample_indices = sample_indices[covered_now]
                uncovered_mask = ~self.covered_sample_mask[
                    i, j, candidate_sample_indices
                ]
                new_samples_in_nonfull_cells += int(np.count_nonzero(uncovered_mask))

        return {
            "candidate_cells": candidate_cells,
            "nonfull_cells_touched": nonfull_cells_touched,
            "new_samples_in_nonfull_cells": new_samples_in_nonfull_cells,
        }

    def would_segment_add_value(self, start: np.ndarray, end: np.ndarray) -> bool:
        gain = self._estimate_swath_gain(start, end)
        return gain["nonfull_cells_touched"] > 0

    def would_point_add_value(self, point: np.ndarray) -> bool:
        point_arr = np.asarray(point, dtype=float)
        return self.would_segment_add_value(point_arr, point_arr)

    def _compute_state_matrices(self):
        self.covered_sample_counts = np.sum(
            self.partition_sample_mask & self.covered_sample_mask, axis=2
        )
        self.coverage_ratio_matrix = np.zeros_like(
            self.partition_sample_counts, dtype=float
        )
        valid_mask = self.partition_sample_counts > 0
        self.coverage_ratio_matrix[valid_mask] = (
            self.covered_sample_counts[valid_mask]
            / self.partition_sample_counts[valid_mask]
        )

        self.state_code_matrix = np.full_like(
            self.partition_sample_counts, -1, dtype=int
        )
        self.state_code_matrix[valid_mask & (self.coverage_ratio_matrix == 0.0)] = 0
        self.state_code_matrix[
            valid_mask
            & (self.coverage_ratio_matrix > 0.0)
            & (self.coverage_ratio_matrix < self.full_threshold)
        ] = 1
        self.state_code_matrix[
            valid_mask & (self.coverage_ratio_matrix >= self.full_threshold)
        ] = 2
        return valid_mask

    def iter_render_cells(self):
        """生成用于绘图的统一网格 cell 渲染数据。"""
        valid_mask = self._compute_state_matrices()
        half = self.cell_size / 2.0
        for i in range(len(self.y_centers)):
            for j in range(len(self.x_centers)):
                if not valid_mask[i, j]:
                    continue
                x0 = max(float(self.x_centers[j] - half), self.domain_x_min)
                x1 = min(float(self.x_centers[j] + half), self.domain_x_max)
                y0 = max(float(self.y_centers[i] - half), self.domain_y_min)
                y1 = min(float(self.y_centers[i] + half), self.domain_y_max)
                width = x1 - x0
                height = y1 - y0
                if width <= 0 or height <= 0:
                    continue

                state_code = int(self.state_code_matrix[i, j])
                state_name = {0: "uncovered", 1: "partial", 2: "full"}.get(
                    state_code, "invalid"
                )
                if state_name == "invalid":
                    continue

                yield {
                    "x": x0,
                    "y": y0,
                    "width": width,
                    "height": height,
                    "state": state_name,
                    "partition_area_ratio": float(
                        self.partition_area_ratio_matrix[i, j]
                    ),
                    "coverage_ratio": float(self.coverage_ratio_matrix[i, j]),
                    "effective_area": float(self.cell_domain_area[i, j]),
                }

    def summarize(self) -> PartitionCoverageSummary:
        valid_mask = self._compute_state_matrices()

        covered_ratio_against_domain = np.zeros_like(
            self.partition_sample_counts, dtype=float
        )
        valid_domain_cells = self.domain_sample_counts > 0
        covered_ratio_against_domain[valid_domain_cells] = (
            self.covered_sample_counts[valid_domain_cells]
            / self.domain_sample_counts[valid_domain_cells]
        )

        covered_area = float(
            np.sum(covered_ratio_against_domain * self.cell_domain_area)
        )
        covered_area = min(covered_area, self.total_partition_area)
        uncovered_area = max(self.total_partition_area - covered_area, 0.0)

        coverage_rate = (
            covered_area / self.total_partition_area
            if self.total_partition_area > 0
            else 0.0
        )
        coverage_rate = self._clip(coverage_rate, 0.0, 1.0)
        miss_rate = 1.0 - coverage_rate

        full_cells = int(
            np.sum(valid_mask & (self.coverage_ratio_matrix >= self.full_threshold))
        )
        partial_cells = int(
            np.sum(
                valid_mask
                & (self.coverage_ratio_matrix > 0.0)
                & (self.coverage_ratio_matrix < self.full_threshold)
            )
        )
        uncovered_cells = int(np.sum(valid_mask & (self.coverage_ratio_matrix == 0.0)))

        return PartitionCoverageSummary(
            partition_id=self.partition_id,
            cell_size=self.cell_size,
            planning_step=self.planning_step,
            microstep=self.microstep,
            total_area=self.total_partition_area,
            covered_area=covered_area,
            uncovered_area=uncovered_area,
            coverage_rate=coverage_rate,
            miss_rate=miss_rate,
            total_cells=self.total_cells,
            full_cells=full_cells,
            partial_cells=partial_cells,
            uncovered_cells=uncovered_cells,
            max_cells_guard_triggered=self.max_cells_guard_triggered,
            reference_point_x=self.reference_point_x,
            reference_point_y=self.reference_point_y,
            reference_depth=self.reference_depth,
            reference_width=self.reference_width,
        )
