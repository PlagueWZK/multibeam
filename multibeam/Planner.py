"""
测线规划核心模块（重构版）

本模块负责测线生成的核心调度逻辑。
数据类、几何计算和可视化逻辑已拆分到独立模块。
"""

import datetime
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# 从拆分后的模块导入
from multibeam.coverage_state_grid import (
    GlobalCoverageMetricsGrid,
    infer_uniform_cell_size,
)
from multibeam.models import (
    LinePartitionContribution,
    LineRecord,
    PartitionResult,
    TerminationReason,
)
from multibeam.Partition import get_partition_for_point, is_point_in_partition
from multibeam.planner_visualizer import SurveyVisualizer, save_dot_csv
from tool.Data import (
    figure_length,
    figure_width,
    forward_direction,
    get_alpha,
    get_gx,
    get_gy,
    get_height,
    get_total_width,
)


# ---------------------------------------------------------------------------
# SurveyPlanner — 测线规划核心类
# ---------------------------------------------------------------------------


@dataclass
class _ChildLineCandidate:
    """垂直扩展候选真实测线段。"""

    line: np.ndarray
    initial_line: np.ndarray
    overlap_rates: np.ndarray
    terminated_reason: TerminationReason = TerminationReason.NONE
    gain_ratio: float = 0.0
    new_area: float = 0.0
    hit_area: float = 0.0
    parent_gain_ratio: float = 1.0
    gain_threshold: float = 0.0


class SurveyPlanner:
    """测线规划器：封装分区测线生成与即时指标记录"""

    def __init__(
        self,
        xs,
        ys,
        cluster_matrix,
        step=50,
        theta=120,
        n=0.1,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        boundary_mask=None,
        cell_effective_area=None,
        grid_cell_size=None,
        depth_matrix=None,
        start_point_strategy="deepest",
        jump_line_gain_threshold=0.75,
        child_line_parent_gain_factor=0.80,
        child_line_min_gain_threshold=0.30,
        main_line_parent_gain_ratio=1.0,
    ):
        self.xs = xs
        self.ys = ys
        self.cluster_matrix = cluster_matrix
        self.default_step = float(step)
        self.step = float(step)
        self.theta = theta
        self.theta_rad = np.radians(theta)
        self.n = n

        # 统一网格参数（直接复用 Coverage 阶段网格）
        self.fine_grid_sampling_points = 9
        self.fine_grid_full_threshold = 1.0
        self.step_scale = 1.0
        self.microstep_min = 10.0
        self.microstep_max = 70.0
        self.legacy_microstep = 35.0
        self.current_microstep = self.legacy_microstep
        self.partition_coverage_summaries = {}
        self.partition_start_points = {}
        self.jump_line_gain_threshold = float(jump_line_gain_threshold)
        if not 0.0 <= self.jump_line_gain_threshold <= 1.0:
            raise ValueError(
                "测线取舍收益率阈值必须位于 [0, 1] 区间："
                f"{jump_line_gain_threshold}"
            )
        self.child_line_parent_gain_factor = float(child_line_parent_gain_factor)
        if not 0.0 <= self.child_line_parent_gain_factor <= 1.0:
            raise ValueError(
                "子测线父收益折减系数必须位于 [0, 1] 区间："
                f"{child_line_parent_gain_factor}"
            )
        self.child_line_min_gain_threshold = float(child_line_min_gain_threshold)
        if not 0.0 <= self.child_line_min_gain_threshold <= 1.0:
            raise ValueError(
                "子测线最低收益率阈值必须位于 [0, 1] 区间："
                f"{child_line_min_gain_threshold}"
            )
        self.main_line_parent_gain_ratio = float(main_line_parent_gain_ratio)
        if not 0.0 <= self.main_line_parent_gain_ratio <= 1.0:
            raise ValueError(
                "主测线参考父收益率必须位于 [0, 1] 区间："
                f"{main_line_parent_gain_ratio}"
            )
        self.max_consecutive_jump_lines = 3
        self.start_point_strategy = str(start_point_strategy).strip().lower()
        self.supported_start_point_strategies = {
            "deepest",
            "coordinate_center",
            "geometric_center",
        }
        if self.start_point_strategy not in self.supported_start_point_strategies:
            raise ValueError(
                "不支持的起点选择策略："
                f"{start_point_strategy}，可选={sorted(self.supported_start_point_strategies)}"
            )

        # 支持传入真实海域边界
        self.x_min = float(x_min) if x_min is not None else float(xs[0])
        self.x_max = float(x_max) if x_max is not None else float(xs[-1])
        self.y_min = float(y_min) if y_min is not None else float(ys[0])
        self.y_max = float(y_max) if y_max is not None else float(ys[-1])
        self.total_area = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        self.boundary_mask = (
            np.asarray(boundary_mask, dtype=bool)
            if boundary_mask is not None
            else np.ones_like(self.cluster_matrix, dtype=bool)
        )
        self.cell_effective_area = (
            np.asarray(cell_effective_area, dtype=float)
            if cell_effective_area is not None
            else None
        )
        self.depth_matrix = (
            np.asarray(depth_matrix, dtype=float)
            if depth_matrix is not None
            else None
        )
        self.grid_cell_size = (
            float(grid_cell_size)
            if grid_cell_size is not None
            else infer_uniform_cell_size(self.xs, self.ys)
        )
        if self.depth_matrix is not None and self.depth_matrix.shape != self.cluster_matrix.shape:
            raise ValueError(
                "depth_matrix 尺寸必须与 cluster_matrix 一致："
                f"depth={self.depth_matrix.shape}, cluster={self.cluster_matrix.shape}"
            )

        # 全局统计网格只用于记录已保留测线对所有分区的贡献，不能参与测线取舍。
        self.metrics_state_grid = GlobalCoverageMetricsGrid(
            self.xs,
            self.ys,
            self.cluster_matrix,
            theta=self.theta,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            boundary_mask=self.boundary_mask,
            cell_effective_area=self.cell_effective_area,
            cell_size=self.grid_cell_size,
            sampling_points=self.fine_grid_sampling_points,
            full_threshold=self.fine_grid_full_threshold,
            step_scale=self.step_scale,
            microstep_min=self.microstep_min,
            microstep_max=self.microstep_max,
            legacy_microstep=self.legacy_microstep,
        )
        self.line_partition_contributions: list[LinePartitionContribution] = []

        # 即时指标存储
        self.all_records: list[LineRecord] = []
        self._line_counter = 0

        # 初始化可视化器
        self.visualizer = SurveyVisualizer(
            xs, ys, cluster_matrix, self.x_min, self.x_max, self.y_min, self.y_max
        )

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _is_in_partition(self, x, y, partition_id):
        """判断点 (x, y) 是否属于指定分区"""
        is_in, _ = is_point_in_partition(
            x,
            y,
            partition_id,
            self.xs,
            self.ys,
            self.cluster_matrix,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
        )
        return is_in

    @staticmethod
    def _is_timestamped_output_name(output_name: str) -> bool:
        return (
            len(output_name) >= 15
            and output_name[:8].isdigit()
            and output_name[8] == "_"
            and output_name[9:15].isdigit()
        )

    def _get_depth_at_cells(self, rows, cols):
        if self.depth_matrix is not None:
            return np.asarray(self.depth_matrix[rows, cols], dtype=float)
        return np.array(
            [get_height(float(self.xs[c]), float(self.ys[r])) for r, c in zip(rows, cols)],
            dtype=float,
        )

    def _get_boundary_cell_along_gradient(
        self,
        start_row,
        start_col,
        partition_rows,
        partition_cols,
        partition_depths,
    ):
        x0 = float(self.xs[start_col])
        y0 = float(self.ys[start_row])
        gx = float(get_gx(x0, y0))
        gy = float(get_gy(x0, y0))
        norm = float(np.hypot(gx, gy))
        if norm <= 1e-12:
            return None

        ux = gx / norm
        uy = gy / norm
        dx = self.xs[partition_cols] - x0
        dy = self.ys[partition_rows] - y0
        proj = dx * ux + dy * uy
        perp = np.abs(dx * uy - dy * ux)
        ray_tolerance = max(self.grid_cell_size, 1e-6)
        candidate_mask = (proj > 1e-8) & (perp <= ray_tolerance)
        if not np.any(candidate_mask):
            return None

        candidate_indices = np.where(candidate_mask)[0]
        best_local_idx = int(candidate_indices[np.argmax(proj[candidate_mask])])
        return {
            "distance": float(proj[best_local_idx]),
            "depth": float(partition_depths[best_local_idx]),
            "row": int(partition_rows[best_local_idx]),
            "col": int(partition_cols[best_local_idx]),
        }

    def _compute_required_seed_clearance(self, alpha_deg, boundary_depth):
        alpha = float(np.radians(alpha_deg))
        sin_half = float(np.sin(self.theta_rad / 2.0))
        denominator = float(
            np.sin((np.pi - self.theta_rad) / 2.0 - alpha)
            + np.sin(alpha) * sin_half
        )
        if boundary_depth <= 0 or not np.isfinite(boundary_depth):
            return np.inf
        if denominator <= 1e-12:
            return np.inf

        delta_r = float(np.cos(alpha) * boundary_depth * sin_half / denominator)
        if not np.isfinite(delta_r) or delta_r < 0:
            return np.inf
        return delta_r

    def _get_valid_partition_cells(self, partition_id, rows, cols):
        if len(rows) == 0:
            raise ValueError(f"分区 {partition_id} 无有效网格点可用于选择起点")

        valid_mask = self.boundary_mask[rows, cols]
        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]
        if len(valid_rows) == 0:
            valid_rows = rows
            valid_cols = cols
        return valid_rows, valid_cols

    def _select_deepest_partition_start_point(self, partition_id, rows, cols):
        """选择满足 clearance 约束的最深网格中心作为起始点。"""
        valid_rows, valid_cols = self._get_valid_partition_cells(
            partition_id, rows, cols
        )

        depths = self._get_depth_at_cells(valid_rows, valid_cols)
        candidate_order = np.argsort(depths)[::-1]

        fallback_idx = int(candidate_order[0])
        fallback_row = int(valid_rows[fallback_idx])
        fallback_col = int(valid_cols[fallback_idx])
        fallback_depth = float(depths[fallback_idx])

        for idx in candidate_order:
            start_row = int(valid_rows[idx])
            start_col = int(valid_cols[idx])
            start_x = float(self.xs[start_col])
            start_y = float(self.ys[start_row])
            start_depth = float(depths[idx])
            alpha_deg = float(self._get_alpha(start_x, start_y))
            boundary_info = self._get_boundary_cell_along_gradient(
                start_row,
                start_col,
                valid_rows,
                valid_cols,
                depths,
            )
            if boundary_info is None:
                continue

            delta_r = self._compute_required_seed_clearance(
                alpha_deg, boundary_info["depth"]
            )
            if boundary_info["distance"] >= delta_r:
                return {
                    "x": start_x,
                    "y": start_y,
                    "depth": start_depth,
                    "row": start_row,
                    "col": start_col,
                    "alpha_deg": alpha_deg,
                    "boundary_distance": float(boundary_info["distance"]),
                    "boundary_depth": float(boundary_info["depth"]),
                    "delta_r": float(delta_r),
                    "rule": "deepest_cell_with_gradient_clearance",
                    "strategy": "deepest",
                }

        fallback_x = float(self.xs[fallback_col])
        fallback_y = float(self.ys[fallback_row])
        return {
            "x": fallback_x,
            "y": fallback_y,
            "depth": fallback_depth,
            "row": fallback_row,
            "col": fallback_col,
            "alpha_deg": float(self._get_alpha(fallback_x, fallback_y)),
            "boundary_distance": None,
            "boundary_depth": None,
            "delta_r": None,
            "rule": "fallback_deepest_cell",
            "strategy": "deepest",
        }

    def _select_coordinate_center_partition_start_point(
        self, partition_id, rows, cols, strategy="coordinate_center"
    ):
        """选择最接近分区有效网格坐标/几何中心的网格中心作为起始点。"""
        strategy = str(strategy).strip().lower()
        valid_rows, valid_cols = self._get_valid_partition_cells(
            partition_id, rows, cols
        )
        center_x = float(np.mean(self.xs[valid_cols]))
        center_y = float(np.mean(self.ys[valid_rows]))

        dx = self.xs[valid_cols] - center_x
        dy = self.ys[valid_rows] - center_y
        nearest_idx = int(np.argmin(dx * dx + dy * dy))
        start_row = int(valid_rows[nearest_idx])
        start_col = int(valid_cols[nearest_idx])
        start_x = float(self.xs[start_col])
        start_y = float(self.ys[start_row])
        start_depth = float(self._get_depth_at_cells([start_row], [start_col])[0])
        rule = (
            "geometric_center_nearest_cell"
            if strategy == "geometric_center"
            else "coordinate_center_nearest_cell"
        )
        return {
            "x": start_x,
            "y": start_y,
            "depth": start_depth,
            "row": start_row,
            "col": start_col,
            "alpha_deg": float(self._get_alpha(start_x, start_y)),
            "boundary_distance": None,
            "boundary_depth": None,
            "delta_r": None,
            "rule": rule,
            "strategy": strategy,
            "target_center_x": center_x,
            "target_center_y": center_y,
        }

    def _select_partition_start_point(self, partition_id, rows, cols):
        if self.start_point_strategy in {"coordinate_center", "geometric_center"}:
            return self._select_coordinate_center_partition_start_point(
                partition_id, rows, cols, strategy=self.start_point_strategy
            )
        return self._select_deepest_partition_start_point(partition_id, rows, cols)

    def _compute_partition_mean_depth(self, partition_id: int) -> float:
        """计算分区有效网格的平均水深，用于全局规划顺序排序。"""

        partition_id = int(partition_id)
        partition_mask = (self.cluster_matrix == partition_id) & self.boundary_mask
        rows, cols = np.where(partition_mask)
        if len(rows) == 0:
            rows, cols = np.where(self.cluster_matrix == partition_id)
        if len(rows) == 0:
            return float("-inf")

        if self.depth_matrix is not None:
            depths = np.asarray(self.depth_matrix[rows, cols], dtype=float)
        else:
            depths = np.array(
                [
                    get_height(float(self.xs[col]), float(self.ys[row]))
                    for row, col in zip(rows, cols)
                ],
                dtype=float,
            )
        finite_depths = depths[np.isfinite(depths)]
        if len(finite_depths) == 0:
            return float("-inf")
        return float(np.mean(finite_depths))

    def _order_partition_ids_by_mean_depth(self, partition_ids):
        mean_depth_by_pid = {
            int(pid): self._compute_partition_mean_depth(int(pid))
            for pid in partition_ids
        }
        ordered_partition_ids = sorted(
            [int(pid) for pid in partition_ids],
            key=lambda pid: (-mean_depth_by_pid[int(pid)], int(pid)),
        )
        return ordered_partition_ids, mean_depth_by_pid

    def _candidate_point_has_value(self, point, prior_covered_mask=None) -> bool:
        """全局统计网格驱动的点保留判断。"""
        if self.metrics_state_grid is None:
            return True
        return self.metrics_state_grid.would_point_add_value(
            np.asarray(point, dtype=float), prior_covered_mask=prior_covered_mask
        )

    def _candidate_segment_has_value(
        self, start_point, end_point, prior_covered_mask=None
    ) -> bool:
        """全局统计网格驱动的线段保留/终止判断。"""
        if self.metrics_state_grid is None:
            return True
        return self.metrics_state_grid.would_segment_add_value(
            np.asarray(start_point, dtype=float),
            np.asarray(end_point, dtype=float),
            prior_covered_mask=prior_covered_mask,
        )

    @staticmethod
    def _resolve_line_termination(front_reason, back_reason) -> TerminationReason:
        reasons = {front_reason, back_reason}
        if TerminationReason.LOW_VALUE in reasons:
            return TerminationReason.LOW_VALUE
        if TerminationReason.BOUNDARY in reasons:
            return TerminationReason.BOUNDARY
        return TerminationReason.NONE

    def _record_line(
        self,
        points,
        partition_id,
        terminated_by,
        *,
        overlap_excess_length=0.0,
        max_overlap_eta=0.0,
        repeated_area=0.0,
    ):
        """测线生成完毕后立即计算并记录指标"""
        pts = np.array(points)
        if len(pts) < 2:
            return None
        length = figure_length(pts)
        coverage = figure_width(pts)

        record = LineRecord(
            line_id=self._line_counter,
            partition_id=partition_id,
            points=pts,
            length=length,
            coverage=coverage,
            overlap_excess_length=float(overlap_excess_length),
            max_overlap_eta=float(max_overlap_eta),
            repeated_area=float(repeated_area),
            terminated_by=terminated_by,
        )
        self.all_records.append(record)

        if self.metrics_state_grid is not None:
            contributions = self.metrics_state_grid.record_polyline_contribution(
                pts,
                line_id=record.line_id,
                owner_partition_id=partition_id,
            )
            self.line_partition_contributions.extend(contributions)
            record.repeated_area = float(sum(c.repeated_area for c in contributions))

        self._line_counter += 1
        return record

    def _compute_parent_child_overlap_rate(self, parent_point, child_point) -> float:
        parent_arr = np.asarray(parent_point, dtype=float)
        child_arr = np.asarray(child_point, dtype=float)
        x_i, y_i = float(parent_arr[0]), float(parent_arr[1])
        x_j, y_j = float(child_arr[0]), float(child_arr[1])

        d_i = get_height(x_i, y_i)
        d_j = get_height(x_j, y_j)
        alpha_i = np.radians(self._get_alpha(x_i, y_i))
        alpha_j = np.radians(self._get_alpha(x_j, y_j))
        sin_half = np.sin(self.theta_rad / 2.0)

        denom_i_left = np.sin((np.pi - self.theta_rad) / 2.0 + alpha_i)
        denom_i_right = np.sin((np.pi - self.theta_rad) / 2.0 - alpha_i)
        denom_j_right = np.sin((np.pi - self.theta_rad) / 2.0 - alpha_j)
        cos_alpha_i = np.cos(alpha_i)

        if (
            abs(denom_i_left) <= 1e-12
            or abs(denom_i_right) <= 1e-12
            or abs(denom_j_right) <= 1e-12
            or abs(cos_alpha_i) <= 1e-12
        ):
            return 0.0

        overlap_width = (
            d_i * sin_half / denom_i_left
            + d_j * sin_half / denom_j_right
            - np.hypot(x_i - x_j, y_i - y_j) / cos_alpha_i
        )
        coverage_width_i = d_i * sin_half * (1.0 / denom_i_left + 1.0 / denom_i_right)

        if not np.isfinite(overlap_width) or not np.isfinite(coverage_width_i):
            return 0.0
        if coverage_width_i <= 1e-12:
            return 0.0
        return float(max(overlap_width / coverage_width_i, 0.0))

    @staticmethod
    def _compute_overlap_excess_length(
        child_points, overlap_rates, threshold=0.2
    ) -> float:
        line_arr = np.asarray(child_points, dtype=float)
        rates = np.asarray(overlap_rates, dtype=float)
        if line_arr.ndim != 2 or len(line_arr) < 2 or len(rates) < 2:
            return 0.0

        total_length = 0.0
        for point0, point1, rate0, rate1 in zip(
            line_arr[:-1], line_arr[1:], rates[:-1], rates[1:]
        ):
            segment_length = float(
                np.hypot(point1[0] - point0[0], point1[1] - point0[1])
            )
            if segment_length <= 0 or (not np.isfinite(rate0)) or (not np.isfinite(rate1)):
                continue

            if rate0 <= threshold and rate1 <= threshold:
                continue
            if rate0 > threshold and rate1 > threshold:
                total_length += segment_length
                continue
            if abs(rate1 - rate0) <= 1e-12:
                if rate0 > threshold:
                    total_length += segment_length
                continue

            crossing_ratio = (threshold - rate0) / (rate1 - rate0)
            crossing_ratio = float(np.clip(crossing_ratio, 0.0, 1.0))
            if rate0 > threshold and rate1 <= threshold:
                total_length += segment_length * crossing_ratio
            elif rate0 <= threshold and rate1 > threshold:
                total_length += segment_length * (1.0 - crossing_ratio)

        return float(total_length)

    def _snapshot_global_metrics_state(self):
        if self.metrics_state_grid is None:
            return None
        return self.metrics_state_grid.snapshot_state()

    def _snapshot_global_covered_mask(self):
        if self.metrics_state_grid is None:
            return None
        return np.array(self.metrics_state_grid.covered_sample_mask, copy=True)

    def _restore_global_metrics_state(self, snapshot) -> None:
        if self.metrics_state_grid is None or snapshot is None:
            return
        self.metrics_state_grid.restore_state(snapshot)

    def _point_with_width(self, x, y):
        """返回 [x, y, 覆盖宽度(w_total)]。"""
        return [
            x,
            y,
            get_total_width(
                x,
                y,
                theta=self.theta,
                microstep=self.current_microstep,
            ),
        ]

    def _get_alpha(self, x, y):
        """使用当前分区的差分步长计算坡度角。"""
        return get_alpha(x, y, microstep=self.current_microstep)

    def _mark_global_segment_covered_for_planning(self, start_point, end_point):
        """规划期临时写入全局覆盖状态；不累计统计指标。"""
        if self.metrics_state_grid is None:
            return
        self.metrics_state_grid.mark_segment_covered_for_planning(
            np.asarray(start_point, dtype=float), np.asarray(end_point, dtype=float)
        )

    def _mark_global_polyline_covered_for_planning(self, points):
        """规划期临时写入整条测线覆盖状态；不累计统计指标。"""
        if self.metrics_state_grid is None:
            return
        self.metrics_state_grid.mark_polyline_covered_for_planning(points)

    # ------------------------------------------------------------------
    # 核心测线生成算法
    # ------------------------------------------------------------------

    def _extend_line_bidirectional(self, start_x, start_y, target_partition_id, step):
        """
        双向延伸生成主测线

        从起点向正反两个方向交替延伸，按当前分区边界与全局网格收益决定是否继续。
        """
        transaction_snapshot = self._snapshot_global_metrics_state()
        line = [self._point_with_width(start_x, start_y)]

        ext_step = step
        ext_loop = 0
        front_x, front_y = start_x, start_y
        back_x, back_y = start_x, start_y
        front_stopped = False
        back_stopped = False
        front_stop_reason = TerminationReason.NONE
        back_stop_reason = TerminationReason.NONE
        extend_front = True
        terminated_reason = TerminationReason.NONE

        while True:
            if front_stopped and back_stopped:
                break

            if extend_front and front_stopped:
                extend_front = False
                continue
            if not extend_front and back_stopped:
                extend_front = True
                continue

            ext_loop += 1

            if extend_front:
                dx, dy = forward_direction(
                    get_gx(front_x, front_y), get_gy(front_x, front_y)
                )
                new_x = front_x + ext_step * dx
                new_y = front_y + ext_step * dy
            else:
                dx, dy = forward_direction(
                    get_gx(back_x, back_y), get_gy(back_x, back_y)
                )
                new_x = back_x - ext_step * dx
                new_y = back_y - ext_step * dy

            if not self._is_in_partition(new_x, new_y, target_partition_id):
                if extend_front:
                    print(f"  [主测线延伸-前端] 延伸{ext_loop}步后超出边界")
                    front_stopped = True
                    front_stop_reason = TerminationReason.BOUNDARY
                else:
                    print(f"  [主测线延伸-后端] 延伸{ext_loop}步后超出边界")
                    back_stopped = True
                    back_stop_reason = TerminationReason.BOUNDARY
                extend_front = not extend_front
                continue

            new_point = self._point_with_width(new_x, new_y)
            anchor_point = line[-1] if extend_front else line[0]
            if not self._candidate_segment_has_value(anchor_point, new_point):
                if extend_front:
                    print(f"  [主测线延伸-前端] 新线段对应全局网格收益不足，停止该端")
                    front_stopped = True
                    front_stop_reason = TerminationReason.LOW_VALUE
                else:
                    print(f"  [主测线延伸-后端] 新线段对应全局网格收益不足，停止该端")
                    back_stopped = True
                    back_stop_reason = TerminationReason.LOW_VALUE
                extend_front = not extend_front
                continue

            if extend_front:
                self._mark_global_segment_covered_for_planning(line[-1], new_point)
                line.append(new_point)
            else:
                self._mark_global_segment_covered_for_planning(new_point, line[0])
                line.insert(0, new_point)

            if extend_front:
                front_x, front_y = new_x, new_y
            else:
                back_x, back_y = new_x, new_y

            extend_front = not extend_front

        line = np.array(line)
        terminated_reason = self._resolve_line_termination(
            front_stop_reason, back_stop_reason
        )
        status = (
            "全局网格收益终止"
            if terminated_reason == TerminationReason.LOW_VALUE
            else "边界终止"
        )
        terminated_by = (
            terminated_reason.name.lower()
            if terminated_reason != TerminationReason.NONE
            else "boundary"
        )
        print(
            f"[主测线1] 完成 | 共{len(line)}个点 | {status} | "
            f"起点({line[0][0]:.1f},{line[0][1]:.1f}) 终点({line[-1][0]:.1f},{line[-1][1]:.1f})"
        )
        self._restore_global_metrics_state(transaction_snapshot)
        return line, terminated_by, 0.0, 0.0, 0.0

    @staticmethod
    def _line_has_enough_points(line) -> bool:
        return len(np.asarray(line)) >= 2

    def _compute_line_gain_ratio(self, line, prior_covered_mask):
        if self.metrics_state_grid is None or not self._line_has_enough_points(line):
            return 0.0, 0.0, 0.0
        return self.metrics_state_grid.compute_polyline_gain(
            line, prior_covered_mask=prior_covered_mask
        )

    def _compute_group_gain_ratio(self, candidates, prior_covered_mask):
        lines = [candidate.line for candidate in candidates]
        if self.metrics_state_grid is None:
            return 0.0, 0.0, 0.0
        return self.metrics_state_grid.compute_cumulative_polyline_group_gain(
            lines, prior_covered_mask=prior_covered_mask
        )

    def _resolve_child_line_gain_threshold(self, parent_gain_ratio):
        parent_gain_ratio = float(np.clip(parent_gain_ratio, 0.0, 1.0))
        return float(
            max(
                self.child_line_min_gain_threshold,
                parent_gain_ratio * self.child_line_parent_gain_factor,
            )
        )

    def _filter_child_candidates_by_segment_gain(
        self, candidates, prior_covered_mask, parent_gain_ratio
    ):
        retained = []
        filtered = []
        gain_threshold = self._resolve_child_line_gain_threshold(parent_gain_ratio)
        if self.metrics_state_grid is None:
            for candidate in candidates:
                candidate.parent_gain_ratio = float(parent_gain_ratio)
                candidate.gain_threshold = float(gain_threshold)
                candidate.gain_ratio = 1.0
                candidate.new_area = 0.0
                candidate.hit_area = 0.0
                retained.append(candidate)
            return retained, filtered, gain_threshold

        if prior_covered_mask is None:
            simulated_prior = np.array(
                self.metrics_state_grid.covered_sample_mask, copy=True
            )
        else:
            simulated_prior = np.array(prior_covered_mask, dtype=bool, copy=True)

        for candidate in candidates:
            line_hit_mask = self.metrics_state_grid.compute_polyline_hit_mask(
                candidate.line
            )
            hit_area = self.metrics_state_grid._area_from_sample_mask(line_hit_mask)
            if hit_area > 1e-12:
                new_area = self.metrics_state_grid._area_from_sample_mask(
                    line_hit_mask & (~simulated_prior)
                )
                gain_ratio = float(new_area / hit_area)
            else:
                new_area = 0.0
                gain_ratio = 0.0

            candidate.parent_gain_ratio = float(parent_gain_ratio)
            candidate.gain_threshold = float(gain_threshold)
            candidate.gain_ratio = float(gain_ratio)
            candidate.new_area = float(new_area)
            candidate.hit_area = float(hit_area)
            if hit_area > 1e-12 and gain_ratio >= gain_threshold:
                retained.append(candidate)
                simulated_prior |= line_hit_mask
            else:
                filtered.append(candidate)
        return retained, filtered, gain_threshold

    @staticmethod
    def _is_finite_candidate_point(point) -> bool:
        point_arr = np.asarray(point, dtype=float)
        return (
            point_arr.shape[0] >= 3
            and np.all(np.isfinite(point_arr[:3]))
            and float(point_arr[2]) > 0.0
        )

    def _compute_perpendicular_candidate_point(
        self, parent_point, direction, target_partition_id
    ):
        x, y = float(parent_point[0]), float(parent_point[1])
        alpha = float(self._get_alpha(x, y))
        h = float(get_height(x, y))
        if not np.isfinite(alpha) or not np.isfinite(h) or h <= 0:
            return None, None, "low_value"

        if alpha <= 0.005:
            d = 2 * h * np.tan(self.theta_rad / 2) * (1 - self.n)
            tx = d * get_gx(x, y)
            ty = d * get_gy(x, y)
        else:
            alpha_rad = np.radians(alpha)
            sin_alpha = np.sin(alpha_rad)
            if abs(sin_alpha) <= 1e-12:
                return None, None, "low_value"
            A = np.sin(np.radians(90) - self.theta_rad / 2 + alpha_rad)
            B = np.sin(np.radians(90) - self.theta_rad / 2 - alpha_rad)
            if abs(A) <= 1e-12 or abs(B) <= 1e-12:
                return None, None, "low_value"
            C = np.sin(self.theta_rad / 2) / A - 1 / sin_alpha
            D = (
                self.n * np.sin(self.theta_rad / 2) * (1 / A + 1 / B)
                - np.sin(self.theta_rad / 2) / B
                - 1 / sin_alpha
            )
            if abs(D) <= 1e-12:
                return None, None, "low_value"
            next_h = h * C / D
            if not np.isfinite(next_h):
                return None, None, "low_value"
            tx = (h - next_h) / np.tan(alpha_rad) * get_gx(x, y)
            ty = (h - next_h) / np.tan(alpha_rad) * get_gy(x, y)

        new_x = x - direction * tx
        new_y = y - direction * ty
        if not np.isfinite(new_x) or not np.isfinite(new_y):
            return None, None, "low_value"
        if not self._is_in_partition(new_x, new_y, target_partition_id):
            return None, None, "boundary"

        candidate_point = np.asarray(self._point_with_width(new_x, new_y), dtype=float)
        if not self._is_finite_candidate_point(candidate_point):
            return None, None, "low_value"

        overlap_rate = self._compute_parent_child_overlap_rate(
            parent_point, candidate_point
        )
        return candidate_point, overlap_rate, None

    @staticmethod
    def _make_child_candidate(points, overlap_rates) -> _ChildLineCandidate:
        line = np.asarray(points, dtype=float)
        rates = np.asarray(overlap_rates, dtype=float)
        return _ChildLineCandidate(
            line=line.copy(), initial_line=line.copy(), overlap_rates=rates.copy()
        )

    def _generate_initial_child_candidates(
        self, parent_segments, direction, target_partition_id, prior_covered_mask=None
    ):
        candidates: list[_ChildLineCandidate] = []
        current_points = []
        current_rates = []
        valid_points = 0
        boundary_rejects = 0
        low_value_rejects = 0

        def flush_current_run():
            nonlocal current_points, current_rates
            if len(current_points) >= 2:
                candidates.append(
                    self._make_child_candidate(current_points, current_rates)
                )
            current_points = []
            current_rates = []

        for parent_segment in parent_segments:
            flush_current_run()
            parent_arr = np.asarray(parent_segment, dtype=float)
            if len(parent_arr) == 0:
                continue

            for parent_point in parent_arr:
                candidate_point, overlap_rate, reject_reason = (
                    self._compute_perpendicular_candidate_point(
                        parent_point,
                        direction,
                        target_partition_id,
                    )
                )
                if candidate_point is None:
                    if reject_reason == "boundary":
                        boundary_rejects += 1
                    else:
                        low_value_rejects += 1
                    flush_current_run()
                    continue

                if len(current_points) == 0:
                    has_value = self._candidate_point_has_value(
                        candidate_point, prior_covered_mask=prior_covered_mask
                    )
                else:
                    has_value = self._candidate_segment_has_value(
                        current_points[-1],
                        candidate_point,
                        prior_covered_mask=prior_covered_mask,
                    )
                    if not has_value:
                        flush_current_run()
                        has_value = self._candidate_point_has_value(
                            candidate_point, prior_covered_mask=prior_covered_mask
                        )
                if not has_value:
                    low_value_rejects += 1
                    flush_current_run()
                    continue

                current_points.append(candidate_point)
                current_rates.append(overlap_rate)
                valid_points += 1

            flush_current_run()

        return candidates, valid_points, boundary_rejects, low_value_rejects

    def _extend_child_candidate(
        self,
        candidate: _ChildLineCandidate,
        target_partition_id,
        direction_name,
        prior_covered_mask=None,
    ) -> _ChildLineCandidate:
        line = [np.asarray(point, dtype=float) for point in candidate.line]
        if len(line) < 2:
            return candidate

        ext_step = self.step
        ext_loop = 0
        front_x, front_y = float(line[-1][0]), float(line[-1][1])
        back_x, back_y = float(line[0][0]), float(line[0][1])
        front_stopped = False
        back_stopped = False
        front_stop_reason = TerminationReason.NONE
        back_stop_reason = TerminationReason.NONE
        extend_front = True

        while True:
            if front_stopped and back_stopped:
                break
            if extend_front and front_stopped:
                extend_front = False
                continue
            if not extend_front and back_stopped:
                extend_front = True
                continue

            ext_loop += 1
            if extend_front:
                dx, dy = forward_direction(
                    get_gx(front_x, front_y), get_gy(front_x, front_y)
                )
                new_x = front_x + ext_step * dx
                new_y = front_y + ext_step * dy
            else:
                dx, dy = forward_direction(get_gx(back_x, back_y), get_gy(back_x, back_y))
                new_x = back_x - ext_step * dx
                new_y = back_y - ext_step * dy

            if not np.isfinite(new_x) or not np.isfinite(new_y):
                stop_reason = TerminationReason.LOW_VALUE
            elif not self._is_in_partition(new_x, new_y, target_partition_id):
                stop_reason = TerminationReason.BOUNDARY
            else:
                new_point = np.asarray(self._point_with_width(new_x, new_y), dtype=float)
                anchor_point = line[-1] if extend_front else line[0]
                if not self._is_finite_candidate_point(new_point):
                    stop_reason = TerminationReason.LOW_VALUE
                elif not self._candidate_segment_has_value(
                    anchor_point, new_point, prior_covered_mask=prior_covered_mask
                ):
                    stop_reason = TerminationReason.LOW_VALUE
                else:
                    stop_reason = TerminationReason.NONE

            if stop_reason != TerminationReason.NONE:
                side_name = "前端" if extend_front else "后端"
                if stop_reason == TerminationReason.BOUNDARY:
                    print(f"  [测线延伸-{side_name}] 延伸{ext_loop}步后超出边界")
                else:
                    print(
                        f"  [测线延伸-{side_name}] 新线段未覆盖未测网格，停止该端"
                    )
                if extend_front:
                    front_stopped = True
                    front_stop_reason = stop_reason
                else:
                    back_stopped = True
                    back_stop_reason = stop_reason
                extend_front = not extend_front
                continue

            if extend_front:
                line.append(new_point)
                front_x, front_y = new_x, new_y
            else:
                line.insert(0, new_point)
                back_x, back_y = new_x, new_y

            extend_front = not extend_front

        candidate.line = np.asarray(line, dtype=float)
        candidate.terminated_reason = self._resolve_line_termination(
            front_stop_reason, back_stop_reason
        )
        print(
            f"  [{direction_name}子测线段] 延伸完成 | 共{len(candidate.line)}点 | "
            f"终止={candidate.terminated_reason.name.lower()}"
        )
        return candidate

    def _extend_child_group_external_ends(
        self,
        candidates: list[_ChildLineCandidate],
        target_partition_id,
        direction_name,
        prior_covered_mask=None,
    ) -> list[_ChildLineCandidate]:
        """仅延伸测线组作为逻辑整体时的两个外端。"""
        candidates = [c for c in candidates if self._line_has_enough_points(c.line)]
        if not candidates:
            return []

        line_lists = [
            [np.asarray(point, dtype=float) for point in candidate.line]
            for candidate in candidates
        ]
        front_idx = len(line_lists) - 1
        back_idx = 0
        front_x, front_y = float(line_lists[front_idx][-1][0]), float(
            line_lists[front_idx][-1][1]
        )
        back_x, back_y = float(line_lists[back_idx][0][0]), float(
            line_lists[back_idx][0][1]
        )

        ext_step = self.step
        ext_loop = 0
        front_stopped = False
        back_stopped = False
        front_stop_reason = TerminationReason.NONE
        back_stop_reason = TerminationReason.NONE
        extend_front = True

        while True:
            if front_stopped and back_stopped:
                break
            if extend_front and front_stopped:
                extend_front = False
                continue
            if not extend_front and back_stopped:
                extend_front = True
                continue

            ext_loop += 1
            if extend_front:
                dx, dy = forward_direction(
                    get_gx(front_x, front_y), get_gy(front_x, front_y)
                )
                new_x = front_x + ext_step * dx
                new_y = front_y + ext_step * dy
                anchor_point = line_lists[front_idx][-1]
            else:
                dx, dy = forward_direction(get_gx(back_x, back_y), get_gy(back_x, back_y))
                new_x = back_x - ext_step * dx
                new_y = back_y - ext_step * dy
                anchor_point = line_lists[back_idx][0]

            if not np.isfinite(new_x) or not np.isfinite(new_y):
                stop_reason = TerminationReason.LOW_VALUE
            elif not self._is_in_partition(new_x, new_y, target_partition_id):
                stop_reason = TerminationReason.BOUNDARY
            else:
                new_point = np.asarray(self._point_with_width(new_x, new_y), dtype=float)
                if not self._is_finite_candidate_point(new_point):
                    stop_reason = TerminationReason.LOW_VALUE
                elif not self._candidate_segment_has_value(
                    anchor_point, new_point, prior_covered_mask=prior_covered_mask
                ):
                    stop_reason = TerminationReason.LOW_VALUE
                else:
                    stop_reason = TerminationReason.NONE

            if stop_reason != TerminationReason.NONE:
                side_name = "前端" if extend_front else "后端"
                if stop_reason == TerminationReason.BOUNDARY:
                    print(f"  [测线组外端延伸-{side_name}] 延伸{ext_loop}步后超出边界")
                else:
                    print(
                        f"  [测线组外端延伸-{side_name}] 新线段未覆盖未测网格，停止该端"
                    )
                if extend_front:
                    front_stopped = True
                    front_stop_reason = stop_reason
                else:
                    back_stopped = True
                    back_stop_reason = stop_reason
                extend_front = not extend_front
                continue

            if extend_front:
                line_lists[front_idx].append(new_point)
                front_x, front_y = new_x, new_y
            else:
                line_lists[back_idx].insert(0, new_point)
                back_x, back_y = new_x, new_y

            extend_front = not extend_front

        if len(candidates) == 1:
            candidates[0].terminated_reason = self._resolve_line_termination(
                front_stop_reason, back_stop_reason
            )
        else:
            candidates[front_idx].terminated_reason = front_stop_reason
            candidates[back_idx].terminated_reason = back_stop_reason

        for candidate, line in zip(candidates, line_lists):
            candidate.line = np.asarray(line, dtype=float)

        print(
            f"  [{direction_name}子测线组] 外端延伸完成 | 共{len(candidates)}段 | "
            f"前端={front_stop_reason.name.lower()} | 后端={back_stop_reason.name.lower()}"
        )
        return candidates

    def _generate_candidate_group(
        self,
        parent_segments,
        direction,
        target_partition_id,
        direction_name,
        prior_covered_mask=None,
    ):
        candidates, valid_points, boundary_rejects, low_value_rejects = (
            self._generate_initial_child_candidates(
                parent_segments,
                direction,
                target_partition_id,
                prior_covered_mask=prior_covered_mask,
            )
        )
        extended_candidates = self._extend_child_group_external_ends(
            candidates,
            target_partition_id,
            direction_name,
            prior_covered_mask=prior_covered_mask,
        )
        return extended_candidates, valid_points, boundary_rejects, low_value_rejects

    def _record_retained_child_group(
        self,
        candidates,
        partition_id,
        lines_dir,
        line_counter,
        perp_iter,
    ):
        total_points = 0

        for segment_index, candidate in enumerate(candidates, start=1):
            line = np.asarray(candidate.line, dtype=float)
            if not self._line_has_enough_points(line):
                continue

            overlap_excess_length = self._compute_overlap_excess_length(
                candidate.initial_line, candidate.overlap_rates, threshold=0.2
            )
            max_overlap_eta = (
                float(np.max(candidate.overlap_rates))
                if len(candidate.overlap_rates) > 0
                else 0.0
            )
            terminated_by = (
                candidate.terminated_reason.name.lower()
                if candidate.terminated_reason != TerminationReason.NONE
                else "boundary"
            )
            record = self._record_line(
                line,
                partition_id,
                terminated_by,
                overlap_excess_length=overlap_excess_length,
                max_overlap_eta=max_overlap_eta,
            )
            if record is None:
                continue

            color = (
                "orange"
                if candidate.terminated_reason == TerminationReason.LOW_VALUE
                else "purple"
            )
            self.visualizer.draw_line(line, color, 1.5)
            total_points += len(line)
            line_counter += 1
            suffix = (
                f"_seg{segment_index:02d}" if len(candidates) > 1 else ""
            )
            snap_path = lines_dir / f"line_{line_counter:04d}_iter{perp_iter}{suffix}.png"
            self.visualizer.save_snapshot(snap_path)
            print(f"  >> 已保存测线: {snap_path}")

        return total_points, line_counter

    def _draw_parent_group_light(self, parent_segments, draw_allowed, draw_first_line, perp_iter):
        if not draw_allowed:
            return
        if not (draw_first_line or perp_iter > 1):
            return
        for segment in parent_segments:
            line = np.asarray(segment, dtype=float)
            if len(line) > 1:
                self.visualizer.draw_light_line(line)

    def _generate_perpendicular_lines(
        self,
        main_line,
        direction,
        target_partition_id,
        lines_dir,
        t0,
        line_counter,
        partition_id,
        draw_first_line=True,
    ):
        """生成垂直扩展测线，按连续有效点切分子测线组。"""
        if not self._line_has_enough_points(main_line):
            return 0, line_counter

        parent_segments = [np.asarray(main_line, dtype=float)]
        parent_draw_allowed = True
        consecutive_jump_groups = 0
        perp_iter = 0
        total_points = 0
        parent_gain_ratio = self.main_line_parent_gain_ratio

        direction_name = "正向" if direction == 1 else "反向"
        print(
            f"\n[{direction_name}扩展] 开始从主测线扩展... | "
            f"初始父收益率={parent_gain_ratio:.2%}"
        )

        while True:
            perp_iter += 1
            transaction_snapshot = self._snapshot_global_metrics_state()
            prior_covered_mask = (
                np.array(transaction_snapshot["covered_sample_mask"], copy=True)
                if transaction_snapshot is not None
                else None
            )

            self._draw_parent_group_light(
                parent_segments, parent_draw_allowed, draw_first_line, perp_iter
            )

            candidates, valid_points, boundary_rejects, low_value_rejects = (
                self._generate_candidate_group(
                    parent_segments,
                    direction,
                    target_partition_id,
                    direction_name,
                    prior_covered_mask=prior_covered_mask,
                )
            )

            elapsed = time.time() - t0
            print(
                f"[{direction_name}扩展 第{perp_iter}轮] 父对象{len(parent_segments)}段 -> "
                f"有效点{valid_points}个 / 子测线{len(candidates)}段 | 已耗时{elapsed:.1f}s"
            )

            if len(candidates) == 0:
                self._restore_global_metrics_state(transaction_snapshot)
                print(
                    f"[{direction_name}扩展] 第{perp_iter}轮无有效子测线，结束规划 | "
                    f"边界拒绝={boundary_rejects} | 低收益拒绝={low_value_rejects}"
                )
                break

            all_gain_ratio, all_new_area, all_hit_area = self._compute_group_gain_ratio(
                candidates, prior_covered_mask
            )
            if all_hit_area <= 1e-12:
                self._restore_global_metrics_state(transaction_snapshot)
                print(
                    f"[{direction_name}扩展] 第{perp_iter}轮候选组全局扫中面积为0，结束规划"
                )
                break

            retained_candidates, filtered_candidates, gain_threshold = (
                self._filter_child_candidates_by_segment_gain(
                    candidates, prior_covered_mask, parent_gain_ratio
                )
            )
            if len(filtered_candidates) > 0:
                print(
                    f"[{direction_name}扩展 第{perp_iter}轮] 累计过滤低收益子段"
                    f"{len(filtered_candidates)}段 | 保留候选{len(retained_candidates)}段 | "
                    f"父收益率={parent_gain_ratio:.2%} | 阈值={gain_threshold:.2%}"
                )

            if len(retained_candidates) == 0:
                print(
                    f"[{direction_name}扩展 第{perp_iter}轮] 无子段达到累计收益阈值"
                    f"{gain_threshold:.2%}，候选组仅作为跳板评估"
                )
                consecutive_jump_groups += 1
                self._restore_global_metrics_state(transaction_snapshot)
                if consecutive_jump_groups > self.max_consecutive_jump_lines:
                    print(
                        f"[{direction_name}扩展] 连续第{consecutive_jump_groups}个跳板组仍无子段达标，"
                        "终止该方向测线生成"
                    )
                    break

                parent_segments = [
                    candidate.line.copy() for candidate in candidates
                ]
                parent_draw_allowed = False
                print(
                    f"[{direction_name}扩展] 第{perp_iter}轮作为跳板组继续迭代 | "
                    f"连续跳板={consecutive_jump_groups}/{self.max_consecutive_jump_lines} | "
                    f"父收益率保持={parent_gain_ratio:.2%}"
                )
                continue

            decision_candidates = retained_candidates
            gain_ratio, new_area, hit_area = self._compute_group_gain_ratio(
                decision_candidates, prior_covered_mask
            )
            print(
                f"[{direction_name}扩展 第{perp_iter}轮] 保留组累计收益率={gain_ratio:.2%} "
                f"(新增={new_area:.2f}m² / 扫中={hit_area:.2f}m²) | "
                "仅用于下一轮父收益率"
            )

            consecutive_jump_groups = 0
            self._restore_global_metrics_state(transaction_snapshot)
            added_points, line_counter = self._record_retained_child_group(
                decision_candidates,
                partition_id,
                lines_dir,
                line_counter,
                perp_iter,
            )
            total_points += added_points
            parent_segments = [candidate.line.copy() for candidate in decision_candidates]
            parent_gain_ratio = gain_ratio if hit_area > 1e-12 else parent_gain_ratio
            parent_draw_allowed = True

        print(f"[{direction_name}扩展] 完成 | 共{perp_iter}轮 | 保留{total_points}点")
        return total_points, line_counter

    # ------------------------------------------------------------------
    # 分区级规划
    # ------------------------------------------------------------------

    def _plan_partition(self, partition_id, lines_dir, dot_dir, t0):
        """为指定分区规划测线"""
        lines_dir.mkdir(parents=True, exist_ok=True)
        dot_dir.mkdir(parents=True, exist_ok=True)

        rows, cols = np.where(self.cluster_matrix == partition_id)
        if len(rows) == 0:
            print(f"[分区{partition_id}] 未找到有效网格点, 跳过")
            return PartitionResult(partition_id, [], [], 0.0, 0.0)

        start_info = self._select_partition_start_point(partition_id, rows, cols)
        start_x = float(start_info["x"])
        start_y = float(start_info["y"])
        start_depth = float(start_info["depth"])
        self.partition_start_points[partition_id] = start_info

        print(f"\n{'=' * 60}")
        if start_info["rule"] == "deepest_cell_with_gradient_clearance":
            print(
                f"[分区{partition_id}] 开始规划 | 起点: ({start_x:.1f}, {start_y:.1f}) | "
                f"最深可行网格中心 | depth={start_depth:.2f}m | "
                f"distance={start_info['boundary_distance']:.2f}m | Δr={start_info['delta_r']:.2f}m"
            )
        elif start_info["rule"] in {
            "coordinate_center_nearest_cell",
            "geometric_center_nearest_cell",
        }:
            center_label = (
                "分区几何中心最近网格中心"
                if start_info["rule"] == "geometric_center_nearest_cell"
                else "坐标中心最近网格中心"
            )
            print(
                f"[分区{partition_id}] 开始规划 | 起点: ({start_x:.1f}, {start_y:.1f}) | "
                f"{center_label} | depth={start_depth:.2f}m | "
                f"target=({start_info['target_center_x']:.1f}, {start_info['target_center_y']:.1f})"
            )
        else:
            print(
                f"[分区{partition_id}] 开始规划 | 起点: ({start_x:.1f}, {start_y:.1f}) | "
                f"回退到最深网格中心 | depth={start_depth:.2f}m"
            )

        initial_summary = self.metrics_state_grid.summarize_partition(partition_id)

        previous_step = self.step
        previous_microstep = self.current_microstep
        self.step = self.metrics_state_grid.planning_step
        self.current_microstep = self.metrics_state_grid.microstep

        print(
            f"[分区{partition_id}] 全局统计网格视图 | d={self.metrics_state_grid.cell_size:.2f}m | "
            f"step={self.step:.2f}m | microstep={self.current_microstep:.2f}m | "
            f"cells={initial_summary.total_cells}"
        )

        try:
            # 设置画布
            self.visualizer.setup_figure(partition_id)
            self.visualizer.draw_seed_point(
                start_x,
                start_y,
                label="Seed point",
                annotation=f"S{partition_id}",
            )

            line_counter = 0

            # 主测线
            print(f"[分区{partition_id}] 主测线开始双向延伸")
            line, main_terminated, overlap_excess_length, max_overlap_eta, repeated_area = self._extend_line_bidirectional(
                start_x, start_y, partition_id, self.step
            )
            main_record = self._record_line(
                line,
                partition_id,
                main_terminated,
                overlap_excess_length=overlap_excess_length,
                max_overlap_eta=max_overlap_eta,
                repeated_area=repeated_area,
            )
            if main_record is None:
                print(f"[分区{partition_id}] 主测线不足2点，丢弃并结束该分区规划")
                total_points1 = 0
                total_points2 = 0
                total_points = 0
            else:
                line_counter += 1
                self.visualizer.draw_line(line, "b", 1.5, "Main line")

                snap_path = lines_dir / f"line_{line_counter:04d}_main.png"
                self.visualizer.save_snapshot(snap_path)
                print(f"  >> 已保存测线: {snap_path}")

                # 正向扩展
                total_points1, line_counter = self._generate_perpendicular_lines(
                    line, 1, partition_id, lines_dir, t0, line_counter, partition_id, False
                )

                # 反向扩展
                total_points2, line_counter = self._generate_perpendicular_lines(
                    line, -1, partition_id, lines_dir, t0, line_counter, partition_id, False
                )

                total_points = len(line) + total_points1 + total_points2

            # 保存最终图
            overlay_legend = self.visualizer.draw_fine_grid_overlay(
                self.metrics_state_grid, show_legend=True, partition_id=partition_id
            )
            final_path = lines_dir / "plan_line_final.png"
            handles, labels = self.visualizer.ax.get_legend_handles_labels()
            self.visualizer.ax.legend(handles=handles + overlay_legend)
            self.visualizer.save_snapshot(final_path)
            self.visualizer.close_figure()

            # 收集结果
            partition_records = [
                r for r in self.all_records if r.partition_id == partition_id
            ]
            partition_all_lines = [r.points for r in partition_records]

            save_dot_csv(partition_all_lines, dot_dir)

            coverage_summary = self.metrics_state_grid.summarize_partition(partition_id)
            self.partition_coverage_summaries[partition_id] = coverage_summary

            print(
                f"[分区{partition_id}完成] 共生成{len(partition_records)}条测线 | 总点数约{total_points} | "
                f"细网格覆盖率={coverage_summary.coverage_rate:.2%}"
            )

            return PartitionResult(
                partition_id=partition_id,
                lines=partition_all_lines,
                records=partition_records,
                total_length=sum(r.length for r in partition_records),
                total_coverage=sum(r.coverage for r in partition_records),
            )
        finally:
            self.step = previous_step
            self.current_microstep = previous_microstep

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def plan_line(self, start_x, start_y, output_dir=None):
        """单分区测线规划"""
        target_partition_id = get_partition_for_point(
            start_x,
            start_y,
            self.xs,
            self.ys,
            self.cluster_matrix,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
        )
        print(
            f"[单分区规划] 参考点 ({float(start_x):.1f}, {float(start_y):.1f}) "
            f"映射到分区 {target_partition_id}"
        )
        return self.plan_partition(target_partition_id, output_dir=output_dir)

    def plan_partition(self, partition_id, output_dir=None):
        """按明确分区 ID 执行单分区测线规划。"""
        if output_dir is None:
            output_dir = "./multibeam/output"

        target_partition_id = int(partition_id)
        available_partition_ids = {
            int(pid) for pid in np.unique(self.cluster_matrix).astype(int) if pid >= 0
        }
        if target_partition_id not in available_partition_ids:
            raise ValueError(
                f"目标分区 {target_partition_id} 不存在，可选={sorted(available_partition_ids)}"
            )

        output_path = Path(output_dir)
        is_timestamped = self._is_timestamped_output_name(output_path.name)

        if is_timestamped:
            run_dir = output_path
        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_path / current_time

        lines_dir = run_dir / "lines"
        metrics_dir = run_dir / "metrics"
        lines_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[单分区规划] 仅规划分区 {target_partition_id} | "
            "测线取舍规则=adaptive-parent-gain | "
            f"父收益折减={self.child_line_parent_gain_factor:.0%} | "
            f"最低阈值={self.child_line_min_gain_threshold:.0%}"
        )

        pid_dir = lines_dir / str(target_partition_id)
        result = self._plan_partition(
            target_partition_id, pid_dir, pid_dir, time.time()
        )

        save_dot_csv(result.lines, lines_dir)
        self.save_metrics_excel(metrics_dir=metrics_dir)

        print(f"[测线规划完成] 输出目录: {run_dir}")
        return result

    def plan_all(self, output_dir=None):
        """自动遍历所有分区规划测线"""
        if output_dir is None:
            output_dir = "./multibeam/output"

        output_path = Path(output_dir)
        is_timestamped = self._is_timestamped_output_name(output_path.name)

        if is_timestamped:
            run_dir = output_path
        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_path / current_time

        lines_dir = run_dir / "lines"
        metrics_dir = run_dir / "metrics"
        lines_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        detected_partition_ids = [
            int(pid)
            for pid in sorted(np.unique(self.cluster_matrix).astype(int))
            if pid >= 0
        ]
        partition_ids, mean_depth_by_pid = self._order_partition_ids_by_mean_depth(
            detected_partition_ids
        )
        order_desc = ", ".join(
            f"{pid}(mean_depth={mean_depth_by_pid[pid]:.2f}m)"
            for pid in partition_ids
        )
        print(
            f"[全局规划] 检测到 {len(detected_partition_ids)} 个分区: "
            f"{detected_partition_ids}"
        )
        print(f"[全局规划] 按平均水深由深到浅规划: [{order_desc}]")

        all_results = []
        for pid in partition_ids:
            pid_dir = lines_dir / str(pid)
            result = self._plan_partition(pid, pid_dir, pid_dir, time.time())
            all_results.append(result)

        # 合并输出
        flat_all_lines = [line for r in all_results for line in r.lines]
        save_dot_csv(flat_all_lines, lines_dir)
        self.visualizer.draw_merged_lines(
            all_results,
            lines_dir,
            self.metrics_state_grid,
            self.partition_start_points,
        )
        self.save_metrics_excel(metrics_dir=metrics_dir)

        total_lines = sum(len(r.records) for r in all_results)
        print(f"[全局规划完成] 共 {len(partition_ids)} 个分区 | {total_lines} 条测线")

        return all_results

    def save_metrics_excel(self, output_dir=None, current_time=None, metrics_dir=None):
        """保存指标到 Excel"""
        import pandas as pd

        if metrics_dir is not None:
            metrics_dir = Path(metrics_dir)
        else:
            if output_dir is None:
                output_dir = "./multibeam/output"
            if current_time is None:
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_dir = Path(output_dir) / "metrics" / current_time
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if self.metrics_state_grid is not None:
            self.partition_coverage_summaries = self.metrics_state_grid.summarize_all()

        partition_rows = []
        global_total_length = 0.0
        global_old_proxy_coverage = 0.0
        global_overlap_excess_length = 0.0
        global_max_overlap_eta = 0.0
        global_total_lines = 0
        global_new_partition_area = 0.0
        global_new_covered_area = 0.0
        global_repeated_area = 0.0

        partition_id_set = (
            set(r.partition_id for r in self.all_records)
            | set(self.partition_coverage_summaries.keys())
        )
        if self.metrics_state_grid is not None:
            partition_id_set |= set(self.metrics_state_grid.partition_ids)
        partition_ids = sorted(partition_id_set)

        for pid in partition_ids:
            recs = [r for r in self.all_records if r.partition_id == pid]
            total_length = sum(r.length for r in recs)
            proxy_coverage = sum(r.coverage for r in recs)
            overlap_excess_length = sum(r.overlap_excess_length for r in recs)
            max_overlap_eta = max((r.max_overlap_eta for r in recs), default=0.0)
            coverage_summary = self.partition_coverage_summaries.get(pid)
            repeated_rate = 0.0
            if coverage_summary and coverage_summary.total_area > 0:
                repeated_rate = (
                    coverage_summary.repeated_area / coverage_summary.total_area
                )

            partition_rows.append(
                {
                    "分区ID": int(pid),
                    "测线条数": len(recs),
                    "测线总长度(m)": round(total_length, 2),
                    "积分覆盖面积-旧口径(m²)": round(proxy_coverage, 2),
                    "重叠率超过20%部分长度(m)": round(overlap_excess_length, 2),
                    "parent-child公式下最大η(诊断)": round(max_overlap_eta, 4),
                    "细网格待测面积-新口径(m²)": round(
                        coverage_summary.total_area if coverage_summary else 0.0, 2
                    ),
                    "细网格覆盖面积-新口径(m²)": round(
                        coverage_summary.covered_area if coverage_summary else 0.0, 2
                    ),
                    "累计冗余扫测面积-新口径(m²)": round(
                        coverage_summary.repeated_area if coverage_summary else 0.0, 2
                    ),
                    "累计冗余扫测率-新口径": f"{repeated_rate:.4%}",
                    "细网格覆盖率-新口径": (
                        f"{coverage_summary.coverage_rate:.4%}"
                        if coverage_summary
                        else "0.0000%"
                    ),
                    "细网格漏测率-新口径": (
                        f"{coverage_summary.miss_rate:.4%}"
                        if coverage_summary
                        else "100.0000%"
                    ),
                    "细网格边长l(m)": round(
                        coverage_summary.cell_size if coverage_summary else 0.0, 2
                    ),
                    "规划步长step(m)": round(
                        coverage_summary.planning_step if coverage_summary else 0.0, 2
                    ),
                    "坡度差分microstep(m)": round(
                        coverage_summary.microstep if coverage_summary else 0.0, 2
                    ),
                    "full_cells": coverage_summary.full_cells
                    if coverage_summary
                    else 0,
                    "partial_cells": coverage_summary.partial_cells
                    if coverage_summary
                    else 0,
                    "uncovered_cells": (
                        coverage_summary.uncovered_cells if coverage_summary else 0
                    ),
                }
            )

            global_total_lines += len(recs)
            global_total_length += total_length
            global_old_proxy_coverage += proxy_coverage
            global_overlap_excess_length += overlap_excess_length
            global_max_overlap_eta = max(global_max_overlap_eta, max_overlap_eta)
            if coverage_summary is not None:
                global_new_partition_area += coverage_summary.total_area
                global_new_covered_area += coverage_summary.covered_area
                global_repeated_area += coverage_summary.repeated_area

        df_partition = pd.DataFrame(partition_rows)

        old_effective_coverage = global_old_proxy_coverage * (1 - self.n)
        old_coverage_rate = (
            old_effective_coverage / self.total_area if self.total_area > 0 else 0
        )
        global_new_covered_area = min(
            max(global_new_covered_area, 0.0), self.total_area
        )
        new_coverage_rate = (
            global_new_covered_area / self.total_area if self.total_area > 0 else 0.0
        )
        new_coverage_rate = min(max(new_coverage_rate, 0.0), 1.0)
        new_miss_rate = 1.0 - new_coverage_rate
        repeated_rate = global_repeated_area / self.total_area if self.total_area > 0 else 0.0

        global_rows = [
            {"指标": "测线条数", "值": global_total_lines},
            {"指标": "测线总长度(m)", "值": round(global_total_length, 2)},
            {"指标": "矩形海域总面积-旧分母(m²)", "值": round(self.total_area, 2)},
            {
                "指标": "积分总覆盖面积-旧口径(m²)",
                "值": round(global_old_proxy_coverage, 2),
            },
            {
                "指标": "积分有效覆盖面积-旧口径(m²)",
                "值": round(old_effective_coverage, 2),
            },
            {"指标": "积分覆盖率-旧口径", "值": f"{old_coverage_rate:.4%}"},
            {
                "指标": "积分漏测率-旧口径(可能为负)",
                "值": f"{1 - old_coverage_rate:.4%}",
            },
            {
                "指标": "重叠率超过20%部分总长度(m)",
                "值": round(global_overlap_excess_length, 2),
            },
            {
                "指标": "parent-child公式下最大η(诊断)",
                "值": round(global_max_overlap_eta, 4),
            },
            {
                "指标": "固定矩形海域总面积-新分母(m²)",
                "值": round(self.total_area, 2),
            },
            {
                "指标": "细网格待测面积偏差-诊断值(m²)",
                "值": round(global_new_partition_area - self.total_area, 2),
            },
            {
                "指标": "细网格真实覆盖面积-新口径(m²)",
                "值": round(global_new_covered_area, 2),
            },
            {
                "指标": "累计冗余扫测面积-新口径(m²)",
                "值": round(global_repeated_area, 2),
            },
            {"指标": "累计冗余扫测率-新口径", "值": f"{repeated_rate:.4%}"},
            {"指标": "细网格覆盖率-新口径", "值": f"{new_coverage_rate:.4%}"},
            {"指标": "细网格漏测率-新口径", "值": f"{new_miss_rate:.4%}"},
            {
                "指标": "新口径统计说明",
                "值": "全局统计网格记录所有已保留测线对所有分区的贡献；测线取舍使用全局网格收益，测线端点仍受当前分区边界约束；跳板测线不绘制、不统计、不提交覆盖。",
            },
            {
                "指标": "跨分区贡献记录数",
                "值": len(self.line_partition_contributions),
            },
        ]

        for pid in partition_ids:
            coverage_summary = self.partition_coverage_summaries.get(pid)
            if coverage_summary is None:
                continue
            summary_dict = asdict(coverage_summary)
            for key in ("partition_id",):
                summary_dict.pop(key, None)
            for metric_name, metric_value in summary_dict.items():
                display_value = (
                    round(metric_value, 4)
                    if isinstance(metric_value, float)
                    else metric_value
                )
                global_rows.append(
                    {
                        "指标": f"分区{pid}-{metric_name}",
                        "值": display_value,
                    }
                )

        contribution_rows = []
        for contribution in self.line_partition_contributions:
            contribution_rows.append(
                {
                    "line_id": int(contribution.line_id),
                    "测线所属分区ID": int(contribution.owner_partition_id),
                    "贡献目标分区ID": int(contribution.target_partition_id),
                    "是否跨分区贡献": bool(
                        contribution.owner_partition_id
                        != contribution.target_partition_id
                    ),
                    "扫中面积(m²)": round(contribution.hit_area, 2),
                    "新增覆盖面积(m²)": round(contribution.new_area, 2),
                    "冗余扫测面积(m²)": round(contribution.repeated_area, 2),
                    "扫中样本数": int(contribution.hit_samples),
                    "新增覆盖样本数": int(contribution.new_samples),
                    "冗余扫测样本数": int(contribution.repeated_samples),
                }
            )

        df_global = pd.DataFrame(global_rows)
        df_contribution = pd.DataFrame(
            contribution_rows,
            columns=[
                "line_id",
                "测线所属分区ID",
                "贡献目标分区ID",
                "是否跨分区贡献",
                "扫中面积(m²)",
                "新增覆盖面积(m²)",
                "冗余扫测面积(m²)",
                "扫中样本数",
                "新增覆盖样本数",
                "冗余扫测样本数",
            ],
        )

        metrics_path = metrics_dir / "metrics.xlsx"
        with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
            df_partition.to_excel(writer, sheet_name="分区统计", index=False)
            df_global.to_excel(writer, sheet_name="全局统计", index=False)
            df_contribution.to_excel(writer, sheet_name="跨分区贡献", index=False)

        print(f"\n[指标统计完成] 已保存到: {metrics_path}")
        return metrics_path


# ---------------------------------------------------------------------------
# 向后兼容接口
# ---------------------------------------------------------------------------


def plan_line(start_x, start_y, xs, ys, cluster_matrix, n=0.1, step=50, theta=120):
    """向后兼容: 单分区测线规划"""
    planner = SurveyPlanner(xs, ys, cluster_matrix, theta=theta, n=n, step=step)
    return planner.plan_line(start_x, start_y)


def plan_all_line(xs, ys, cluster_matrix, n=0.1, step=50, theta=120):
    """向后兼容: 全局测线规划"""
    planner = SurveyPlanner(xs, ys, cluster_matrix, theta=theta, n=n, step=step)
    results = planner.plan_all()
    planner.save_metrics_excel()
    return results
