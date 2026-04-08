"""
测线规划工具函数模块

包含几何计算、角度计算等纯数学工具函数。
从 Planner.py 中拆分出来，遵循单一职责原则。
"""

import numpy as np

try:
    from shapely.geometry import LineString, Point
except ImportError:  # pragma: no cover - optional dependency fallback
    LineString = None
    Point = None

from multibeam.models import CoverageSummary, ScoreBreakdown


def compute_signed_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算从向量v1到v2的带符号夹角（弧度），逆时针为正

    参数:
        v1: 向量1 [x, y]
        v2: 向量2 [x, y]

    返回:
        float: 夹角（弧度），范围 [-π, π]
    """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return np.arctan2(cross, dot)


def compute_total_turning_angle(line_arr: np.ndarray) -> float:
    """
    计算整条测线的累计偏转角（所有相邻线段夹角之和）

    参数:
        line_arr: 测线点数组 [N, 3] 或 [N, 2]

    返回:
        float: 累计偏转角（弧度）
    """
    if len(line_arr) < 3:
        return 0.0
    total = 0.0
    for i in range(len(line_arr) - 2):
        v1 = line_arr[i + 1] - line_arr[i]
        v2 = line_arr[i + 2] - line_arr[i + 1]
        total += compute_signed_angle(v1, v2)
    return total


def point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """
    计算点到线段的最短距离

    参数:
        px, py: 点坐标
        x1, y1: 线段起点
        x2, y2: 线段终点

    返回:
        float: 点到线段的最短距离
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def line_to_coverage_mask(
    line: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """将测线近似映射为覆盖网格掩码。"""
    if line is None or len(line) < 2:
        return np.zeros_like(valid_mask, dtype=bool)

    line_xy = np.asarray(line[:, :2], dtype=float)
    mean_width = float(np.mean(np.asarray(line[:, 2], dtype=float))) if line.shape[1] >= 3 else 0.0
    buffer_radius = max(mean_width / 2.0, 1.0)
    mask = np.zeros_like(valid_mask, dtype=bool)
    candidate_rows, candidate_cols = np.where(valid_mask)
    if len(candidate_rows) == 0:
        return mask

    if LineString is None or Point is None:
        for row, col in zip(candidate_rows, candidate_cols):
            px = float(grid_x[row, col])
            py = float(grid_y[row, col])
            for i in range(len(line_xy) - 1):
                x1, y1 = line_xy[i]
                x2, y2 = line_xy[i + 1]
                if point_to_segment_distance(px, py, x1, y1, x2, y2) <= buffer_radius:
                    mask[row, col] = True
                    break
        return mask

    corridor = LineString(line_xy).buffer(
        buffer_radius,
        cap_style="flat",
        join_style="mitre",
    )

    min_x, min_y, max_x, max_y = corridor.bounds
    for row, col in zip(candidate_rows, candidate_cols):
        x = grid_x[row, col]
        y = grid_y[row, col]
        if x < min_x or x > max_x or y < min_y or y > max_y:
            continue
        if corridor.covers(Point(float(x), float(y))):
            mask[row, col] = True

    return mask


def evaluate_candidate_line(
    line: np.ndarray,
    coverage_counts: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    valid_mask: np.ndarray,
    cell_area: float,
    score_weights: dict,
) -> tuple[ScoreBreakdown, np.ndarray]:
    """评估候选测线对全局覆盖的增益与代价。"""
    line_mask = line_to_coverage_mask(line, grid_x, grid_y, valid_mask)
    unique_gain_cells = int(np.sum(line_mask & (coverage_counts == 0)))
    overlap_cells = int(np.sum(line_mask & (coverage_counts > 0)))

    diffs = np.diff(line[:, :2], axis=0)
    length = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1)))) if len(line) >= 2 else 0.0
    bend = float(abs(compute_total_turning_angle(line[:, :2]))) if len(line) >= 3 else 0.0

    gain_term = score_weights["gain"] * unique_gain_cells * cell_area
    overlap_term = score_weights["overlap"] * overlap_cells * cell_area
    length_term = score_weights["length"] * length
    bend_term = score_weights["bend"] * bend
    score = gain_term - overlap_term - length_term - bend_term

    return (
        ScoreBreakdown(
            unique_gain_cells=unique_gain_cells,
            overlap_cells=overlap_cells,
            length=length,
            bend=bend,
            score=score,
        ),
        line_mask,
    )


def build_coverage_summary(
    coverage_counts: np.ndarray,
    valid_mask: np.ndarray,
    cell_area: float,
    raw_coverage_area: float,
) -> CoverageSummary:
    """根据覆盖计数构建可信的全局指标。"""
    valid_cells = int(np.sum(valid_mask))
    covered_once = int(np.sum((coverage_counts > 0) & valid_mask))
    repeated_cells = int(np.sum((coverage_counts > 1) & valid_mask))

    total_area = valid_cells * cell_area
    unique_coverage_area = covered_once * cell_area
    overlap_area = repeated_cells * cell_area
    coverage_rate = unique_coverage_area / total_area if total_area > 0 else 0.0
    coverage_rate = min(max(coverage_rate, 0.0), 1.0)
    leakage_rate = 1.0 - coverage_rate

    return CoverageSummary(
        total_area=total_area,
        raw_coverage_area=raw_coverage_area,
        unique_coverage_area=unique_coverage_area,
        overlap_area=overlap_area,
        coverage_rate=coverage_rate,
        leakage_rate=leakage_rate,
    )
