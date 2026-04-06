"""
几何计算工具模块
基于 Shapely 库实现精确的几何相交检测
"""

import numpy as np
from shapely.geometry import LineString, Point


def check_line_intersection_shapely(
    new_pt_start: np.ndarray, new_pt_end: np.ndarray, prev_lines: list[np.ndarray]
) -> bool:
    """
    使用 Shapely 检测新线段与历史测线是否相交

    相比原有的距离阈值法，此方法能精确检测：
    - 完全不相交
    - 端点相交
    - 中间交叉
    - 共线重叠

    参数:
        new_pt_start: 新线段起点 [x, y] 或 [x, y, w]
        new_pt_end: 新线段终点 [x, y] 或 [x, y, w]
        prev_lines: 历史测线列表，每个元素是 [N, 3] 或 [N, 2] 的 numpy 数组

    返回:
        bool: True 表示相交，False 表示不相交
    """
    # 提取前两列 (x, y)，忽略第三列 (覆盖宽度)
    new_segment = LineString([new_pt_start[:2], new_pt_end[:2]])

    for prev_line in prev_lines:
        if prev_line is None or len(prev_line) < 2:
            continue

        # 构造历史测线的 LineString
        prev_ls = LineString(prev_line[:, :2])

        # 使用 Shapely 的 intersects 方法检测相交
        if new_segment.intersects(prev_ls):
            return True

    return False


def _point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """
    计算点到线段的最短距离

    此函数保留用于向后兼容和其他可能用途

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


def check_self_intersection(
    line: np.ndarray, min_points: int = 10, proximity_threshold: float = 15.0
) -> bool:
    """
    检测测线是否与自身相交（混合方案）

    策略：
    1. 先用 Shapely is_simple 检测精确交叉（快速，O(n log n)）
    2. 再用端点接近检测处理"接近但未交叉"的情况（O(n)）

    参数:
        line: 测线点数组 [N, 2] 或 [N, 3]
        min_points: 最少点数，低于此值不检测
        proximity_threshold: 端点接近阈值（米），默认15m

    返回:
        bool: True 表示自交
    """
    if line is None or len(line) < min_points:
        return False

    line_xy = line[:, :2]
    ls = LineString(line_xy)

    # 第一层：精确自交检测
    if not ls.is_simple:
        return True

    # 第二层：端点接近检测
    # 检测最后一个点是否靠近非相邻线段（跳过最后3个相邻段）
    last_pt = line_xy[-1]
    for i in range(len(line_xy) - 4):
        seg = LineString([line_xy[i], line_xy[i + 1]])
        if seg.distance(Point(last_pt)) < proximity_threshold:
            return True

    # 第三层：起点接近检测（处理闭合环形）
    first_pt = line_xy[0]
    for i in range(max(0, len(line_xy) - 4), len(line_xy) - 1):
        seg = LineString([line_xy[i], line_xy[i + 1]])
        if seg.distance(Point(first_pt)) < proximity_threshold:
            return True

    return False
