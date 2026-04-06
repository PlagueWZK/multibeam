"""
几何计算工具模块
基于 Shapely 库实现精确的几何相交检测
"""

import numpy as np
from shapely.geometry import LineString


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
