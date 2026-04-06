"""
测线规划工具函数模块

包含几何计算、角度计算等纯数学工具函数。
从 Planner.py 中拆分出来，遵循单一职责原则。
"""

import numpy as np


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
