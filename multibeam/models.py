"""
数据模型定义
包含测线规划所需的所有数据类和枚举
"""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class TerminationReason(Enum):
    """测线终止原因枚举"""

    NONE = auto()  # 未终止
    BOUNDARY = auto()  # 超出分区边界
    SPIRAL = auto()  # 累计偏转角 > 360°
    INTERSECTION = auto()  # 与已有测线相交
    SATURATION = auto()  # 测线收缩至质心内侧覆盖范围内
    DEGRADATION = auto()  # 新增点数 < 5
    EMPTY = auto()  # 无有效点


@dataclass
class LineRecord:
    """单条测线的即时指标记录"""

    line_id: int
    partition_id: int
    points: np.ndarray  # [N, 3] = [x, y, w_total]
    length: float
    coverage: float
    terminated_by: str  # "boundary" / "spiral" / "intersection" / "saturation" / "degradation" / "empty"
    unique_gain_area: float = 0.0
    overlap_area: float = 0.0
    score: float = 0.0


@dataclass
class PartitionResult:
    """单个分区的规划结果"""

    partition_id: int
    lines: list[np.ndarray]  # 原始 line 数组（用于绘图）
    records: list[LineRecord]  # 指标记录
    total_length: float
    total_coverage: float


@dataclass
class ScoreBreakdown:
    """候选测线评分拆解"""

    unique_gain_cells: int
    overlap_cells: int
    length: float
    bend: float
    score: float


@dataclass
class CoverageSummary:
    """全局覆盖统计摘要"""

    total_area: float
    raw_coverage_area: float
    unique_coverage_area: float
    overlap_area: float
    coverage_rate: float
    leakage_rate: float
