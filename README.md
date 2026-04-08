# Multibeam — 多波束测深自动测线规划系统

基于机器学习预测模型的多波束测线自动规划工具，实现从海图数据到测线方案的全流程自动化。

## 技术栈

| 组件 | 版本/说明 |
|---|---|
| **Python** | >= 3.12 |
| **NumPy** | >= 2.4 — 核心数值计算 |
| **Pandas** | >= 3.0 — Excel 读写、指标统计 |
| **scikit-learn** | >= 1.8 — RandomForestRegressor、KMeans、StandardScaler |
| **Matplotlib / Seaborn** | >= 3.10 / 0.13 — 可视化输出 |
| **OpenPyXL** | >= 3.1 — Excel 引擎 |
| **Joblib** | >= 1.4 — 模型序列化 (pkl) |
| **Shapely** | >= 2.0 — 精确几何相交检测 |
| **包管理** | `uv` (pyproject.toml + uv.lock) |

## 包结构

```
multibeam/
├── main.py                          # 入口：四阶段流水线编排
├── model/                           # 模型训练（一次性脚本，非运行时模块）
│   ├── TrainHeightModel.py          #   训练深度预测 RF 模型
│   └── TrainGradientModel.py        #   训练梯度 RF 模型（向量化实现）
├── tool/                            # 通用工具层
│   ├── Tool.py                      #   Excel 读取、模型加载、训练辅助
│   ├── Data.py                      #   运行时推理门面（ModelManager 单例懒加载）
│   └── geometry.py                  #   几何计算（Shapely 相交检测）
├── multibeam/                       # 核心业务逻辑
│   ├── models.py                    #   数据类定义（LineRecord, PartitionResult, TerminationReason）
│   ├── planner_utils.py             #   几何计算工具函数
│   ├── planner_visualizer.py        #   可视化逻辑（SurveyVisualizer）
│   ├── GridCell.py                  #   Phase 1: 最优网格边长计算
│   ├── Coverage.py                  #   Phase 2: 覆盖次数矩阵计算
│   ├── Partition.py                 #   Phase 3: K-means 空间聚类分区
│   └── Planner.py                   #   Phase 4: 测线规划核心调度
└── data/                            # 数据与模型资产
    ├── data.xlsx                    #   原始海图网格数据
    └── *.pkl                        #   三个预训练 Random Forest 模型
```

### 各模块职责

| 包/模块 | 职责 | 业务划分 |
|---|---|---|
| **`model/`** | 离线训练脚本 | 从 `data.xlsx` 提取数据，训练 RF 模型并序列化 |
| **`tool.Tool`** | I/O 与模型管理 | Excel 解析、pkl 加载、训练辅助 |
| **`tool.Data`** | 推理门面 | `ModelManager` 单例懒加载，首次调用时才加载模型 |
| **`tool.geometry`** | 几何计算 | Shapely 精确相交检测，解决穿越穿模问题 |
| **`multibeam.models`** | 数据模型定义 | `LineRecord`, `PartitionResult`, `TerminationReason` 枚举 |
| **`multibeam.planner_utils`** | 几何工具 | 角度计算、累计偏转角计算 |
| **`multibeam.planner_visualizer`** | 可视化 | `SurveyVisualizer` 类，封装所有 matplotlib 绘图 |
| **`GridCell`** | 网格参数优化 | 计算最优网格边长 `d` |
| **`Coverage`** | 覆盖矩阵计算 | ML 预测深度 + 邻域覆盖判定 |
| **`Partition`** | 空间聚类分区 | K-means 聚类 + Elbow 法 |
| **`Planner`** | 测线规划引擎 | 核心调度，使用枚举管理终止条件 |

## 核心业务流（四阶段流水线）

```
data.xlsx ──► [Phase 1] GridCell.calculate_optimal_mesh_size()
                   │  输入: 原始 Excel 深度矩阵
                   │  算法: 圆面积拟合误差 < 0.1% 的最优网格边长 d
                   ▼
              d_optimal
                   │
                   ▼
            [Phase 2] Coverage.calculate_coverage_matrix_with_ml()
                   │  输入: 海域边界 (X,Y 范围) + d_optimal
                   │  步骤:
                   │    1. generate_resampled_grid() → 用 RF 模型预测每个网格中心深度
                   │    2. 邻域覆盖判定: 对每个网格检查其邻域内哪些点被覆盖
                   │       - 水平距离 dis_xoy ≤ w_max/2 (圆形邻域)
                   │       - 3D 距离 dis_xyz ≤ dis_sp (换能器开角约束)
                   │       - 坡度角 alpha 影响覆盖宽度
                   ▼
              xs, ys, depth_matrix, coverage_matrix
                   │
                   ▼
            [Phase 3] Partition.partition_coverage_matrix()
                   │  输入: 坐标 + 覆盖次数矩阵
                   │  特征: (X, Y, coverage_count) 三维
                   │  权重: 自动计算覆盖次数 vs 空间坐标的尺度比
                   │  聚类: KMeans + Elbow 法自动确定 U（或手动指定）
                   ▼
              cluster_matrix (2D), U
                   │
                   ▼
            [Phase 4] SurveyPlanner.plan_all()
                   │  对每个分区:
                   │    1. 主测线: 从质心双向延伸（沿梯度垂直方向）
                   │    2. 垂直扩展: 正向/反向逐轮生成垂直测线
                   │    3. 终止条件: boundary / spiral / intersection /
                   │                saturation / degradation / empty
                   │  输出: 测线图 + dot.csv + metrics.xlsx
                   ▼
              output/<timestamp>/
                ├── lines/<pid>/*.png, dot.csv
                ├── metrics/metrics.xlsx
                └── partition/*.png
```

## 关键实体/模型

### 数据模型

| 实体 | 结构 | 说明 |
|---|---|---|
| **`LineRecord`** (dataclass) | `line_id, partition_id, points[N,3], length, coverage, terminated_by` | 单条测线的即时指标 |
| **`PartitionResult`** (dataclass) | `partition_id, lines[], records[], total_length, total_coverage` | 单个分区的规划结果 |
| **`TerminationReason`** (Enum) | `NONE, BOUNDARY, SPIRAL, INTERSECTION, SATURATION, DEGRADATION, EMPTY` | 测线终止原因枚举 |
| **覆盖次数矩阵** | `int[rows, cols]` | 每个网格被覆盖的次数 |
| **聚类矩阵** | `int[rows, cols]` | 每个网格所属的分区 ID |
| **测线点数组** | `float[N, 3]` | `[x, y, w_total]`，第三列为预计算覆盖宽度 |

### 预训练模型（3 个 RandomForest）

| 模型文件 | 预测目标 | 特征 | 训练数据源 |
|---|---|---|---|
| `height_random_forest_model.pkl` | 深度 (z) | (x, y) | `data.xlsx` 深度矩阵 |
| `gx_random_forest_model.pkl` | X 方向归一化梯度 | (x, y) | 深度矩阵的中心差分归一化 |
| `gy_random_forest_model.pkl` | Y 方向归一化梯度 | (x, y) | 深度矩阵的中心差分归一化 |

### 输入数据格式 (`data.xlsx`)

```
(无表头)
列0: 空        列1: Y坐标    列2+: X坐标
行0: 空
行1: 空        空           X坐标值 (海里)
行2+: Y坐标值   空           深度矩阵 Z[j][i]
```

## 配置参数

所有参数硬编码在代码中，无外部配置文件。

| 参数 | 默认值 | 所在文件 | 说明 |
|---|---|---|---|
| `theta` | 120° | 多处 | 换能器开角 |
| `n` | 0.1 | Planner.py | 重叠率 |
| `step` | 50 | Planner.py | 测线延伸步长 |
| `min_error` | 0.001 | GridCell.py | 网格拟合误差阈值 |
| `U` | None (自动) | main.py | 分区数量（可手动指定） |
| `microstep` | 35 | Data.py | 坡度角计算的微步长 |
| 海域范围 | 0~5×1852, 0~4×1852 | main.py | X/Y 边界（米） |

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 训练模型（首次运行或数据更新时）
python model/TrainHeightModel.py
python model/TrainGradientModel.py

# 3. 运行测线规划
python main.py
```

输出位于 `multibeam/output/<timestamp>/` 目录下。
