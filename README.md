# Multibeam — 多波束测深自动测线规划系统

基于机器学习预测模型的多波束测深自动测线规划工具，实现从原始海图网格数据到覆盖次数分析、聚类分区、测线规划、统一统计与可视化输出的全流程自动化。

---

## 当前架构要点

当前版本已经完成**统一网格重构**，核心原则是：

1. 先根据原始数据和专利误差项求最优网格边长 `d`
2. 后续覆盖次数、聚类分区、收益判断、覆盖统计、细网格状态图**全部复用这套统一网格**
3. 不再在每个分区内部重新生成第二套局部细网格

这意味着项目现在采用的是：

> **单一统一网格真相（single grid truth）**

---

## 技术栈

| 组件 | 版本/说明 |
| --- | --- |
| **Python** | >= 3.12 |
| **NumPy** | >= 2.4 — 核心数值计算 |
| **Pandas** | >= 3.0 — Excel 读写、指标统计 |
| **scikit-learn** | >= 1.8 — RandomForestRegressor、KMeans、StandardScaler |
| **Matplotlib / Seaborn** | >= 3.10 / 0.13 — 可视化输出 |
| **OpenPyXL** | >= 3.1 — Excel 引擎 |
| **Joblib** | >= 1.4 — 模型序列化 |
| **Shapely** | >= 2.0 — 几何相交检测辅助 |
| **包管理** | `uv` |

---

## 包结构

```text
multibeam/
├── main.py                          # 入口：四阶段流水线编排
├── model/                           # 模型训练（一次性脚本）
│   ├── TrainHeightModel.py
│   └── TrainGradientModel.py
├── tool/
│   ├── Tool.py                      # Excel 读取、模型加载、训练辅助
│   ├── Data.py                      # 运行时推理门面（ModelManager 懒加载）
│   └── geometry.py                  # 几何计算辅助
├── multibeam/
│   ├── GridCell.py                  # Phase 1: 最优网格边长 d 计算
│   ├── Coverage.py                  # Phase 2: 统一网格重采样 + 覆盖次数矩阵
│   ├── Partition.py                 # Phase 3: 基于统一网格的 KMeans 分区
│   ├── coverage_state_grid.py       # 统一网格覆盖状态管理与分区视图
│   ├── Planner.py                   # Phase 4: 测线规划核心调度
│   ├── planner_visualizer.py        # 绘图与最终图输出
│   ├── planner_utils.py             # 规划几何工具
│   └── models.py                    # 数据类与枚举
└── data/
    ├── data.xlsx                    # 原始海图网格数据（坐标单位：海里）
    └── *.pkl                        # 预训练随机森林模型
```

---

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `tool.Tool` | 读取 `data.xlsx`、加载模型、训练辅助 |
| `tool.Data` | 深度 / 梯度 / 覆盖宽度等运行时查询接口 |
| `multibeam.GridCell` | 计算最优网格边长 `d`，并处理统一网格边界截断 |
| `multibeam.Coverage` | 在统一网格上预测深度并计算覆盖次数矩阵 |
| `multibeam.Partition` | 对统一网格上的有效海域格进行聚类分区 |
| `multibeam.coverage_state_grid` | 统一网格上的覆盖状态管理、收益判断、统计与渲染数据生成 |
| `multibeam.Planner` | 基于统一网格与分区结果执行测线规划 |
| `multibeam.planner_visualizer` | 输出分区图、测线图、统一细网格状态图 |
| `multibeam.models` | `LineRecord`、`PartitionResult`、`TerminationReason` 等数据模型 |

---

## 核心业务流（四阶段流水线）

```text
data.xlsx
  │
  ├─► [Phase 1] GridCell.calculate_optimal_mesh_size()
  │       输入: 原始 Excel 深度网格（坐标单位海里）
  │       输出: 最优网格边长 d（米）
  │       规则:
  │         - 使用专利误差项 E(d, ξ)
  │         - d 严格小于原始数据网格边长
  │
  ├─► [Phase 2] Coverage.calculate_coverage_matrix_with_ml()
  │       输入: 海域边界 + d
  │       输出:
  │         - xs, ys                    统一网格坐标
  │         - depth_matrix             重采样深度矩阵
  │         - coverage_matrix          覆盖次数矩阵
  │         - boundary_mask            有效海域掩码
  │         - cell_effective_area      每格真实有效面积
  │         - cell_area_ratio          每格有效面积比例
  │
  ├─► [Phase 3] Partition.partition_coverage_matrix()
  │       输入: xs, ys, coverage_matrix, boundary_mask
  │       特征: (X, Y, coverage_count, gx, gy)
  │       注意:
  │         - 仅对 boundary_mask=True 的有效格聚类
  │         - 无效格在 cluster_matrix 中标记为 -1
  │
  └─► [Phase 4] Planner.SurveyPlanner.plan_all()
          输入: 统一网格 + 分区矩阵
          规则:
            - 规划算法主体沿用当前实现
            - 不再在分区内重建第二套网格
            - step 从统一网格 d 派生
            - microstep 从统一网格 d/2 派生并裁剪到 [10, 70]
            - full 判定为 9/9 采样点全部覆盖
          输出:
            - 分区测线图
            - 全局分区视图
            - 全局细网格状态视图
            - dot.csv
            - metrics.xlsx
```

---

## 统一网格与边界处理

### 为什么不要求海域边长能整除 `d`

`d` 由专利误差项和原始网格分辨率约束共同确定，**不会为了整除海域边长而二次修改**。

### 边界处部分网格如何处理

若海域边长不能被 `d` 整除，则边界处会出现部分超出海域的格子。当前项目采用：

1. **固定 `d` 不变**
2. 使用 `arange` 生成覆盖整个海域的统一网格
3. 对边界格使用：
   - `boundary_mask`：该格是否与海域有正面积重叠
   - `cell_effective_area`：该格在海域内的真实有效面积
   - `cell_area_ratio`：有效面积占满格面积比例
   - `sample_mask`：采样点中哪些点真正落在海域内

因此：

- 覆盖次数计算不需要改理论公式
- 聚类只针对有效海域格
- 统计不会把边界 partial cell 当成完整 `d²` 硬算

---

## 网格边长 `d` 的计算原则

当前采用：

- 理论圆面积：`A = π ξ²`
- 离散邻域面积：`B = N(d, ξ) · d²`
- 半径设定：`ξ = w_max / 2`
- 专利误差项：`E(d, ξ) = (1 - |A - B| / A) ξ²`

实现上：

1. 保留专利误差项不变
2. `min_error` 仍表示相对面积误差阈值
3. 内部将其映射为专利误差项下限
4. 候选 `d` 上界限制为**严格小于原始数据网格边长**

当前默认：

- `min_error = 0.001 (= 0.1%)`

---

## 聚类分区说明

`Partition.partition_coverage_matrix()` 当前不是只看 `(X, Y, coverage_count)`，而是使用：

- `X`
- `Y`
- `coverage_count`
- `gx`
- `gy`

这样做的目的是：

1. 保证空间连续性
2. 让地形剧变区域更容易形成合理边界

同时当前版本已经明确：

- **只对 `boundary_mask=True` 的有效格聚类**
- **无效格统一标记为 `-1`**

---

## 测线规划说明

当前 `SurveyPlanner` 的核心几何规划逻辑保持不变，主要包括：

1. **主测线双向延伸**
   - 从分区中心附近起点出发
   - 沿梯度垂线方向向前后两端延伸

2. **正向 / 反向垂直扩展**
   - 围绕主测线向两侧扩展从测线

3. **收益驱动终止**
   - 新点或新线段若不再触达任何 non-full cell，则停止该方向扩展

### 当前规划参数派生规则

- `cell_size = d`
- `step = d`
- `microstep = clip(d / 2, 10, 70)`
- `sampling_points = 9`
- `full_threshold = 1.0`

即：

> 统一网格不仅用于统计，也直接驱动规划收益判断与步长尺度。

---

## 关键实体 / 数据结构

### 数据模型

| 实体 | 结构 | 说明 |
| --- | --- | --- |
| `LineRecord` | `line_id, partition_id, points[N,3], length, coverage, terminated_by` | 单条测线即时指标 |
| `PartitionResult` | `partition_id, lines[], records[], total_length, total_coverage` | 单个分区规划结果 |
| `TerminationReason` | `NONE, BOUNDARY, LOW_VALUE, ...` | 测线终止原因枚举 |

### 统一网格核心矩阵

| 名称 | 结构 | 说明 |
| --- | --- | --- |
| `depth_matrix` | `float[rows, cols]` | 统一网格重采样深度矩阵 |
| `coverage_matrix` | `int[rows, cols]` | 每个统一网格被覆盖的次数 |
| `boundary_mask` | `bool[rows, cols]` | 是否属于有效海域 |
| `cell_effective_area` | `float[rows, cols]` | 每格真实有效面积 |
| `cluster_matrix` | `int[rows, cols]` | 分区矩阵；无效格为 `-1` |
| 测线点数组 | `float[N, 3]` | `[x, y, w_total]` |

---

## 预训练模型

| 模型文件 | 预测目标 | 特征 | 来源 |
| --- | --- | --- | --- |
| `height_random_forest_model.pkl` | 深度 `z` | `(x, y)` | `data.xlsx` |
| `gx_random_forest_model.pkl` | `gx` | `(x, y)` | 深度矩阵中心差分归一化 |
| `gy_random_forest_model.pkl` | `gy` | `(x, y)` | 深度矩阵中心差分归一化 |

---

## 输入数据格式 (`data.xlsx`)

```text
(无表头)
列0: 空        列1: Y坐标    列2+: X坐标
行0: 空
行1: 空        空           X坐标值 (海里)
行2+: Y坐标值   空           深度矩阵 Z[j][i]
```

注意：

- Excel 中原始坐标单位是**海里**
- 运行时统一换算到**米**

---

## 配置参数（当前实现）

| 参数 | 默认值 | 所在文件 | 说明 |
| --- | --- | --- | --- |
| `theta` | `120°` | 多处 | 换能器开角 |
| `n` | `0.1` | `Planner.py` | 重叠率 |
| `min_error` | `0.001 (=0.1%)` | `GridCell.py` | 相对面积误差阈值（内部映射到专利误差项） |
| `U` | `None` | `main.py` / `Partition.py` | 分区数；默认通过 Elbow 自动确定 |
| `sampling_points` | `9` | `coverage_state_grid.py` | 每格采样点数 |
| `full_threshold` | `1.0` | `Planner.py` / `coverage_state_grid.py` | 9/9 采样点全部覆盖才算 full |
| `step` | `d` | `Planner.py` | 规划步长直接由统一网格边长派生 |
| `microstep` | `clip(d/2, 10, 70)` | `Planner.py` | 坡度差分步长由统一网格派生 |
| 海域范围 | `X: 0~5×1852`, `Y: 0~4×1852` | `main.py` | 真实海域边界（米） |

---

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. （首次运行或模型需要更新时）训练模型
python model/TrainHeightModel.py
python model/TrainGradientModel.py

# 3. 运行完整流程
python main.py
```

---

## 输出说明

输出目录默认位于：

```text
multibeam/output/<timestamp>/
```

典型内容：

```text
multibeam/output/<timestamp>/
├── lines/
│   ├── <partition_id>/line_*.png
│   ├── all_partitions_final_partition_view.png
│   ├── all_partitions_final_fine_grid_view.png
│   └── dot.csv
├── metrics/
│   └── metrics.xlsx
└── partition/
    ├── elbow_method.png
    └── partition_U=<U>.png
```

其中：

- `all_partitions_final_partition_view.png`：全局分区底图 + 测线
- `all_partitions_final_fine_grid_view.png`：全局统一细网格状态 + 测线
- `metrics.xlsx`：分区统计与全局统计

---

## 当前版本特征总结

当前版本的关键特征可以概括为：

- **专利误差项选取统一网格 `d`**
- **统一网格贯穿覆盖次数、聚类、规划、统计、可视化**
- **边界 partial cell 通过有效面积修正处理**
- **聚类排除无效海域格**
- **规划步长与微步长由统一网格直接派生**
- **full 判定采用 9/9 全覆盖严格口径**
