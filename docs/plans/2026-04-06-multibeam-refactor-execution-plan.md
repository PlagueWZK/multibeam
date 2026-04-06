# Multibeam 重构执行计划

**计划ID**: PLAN-2026-04-06-001  
**关联需求**: REQ-2026-04-06-001  
**创建时间**: 2026-04-06  
**内部执行等级**: L (Serial Native Execution)

---

## 1. 执行策略

### 1.1 内部等级决策
- **等级**: L
- **理由**: 
  - 重构涉及 5 个核心模块，有明确的依赖关系
  - 需要串行执行以保证每步验证
  - 不需要大规模并行或子代理协调

### 1.2 Wave 结构

```
Wave 1: 基础设施准备
  ├─ Step 1.1: 添加 shapely 依赖到 requirements.txt
  ├─ Step 1.2: 创建 multibeam/models.py (数据类定义)
  └─ Step 1.3: 重构 model/TrainGradientModel.py (向量化)

Wave 2: 工具层重构
  ├─ Step 2.1: 创建 tool/geometry.py (Shapely 相交检测)
  └─ Step 2.2: 重构 tool/Data.py (ModelManager 单例)

Wave 3: Planner 核心拆分
  ├─ Step 3.1: 创建 multibeam/planner_utils.py (几何计算函数)
  ├─ Step 3.2: 创建 multibeam/planner_visualizer.py (可视化类)
  └─ Step 3.3: 重构 multibeam/Planner.py (核心调度)

Wave 4: 集成与验证
  ├─ Step 4.1: 修复所有导入关系
  ├─ Step 4.2: 运行完整测试
  └─ Step 4.3: 更新 README.md
```

---

## 2. 详细步骤

### Wave 1: 基础设施准备

#### Step 1.1: 添加 shapely 依赖
**所有权**: native  
**验证命令**: `python -c "import shapely; print(shapely.__version__)"`  
**回滚**: 从 requirements.txt 移除 shapely 行

```python
# 文件: requirements.txt
# 操作: 追加 shapely>=2.0.0
```

#### Step 1.2: 创建 multibeam/models.py
**所有权**: native  
**验证命令**: `python -c "from multibeam.models import LineRecord, PartitionResult, TerminationReason; print('OK')"`  
**回滚**: 删除文件

```python
# 文件: multibeam/models.py (新建)
# 内容:
# - from dataclasses import dataclass
# - from enum import Enum, auto
# - TerminationReason(Enum) with 7 states
# - LineRecord dataclass (move from Planner.py)
# - PartitionResult dataclass (move from Planner.py)
```

#### Step 1.3: 重构梯度计算
**所有权**: native  
**验证命令**: 
- `python model/TrainGradientModel.py` (成功生成模型)
- 手动比对新旧梯度矩阵差异 < 1e-10  
**回滚**: 恢复原始 `compute_normalized_gradient` 函数

```python
# 文件: model/TrainGradientModel.py
# 操作:
# 1. 删除 compute_normalized_gradient 函数 (60+ 行 for 循环)
# 2. 添加 compute_normalized_gradient_vectorized 函数 (使用 np.gradient)
# 3. 更新调用点
```

---

### Wave 2: 工具层重构

#### Step 2.1: 创建 tool/geometry.py
**所有权**: native  
**验证命令**: `python -c "from tool.geometry import check_line_intersection_shapely; print('OK')"`  
**回滚**: 删除文件

```python
# 文件: tool/geometry.py (新建)
# 内容:
# - from shapely.geometry import LineString
# - def check_line_intersection_shapely(new_pt_start, new_pt_end, prev_lines) -> bool
```

#### Step 2.2: 重构 tool/Data.py
**所有权**: native  
**验证命令**: 
- `python -c "import tool.Data; print('OK')"` (不应打印加载日志)
- `python -c "from tool.Data import get_height; print(get_height(100, 100))"` (首次调用才加载)  
**回滚**: 恢复原始模块级加载代码

```python
# 文件: tool/Data.py
# 操作:
# 1. 删除模块顶层的 height_rf, gx_rf, gy_rf = load_model()
# 2. 添加 ModelManager 类 (_height_rf, _gx_rf, _gy_rf 类属性)
# 3. 实现 get_models() @classmethod
# 4. 修改 get_height/get_gx/get_gy 使用 ModelManager.get_models()
```

---

### Wave 3: Planner 核心拆分

#### Step 3.1: 创建 multibeam/planner_utils.py
**所有权**: native  
**验证命令**: `python -c "from multibeam.planner_utils import _compute_signed_angle, _compute_total_turning_angle; print('OK')"`  
**回滚**: 删除文件

```python
# 文件: multibeam/planner_utils.py (新建)
# 内容:
# - _compute_signed_angle(v1, v2) -> float
# - _compute_total_turning_angle(line_arr) -> float
# - _point_to_segment_distance(...) -> float (保留，可能其他地方使用)
```

#### Step 3.2: 创建 multibeam/planner_visualizer.py
**所有权**: native  
**验证命令**: `python -c "from multibeam.planner_visualizer import SurveyVisualizer; print('OK')"`  
**回滚**: 删除文件

```python
# 文件: multibeam/planner_visualizer.py (新建)
# 内容:
# - class SurveyVisualizer:
#     - __init__(self, xs, ys, cluster_matrix, x_min, x_max, y_min, y_max)
#     - setup_figure(self, partition_id) -> fig, ax
#     - draw_line(self, line, color, label)
#     - draw_partition_background(self)
#     - save_snapshot(self, path)
#     - draw_merged_lines(self, all_results)
```

#### Step 3.3: 重构 multibeam/Planner.py
**所有权**: native  
**验证命令**: 
- `python main.py` (完整运行无报错)
- 检查输出目录结构完整性  
**回滚**: 恢复原始 Planner.py (需要备份)

```python
# 文件: multibeam/Planner.py
# 操作:
# 1. 删除 LineRecord, PartitionResult 定义 (移至 models.py)
# 2. 删除几何计算函数 (移至 planner_utils.py)
# 3. 删除绘图逻辑 (移至 planner_visualizer.py)
# 4. 添加导入: from multibeam.models import ...
# 5. 添加导入: from multibeam.planner_utils import ...
# 6. 添加导入: from multibeam.planner_visualizer import ...
# 7. 添加导入: from tool.geometry import check_line_intersection_shapely
# 8. 替换 _check_line_intersection 为 check_line_intersection_shapely
# 9. 实现 _evaluate_termination() 方法 (返回 TerminationReason)
# 10. 修改主循环使用 TerminationReason 枚举
# 11. 在 SurveyPlanner.__init__ 中初始化 SurveyVisualizer
# 12. 修改绘图调用使用 visualizer
```

---

### Wave 4: 集成与验证

#### Step 4.1: 修复导入关系
**所有权**: native  
**验证命令**: 
- `python -c "from multibeam.GridCell import *; from multibeam.Coverage import *; from multibeam.Partition import *; from multibeam.Planner import *; print('All imports OK')"`  
**回滚**: 逐个文件回滚

```python
# 可能需要修改的文件:
# - multibeam/Coverage.py: from tool import Data -> from tool.Data import get_height
# - multibeam/GridCell.py: 同上
# - multibeam/Partition.py: 无依赖变化
```

#### Step 4.2: 运行完整测试
**所有权**: native  
**验证命令**: 
- `python main.py` (成功运行)
- 检查 `output/<timestamp>/` 目录结构:
  - lines/<pid>/*.png 存在
  - metrics/metrics.xlsx 存在
  - partition/*.png 存在
- 手动检查测线图无穿越穿模  
**回滚**: 整体回滚到重构前状态

#### Step 4.3: 更新 README.md
**所有权**: native  
**验证命令**: 无  
**回滚**: 恢复原 README.md

```markdown
# 文件: README.md
# 操作:
# 1. 更新 "包结构" 部分，添加新文件
# 2. 更新 "关键实体/模型" 部分，添加 TerminationReason
# 3. 更新 "当前挑战" 部分，标记已解决的问题
# 4. 添加 "重构说明" 章节
```

---

## 3. 验证命令汇总

| Step | 验证命令 | 期望输出 |
|---|---|---|
| 1.1 | `python -c "import shapely; print(shapely.__version__)"` | 版本号 |
| 1.2 | `python -c "from multibeam.models import LineRecord, PartitionResult, TerminationReason; print('OK')"` | OK |
| 1.3 | `python model/TrainGradientModel.py` | 成功生成 .pkl |
| 2.1 | `python -c "from tool.geometry import check_line_intersection_shapely; print('OK')"` | OK |
| 2.2 | `python -c "import tool.Data; print('OK')"` | OK (无加载日志) |
| 3.1 | `python -c "from multibeam.planner_utils import _compute_signed_angle; print('OK')"` | OK |
| 3.2 | `python -c "from multibeam.planner_visualizer import SurveyVisualizer; print('OK')"` | OK |
| 4.1 | 全模块导入测试 | All imports OK |
| 4.2 | `python main.py` | 完整输出 |

---

## 4. 回滚规则

| 级别 | 触发条件 | 操作 |
|---|---|---|
| Step 级 | 单步验证失败 | 回滚该 Step 的文件变更 |
| Wave 级 | Wave 内多个 Step 失败 | 回滚整个 Wave，保留已成功的 Wave |
| 全局级 | 集成测试失败 | 恢复所有原始文件 |

---

## 5. 清理期望

重构完成后需要清理：
- [ ] 删除旧的 `compute_normalized_gradient` 函数（已替换）
- [ ] 删除 Planner.py 中的冗余辅助函数（已移动）
- [ ] 更新 `.gitignore`（如有新输出目录）
- [ ] 清理临时备份文件（如有）

---

## 6. 交付接受计划

1. **代码审查**: 检查所有新建/修改文件符合 Python 代码规范
2. **功能验证**: 运行 `python main.py`，输出与重构前一致
3. **性能验证**: 梯度计算速度提升 10x+
4. **文档更新**: README.md 反映新架构

---

**创建时间**: 2026-04-06 21:00  
**批准状态**: Pending User Approval
