# Multibeam 重构需求文档

**文档ID**: REQ-2026-04-06-001  
**创建时间**: 2026-04-06  
**状态**: Frozen  

---

## 1. Goal (目标)

对 `multibeam` 项目进行系统性重构，解决以下核心问题：
1. **测线相交检测不准确** - 替换为 Shapely 精确几何检测
2. **模型加载耦合** - 实现懒加载单例模式
3. **Planner.py 膨胀** - 按 SRP 拆分为多个模块
4. **终止条件状态机复杂** - 引入枚举和评估器
5. **梯度计算低效** - 向量化实现

---

## 2. Deliverable (交付物)

### 2.1 代码交付物

| 文件/模块 | 变更类型 | 说明 |
|---|---|---|
| `requirements.txt` | 修改 | 添加 `shapely` 依赖 |
| `tool/Data.py` | 重构 | 引入 `ModelManager` 单例类 |
| `tool/geometry.py` | 新增 | 几何计算工具（Shapely 相交检测） |
| `multibeam/models.py` | 新增 | 数据类定义（`LineRecord`, `PartitionResult`, `TerminationReason`） |
| `multibeam/planner_utils.py` | 新增 | 几何计算函数 |
| `multibeam/planner_visualizer.py` | 新增 | 可视化逻辑 |
| `multibeam/Planner.py` | 重构 | 精简为核心调度逻辑 |
| `model/TrainGradientModel.py` | 重构 | 向量化梯度计算 |

### 2.2 文档交付物

- 更新后的 `README.md`（反映新架构）
- 重构验证报告

---

## 3. Constraints (约束条件)

### 3.1 功能约束
- **向后兼容**: 保持 `main.py` 入口不变，外部调用 API 兼容
- **模型兼容**: 已训练的 `.pkl` 模型文件继续使用，无需重新训练
- **输出格式**: 生成的测线图、dot.csv、metrics.xlsx 格式保持不变

### 3.2 技术约束
- Python >= 3.12
- 新增依赖仅限 `shapely`
- 不引入额外 Web 框架或数据库

### 3.3 依赖约束
- `tool/Data.py` 模型加载重构后，`model/TrainHeightModel.py` 和 `model/TrainGradientModel.py` 仍可独立运行
- `Planner.py` 拆分后，子模块间依赖清晰，避免循环导入

---

## 4. Acceptance Criteria (验收标准)

### 4.1 Shapely 相交检测
- [ ] `tool/geometry.py` 中实现 `check_line_intersection_shapely()` 函数
- [ ] 测线延伸时检测新线段与历史测线的相交
- [ ] 单元测试：覆盖以下场景
  - 完全不相交的线段
  - 端点相交
  - 中间交叉
  - 共线重叠

### 4.2 模型懒加载
- [ ] `tool/Data.py` 中实现 `ModelManager` 单例类
- [ ] `import tool.Data` 不触发模型加载
- [ ] 首次调用 `get_height()` 时才加载模型
- [ ] 后续调用复用已缓存模型
- [ ] `model/TrainHeightModel.py` 可独立运行不报错

### 4.3 Planner 拆分
- [ ] `multibeam/models.py` 定义 `LineRecord`, `PartitionResult`, `TerminationReason`
- [ ] `multibeam/planner_utils.py` 包含 `_compute_signed_angle`, `_compute_total_turning_angle`
- [ ] `multibeam/planner_visualizer.py` 包含 `SurveyVisualizer` 类
- [ ] `multibeam/Planner.py` 行数 < 400
- [ ] 运行 `python main.py` 生成与重构前一致的输出

### 4.4 终止条件状态机
- [ ] `TerminationReason` 枚举定义 7 种状态
- [ ] `_evaluate_termination()` 独立方法返回枚举值
- [ ] 主循环使用 `match-case` 或清晰 `if` 处理终止
- [ ] `LineRecord.terminated_by` 正确记录终止原因

### 4.5 梯度计算向量化
- [ ] `compute_normalized_gradient_vectorized()` 使用 `np.gradient()`
- [ ] 删除原有双重 `for` 循环
- [ ] 运行 `python model/TrainGradientModel.py` 成功生成模型
- [ ] 新旧方法输出结果一致（相对误差 < 1e-10）

### 4.6 集成测试
- [ ] 完整运行 `python main.py` 无报错
- [ ] 输出目录结构正确：`output/<timestamp>/lines/`, `metrics/`, `partition/`
- [ ] 所有测线图中无穿越穿模现象

---

## 5. Product Acceptance Criteria (产品验收标准)

- 功能正确性：重构后输出与重构前完全一致（测线坐标、覆盖面积、分区边界）
- 性能：梯度计算速度提升 10x+（向量化效果）
- 可维护性：Planner.py 可读性提升，各模块职责清晰

---

## 6. Manual Spot Checks (手动验证点)

1. **相交检测可视化**: 检查生成的测线图，确认无交叉穿越
2. **模型加载时机**: 在 `get_height()` 前后打印日志，确认首次调用才加载
3. **终止原因统计**: 查看 `metrics.xlsx` 中各终止原因的分布

---

## 7. Non-goals (非目标)

- 不改变测线规划的核心算法逻辑
- 不重新训练模型或调整模型参数
- 不修改输入数据格式 (`data.xlsx`)
- 不优化 `Coverage.py` 的性能瓶颈（不在本次范围）

---

## 8. Autonomy Mode (自主模式)

- **Mode**: `interactive_governed`
- **Internal Grade**: `L` (Serial native execution)
- **理由**: 重构涉及多个模块，有明确的依赖关系，适合串行执行

---

## 9. Inferred Assumptions (推断假设)

1. 用户已阅读 `README.md` 并理解当前架构
2. 用户希望保持向后兼容，不破坏现有工作流
3. 用户接受引入 `shapely` 作为新依赖
4. 重构优先级：正确性 > 性能 > 可读性

---

## 10. Execution Phases (执行阶段)

```
Phase 1: 基础设施层
  ├─ 1.1 添加 shapely 依赖
  ├─ 1.2 创建 multibeam/models.py (数据类 + 枚举)
  └─ 1.3 梯度计算向量化

Phase 2: 工具层重构
  ├─ 2.1 创建 tool/geometry.py (Shapely 相交检测)
  └─ 2.2 重构 tool/Data.py (ModelManager 单例)

Phase 3: Planner 拆分
  ├─ 3.1 创建 multibeam/planner_utils.py
  ├─ 3.2 创建 multibeam/planner_visualizer.py
  └─ 3.3 精简 multibeam/Planner.py

Phase 4: 集成与验证
  ├─ 4.1 修复导入关系
  ├─ 4.2 运行集成测试
  └─ 4.3 更新 README.md
```

---

**冻结时间**: 2026-04-06 20:50  
**批准状态**: Pending User Approval
