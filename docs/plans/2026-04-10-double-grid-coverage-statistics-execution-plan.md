# 双网格覆盖统计与规划状态执行计划

**计划ID**: PLAN-2026-04-10-001  
**关联需求**: REQ-2026-04-10-001  
**创建时间**: 2026-04-10  
**内部执行等级**: L (Serial Native Execution)

---

## 1. 执行策略

### 1.1 等级决策

- **等级**: L
- **原因**:
  - 该方案涉及 `Planner.py`、统计模块、参数传递链与新细网格模块
  - 逻辑依赖明确，适合串行推进
  - 需要先冻结口径，再逐步替换统计实现

### 1.2 Wave 结构

```
Wave 1: 参数与状态面冻结
  ├─ Step 1.1 冻结细网格参数面（c_l=1.5, c_step=1.0, microstep=[10,70], 9-point）
  ├─ Step 1.2 定义细网格数据结构与状态枚举
  └─ Step 1.3 明确分区级 w_min_partition 计算口径

Wave 2: 细网格模块实现
  ├─ Step 2.1 新建分区细网格生成模块
  ├─ Step 2.2 新建 cell 采样覆盖判定与部分覆盖比例计算
  └─ Step 2.3 提供分区面积/覆盖面积/漏测面积统计接口

Wave 3: Planner 联动改造
  ├─ Step 3.1 在分区规划启动时初始化细网格
  ├─ Step 3.2 每新增测线段后更新细网格状态
  ├─ Step 3.3 将 step 改为 c_step * l 的参数化形式
  └─ Step 3.4 将 get_alpha 使用的 microstep 改为 clip(l/2, min, max)

Wave 4: 指标模块替换
  ├─ Step 4.1 用细网格统计替换当前 figure_width 累加口径
  ├─ Step 4.2 分区统计按真实覆盖面积输出
  ├─ Step 4.3 全局统计做二次汇总与边界裁剪
  └─ Step 4.4 确保 coverage_rate / miss_rate 落在 [0,1]

Wave 5: 验证与回归检查
  ├─ Step 5.1 高重叠案例检查
  ├─ Step 5.2 相交案例检查
  ├─ Step 5.3 参数灵敏度检查（c_l, c_step）
  └─ Step 5.4 输出指标与可视化核对
```

---

## 2. 设计落地细节

### 2.1 粗网格与细网格职责划分

| 层级 | 用途 | 保留/新增 |
|---|---|---|
| 粗网格 | 覆盖次数矩阵、KMeans 分区、全局前处理 | 保留 |
| 细网格 | 分区内部覆盖状态、去重面积统计、规划过程状态更新 | 新增 |

### 2.2 `w_min_partition` 计算口径

实现建议：

1. 使用当前分区在粗网格上的有效点集
2. 对这些粗网格点计算深度
3. 找到**深度最小的点**（最浅点）
4. 以该点的 `w_total = get_w_left + get_w_right` 作为 `w_min_partition`
5. 若分区样本过少或存在异常值，可在实现期加入保护与预警，但不改变此冻结口径

### 2.3 细网格边长与参数化

- 公式：`l = 1.5 * w_min_partition`
- 保护策略：若预估细网格 cell 数超过 `100000 / partition`，则自动增大 `l` 直到落入上限内
- 实现要求：保留 `c_l` 为参数字段，但首版默认值固定为 `1.5`

### 2.4 cell 覆盖状态模型

建议不要只存三态枚举，而应同时保存：

- `coverage_ratio` ∈ [0, 1]
- `state` = uncovered / partial / full

推荐映射：

- `coverage_ratio = 0` -> `uncovered`
- `0 < coverage_ratio < 0.9` -> `partial`
- `coverage_ratio >= 0.9` -> `full`

### 2.5 覆盖判定方式

冻结选择：**采样判定**。

实现建议：

1. 对每个受影响 cell 进行固定 9 点采样（3×3）
2. 对每个采样点判断其是否落入某测线段对应的覆盖走廊
3. `coverage_ratio = 覆盖采样点数 / 总采样点数`

### 2.6 规划过程中的状态更新粒度

冻结建议：**按新增测线段更新，而不是按单点更新**。

理由：

- 单点只代表离散采样位置，不能表达相邻点之间的连续扫幅
- 按线段更新更接近真实测线覆盖走廊

实现建议：

1. 当 `line` 新增点 `p_{i+1}` 时，取新线段 `(p_i, p_{i+1})`
2. 用端点局部宽度构造该线段的近似覆盖走廊
3. 对其包围盒内的细网格 cell 执行采样判定
4. 更新 `coverage_ratio` 和状态

### 2.7 `step` 与 `microstep` 联动

- `step = 1.0 * l`
- `microstep = clip(l/2, 10, 70)`

实现约束：

- `c_step` 以参数字段实现，首版默认值为 `1.0`
- `microstep_min/max` 以参数字段实现，首版默认值为 `[10, 70]`
- 保持对现有 `get_alpha()` 调用方的兼容性，可通过包装或显式传参实现

### 2.8 统计口径替换

现有错误口径：

- `sum(line_length * width)`

新口径：

- 分区真实覆盖面积 = `sum(cell_area * coverage_ratio)`
- 分区漏测面积 = `sum(cell_area * (1 - coverage_ratio))` 或基于状态定义的未覆盖面积
- 全局覆盖面积 = 全局统一去重后的分区汇总面积

说明：

- 若分区互斥，则全局面积可由分区细网格面积直接汇总
- 覆盖率 = `covered_area / total_area`
- 漏测率 = `1 - coverage_rate`
- 需在输出时强制裁剪到 `[0,1]` 作为保护线

---

## 3. 所有权边界

- **root_governed**: 当前主执行体
- **child_governed**: 不启用
- **目标文件范围（后续实现期）**:
  - 新增：`multibeam/coverage_state_grid.py`（建议名，可调整）
  - 修改：`multibeam/Planner.py`
  - 修改：`tool/Data.py`（或仅调用参数层）
  - 视实现需要修改：`multibeam/models.py`、`README.md`

---

## 4. Verification Commands / Methods (验证方式)

### 4.1 静态验证
- 检查 `Planner.py` 不再以 `sum(r.coverage)` 作为全局真实覆盖面积口径
- 检查细网格状态结构存在且有 `coverage_ratio`

### 4.2 行为验证
- 构造高重叠测线：覆盖率不应超过 100%
- 构造相交测线：覆盖率不应超过 100%
- 构造正常覆盖案例：覆盖率随测线增多单调上升，但逐渐饱和

### 4.3 数值验证
- `0 <= coverage_rate <= 1`
- `0 <= miss_rate <= 1`
- `covered_area <= total_area`

---

## 5. Delivery Acceptance Plan (交付验收计划)

后续实现完成时，至少需输出：

1. 分区细网格统计结果
2. 全局细网格统计结果
3. `metrics.xlsx` 中新的覆盖面积/覆盖率/漏测率
4. 至少两类问题案例的验证记录：
   - 高重叠
   - 相交

---

## 6. Completion Language Rules (完成措辞规则)

- 当前允许：宣称“规划方案已冻结”
- 当前禁止：宣称“bug 已修复”
- 只有完成代码实现与验证后，才能宣称统计模块替换成功

---

## 7. Rollback Rules (回滚规则)

若后续实现阶段出现问题：

1. 保留当前粗网格与原规划主线不动
2. 若联动过强，可先保留新统计模块并暂缓 `step` / `microstep` 切换
3. 若新统计模块不稳定，可先并行输出旧指标与新指标做对照

---

## 8. Phase Cleanup Expectations (阶段清理要求)

- 规划阶段不产生业务代码临时改动
- 参数试验脚本若临时创建，必须在阶段结束时清理或转为正式验证脚本
- 输出应保留需求文档、计划文档与 runtime receipts

---

## 9. 当前仍开放但已受控的参数面

以下点已经冻结为首版默认值，但仍须实现为参数字段以便后续调优：

- `c_l = 1.5`
- `max_cells_per_partition = 100000`
- `c_step = 1.0`
- `microstep_min/max = [10, 70]`
- 采样模式 = 9-point
- `partial/full` 阈值 = `0+ / 0.9`

后续实现只允许在“参数调优”层面变更这些值，不得改变本计划冻结的结构。

---

**计划状态**: Frozen For Implementation Planning
