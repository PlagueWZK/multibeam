# 环形测线死循环修复方案

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复螺旋测线重合问题和饱和终止失效问题

**Architecture:** 
1. 用几何自交检测（is_simple + 端点接近）替代累计偏转角检测
2. 在螺旋终止后，用测线质心到自身和父测线的平均距离比较来判断收敛状态

**Tech Stack:** Python, NumPy, Shapely

---

## 问题诊断

### 问题1：累计偏转角检测不准确
- **当前**：`compute_total_turning_angle(line) > 2π` 判定螺旋
- **缺陷**：测线可能在偏转角未达360°时已自交，或达360°时已绕多圈
- **根因**：用角度间接推断，而非直接检测几何自交

### 问题2：收敛状态判断依赖测线长度
- **当前**：`current_length < parent_line_length` 判断是否进入收敛
- **缺陷**：测线长度受分区边界影响，不完全反映收缩趋势
- **根因**：长度不是衡量收敛的可靠指标

---

## 实施任务

### Task 1: 添加几何自交检测函数

**Files:**
- Modify: `tool/geometry.py` — 添加 `check_self_intersection` 函数

**Step 1: 在 geometry.py 末尾添加自交检测函数**

```python
def check_self_intersection(line: np.ndarray, min_points: int = 10, proximity_threshold: float = 15.0) -> bool:
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
```

**Step 2: 更新 geometry.py 的 import**

在文件顶部添加 `Point` 导入：
```python
from shapely.geometry import LineString, Point
```

**Step 3: 提交**

```bash
git add tool/geometry.py
git commit -m "feat: 添加测线自交检测函数（is_simple + 端点接近）"
```

---

### Task 2: 替换螺旋检测 + 重构收敛状态判断

**Files:**
- Modify: `multibeam/Planner.py` — 多处修改

**Step 1: 在 SurveyPlanner 类中添加收敛判断方法**

在 `_evaluate_termination` 方法之前（约第 98 行后）添加：

```python
def _is_converging(self, line: np.ndarray, parent_line: np.ndarray) -> bool:
    """
    判断测线是否在收敛（收缩）
    
    算法：计算当前测线质心，比较质心到自身和到父测线的平均距离。
    若前者小，说明测线在收缩，视为收敛状态。
    
    参数:
        line: 当前测线 [N, 3] 或 [N, 2]
        parent_line: 父测线 [M, 3] 或 [M, 2]
    
    返回:
        bool: True 表示收敛
    """
    centroid = np.mean(line[:, :2], axis=0)
    
    # 当前测线到自身质心的平均距离
    dist_current = np.mean(np.sqrt(
        (line[:, 0] - centroid[0]) ** 2 + (line[:, 1] - centroid[1]) ** 2
    ))
    
    # 父测线到当前测线质心的平均距离
    dist_parent = np.mean(np.sqrt(
        (parent_line[:, 0] - centroid[0]) ** 2 + (parent_line[:, 1] - centroid[1]) ** 2
    ))
    
    return dist_current < dist_parent
```

**Step 2: 修改 `_evaluate_termination` 方法（第 121-186 行）**

将第一个检查（偏转角）替换为自交检测：

```python
def _evaluate_termination(
    self,
    line: np.ndarray,
    new_point: np.ndarray,
    prev_lines: list[np.ndarray],
    in_convergence_state: bool,
    parent_line_length: float,
) -> TerminationReason:
    """
    评估测线终止原因

    参数:
        line: 当前测线（包含新点）
        new_point: 最新添加的点
        prev_lines: 历史测线列表
        in_convergence_state: 是否处于收敛状态
        parent_line_length: 父测线长度

    返回:
        TerminationReason: 终止原因枚举
    """
    from tool.geometry import check_self_intersection

    # 1. 检查测线自交（螺旋终止）
    if check_self_intersection(line, min_points=10):
        return TerminationReason.SPIRAL

    # 2. 检查测线相交（迷路终止）
    if len(line) >= 2 and prev_lines:
        if check_line_intersection_shapely(line[-2], line[-1], prev_lines):
            return TerminationReason.INTERSECTION

    # 3. 检查测线退化（点数过少）
    if len(line) < 5:
        return TerminationReason.DEGRADATION

    # 4. 检查测线饱和（收敛状态下）
    if in_convergence_state:
        current_length = figure_length(line)
        if current_length < parent_line_length:
            # 计算质心到最近点的距离
            centroid = np.mean(line, axis=0)
            dists = np.sqrt(
                (line[:, 0] - centroid[0]) ** 2 + (line[:, 1] - centroid[1]) ** 2
            )
            min_dist = np.min(dists)
            nearest_idx = np.argmin(dists)
            nearest_point = line[nearest_idx]

            # 计算内侧覆盖宽度
            gx_val = get_gx(nearest_point[0], nearest_point[1])
            gy_val = get_gy(nearest_point[0], nearest_point[1])
            to_centroid = centroid[:2] - nearest_point[:2]
            grad_dir = np.array([gx_val, gy_val])

            dot_product = np.dot(grad_dir, to_centroid)
            inner_width = (
                get_w_right(nearest_point[0], nearest_point[1], self.theta)
                if dot_product > 0
                else get_w_left(nearest_point[0], nearest_point[1], self.theta)
            )

            if min_dist < inner_width:
                return TerminationReason.SATURATION

    return TerminationReason.NONE
```

**Step 3: 更新主测线延伸中的打印信息（第 255-258 行）**

```python
if reason == TerminationReason.SPIRAL:
    print(f"  [主测线延伸] 检测到自交，螺旋终止")
    terminated_reason = reason
    break
```

**Step 4: 更新扩展测线延伸中的打印信息（第 426-428 行）**

```python
if reason == TerminationReason.SPIRAL:
    print(f"  [测线延伸] 检测到自交，螺旋终止")
    terminated_reason = reason
    break
```

**Step 5: 重构 `_generate_perpendicular_lines` 中的收敛判断（第 460-483 行）**

将原来的绘制和保存 + 饱和检查部分替换为：

```python
            # 绘制和保存
            if terminated_reason == TerminationReason.SPIRAL:
                self.visualizer.draw_line(line, "red", 1.5)
                # 使用质心距离比较判断收敛状态
                if self._is_converging(line, main_line):
                    in_convergence_state = True
                    print(f"  [向内螺旋] 进入收敛状态，后续将进行饱和检查")
                else:
                    print(f"  [向外螺旋] 继续正常扩展")
            elif terminated_reason == TerminationReason.INTERSECTION:
                self.visualizer.draw_line(line, "purple", 1.5)

            line_counter += 1
            snap_path = lines_dir / f"line_{line_counter:04d}_iter{perp_iter}.png"
            self.visualizer.save_snapshot(snap_path)
            print(f"  >> 已保存测线: {snap_path}")

            parent_line_length = figure_length(line)

            # 检查饱和终止
            if in_convergence_state:
                reason = self._evaluate_termination(
                    line, line[0], prev_lines, True, parent_line_length
                )
                if reason == TerminationReason.SATURATION:
                    print(f"[{direction_name}扩展] 测线饱和，终止该方向扩展")
                    break
```

**Step 6: 提交**

```bash
git add multibeam/Planner.py
git commit -m "refactor: 用自交检测替代偏转角，用质心距离判断收敛状态"
```

---

## 验收标准

- [ ] 程序在合理时间内完成（原问题：死循环）
- [ ] 红色测线后不再有大量重合测线
- [ ] 控制台输出显示"向内螺旋"或"向外螺旋"的判断结果
- [ ] 饱和终止成功触发（控制台有输出）
- [ ] 测线条数合理（< 100 条/分区）

---

## 风险与回退

**风险**：
1. 自交检测的 `proximity_threshold=15m` 可能不适用于所有场景
   - **缓解**：可根据实际测线间距调整该参数
2. `_is_converging` 在螺旋初期可能误判
   - **缓解**：只在螺旋终止后才调用，此时测线形状已稳定

**回退方案**：
- 保留原有偏转角检测代码（注释掉），必要时可切换回
