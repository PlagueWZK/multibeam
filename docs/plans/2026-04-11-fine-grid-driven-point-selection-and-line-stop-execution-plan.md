# 细网格驱动的点取舍与测线停止执行计划

**计划ID**: PLAN-2026-04-11-001  
**关联需求**: REQ-2026-04-11-001  
**创建时间**: 2026-04-11  
**内部执行等级**: L

---

## 1. Internal Grade Decision (内部等级决策)

- **等级**: L
- **原因**:
  - 主要涉及 `Planner.py` 与 `coverage_state_grid.py` 两个强耦合文件
  - 逻辑需要串行冻结：先定义收益判断接口，再替换主/从测线停止逻辑
  - 不需要 XL 多代理拆分

---

## 2. Wave Structure (波次结构)

### Wave 1: 细网格收益判断面补齐
- Step 1.1 为细网格补充“候选点/候选线段是否还能新增覆盖”的查询接口
- Step 1.2 统一低收益判定输出，避免规划器自行重复计算细网格状态

### Wave 2: 主测线逻辑切换
- Step 2.1 移除主测线中的环形终止依赖
- Step 2.2 改为边界 + 细网格低收益双端停止

### Wave 3: 从测线逻辑切换
- Step 3.1 平移生成候选从测线时先按边界和细网格收益筛点
- Step 3.2 自延伸阶段改为边界 + 细网格低收益双端停止
- Step 3.3 删除收敛状态/饱和算法相关分支与日志

### Wave 4: 验证与清理
- Step 4.1 运行静态编译验证
- Step 4.2 运行一次主程序级验证，记录输出目录
- Step 4.3 写入 phase/cleanup receipt

---

## 3. Ownership Boundaries (所有权边界)

- 修改：`multibeam/coverage_state_grid.py`
- 修改：`multibeam/Planner.py`
- 视兼容性需要调整：`multibeam/models.py`
- 不改：分区流程、统计口径、主入口参数面

---

## 4. Verification Commands (验证命令)

1. `python -m compileall multibeam main.py tool`
2. `python main.py`

---

## 5. Delivery Acceptance Plan (交付验收计划)

- 代码层面确认旧终止逻辑已退出主流程
- 运行层面确认主流程可完成一次输出
- 日志/结果层面确认不再以 `spiral` / `saturation` 作为主要停止判定

---

## 6. Completion Language Rules (完成措辞规则)

- 允许：已实现细网格驱动点取舍与测线停止，并完成有限验证
- 禁止：在未运行验证前宣称算法已完全证明优于旧方案

---

## 7. Rollback Rules (回滚规则)

- 若细网格收益判断导致全部候选点被过度拒绝，可回退为“只要能新增未覆盖采样点即保留”的最宽松规则
- 若运行级验证暴露严重兼容问题，可先保留接口层改造，回退 Planner 中部分停止策略

---

## 8. Phase Cleanup Expectations (阶段清理要求)

- 不新增临时脚本
- 保留本次 requirement / plan / runtime receipts
- 仅保留验证产生的正式输出目录，不保留额外临时文件
