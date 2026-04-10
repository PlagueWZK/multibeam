# 漏测率负数问题分析执行计划

**计划ID**: PLAN-2026-04-09-003  
**关联需求**: REQ-2026-04-09-003  
**创建时间**: 2026-04-09  
**内部执行等级**: M

---

## 1. 执行策略

### Wave 1: skeleton_check
- 确认仓库与治理目录状态

### Wave 2: 代码证据收集
- 检查 `tool/Data.py` 中覆盖面积代理公式
- 检查 `multibeam/Planner.py` 中全局统计与漏测率公式

### Wave 3: 分析与交付
- 归纳统计口径缺陷
- 提出替代统计方案
- 给出推荐实现路径

---

## 2. Verification Plan (验证计划)

- 静态核对 `figure_width()` 与 `save_metrics_excel()`
- 验证覆盖率使用 `effective_coverage / total_area`
- 验证漏测率使用 `1 - coverage_rate`

---

## 3. Completion Language Rules (完成措辞规则)

- 允许：问题根因分析、修复建议、推荐方案
- 禁止：宣称已修复或已验证无回归
