# 分支清理与差异分析需求文档

**文档ID**: REQ-2026-04-10-005  
**创建时间**: 2026-04-10  
**状态**: Frozen

---

## 1. Goal (目标)

完成两项 git 相关任务：

1. 删除用户指定的 `list` 与 `lsit` 分支
2. 分析 `main`、`feature`、`wzk` 三个分支之间的差异

---

## 2. Deliverable (交付物)

- 删除结果说明：
  - 分支是否存在
  - 删除的是本地分支还是远程分支（若有）
  - 删除是否成功
- 一份中文差异总结，覆盖：
  - 三个分支各自相对位置
  - 哪些提交/改动只存在于某分支
  - 哪个分支明显更靠前或包含更多新工作

---

## 3. Constraints (约束条件)

- 不做 `push --force`、`reset --hard` 等破坏性操作
- 默认只删除明确存在且用户点名的分支
- 差异分析以 git 历史和 diff 为依据，不臆测未提交内容
- 不创建 commit

---

## 4. Acceptance Criteria (验收标准)

- [x] 检查 `list` / `lsit` 是否存在
- [x] 安全删除用户要求的分支（如存在）
- [x] 检查 `main` / `feature` / `wzk` 的相对差异
- [x] 输出明确、可读的中文总结

---

## 5. Delivery Truth Contract (交付真实性约定)

- 结论基于 git branch / log / diff 结果
- 不声称已处理未明确要求的远程分支，除非实际执行并说明

---

## 6. Non-goals (非目标)

- 不合并分支
- 不推送远程
- 不改写历史

---

## 7. Autonomy Mode (自主模式)

- **Mode**: `interactive_governed`
- **Internal Grade**: `M`

---

**冻结时间**: 2026-04-10 23:04  
**批准状态**: Approved By User Instruction
