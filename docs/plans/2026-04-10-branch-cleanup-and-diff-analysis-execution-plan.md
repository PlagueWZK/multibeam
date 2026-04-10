# 分支清理与差异分析执行计划

**计划ID**: PLAN-2026-04-10-005  
**关联需求**: REQ-2026-04-10-005  
**创建时间**: 2026-04-10  
**内部执行等级**: M

---

## 1. 执行策略

### Wave 1: skeleton_check
- 确认当前分支与工作树状态
- 确认 docs / runtime 证据目录可用

### Wave 2: 分支检查与删除
- 检查本地与远程是否存在 `list` / `lsit`
- 在不影响当前分支的前提下删除用户要求的分支

### Wave 3: 差异分析
- 检查 `main` / `feature` / `wzk` 的最近提交与共同祖先关系
- 检查各分支独有提交与主要 diff 范围

### Wave 4: 交付
- 汇总删除结果
- 汇总三分支差异

---

## 2. Verification Plan (验证计划)

- 通过 `git branch --list` / `git branch -r --list` 确认分支存在性
- 通过删除后再次列出分支确认删除结果
- 通过 `git log --oneline --left-right --cherry-pick` 和 `git diff --stat` 分析差异

---

## 3. Completion Language Rules (完成措辞规则)

- 允许：说明哪些分支被删、哪些改动在哪个分支
- 禁止：把未提交工作树变化混同为分支历史的一部分
