# vibe 技能有效性检查执行计划

**计划ID**: PLAN-2026-04-09-002  
**关联需求**: REQ-2026-04-09-002  
**创建时间**: 2026-04-09  
**内部执行等级**: M (Single-Agent Validation)

---

## 1. 执行策略

### 1.1 Internal Grade Decision

- **等级**: M
- **原因**:
  - 单一技能检查
  - 不涉及代码实现或并行任务
  - 主要由静态检查与一次最小校验脚本组成

### 1.2 Wave Structure

```
Wave 1: skeleton_check
  ├─ 确认仓库治理目录可用
  └─ 确认 OpenCode 下 vibe 技能安装目录存在

Wave 2: 证据收集
  ├─ 读取 SKILL.md / README.md / check.ps1
  └─ 确认 HostId 支持 opencode

Wave 3: plan_execute
  ├─ 运行最小非破坏性校验
  └─ 记录 PASS / FAIL / WARN 结果

Wave 4: phase_cleanup
  └─ 留存 runtime receipts，无额外临时文件
```

---

## 2. Ownership Boundaries (所有权边界)

- **root_governed**: 当前对话主执行体
- **child_governed**: 不启用
- **写入范围**:
  - `docs/requirements/2026-04-09-vibe-skill-validation.md`
  - `docs/plans/2026-04-09-vibe-skill-validation-execution-plan.md`
  - `outputs/runtime/vibe-sessions/20260409-200200-vibe-skill-validation/`

---

## 3. Verification Commands (验证命令)

| 阶段 | 命令/方式 | 目的 |
|---|---|---|
| skeleton_check | `git status -sb` | 记录当前仓库治理上下文 |
| static check | 读取 `SKILL.md` / `check.ps1` | 确认技能元数据与脚本入口 |
| runtime check | `./check.ps1 -HostId opencode -Profile minimal` | 验证安装最小有效性 |

---

## 4. Delivery Acceptance Plan (交付验收计划)

最终输出必须包含：

1. 安装位置
2. 关键文件完整性
3. 最小校验命令是否执行成功
4. PASS/FAIL/WARN 概览
5. 最终判断与后续建议

---

## 5. Completion Language Rules (完成措辞规则)

- 若脚本通过：可说“当前安装基本有效”或“最小校验通过”
- 若脚本失败：必须明确失败项，不能只说“无效”
- 不得把最小校验通过夸大为“所有能力都验证通过”

---

## 6. Rollback Rules (回滚规则)

- 本任务不修改 OpenCode 配置，无需回滚
- 若治理文件路径命名错误，仅删除本次新增治理文件即可

---

## 7. Phase Cleanup Expectations (阶段清理要求)

- 不产生额外临时脚本
- 不保留无关输出
- 保留 requirement / plan / runtime receipts 作为证据
