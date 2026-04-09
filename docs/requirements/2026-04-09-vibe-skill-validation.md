# vibe 技能有效性检查需求文档

**文档ID**: REQ-2026-04-09-002  
**创建时间**: 2026-04-09  
**状态**: Frozen

---

## 1. Goal (目标)

检查当前安装在 OpenCode 环境中的 `vibe` 技能是否处于“有效可用”状态。

这里的“有效”定义为：

1. 技能目录存在且结构完整
2. `SKILL.md` 元数据可识别为 `vibe`
3. 官方校验脚本存在并可针对 `opencode` 主机运行
4. 若校验脚本返回失败，能够指出失败点；若通过，则可据此认为安装基本有效

---

## 2. Deliverable (交付物)

- 一份中文检查结论，说明：
  - 技能是否已安装
  - 关键文件是否齐全
  - 校验脚本是否可运行
  - 当前结论是“有效 / 基本有效 / 存在问题”中的哪一种
  - 若有问题，具体卡在哪一步

---

## 3. Constraints (约束条件)

- 尽量使用技能自带的非破坏性检查方式
- 不修改用户的 OpenCode 全局配置
- 不重新安装技能，除非用户后续明确要求
- 结论必须区分“文件存在”与“功能校验通过”

---

## 4. Acceptance Criteria (验收标准)

- [x] 确认 `vibe` 技能目录存在
- [x] 确认 `SKILL.md` 中的技能名为 `vibe`
- [x] 确认存在 `check.ps1`
- [x] 尝试以 `opencode` host 执行非破坏性检查
- [x] 输出明确结论与证据

---

## 5. Product Acceptance Criteria (产品验收标准)

- 用户能够知道：当前 OpenCode 环境里的 `vibe` 是否可被视为“安装有效”
- 若无效，用户能够知道下一步该修什么，而不是只看到模糊失败

---

## 6. Manual Spot Checks (手动验证点)

1. 检查技能目录是否位于 `C:\Users\王政凯\.config\opencode\skills\vibe`
2. 检查 `SKILL.md` frontmatter 是否声明 `name: vibe`
3. 检查 `check.ps1` 是否支持 `-HostId opencode`
4. 运行最小校验并查看 PASS/FAIL/WARN

---

## 7. Completion Language Policy (完成措辞策略)

- 允许：
  - “已安装且最小校验通过”
  - “文件存在，但校验失败于某步骤”
  - “从当前证据看基本有效”
- 禁止：
  - “100% 完全没问题”
  - “所有运行时能力都已验证”

---

## 8. Delivery Truth Contract (交付真实性约定)

- 以静态文件检查 + 技能自带校验脚本结果为依据
- 不做重新安装
- 不修改 OpenCode 配置
- 不验证所有高级工作流，仅验证安装有效性

---

## 9. Non-goals (非目标)

- 不修复安装
- 不升级技能版本
- 不检查其他技能

---

## 10. Autonomy Mode (自主模式)

- **Mode**: `interactive_governed`
- **Internal Grade**: `M`
- **理由**: 单技能安装检查，范围明确，适合单代理只读/轻执行验证

---

## 11. Inferred Assumptions (推断假设)

1. 用户关心的是 OpenCode 主机上的 `vibe` 是否可用，而不是 Codex/Claude Code 的安装状态
2. 用户更需要“当前是否有效 + 证据”，而不是生态仓库完整介绍

---

**冻结时间**: 2026-04-09 20:02  
**批准状态**: Implicitly Approved For Non-destructive Validation
