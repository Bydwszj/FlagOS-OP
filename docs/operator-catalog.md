# 算子目录维护指南

这份指南说明 FlagOS-OP 如何维护持续增长的算子 / 平台矩阵，并避免把
`ops/README.md` 变成难以维护的手工宽表。

## 事实来源

事实来源：

```text
ops/catalog.yaml
```

生成文件：

```text
ops/README.md
```

修改 catalog 后重新生成：

```bash
python3 scripts/build_ops_index.py
```

检查生成文件是否最新：

```bash
python3 scripts/build_ops_index.py --check
```

不要手工编辑 `ops/README.md`。文件内包含 AUTO-GENERATED 标记；生成结果
直接入库，是为了让 GitHub 页面保持可读。

## 目标范围

catalog 当前建模六个厂商平台家族：

| Family key | 显示名 | 当前状态 |
|---|---|---|
| `ascend` | 华为昇腾 | partial |
| `cambricon` | 寒武纪 | partial |
| `kunlunxin` | 昆仑芯 | partial |
| `thead` | 平头哥 | partial |
| `enflame` | 燧原 GCU | planned；已知型号 GCU300 / GCU400 / GCU410 |
| `biren` | 壁仞 | planned |

状态语义：

- **partial**：至少一个算子在该平台家族上有已合并证据；不表示所有算子都支持该平台。
- **planned**：只表示目标范围；没有算子条目、profile、dispatch key 或支持承诺。

平台矩阵会增量补齐。不要为了表格看起来完整而预填空平台单元格。

## 数据结构

### `platform_families`

每个平台家族记录：

```yaml
<family-key>:
  display_name: <厂商显示名>
  status: partial | planned
  current_profiles: [<已有 configs/devices profile>]
  known_variants: [<可选硬件型号>]
  roadmap_note: <简短非承诺说明>
```

校验规则：

- 每个 `current_profiles` 项必须存在 `configs/devices/<name>.yaml`。
- `planned` 家族不能声明 current profile。
- 家族只有在至少一个算子登记后才可变为 `partial`。

### `operator_groups`

分组描述算子存在的读者原因，而不是提交历史。当前有两类：

- **core**：多平台计算算子，例如 attention、embedding。
- **foundation**：用于验证执行链路和集成链路的基础 ATen 语义。

生成页按 `rank` 输出分组。现有分类不能准确描述新算子时，应新增分组，
不要把未来算子硬塞进语义不清的组。

### `operators`

每个算子记录：

```yaml
name: <算子名>
directory: <ops 子目录>
group: <分组 key>
purpose: <面向读者的一句话作用>
key_result: <有证据支撑的简短结果>
evidence: C0 | C1 | C2          # 汇总，必须等于 profile 记录的最高等级（派生，不能独立填写）
evidence_basis: <仅顶层 C2：一句范围诚实的说明（平台/日期/消费点）>
evidence_paths:
  - <仅顶层 C2：仓库内证据文件路径，至少一项；C2 记录的 report 必须在列>
platforms:
  <family-key>:
    profiles: [<已交付 profile>]
    implementation: <实现摘要>
    evidence:
      profiles:
        <profile>:
          level: C0 | C1 | C2   # 算子 + profile 是证据记录的最小单位
          basis: <C1/C2 必填：consumer 证据范围>
          report: <C2 必填：仓库内结构化证据报告>
          consumer: <可选：consumer/engine 标识，与结构化报告核对>
          status: <可选：验证状态，与结构化报告核对>
```

foundation 算子还需要顶层 `route`，除非每个平台条目单独声明 `route`。
算子里的 platform 条目表示已交付声明，必须由该算子的测试和报告支撑。

证据等级定义见 [docs/testing.md](testing.md) 证据等级，本目录的口径：

- **C0**：契约/直调范围——接口、数值语义或注册契约满足；无 consumer
  证据。只跑原生 torch 前向、`output_match` 固定返回、未消费仓库目标
  实现的脚本停在这一级。
- **C1**：有最小 consumer 证据——`ops/<directory>/test/framework_level.py`
  里基线/注入双跑或 mini-consumer 对拍，证明目标消费者能调用仓库实现
  并完成局部输出比对。**文件存在只是结构前提，不单独构成 C1**。
- **C2**：真实引擎端到端调用与输出结论——必须给出
  `evidence_basis`、仓库内 `evidence_paths`，并按 profile 记录
  `report`（结构化 JSON 报告须声明相同 `profile` 与通过状态）。
  C2 只按已实测消费点的 profile 标注；「注册进引擎但未被调用」
  停在 C1。

`evidence`（operator 顶层）是各 `platforms.<family>.evidence.profiles`
记录的**派生汇总**（最高等级），门禁校验二者一致；单个 family 下多个
profile（如 `thead-ppu` 与 `thead-ppu-c2`、`p800-kunlunxin` 与
`p800-kunlunxin-gems5`）必须各自有独立记录，不共享未区分的等级。
`ops/README.md` 按 profile 展示等级（平台列与证据列均带 profile 明细）。

`python3 scripts/evidence_check.py`（CI `ops-catalog` job）复核上述字段、
平台状态三方一致（catalog ↔ operator-catalog.md ↔ 生成的 `ops/README.md`）
以及 `updated`/环境锁日期不晚于当天，并交叉复核：

- 聚合页 `ops/<算子>/REPORT.md` 里宣称通过的 FlagOS E2E 结果行必须出现
  已知 profile 的**完整 token**（边界匹配：`demo-10`、`NOT-demo-1-invalid`
  都不算 `demo-1`；文件名内嵌的完整 profile 名算命中）；已知 profile 集
  为空时存在成功声明即报错——P800 的结果不能自动代表 PPU；行内引用的
  结构化 JSON 证据若声明 `profile`，必须与行内 profile 范围一致（拿
  别的环境的报告给成功行背书即报错；未声明 profile 的存量文件不在
  本层臆断范围，见「已知未覆盖范围」）；
- 所有证据路径（catalog `evidence_paths`/`report`、环境锁
  `evidence[].path`）经统一解析：禁外链、禁绝对路径、`resolve()` 后必须
  仍在仓库根内、必须是文件（目录、`..` 越界、仓库外 symlink 均拒绝），
  算子归属基于规范化后的仓库相对路径；
- `configs/env/*.lock.yaml` 的 `evidence[].path` 必须是仓库内当前存在的
  报告（禁外链、禁占位）；结构化 JSON 证据必须**可解析且声明与锁
  `name` 一致的 `profile`**——损坏、缺 profile、跨 profile 的 JSON
  单独报错、不再无声跳过，也不能成为任何支撑（P800 报告不能给 PPU
  的 C2 声明背书）；`scope` 行里的 `ops/<算子>` 必须登记在 catalog，
  `ops/<算子> C[012]` 声明必须等于**该锁 profile** 记录的等级（而不是
  operator 全局汇总），并由本锁 `evidence` 中该算子的**合格**报告支撑
  ——C2 声明的支撑报告必须有效且通过（JSON 可解析、声明本锁 profile、
  `status`/`ok` 为通过态，与 catalog 侧报告通过状态检查同口径），并
  与该 profile 的 C2 记录 `report`/`consumer` 同源关联（等于记录
  `report`、列入算子 `evidence_paths`、或 consumer 与记录一致）。

基础算子当前的实测报告以 `cambricon` profile 的 2026-09-18 历史快照为主；
没有独立 `mlu590` 报告时，不应把 `mlu590` 写入算子条目的 `profiles`。环境锁的
版本升级或同硬件别名不能替代该 profile 的复验。

跨多个平台家族的算子必须维护：

```text
ops/<operator>/PLATFORM.md
ops/<operator>/MERGE.md
```

### `registration_contracts`

注册契约按平台族记录，避免把一个厂商或一个算子族的 dispatch key 复制到
无关代码。未来新增契约时应记录 canonical key、兼容 key、可注册算子数、
不可注册算子和实现约束。

## 已知未覆盖范围（issue #41 跟踪，勿把门禁 OK 写成全部完成）

证据门禁当前**有意不覆盖**以下内容，处理方式在此登记，避免把脚本返回
OK 表述成 #41 验收范围全部闭环：

1. `ops/sdpa_math/reports/development.md` 的 `| FlagOS E2E | ☑ PASS | … |`
   分册行未标 profile——门禁只扫聚合页 `ops/*/REPORT.md`；补齐需要先补
   该行的真实环境数据，属 ops 报告数据改动，单独跟踪。
2. `ops/sdpa_math/REPORT.md` 关键数字表里的 E2E 性能倍率行（无成功标记、
   无 profile）——口径是「宣称通过的 E2E 结果行必须标 profile」，纯数字
   行不在其列；若收紧到「所有 E2E 行必标 profile」需同步补这两行数据。
3. 日期门禁只检查格式合法与不晚于当天，**不判定** stale roadmap/历史结论
   失效；历史证据可以保留，但不能冒充当前环境结论——该内容检查需要逐页
   人工复核，作为 #41 后续项而不是脚本规则。
4. REPORT E2E 行出现 profile token 只是第一层；行内引用的结构化 JSON
   证据声明的 `profile` 已与行内范围核对，但**未声明 profile 的引用
   文件**（存量：`ops/sdpa_math/reports/e2e_flagos_p800-kunlunxin.json`
   无 `profile` 字段）不在本层臆断其范围，补字段属报告数据改动、
   单独跟踪；行结论与 catalog 按 profile 记录的证据范围是否一致，仍由
   catalog 记录 + 环境锁交叉校验兜底，脚本不解析任意自然语言叙述。

## 新增算子

1. 从 `templates/operator/` 复制并完成算子目录。
2. 通过 kernel、op、framework/application 三层测试。
3. 补齐 golden 与性能证据。
4. 在 `ops/catalog.yaml` 的 `operators` 中登记算子、按 profile 回填
   `platforms.<family>.evidence.profiles`（顶层 `evidence` 汇总随记录
   派生；C2 需 `evidence_basis` + `evidence_paths` + 记录 `report`）。
5. 运行 `python3 scripts/build_ops_index.py`。
6. 确认 `python3 scripts/build_ops_index.py --check` 通过。
7. 确认 `python3 scripts/evidence_check.py` 通过。

校验器要求 `ops/` 下每个目录都出现在 catalog 中，防止新算子合入后没有
库存索引记录。

## 新增平台实现

1. 新增 `configs/devices/<profile>.yaml`。
2. 把 profile 加入厂商家族的 `current_profiles`。
3. 只有当首个算子的实现和验证合入后，才把家族从 `planned` 改为 `partial`。
4. 在对应算子的 `platforms` 中登记该家族。
5. 更新算子的 `PLATFORM.md`；第二个及后续平台还需维护 `MERGE.md`。
6. 重新生成 `ops/README.md`。

新增厂商家族时，先登记 `planned` 家族和不承诺支持的 roadmap 说明；算子
条目等到实现和验证落地后再添加。

## 文档边界

生成页只回答：

- 当前有哪些算子；
- 哪些平台家族已有交付证据；
- 详细信息在哪里。

它不替代算子内文档。详细契约、fallback 边界、数值容差、性能协议和已知
限制仍维护在各算子的 README、REPORT、PLATFORM 和报告分册中。
