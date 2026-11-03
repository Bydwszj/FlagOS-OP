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
platforms:
  <family-key>:
    profiles: [<已交付 profile>]
    implementation: <实现摘要>
    evidence:            # 可选；建议所有新登记平台都写（issue #41）
      level: C0|C1|C2|C2-routing-boundary
      report: <C2* 必填，仓库内存在的报告路径>
```

证据等级语义（与 [docs/testing.md](testing.md) 一致）：

| level | 含义 | 校验 |
|---|---|---|
| `C0` | 契约 / 黄金 / 直调 | 可不带 report |
| `C1` | mini consumer / 等价消费方 | 可不带 report；报告里不得写成完整 vLLM |
| `C2` | 真实引擎命中目标算子 | **必须** `evidence.report` 且文件存在 |
| `C2-routing-boundary` | 真实引擎可跑、输出可比，但默认路由未命中该算子 | **必须** `evidence.report` 且文件存在 |

`scripts/build_ops_index.py --check` 会拒绝非法 level、缺失的 C2 report
和不存在的 report 路径（禁止占位证据）。

foundation 算子还需要顶层 `route`，除非每个平台条目单独声明 `route`。
算子里的 platform 条目表示已交付声明，必须由该算子的测试和报告支撑。

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

## 新增算子

1. 从 `templates/operator/` 复制并完成算子目录。
2. 通过 kernel、op、framework/application 三层测试。
3. 补齐 golden 与性能证据。
4. 在 `ops/catalog.yaml` 的 `operators` 中登记。
5. 运行 `python3 scripts/build_ops_index.py`。
6. 确认 `python3 scripts/build_ops_index.py --check` 通过。

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
