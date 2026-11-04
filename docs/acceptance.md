# 交付验收标准: 精度与性能的定义

[← 返回文档中心](index.md)

> 交付前要回答两个问题: 精度够不够、性能够不够。这一页给出判定
> 口径: 容差怎么分档、必测哪些用例、性能怎么测才算数。文末给出
> 验收分级和环境变更后的重验流程。

<a id="accuracy"></a>
## 精度验收定义

### 参考实现约定（判卷标准本身先要站得住）

参考实现内部用 **fp32 计算，最后 cast 回输入 dtype**。这一步不能省:
不 cast 回去，bf16 自身的量化噪声（约 1e-2）就会被当成实现误差。
本库实测踩过——`*_and_mul` 类算子三方误差同为 6e-2，看似全错，
其实只是参考没 cast。

误差类型按算子选:

| 算子形态 | 用哪种误差 |
|---|---|
| pointwise，输出有界 | 绝对误差（abs） |
| GEMM / 输出随输入放大 | 相对误差（除以 max\|ref\|） |

### 容差分档（唯一权威表）

| 算子类型 | dtype | 容差 | 依据 |
|---|---|---|---|
| pointwise（有界输出） | fp32 | 1e-5 abs | fp32 本征精度 |
| pointwise | bf16 / fp16 | 1e-2 abs | bf16 量化噪声本身约 1e-2，更严会误杀正确实现 |
| GEMM / 乘法放大 | 全部 | **5e-2 rel** | 输出量级随输入放大，abs 不可比 |

### 必测矩阵（少一项就不算验收）

| 维度 | 要求 | 防的坑 |
|---|---|---|
| shape | ≥3 组，**必须含非 BLOCK 整数倍的维度**（如 3072/5000/5120） | [#15a](known-issues.md) 尾块污染——全 2 的幂时漏网 |
| dtype | bf16 / fp16 / fp32 三档 | 单 dtype 掩盖精度路径差异 |
| 随机种子 | 每组合 ≥3 | 单次幸运通过 |
| 特殊用例 | zeros / large / boundary 至少覆盖本算子相关的 | 除零、exp 溢出、饱和截断 |
| 三方对比 | 自研 / 原生 / FlagGems（缺某方时注明原因） | FlagGems 快可能靠降精度（softmax 实测 fp32 差 5 个量级） |

### 精度判定

- **Must**: 必测矩阵全绿
- **Must**: 哨兵通过——同输入两次结果一致；输入变了，输出必须变
- **Must**: 黄金回归 100% pass
- **Should**: 三方对比表齐备。自研若劣于某方，要给理由——"用精度
  换速度"是合法理由，"没测"不是
- 复现: 样板内 `script/check_accuracy.py`；仓库级
  `scripts/accuracy_report.py`

<a id="performance"></a>
## 性能验收定义

### 测量口径（不满足口径的数字无效）

| 项 | 要求 | 防的坑 |
|---|---|---|
| 采样 | 短采样 warmup 20 + iters ≤100 | [#10](known-issues.md) 分配器池增长 |
| 进程 | 多用例**每用例独立子进程** | 同进程混跑实测 0.311→1.688ms 污染 |
| 同步 | 计时区段前后 synchronize | 假异步计时 |
| 记录 | 数字必须可由 `perf_run` 复现并已入基线 | 口说无凭 |

### 判定

- **Must**: [性能回归门禁](performance-regression.md) 通过
  （比基线慢 >30% 即 FAIL；20%~30% WARN 需解释）
- **Must（微算子附加）**: 延迟 <100µs 的算子跑一次
  `bench_dispatch.py`，分发开销占比写进报告（A1 ≈3µs 可忽略；
  A2 ≈20µs 对微 kernel 有感）
- **Should**: 三方对比齐备；落后时给出瓶颈判断与优化方向——
  softmax 落后原生 6x 但"单遍在线归约"方向明确，就是合格表述
- **Info**: 本栈 kernel 启动 floor ~50µs——优化微算子前先确认瓶颈
  不在启动本身

<a id="levels"></a>
## 交付判定分级总表

| 级别 | 含义 | 不满足时 |
|---|---|---|
| **Must** | 正确性与门禁红线 | **不可交付** |
| **Should** | 完整性与可解释性 | 可交付但需在报告"遗留"中记录原因 |
| **Info** | 供决策参考 | 不影响交付 |

Must 清单（全勾才可进 [ops/](../ops/README.md)）:

- [ ] 精度必测矩阵全绿（含非整倍数 shape + 特殊用例 + 三 dtype）
- [ ] 哨兵: 确定性 + 输入敏感
- [ ] 黄金回归 100%
- [ ] 三层测试（kernel/op/framework）全绿
- [ ] 性能门禁 0 FAIL（微算子另加分发开销检查）
- [ ] 交付报告四件套齐备（REPORT + development + test-report + 分册）
- [ ] **证据等级如实登记**: `ops/catalog.yaml` 的
      `platforms.<family>.evidence.level` 与报告一致；C1（mini consumer）
      不得写成完整 vLLM 支持；C2 / C2-routing-boundary 必须带
      `evidence.report` 且文件存在（issue #41，`build_ops_index.py --check` 强制）

C2 合并门禁（针对声称真实引擎交付的平台）:

- 有真实 vLLM / 目标引擎双跑记录（版本、模型、profile、commit、
  调用计数、输出对比）
- 无真实引擎 runtime 时，只能标 C0/C1；若已在真实引擎环境内尝试
  runtime 但默认路由未走到目标算子，才可标 C2-routing-boundary——
  该级别必须附真实引擎尝试记录（引擎版本、注入方式、命中计数、
  未命中原因），不得以「本环境无 runtime」为由直接标注；
  并在 REPORT 写明环境边界
- mini consumer 结论只能支撑 C1

<a id="revalidate"></a>
## 环境变更后的重验 runbook（维护层）

驱动、固件、共享镜像或依赖升级后，旧数字与旧黄金全部失去可比性。
按序执行:

```bash
python3 scripts/check_env.py --device <profile>                       # ① 匹配本次软件栈的环境锁
python3 <算子>/script/gen_golden.py --device cpu                       # ② 黄金重建
python3 <算子>/script/check_accuracy.py --impl triton --device <profile>
python3 scripts/perf_run.py --update-baseline --device <profile>       # ③ 基线重采
python3 scripts/perf_compare.py --device <profile>                     # ④ 门禁
python3 scripts/accuracy_report.py --device <profile>                  # ⑤ 三方复测
```

昆仑芯 4.2 使用 `p800-kunlunxin`，5.0 使用 `p800-kunlunxin-gems5`，
具体入口和可验证范围见[环境配置](installation.md#kunlunxin)。真实框架验证
前额外执行 `check_env.py --device <profile> --require-model`；该检查只验证
模型目录存在，仍需确认模型与引擎参数匹配。

要点: ②黄金在 **CPU** 重建（最可信参考不受设备栈影响）；③基线重采
是**有意动作**，提交信息必须写明触发原因（镜像升级/驱动变更），
日常代码改动不允许顺手更新——与[性能基线规范](performance-regression.md)
一致。

---

**下一步**: [集成指南](integration.md)——验收通过后，选择交付路径。
