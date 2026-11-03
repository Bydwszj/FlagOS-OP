# Changelog

## [Unreleased]

### Added
- **catalog 证据等级字段（issue #41）**：`ops/catalog.yaml` 的
  `platforms.<family>.evidence` 支持 `level`（C0/C1/C2/C2-routing-boundary）
  与可选 `report`；`scripts/build_ops_index.py` 校验非法 level、C2 缺
  report 与不存在的 report 路径；生成页平台单元格显示证据等级；
  [acceptance.md](docs/acceptance.md) 增加 C2 合并门禁清单
- **sdpa_math 真实 vLLM-Ascend C2 双跑（issue #38）**：
  `ops/sdpa_math/script/c2_real_vllm_ascend.py` 在 ascend910 上完成
  baseline/注入双跑（vLLM 0.20.2 + vllm-ascend + SmolLM2-135M），
  token_ids 逐位一致；A1 probe 命中、真实前向 `forward_hits=0`
  （`npu_fusion_attention` 路由，known-issues #19）→ 标记
  `C2-routing-boundary`，证据
  [reports/c2_ascend910.md](ops/sdpa_math/reports/c2_ascend910.md)；
  `injection/a1_aten_ops` 增加 `FLAGOS_C2_COUNT_DIR` pid 分片计数
- **13 个基础 ATen 算子接入 p800-kunlunxin（2026-10-22）**：
  `clone/contiguous/copy_/detach/detach_/dropout/empty/empty_like/
  empty_strided/item/_local_scalar_dense/result_type/type_as` 完成
  kernel/op/应用等价三层验证、黄金与性能基线；注册 key 为
  `AutogradCUDA`/`CUDA`，设备探测走 `torch_xmlir`；P800 平台差异
  （type_as/copy_ 目标 bf16 兜底、dropout 随机分支委托原生、
  result_type fp64 skip）记入各算子 `PLATFORM.md`/`MERGE.md` 与
  四件套报告；性能基线在空闲 P800 上两轮复测后重采（旧数据受共享卡
  争用影响约 2-4x）；`p800-kunlunxin` 环境锁校准到 FlagTree 0.6.1 /
  Triton 3.6.0
- 新增 [软硬件环境版本汇总](docs/environments.md) 与
  `scripts/build_env_index.py`：从 `configs/env/*.lock.yaml` 生成各实测
  profile 的硬件、系统、Python、厂商运行时和软件包版本索引，并声明
  CPU / NVIDIA / A100 与规划平台的证据边界；补齐 Ascend 环境锁，
  P800 锁增加 XMLIR / XTORCH_OPS 与硬件快照
- 新增 [基座镜像索引](docs/base-images.md) 与
  `configs/images/catalog.yaml` / `scripts/build_base_image_index.py`：
  按 FlagRT runtime-team 镜像索引建立六平台协作视图，登记当前外部
  基座 tag、用途、复现状态、分发方式和 FlagOS-OP 环境锁关系；平台方
  可按 schema 追加历史版本或新镜像，并用 CI 校验生成页一致
- 新增 [安装与 FlagGems 配合使用](docs/installation.md)：源码仓库
  `PYTHONPATH` 口径、FlagGems 运行时 / 源码共用方式、A1 注册示例、
  拦截计数、注册对象生命周期与进程隔离边界、常见 import / 命中问题；
  根目录新增 `INSTALL.md` 入口；新增 `examples/operator-usage`，用同一条
  连续调用链对比 native、direct kernel、FlagGems + 多算子 A1 混合映射，
  并以子进程隔离 baseline 与注册路径、用 counter 验证命中；CPU CI 增加
  import 与 `--help` smoke test，硬件路径仍按算子 profile 复验
- 新增 [FlagGems 5.0 兼容性复测（P800）](docs/flaggems5-p800-retest.md)：
  环境对照、三层与黄金、微基准与 e2e 性能、路线样例矩阵、发现与修复
  状态与复现命令——已完成的 kernel/A1/mini-consumer 证据与 4.2 基线一致、
  推理/生成一致、训练收益收窄归因 xtorch_ops/plugin；A2 插件路线
  BLOCKED、framework 矩阵因冒烟模型缺口未复测（待回填）；配套登记
  `p800-kunlunxin-gems5` 设备 profile 与环境锁（4.2 基线锁保留）；
  known-issues 新增 #20-#23（插件 per_op_order 覆盖钉选、bench
  ValueError 容错、FlagGems 5.0 pointwise codegen `__future__` 回归、
  A2 插件与内置 backend impl_id 冲突）与 #5 的 XPU1 Xid 补注

### Changed
- **known-issues #13 / issue #30 跟进（2026-10-27）**: 补 GELU
  `approximate="tanh"` 链接失败的组件归属与修复状态——4.2.1rc0 根因为
  xtriton XPU libdevice 表 fp32 tanh 占位符 `Unsupported`（报错另会被
  #14 掩盖为 `NameError: sys`）；同硬件 FlagGems 5.0.0 实测 tanh 路径
  已修复（`max_diff=0.000e+00`），4.2 栈规避为升级 5.0 或自研手写 tanh
- 昆仑芯环境指南区分 FlagGems 4.2 基线与 5.0 复测栈，明确中央 profile、
  独立算子别名、模型核验和框架子进程卡号覆盖；安装/快速开始/环境与镜像
  索引同步链接，复测页区分历史与整改后单测数，说明 5.0 框架黄金缺失、
  SDPA 中央性能用例未覆盖 gems5，以及性能基线单独建档的要求
- 文档统一区分 C0 契约、C1 最小消费者和 C2 真实引擎证据等级；framework
  层示例需注明 mini consumer 或真实 vLLM，性能数字需带 profile 和测量协议
- **ops/ 文档契约同步（2026-10-13）**: 13 个 MLU 算子的
  README / REPORT / development / test-report 补记 PR #25 后的
  `counter + platform` 注册契约、canonical `AutogradPrivateUse1` /
  兼容 `PrivateUse1`、包态优先导入与 clone 包态执行用例；修正
  `empty` 的自用/实验路线与 sdpa_math 黄金口径描述；`ops/README`
  重组为核心多平台算子与 MLU 基础 ATen 算子两组，基础算子再按
  布局/随机/分配/标量用途分组
- **可扩放算子目录（2026-10-14）**: 新增 `ops/catalog.yaml` 作为
  算子 / 平台矩阵唯一事实来源，登记六个目标厂商平台家族与当前增量
  交付状态；`scripts/build_ops_index.py` 生成 `ops/README.md` 并校验
  profile、目录、平台声明和多平台文档，配套
  `docs/operator-catalog.md` 与单元测试；新增 `ops-catalog` CI 门，
  并恢复完整的 docs-and-syntax 链接 / 语法 / YAML / 基线检查
- sdpa `PLATFORM.md` 补记多算子 perf registry 顺序加载时的顶层
  `reference` 遮蔽约束；backend 导入回归测试同步覆盖 P800 与 MLU

### Fixed
- **embedding PLATFORM 降级对齐（issue #38）**：`ops/embedding/PLATFORM.md`
  头部与 §2 由「三平台交付」改为「两平台交付 + ascend910 历史实验」，
  与 `ops/catalog.yaml` / `ops/README.md` / `REPORT.md` 口径一致
- 环境与基座索引复核修正：embedding 的 MLU 证据归入匹配的
  `mlu590` 环境锁；PPU 镜像引用补全 Harbor registry / namespace；
  环境锁与镜像 catalog 增加双向关系校验；历史镜像保留稳定文档锚点；
  Ascend 不可用范围收窄为真实 vLLM 入口
- catalog foundation 算子的 route 校验与文档契约对齐：顶层 route 与
  “每个平台条目均声明 route”两种形式均可用；贡献指南改为维护
  `ops/catalog.yaml` 并重新生成 `ops/README.md`
- P800 性能门禁多算子加载时，`ops.sdpa` backend 的裸 `reference`
  导入可能命中 `sdpa_math.reference` 等同名顶层模块，导致
  `ops.sdpa.p800.forward` SKIP；改为包态相对导入优先并保留单算子
  脚本 fallback。`perf_run` 会先清理本次选中用例的旧记录，且任一
  SKIP 直接返回失败，避免 compare 消费历史 JSON 形成“假绿”
- **FlagGems 5.0 复测适配（known-issues #20/#21/#22）**：
  `tests/op_level/test_b`、`examples/b-op`、`examples/b-fullstack`(stage2)
  改用 `VLLM_FL_PER_OP` + `reset_global_policy()` 钉住 silu_and_mul 的
  vendor 实现，对抗插件 0.2.0 平台 `op_backends` 钉序（修复前选中
  `default.flagos`）——钉选前清掉 `VLLM_FL_OP_CONFIG`（优先级更高会
  覆盖钉选）、断言最终 impl_id、运行结束恢复环境变量避免 `--all`
  串跑污染；`ops/embedding/script/bench_perf.py` 容错扩为
  `except (RuntimeError, ValueError)`，`scale_grad_by_freq=True` 原生
  `ValueError` 不再崩整个基准（记 `native_error`）；`examples/a1-op`、
  `examples/a2-op` 移除模块级 `from __future__ import annotations`，
  本仓库侧规避 FlagGems 5.0 pointwise_dynamic codegen SyntaxError
  （FlagGems 根因未修）

### Added
- **sdpa_math 第二平台: p800-kunlunxin**（vendor O/LSE + exact-P + A1）
  - `_profile` / benchmark / framework 消费方接入 P800；A1 使用
    `AutogradCUDA+CUDA` 成对注册；fast path 由 vendor efficient attention
    产出 `O/LSE`，再物化 `P=exp(scale·QKᵀ-LSE)`，fp32/mask/dropout 回退
    `torch_level`
  - A1 backward P800 分支复用 vendor efficient backward，`dprobs` 用已
    物化 P 单独求 Jacobian；fp16 GQA dO+dP 对照 native 通过
  - kernel 52/52、黄金 175/175、op 22 项、FlagGems 应用层 9 项、
    P800 perf gate 全绿；direct 1.46-2.21x native
  - 新增 FlagOS mini-LLM 端到端：FlagGems GELU + A1 sdpa_math +
    概率图可选熵正则，forward 1.07x、train-out 1.45x、
    train+probs 1.48x、generate 1.09x；fused backward off/on 1.45-1.50x
  - `ops.sdpa_math` 包化，支持 `from ops.sdpa_math import register_a1`

- **embedding 第二平台: cambricon**（Cambricon MLU590 / torch_mlu）
  - 新增 [ops/embedding/kernel/cambricon.py](ops/embedding/kernel/cambricon.py)：
    前向委托原生 `aten::index_select`，反向委托原生 `aten::embedding_backward`
    ——MLU 原生 `padding_idx` + `scale_grad_by_freq` 完整（6/6 组合 err=0），
    **无需** P800/XPU 那套 inverse-frequency 补偿；`sparse=True` backward
    显式 `NotImplementedError`（否则 torch_mlu 返回 COO 稀疏梯度）
  - 新增多平台门面 [ops/embedding/kernel/platform.py](ops/embedding/kernel/platform.py)：
    按 `tensor.device.type` 路由 `mlu→cambricon` / `cuda→p800`，
    附 `dispatch_key_for`/`synchronize`；`p800_kunlunxin.py` 逐字节未改
  - A1 key = `AutogradPrivateUse1`：两个 key 实测都命中，但 `PrivateUse1`
    会覆盖 functorch 的 batch rule（`BatchRulesModules.cpp`）；Autograd key
    同时与 p800 `AutogradCUDA` 同构
  - 平台命名统一 `cambricon`（别名 `mlu`/`mlu590`/`cambricon-mlu590` 归一），
    与 13 个既有 MLU 算子、`configs/devices/cambricon.yaml`、
    `perf/baselines/cambricon.json` 同名
  - 验证全绿：kernel 20/20 前向 + 6/6 反向 err 0、黄金 **174/174 逐位**、
    A1 op 拦截 5 次（注册=直调）、应用层拦截 8 次 logits/梯度 0 diff、
    greedy 1.00、`pytest tests/unit` 20 passed、`example.py cambricon` 三层 PASS
  - perf gate 注册 `ops.embedding.cambricon.forward`（0.060ms / 142.8 GB/s）
    与 `ops.embedding.native.forward`（0.045ms / 187.9 GB/s），
    `perf/baselines/cambricon.json` 增至 33 条 case，门禁 `FAIL 0 · WARN 0 · NEW 0`
  - 新增 [ops/embedding/reports/cambricon.md](ops/embedding/reports/cambricon.md)
    与 `perf_fp16_cambricon.json` / `dispatch_cambricon.json`
- **新算子: [ops/embedding](ops/embedding/)**（`aten::embedding`，
  A1 路线，p800-kunlunxin）
  - forward 复用 XMLIR native `index_select` row-gather；任意 rank indices、
    FP32/FP16/BF16、int64/int32、padding 与 empty 全覆盖
  - dense backward 复用 native `embedding_backward`；为 XPU 未实现的
    `scale_grad_by_freq=True` 增加 inverse-frequency fallback
  - `AutogradCUDA` A1 拦截 + 数学 backward；174 组黄金、20 组 forward
    （含 `sparse=True` 前向）、6 组反向、`nn.Embedding` 应用层前向/反向
    全部通过；perf gate 注册与 P800 基线更新，硬件级显式置空
  - 新增 A1 dispatch 独立子进程基准，native→A1 附加开销约 0.024ms
    （⚠️ 该 a1 数字因下述 Library GC 缺陷作废，待复测）
- **SDPA 第三平台: mlu590**（Cambricon MLU590 / torch_mlu）
  - 新增 `kernel/backends/mlu590.py`：分层委托——
    **TMO** `torch_mlu_ops.flash_attention`（半精度快路径）→
    **fused overrideable**（CNNL FA v2，与原生 `F.sdpa` 同路径）→
    math/FlagGems/reference 兜底；bool→additive；causal 折叠；NaN 恢复；
    GQA `repeat_interleave`；`SDPA_MLU_TMO=0` 可关 TMO
  - 更正早期结论：原生 F.sdpa 走 fused overrideable **不是** math；
    math-only 主路径实测仅 0.11-0.55x 原生
  - facade 增加 `"mlu"` 分发；`register_a1` 守卫放宽为 torch_npu **或**
    torch_mlu 可用；`_profile`/`configs/devices/mlu590.yaml`/`mlu590.lock.yaml` 接入
  - A1 `AutogradPrivateUse1` 拦截；kernel 37/37、黄金 397/397、
    op/framework/guard 三层全绿
  - 实测修订后 **1.00-1.33x** 原生（prefill 1k D64 1.33x；FA2 协议
    多点 TFLOPS 高于 native）；FlagGems 仍慢 10-40x 仅兜底
  - 新增 [ops/sdpa/reports/mlu590.md](ops/sdpa/reports/mlu590.md) 与
    `perf_fp16_mlu590.json`
- **SDPA 第二平台: p800-kunlunxin**（Kunlun XPU / torch_xmlir）
  - 新增多平台 backend facade：旧 `kernel.triton_level.sdpa_triton` 引用零改动
  - fp16/bf16 委托厂商 `aten::_scaled_dot_product_efficient_attention`；
    bool mask / 全遮蔽行语义补偿；fp32 走精确 ATen 组合并绕开 XMLIR
    `S∈(320,640]` bmm JIT 缺陷
  - direct autograd 按需计算 log-sumexp；可微 float mask 复用 A1 数学
    backward，规避厂商 `bias_requires_grad` 限制
  - A1 `AutogradCUDA` 拦截 + 数学 backward；P800 kernel 37/37、黄金
    397/397、mini-decoder 与平台守卫全绿
- [ops/sdpa/reports/p800-kunlunxin.md](ops/sdpa/reports/p800-kunlunxin.md)
  与平台化性能 JSON，沉淀 FlagGems/Triton 平移失败原因与复现命令
- SDPA perf gate 用例注册与 P800 基线更新；硬件级目录显式置空说明
- P800 自研固定调度 Triton forward 实验与复现脚本：no-mask causal、
  GQA、非 causal 与尾块正确，但 mask launch 失败且慢于厂商路径
  2.1-6.5x，因此暂不接入生产 backend

### Fixed
- `ops/embedding/script/bench_dispatch.py` a1 分支裸调 `register_a1`：
  返回的 `torch.library.Library` 无引用时被 GC，**析构即反注册**，旧写法
  测的是未注册状态（`F.embedding` 落回原生）。改为持有 `_a1_lib` 并断言；
  全仓扫描确认为唯一裸调用点（sdpa 与各测试均持有 `lib`）
  - 作废待复测：P800 a1 数字 `A1−native ≈0.024ms`、"A1 与 direct 同档/略快"
    （本机无 `torch_xmlir` 无法重测；native / direct 两行仍有效）
  - 三层精度结论不受影响（测试持有 `lib`，拦截计数 5/8 均 > 0）
  - MLU 修复后复测：A1 − native ≈ **+0.059ms**、A1 − direct ≈ **+0.044ms**
- `ops/embedding/script/bench_perf.py` 反向分支原为 XPU 专用的
  `if not scale:` 跳过，MLU 上拿不到 `scale_grad_by_freq` 对照——
  改为按平台 try/except 逐个尝试原生 scale，两平台反向表均含 native 对照

## [0.16.1] - 2026-10-02

### Changed
- **算子模板范式更新**（对齐 ops/ 实际交付）: 补 `__init__.py` 包化
  入口、register 平台分发签名（impl/platform + AutogradPrivateUse1
  注记）、kernel/backends/ 选择器模板、"与 ops/ 对齐"速览节——
  新算子照模板开发即与 sdpa/embedding 同构

## [0.16.0] - 2026-10-02

### Added
- **算子目录包化**（多算子同进程冲突的根治）: ops/sdpa、ops/embedding
  加 `__init__.py`（惰性导出 register_a1/kernel 入口），包态导入
  `from ops.sdpa import register_a1` 与单算子脚本模式兼容并存；
  ops/test_packaging.py 四项验证全绿（双包导入/双注册同进程/双算子
  消费数值/旧用法兼容）
- sdpa register 的模块加载器升级"包态优先"（ops.sdpa.kernel.* 包
  导入, 文件锚定 fallback）

### Fixed
- op_level_auto 测试的实例分裂（stats/patch 状态统一到 register
  锚定实例）

## [0.15.0] - 2026-09-30

### Added
- **embedding 第二平台 ascend910**（kernel/ascend910.py，厂商委托
  index_select + dense_backward；register/_profile/tests 按
  dispatch_key 平台分发；三层 + 黄金 350 PASS + Triton 探针数据
  ——P800"生产委托"结论在 Ascend 复现，Triton gather 慢 18-69x）

## [0.13.0] - 2026-09-17

### Added
- **首个正式算子: [ops/sdpa](ops/sdpa/)**（aten::scaled_dot_product_attention，
  Triton 级，A1 路线，ascend910）
  - online-softmax two-pass kernel + causal 循环截断（1.93x）+ GQA/双 mask/
    尾块安全；黄金 265/265、kernel 层 37/37、A1 拦截+梯度验证全绿
  - 首个算子级平台文档: [PLATFORM.md](ops/sdpa/PLATFORM.md)（8 项 Ascend
    绑定 + 移植指引 + 引用防误用三层机制）与 [MERGE.md](ops/sdpa/MERGE.md)
    （第二平台七步合并指南）
- **设备 profile: [ascend910](configs/devices/ascend910.yaml)**（CANN 9.0.0 /
  torch_npu 2.10.0；quirks.no_vllm_runtime 标注空壳 vllm 环境）

### Discovered（随算子交付的实测结论，上游有价值）
- torch_npu 栈 A1 注册点为 AutogradPrivateUse1（PrivateUse1 永不命中，
  C++ 包装不 redispatch）——与 CUDA 系后端关键差异
- flag_gems 5.3.5 SDPA 未被 aten 分发接入（enable 后 diff=0.0）
- triton-ascend 三硬约束: dot 强制同 dtype / fp32 默认 tf32 / exp2 慢路径

## [0.12.3] - 2026-09-14

### Fixed
- getting-started 环境要求中 Triton 重复列出两次

### Changed
- 可读性润色（第一读者视角通读）:
  - getting-started 环境要求改为三行表（替代嵌套 bullet）
  - acceptance 开场去双破折号嵌套；参考实现约定拆为两段+误差类型表；
    精度 Must 拆为三条独立判定；容差表依据列瘦身
  - architecture run() 约定拆两行；fullstack-guide 起点句拆短；
    testing 容差句改为指向验收权威表（消除口径重复）

## [0.12.2] - 2026-09-14

### Fixed
- 阅读链顺序矛盾: README 表"集成(6)→验收(7)"与 acceptance 的
  下一步（验收→集成）及 index 交付线相反——交换为 验收(6)→集成(7)
- perf-regression 下一步跳过验收（直连 AI 接入）——改接验收标准，
  主链闭合为 性能→验收→集成

### Changed
- architecture 模板子树补 example.py（一键编排+perf_cases）
- perf"新增用例"明确推荐位置为算子 example.py（样板既定约定）
- 全链路第 6 步补 example.py 一键复跑、交付四件套与验收标准链接

## [0.12.1] - 2026-09-14

### Added
- 算子样板向 softmax-fullstack 范本对齐（补 3 件）:
  - example.py 一键编排（三层一次跑完）+ perf_cases 性能回归入口
  - reports/development.md 开发报告模板（7 章，链验收标准分档）
  - reports/test-report.md 测试报告模板（范围矩阵 + 一键命令）
- REPORT.md 增"一键三层"行与交付物清单（四件套）

### Changed
- 样板 README 目录树/开发顺序同步（一键命令、perf_cases 登记方式、
  四件套填写步骤）并注明与 softmax 范本"空壳↔实肉"关系
- templates/README、ops/README（流程 4 步）、examples/README
  （双向同构表述）同步

## [0.12.0] - 2026-09-14

### Added
- docs/acceptance.md 交付验收标准（收拢此前散在 5 处的定义为唯一口径）:
  精度（参考约定 / 容差权威表 / 必测矩阵 5 维 / 判定）、性能（测量
  口径 4 条 / 门禁 / 微算子分发开销附加项）、Must/Should/Info 分级
  与 ops/ 合入门槛、环境变更重验 runbook（维护层，此前无家）
- integration 交付 checklist 精简为指向验收标准；index/README 接入

## [0.11.1] - 2026-09-14

### Added
- ops/ 目录: 存放按样板开发完成的算子（结构同构 + 索引表 + 约定:
  未实现级别置空说明、#11/#15 硬约束、合入前三层全绿+黄金+门禁）
- 全站 cp 目标统一为 ops/<算子名>/（README 快速开始、样板 README、
  全链路指南起点、CONTRIBUTING）
- README 仓库分区表、architecture 目录树、index 任务表与交付线、
  集成指南交付 checklist 均接入 ops/

## [0.11.0] - 2026-09-14

### Added
- softmax-fullstack 交付报告补齐（此前仅提纲级 90 行，缺成套交付物）:
  - reports/development.md 开发报告（7 章模板全填: 环境锁定/算子定义/
    实现说明含 #15 两条硬约束/三层验证明细/三方性能/风险/结论/复现命令）
  - reports/test-report.md 测试报告（测试范围矩阵 5✅1◐、环境 7/7、
    三方精度、三方性能、#15 修复史风险表、可交付结论）
  - REPORT.md 新增"交付物清单"；精度/性能分册补复现命令
  - reports/examples 索引收录两份新报告

## [0.10.2] - 2026-08-28

### Added
- known-issues #16: A2 call_op 长循环偶发挂起（~300 次触发，100 次
  稳定；缓解为短循环，复现条件待稳定后深挖）

### Changed
- docs index 阅读路径新增性能/生成/交付三线中的"交付"线，分层说明
  升为四层
- architecture 路线节链到集成指南的"两张调度网"（此前 0 处互链）
- route-b 下一步补集成指南出口

## [0.10.1] - 2026-08-28

### Changed
- 集成指南补"两张调度网"架构（实测源码验证）: FlagGems 不是调度器，
  是 kernel 集合；它同时出现在两张网——aten dispatcher（enable 注册）
  与 vllm_fl OpManager（backends/flaggems/ 包装为 default.flagos）。
  修正"A2/B 无法统一调度"的误解: 它们的统一调度网是 vllm_fl，
  且与 FlagGems 实现同网竞争

## [0.10.0] - 2026-08-28

### Added
- 算子集成指南 docs/integration.md: 四条交付路径选择（FlagGems 贡献 /
  A1 aten / A2 插件 / B 厂商）、"要不要进 FlagGems"的判断依据、
  交付 checklist
- scripts/bench_dispatch.py: 分发开销可复现实测
- P800 分发开销首次量化: A1 aten ≈2-3µs（可忽略，即 FlagGems 机制）；
  A2 call_op ≈17-21µs（微 kernel 需注意，大 kernel 可忽略）；
  本栈 kernel 启动 floor ~50µs（比两条分发开销都大，此前被忽视）

## [0.9.3] - 2026-08-28

### Changed
- 叙事水位拉平（0.9.2 只覆盖了 a2，本轮补齐其余干燥点）:
  - route-a1: dispatch key 的两个误解（CPU 参考天然可用 / 同 key
    覆盖竞争的 flag_gems 实战）、步骤排序理由（拦截确认先于精度）
  - route-b: KernelSpec 声明式的动机（新 kernel = 一行 YAML 而非
    新测试，5 个厂商 kernel 3 个缺陷全由此抓出）、audit 分离
    "链路通/数值对"的归因价值
  - getting-started: 补"成功长什么样"示例输出与 FAIL 排障入口
  - reporting: 三类报告的分工与典型工作流（总报告随手维护 /
    分册沉淀数据 / 开发报告评审时才写）

## [0.9.2] - 2026-08-28

### Changed
- 文档"去干燥化"扩写: 为核心表格补上解释性段落（7 个文件，约 +90 行）
  - architecture: 矩阵的存在动机（注册未选中/语义被改/端到端破坏三类
    真实关口）与"先列后行"读法、三级别的效率/天花板取舍叙事
  - testing: kernel 层三子项各自防什么、尾块 shape 的必要性（#15
    漏网教训）、哨兵的 swiglu 真实故障叙事、framework 双跑归因逻辑、
    pid 分片 35 倍教训、断言分档的混沌放大器解释
  - performance-regression: 双档阈值的噪声/回归分界依据、基线携带
    环境信息的归因价值
  - examples/README: 定位图/性能/精度三节各补"怎么读"导语
  - ai-intake: 三入口收敛点、正负例的价值主张
  - fullstack-guide: 第 1/2/4 步补"为什么这样排序"连接段
  - route-a2: 分发模型补三层排障定位叙事

## [0.9.0] - 2026-08-28

### Added
- softmax-fullstack 重组为与算子样板**完全同构**的教学样例:
  kernel/ 三级（torch✅/Triton✅/硬件级⬜置空+说明）、test/ 三层、
  goldendata/（规格含 #15 尾块形状）、script/（gen_golden 39 组 ✓ /
  check_accuracy triton 39/39 ✓）、REPORT.md 总报告 + reports/ 分册
  （真实数据已填）
- examples/README 新增"与算子样板的对应"映射表（每个样例对应样板
  哪个槽位；格级样例保持单文件聚焦的理由）

### Fixed
- test/ 目录名与标准库 test 包同名导致导入遮蔽——编排器按路径加载

## [0.8.3] - 2026-08-28

### Changed
- 文字与组织同步样板新结构: templates/README"7 个文件"描述改为单算子
  自包含结构; architecture 目录树展开 operator/ 子树
- docs index 按任务表新增"从样板开始开发新算子"入口
- reporting 组成表纳入测试报告模板与算子总报告+分册
- CONTRIBUTING 新增样例 checklist 补黄金数据项（含特殊用例要求）

## [0.8.2] - 2026-08-28

### Changed
- 硬件级模板改为占位符 + 文字说明形态（模板库定位: 骨架与指引，
  不预写实现）: kernel.cu 保留结构要点（网格划分/访存/精度/尾块提示），
  xtorch_binding.cpp 标注厂商 API 位置与 #2/#5 实测坑

## [0.8.1] - 2026-08-28

### Changed
- 算子样板按"单算子自包含"重组（采纳用户建议结构 + 相关工作对标）:
  kernel/（三级实现: torch/Triton/硬件级类 C）· test/（三层测试）·
  goldendata/（声明式规格 + 黄金数据）· script/（生成/精度/性能脚本）·
  REPORT.md（总体报告）+ reports/（精度/性能分册）

### Added
- 硬件级模板: kernel.cu（CUDA C++，标注 #12 限制）+ xtorch_binding.cpp
  （P800 当前可跑的厂商 C++ 绑定，含已验证 include 三链）+ BUILD.md
- goldendata/inputs_spec.yaml: 维度/精度/数量/随机分布/特殊功能
  （zeros/large/boundary）声明式规格（参考 KernelBench problem 声明）
- script/gen_golden.py: 按规格生成黄金输入+参考输出+sha256 索引
  （CPU 实测 24 组生成 ✓）
- script/check_accuracy.py: 读黄金按容差判定（实测 24/24 pass ✓）
- script/bench_perf.py: 单算子性能测试（复用 common.perf 口径）
- REPORT.md 总体报告 + reports/accuracy|performance.md 分册模板

## [0.8.0] - 2026-08-28

### Added
- 算子样板 templates/operator/（7 文件 copy-paste 起点 kit）: 语义参考 /
  Triton kernel（内置 #11 device 上下文与 #15 尾块 pad 两条硬约束）/
  kernel 层测试（metrics 落盘约定）/ 三路线注册 / op 与应用层测试骨架 /
  性能回归用例
- 测试报告模板 templates/op-test-report.md（测试范围/精度/性能/风险/结论）
- reports/examples/ 入库: gelu_and_mul 已填测试报告（真实数据）+ 2 份
  开发报告骨架 + 索引（原 reports/ 整体 gitignore，报告样例不可见）

### Changed
- 根 README 新增"仓库分区"表（样板/正式实现/教学样例/自动化/文档五区）
- 全链路指南第 0 步、CONTRIBUTING、architecture 目录树接入样板入口
- .gitignore: reports/* 但保留 reports/examples/

## [0.7.0] - 2026-08-28

### Added
- scripts/accuracy_report.py: 自研/FlagGems/原生 三方同输入同参考精度对比
- 样例索引新增"精度速览"表: 自研 15/15 全过；FlagGems softmax 三 dtype
  全超差（fp32 1.5e-2）——其性能优势为低精度换取

### Fixed
- ⚠️ softmax 尾块正确性 bug（#15a，由精度探针发现）: N 非 BLOCK 整数倍时
  masked load+tl.sum 污染结果（三种防护无效）；修复为 pad 到 2048 倍数
- ⚠️ 移除 @triton.autotune（#15b）: 本栈 autotuner 选出非法 num_warps=5，
  同 kernel 不可复现地时对时错；改固定 BLOCK_N=2048
- softmax 测试形状补非整倍数 N（3072/5000/5120）——原测试全整倍数漏测

## [0.6.2] - 2026-08-28

### Added
- FlagGems 生产基线接入性能对比与回归体系: softmax / bmm 直调，
  silu_and_mul 用 FlagGems silu 单算子组合（无该融合算子）
- known-issues #13（FlagGems gelu(tanh) 链接失败，XPU libdevice
  tanh → "Unsupported"）与 #14（编译错误被 NameError:sys 掩盖）
- common/xpu_compat.py: 经 backends 注册表给隐藏的 xpu compiler
  第二实例注入 sys，编译失败暴露真实原因

### Changed
- bmm/softmax/b-fullstack/hw 样例的性能段输出 FlagGems 基线行（防御式）
- A1/A2 因 #13 无同语义 FlagGems 基线，文档注明而非用 gelu(none) 冒充
- 样例性能速览表增加 FlagGems 基线列（softmax 上 FlagGems 反超自研）

## [0.6.1] - 2026-08-27

### Fixed
- 文档-代码矛盾: 4 处仍写 run() 只返回 bool，与 0.6.0 的 dict 约定不一致
- b-kernel 性能段静默吞异常（perf=null 无原因）→ 打印 SKIP 原因
- perf_compare 常驻 "NEW 1" 噪音: intake 用例单独列出，不参与门禁与 NEW 统计
- b-kernel 负例 flaky: gelu_tanh_and_mul 缺陷偶发（#5），"负例未复现"
  误判整格 FAIL → 降级 WARN，通过不再依赖坏 kernel 每次都坏
- p800 profile 该负例的 known-issues 引用误写 #10 → #5

### Changed
- device-profiles 补冒烟模型要求（结构/权重格式/词表与 tokenizer）

## [0.6.0] - 2026-08-27

### Added
- 环境锁定: configs/env/p800-kunlunxin.lock.yaml（共享镜像实测版本）
  + scripts/check_env.py 核对（--require-model 一并检查模型路径）
- 模型路径泛化: 环境变量 FLAGOS_MODEL_PATH 覆盖 profile，换机器不改 YAML
- tests/unit/ 基础设施单测 20 个（intake 契约 / perf 记录与基线合并 /
  门禁分级 / KernelSpec 适配器与哨兵 / profile 覆盖），CPU 即可运行
- CI 新增 unit-tests job（CPU torch + pytest），从纯结构检查升级为跑真实测试

### Changed
- run() 约定支持返回 {"ok": bool, ...指标}，结果 JSON 新增 metrics 字段
- kernel 层 a1/a2/b 测试落盘精度 max_err / 哨兵结论 / 性能延迟
- 报告脚手架自动渲染 metrics（第 4 章不再全部手填）
- perf_compare 门禁分级抽为 classify_delta 纯函数（可单测）
- kernel_spec 哨兵检查的 synchronize 加可用性保护（CPU 单测可用）

## [0.5.6] - 2026-08-27

### Fixed
- b-fullstack 三处历史重命名残留: "kernel 层 层"/"op 层 层"（README+docstring）
  与 "厂商语言 kernel/厂商语言路线" 误称（该样例是 torch 级 C++ 调 ATen，
  路线名为厂商算子注册）

## [0.5.5] - 2026-08-27

### Changed
- 消除模板句口癖（"读这一页能得到什么"×4、"典型开发级别:"引言×3、
  "## 定位"标题），全部改写为自然连贯的说明段落
- 根 README 开篇从名词堆叠改写为两段完整叙述（库解决什么问题、
  3×3 矩阵如何组织）
- 路线文档把"级别与路线正交"从事故在引言里的标签，改写成
  适用场景内的正文段落，并删除文末重复的级别标签行
- docs index 补一段文档分层说明（主线/专题/参考）；样例索引开篇
  改写为可执行指引

## [0.5.4] - 2026-08-27

### Changed
- architecture/testing/reporting 补定位开篇句（与其余 10 篇风格对齐）
- testing 矩阵表归入"总览"小节（消除目录后表格悬空）
- 样例索引回链全链路指南覆盖矩阵（样例视角 ↔ 步骤视角互达）
- index 新手线注明性能/AI 生成走专题线

## [0.5.3] - 2026-08-27

### Fixed
- known-issues 错误引用 3 处: swiglu 不写输出是 #2（route-b/b-op/b-framework 误写 #5）
- route-a1 错乱文本（"torch torch 算子替换"、"典型 kernel 层级: Triton（Triton）"）
- route-b audit 段重复句、"kernel 层级"→"开发级别"
- route 三篇 breadcrumb 残留旧路线名尾巴
- device-profiles: 循环引言重写、"字段速查"空壳章节（表归位）
- getting-started "3 步接入" 与 device-profiles "4 步" 不一致
- reporting: 全链路步骤编号 4→6（指南重构后）

### Changed
- route-a1 样例补 bmm/softmax（A1 全链路）；route-b 样例补 b-kernel/b-fullstack/hw-kernel
- routes README 标题统一正式路线命名
- 新增 tests/kernel_level/README.md（与其他两层测试对称）

## [0.5.2] - 2026-08-27

### Changed
- 全链路指南重构: 从"C++ 单路线 4 步说明"改为"七步抽象流水线"
  （每步: 抽象动作 + 完成标志 + 分级/分路线变体 + 范本映射）
- 新增"步骤 × 样例覆盖矩阵"（14 样例 × 7 步）与"按需求快速选范本"表
- 修正 a1-op README 重复词（"torch torch"）

## [0.5.1] - 2026-08-27

### Fixed
- 根 README 已知问题计数 11→12
- architecture: 全链路样例列表去重（softmax/backward 重复两次）
- architecture: 路线图补 AI 生成源旁路（与根 README/docs index 三处一致）

### Changed
- architecture 目录树同步 intake/ 与 perf/ 目录、14 样例数、perf harness
- docs index 阅读路径新增"性能"与"生成"两条线
- testing/getting-started 挂接性能回归与 intake 产物入口
- testing/perf/intake/reporting/device-profiles/known-issues 补齐"下一步"导航链
- known-issues #10 补短采样不足的实测证据与子进程隔离防护；#11 补 intake 负例自动拦截

## [0.5.0] - 2026-08-27

### Added
- 性能回归追踪: 统一基准 harness（common/perf.py）+ 用例注册表 + 15 个首批用例（6 样例 + kernel 层矩阵格）
- perf_run / perf_compare 脚本: 结构化运行记录（results/perf/runs/）+ 入库基线（perf/baselines/）+ 双档门禁（慢 20% WARN / 慢 30% FAIL）
- AI 生成算子 intake 通道: manifest 契约（JSON Schema）+ 三级验证脚本 + KernelGen 正/负示例 case
- 负例 kernelgen-gelu-no-device-context: 哨兵检查在 intake 阶段拦截 known-issues #11 类静默 no-op
- 文档: 性能回归追踪 / AI 生成算子接入（含生命周期图与阈值语义）
- CI: intake 契约校验 + 性能基线结构检查

### Fixed
- examples/README 性能速览表错行（BMM 行链到 softmax，混入 3 行样例表内容）

## [0.4.0] - 2026-08-27

### Added
- 硬件级开发样例 hw-kernel-example: xtorch_ops 厂商原语组合 + CUDA C++ 参考
- SDK 就绪模板 sdk_template/: XPU kernel 源码 + 编译指南 + 自动检测
- reduction 样例 softmax-fullstack（流式三遍归约 + @triton.autotune）
- backward 样例 backward-example（autograd fwd+bwd + 训练冒烟）
- 样例定位图: 14 个样例映射物理栈层 × 路线
- 物理栈架构图: 应用→框架→编译→算子库→硬件 五层垂直视图

### Changed
- 路线名: aten dispatcher→torch 算子替换 / FlagOS dispatch→FlagOS 融合算子 / vendor backend→厂商算子注册
- 开发级别: FW/TR/HW→torch 级/Triton 级/硬件级
- 验证层级: L0/L2/L4 编号→物理栈层名（算子库层/框架层/应用层）
- 体系图: 四子图平行结构→物理栈垂直视图
- 根 README: 139→61 行（着陆页标准）
- 全文档加"下一步"导航（6 个文档形成阅读链路）
- 核心心智模型在需要处重复出现（不再只靠索引跳转）

### Fixed
- known-issues #12: CUDA C++(NVIDIA) 无法在 P800/XPU 执行
- 报告模板与脚手架同步物理栈视角

## [0.3.0] - 2026-08-26

### Added
- 开发三级重构: torch 级 / Triton 级 / 硬件级
- 样例定位图: 13 个样例直接映射 3×3 矩阵与开发层级
- 算子开发报告模板 + 环境快照 + 骨架生成器
- BMM 全链路样例（Triton 路线）+ 发现 known-issues #11
- CI 工作流（markdown 链接检查 + Python 语法）
- CONTRIBUTING.md / LICENSE (Apache-2.0)

### Fixed
- known-issues #11（裸 Triton 缺 device 上下文→静默 no-op）
- 路线名与开发层级解耦（消除正交性矛盾）
- 跨层一致性结果落盘 / TP 计数文件并发写竞争

## [0.2.0] - 2026-08-25

### Added
- 跨设备黄金输出（CPU transformers / 加速卡 vLLM）
- 漂移实验工具 + 逐 prompt 自适应断言前缀
- 全链路样例 b-fullstack（C++ 路线）+ 开发报告
- 测试输入声明式模板 + token 钳制（规避 embedding 越界）
- 文档体系: docs/ 11 篇中文文档 + 交叉链接 + Mermaid 图

## [0.1.0] - 2026-08-24

### Added
- 初始版本: 3 路线 × 3 层级 = 9 格矩阵 + 跨层一致性
- 设备 profile 芯片泛化（P800 参考实例）
- 厂商 kernel 直测 + C++ JIT 编译闭环 + 哨兵检查
- 11 条已知问题（含 swiglu 不写输出等重大发现）
