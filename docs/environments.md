# 软硬件环境版本汇总

[← 返回文档中心](index.md)

这一页汇总当前有交付证据的设备 profile、硬件、系统、Python 和核心软件栈版本。
它是环境声明索引，不是兼容性承诺；性能与精度结论仍必须回到对应算子报告。
本页由 `configs/env/*.lock.yaml` 生成，修改环境后不要手工编辑本页。

<a id="sources"></a>
## 事实来源与使用方式

| 事实 | 事实来源 | 用途 |
|---|---|---|
| 设备映射、dispatch key、框架参数 | [configs/devices/](../configs/devices/) | 运行测试和注册算子 |
| 硬件、系统、Python、软件包版本 | [configs/env/](../configs/env/) | 环境声明与 `check_env` 核对 |
| 单次精度、性能和应用验证 | 算子 `REPORT.md`、`reports/` 与性能 JSON | 限定每条结论的适用范围 |
| 算子 / 平台覆盖状态 | [算子 / 平台矩阵](operator-catalog.md) | 判断某平台是否已有交付算子 |
| 外部基座镜像线索 | [基座镜像索引](base-images.md) | 查看协作方镜像 tag 与分发状态 |

核对当前运行时：

```bash
python3 scripts/check_env.py --device <profile>
```

修改环境声明后重新生成本页：

```bash
python3 scripts/build_env_index.py
python3 scripts/build_env_index.py --check
```

<a id="current"></a>
## 已登记实测环境

<a id="ascend910-c2-vllm"></a>
### Ascend 910 真实 vLLM-Ascend C2 栈（editable vllm + vllm-ascend 源码树）实测版本

| 项 | 声明 |
|---|---|
| profile | [`ascend910-c2-vllm`](../configs/devices/ascend910-c2-vllm.yaml) |
| 平台家族 | `ascend` |
| 硬件 | Ascend 910（910_9382 同代） ×16 · 64 GB HBM/卡 · 运行时: CANN 9.0.0（V100R001C10SPC001B250） · 监控工具: npu-smi · 可见设备: 容器内 16 卡全见；C2 实测用卡 0（其余卡他租户占用勿动） |
| 系统 / 内核 | Linux / 5.10.0-216 |
| Python | `3.11.15` |
| 声明采集时间 | 软件栈 2026-10-30 · 硬件快照 2026-10-30 |
| 环境锁 | [`ascend910-c2-vllm.lock.yaml`](../configs/env/ascend910-c2-vllm.lock.yaml) |
| 基座镜像 | 未登记 |

适用范围：

- ops/embedding C2（真实 vLLM EngineCore 消费 aten::embedding）
- ops/sdpa_math C2（真实 vLLM EngineCore 经 FlagOSMathAttentionBackend 显式 MATH 路径消费 aten::_scaled_dot_product_attention_math）

| 软件包 / 组件 | 锁定版本 |
|---|---|
| `torch` | `2.10.0+cpu` |
| `torch_npu` | `2.10.0` |
| `triton` | `3.5.1` |
| `flag_gems` | `5.3.5` |
| `vllm` | `0.20.2+empty` |
| `transformers` | `5.5.3` |

证据入口：

- [ops/sdpa_math/reports/c2_ascend910/summary.json](../ops/sdpa_math/reports/c2_ascend910/summary.json)：sdpa_math C2 三跑（baseline/baseline_math/inject）摘要
- [ops/embedding/reports/c2_ascend910/summary.json](../ops/embedding/reports/c2_ascend910/summary.json)：embedding C2 双跑摘要
- [ops/sdpa_math/reports/c2_ascend910.md](../ops/sdpa_math/reports/c2_ascend910.md)：sdpa_math C2 叙述报告
- [ops/embedding/reports/c2_ascend910.md](../ops/embedding/reports/c2_ascend910.md)：embedding C2 叙述报告
- [ops/sdpa_math/reports/perf_compare_ascend910_20261030.md](../ops/sdpa_math/reports/perf_compare_ascend910_20261030.md)：算子级性能回归对比（ops.sdpa_math.* 对基线 OK；embedding NEW 未入基线，同文另存 ops/embedding/reports/）

备注：与历史 ascend910.lock.yaml（2026-09-16, vllm 空壳、框架级入口不可用） 的关键差异: 本栈 vllm 为 editable 安装（/vllm-workspace/vllm @ bc150f5 clean），并有 vllm-ascend 源码树（@367b8e62，M csrc/utils/inc/kernel/ moe_distribute_base.h + untracked csrc/build_out/——采集时如实记录， 非本仓库可控）作为真实框架入口；VLLM_PLUGINS=ascend。 C2 harness 要求 VLLM_WORKER_MULTIPROC_METHOD=spawn（fork 后 torch.npu 再初始化失败）；enforce_eager=True 禁 graph replay 绕过 Python 路径。 flag_gems 5.3.5 已安装但 C2 harness 未启用（范围 = torch 组合实现， 非 Triton/FlagGems kernel 对照）。 模型权重: HuggingFaceTB/SmolLM2-135M 快照 93efa2f097d58c2a74874c7e644dbc9b0cee75a2（HF cache 内，不在镜像内）。 卡况（2026-10-30 快照）: C2 用卡 0；1-5 等他租户占用。

<a id="ascend910"></a>
### Ascend 910_9382 SDPA 交付环境

| 项 | 声明 |
|---|---|
| profile | [`ascend910`](../configs/devices/ascend910.yaml) |
| 平台家族 | `ascend` |
| 硬件 | Ascend 910_9382 ×8 · 64 GB HBM/卡 · 计算拓扑: 24 AICore/卡 @ 1800MHz · 运行时: CANN 9.0.0（V100R001C10SPC001B250） |
| 系统 / 内核 | openEuler aarch64 / 5.10.0-216 |
| Python | `3.11.15` |
| 声明采集时间 | 软件栈 2026-09-16 · 硬件快照 2026-09-16 |
| 环境锁 | [`ascend910.lock.yaml`](../configs/env/ascend910.lock.yaml) |
| 基座镜像 | 未登记 |

适用范围：

- ops/sdpa Ascend 910 backend

| 软件包 / 组件 | 锁定版本 |
|---|---|
| `torch` | `2.10.0+cpu` |
| `torch_npu` | `2.10.0` |
| `triton` | `3.5.1` |
| `flag_gems` | `5.3.5` |
| `vllm` | `0.20.2+empty` |

证据入口：

- [ops/sdpa/reports/development.md](../ops/sdpa/reports/development.md)：环境声明与实现报告
- [ops/sdpa/reports/performance.md](../ops/sdpa/reports/performance.md)：性能环境与结果

备注：vLLM 为空壳安装，真实 vLLM 框架级与应用级入口不可用；ops/sdpa 使用 kernel 层、op 层、黄金与性能对照收口，并包含 mini-decoder 应用验证。共享 8 卡服务器的绝对延迟存在 ±15µs 级扰动。

<a id="cambricon"></a>
### Cambricon MLU 本地实测版本（v26.08.0 容器）

| 项 | 声明 |
|---|---|
| profile | [`cambricon`](../configs/devices/cambricon.yaml) |
| 平台家族 | `cambricon` |
| 硬件 | MLU590-M9 ×8 · 98304 MiB/卡 · 驱动: 6.2.29 · 固件: v1.5.0 · 运行时: Neuware 4.7.6 · 监控工具: CNMON v6.5.48 |
| 系统 / 内核 | Ubuntu 24.04 容器 / 未登记 |
| Python | `3.12.3` |
| 声明采集时间 | 软件栈 2026-09-23 · 硬件快照 2026-09-23 |
| 环境锁 | [`cambricon.lock.yaml`](../configs/env/cambricon.lock.yaml) |
| 基座镜像 | 未登记 |

适用范围：

- 13 个基础 ATen 算子的 PR #25 契约复验
- 真实 vLLM 注入链路复验

| 软件包 / 组件 | 锁定版本 |
|---|---|
| `torch` | `2.12.1+cpu` |
| `torch_mlu` | `1.34.1+torch2.12.1` |
| `triton` | `3.4.0+mlu2.1.3` |
| `flag_gems` | `5.0.2` |
| `numpy` | `2.2.6` |
| `vllm` | `0.25.1+empty` |
| `vllm_mlu` | `0.25.1+mlu0.16.0.pt212` |

证据入口：

- [ops/clone/REPORT.md](../ops/clone/REPORT.md)：基础 ATen 算子契约复验入口
- [ops/type_as/reports/development.md](../ops/type_as/reports/development.md)：基础算子原始交付历史环境

备注：容器 cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312 （Neuware 4.7.6，driver 6.2.29）。flag_gems 为本地 FlagGems main 的 editable 安装（SETUPTOOLS_SCM_PRETEND_VERSION=5.0.2，仓库无 tag）。 容器内已装 vLLM 0.25.1+empty / vllm_mlu 0.25.1+mlu0.16.0.pt212（MLUPlatform）， 应用层可跑真实 vLLM（见 ops/<op>/test/framework_level.py）。

历史环境：13 个基础算子 2026-09-18 的原始交付报告使用 Python 3.10.20、 torch 2.7.1+cpu、torch_mlu 1.29.2+torch2.7.1、Triton 3.2.0+mlu1.7.2。 引用原始性能数字时以各算子 reports/development.md 的历史环境为准； 本文件是 2026-09-23 后的契约复验环境。

<a id="mlu590"></a>
### Cambricon MLU590 实测版本

| 项 | 声明 |
|---|---|
| profile | [`mlu590`](../configs/devices/mlu590.yaml) |
| 平台家族 | `cambricon` |
| 硬件 | MLU590-M9 ×8 · 96 GiB/卡 · 驱动: 6.2.29 · 固件: v1.5.0 · 运行时: CNRT 7.7.2 · CNNL 2.2.14 · CNNLExtra 2.4.0 · CNCL 1.30.8 · 监控工具: CNMON v6.5.48 · 可见设备: MLU_VISIBLE_DEVICES=7 → mlu:0 |
| 系统 / 内核 | Linux / 5.15.0-139-generic |
| Python | `3.12.13` |
| 声明采集时间 | 软件栈 2026-09-23 · 硬件快照 2026-09-23 |
| 环境锁 | [`mlu590.lock.yaml`](../configs/env/mlu590.lock.yaml) |
| 基座镜像 | 未登记 |

适用范围：

- ops/sdpa MLU590 backend
- ops/embedding Cambricon backend（算子运行 profile 为 cambricon）
- mini-decoder 应用层验证

| 软件包 / 组件 | 锁定版本 |
|---|---|
| `torch` | `2.11.0+cpu` |
| `torch_mlu` | `1.33.1+torch2.11.0` |
| `triton` | `3.4.0+mlu2.1.1` |
| `flag_gems` | `5.3.5` |

证据入口：

- [ops/embedding/reports/cambricon.md](../ops/embedding/reports/cambricon.md)：embedding Cambricon 平台环境与验证报告
- [reports/examples/sdpa_mlu590_report.md](../reports/examples/sdpa_mlu590_report.md)：SDPA MLU590 环境与实现报告
- [reports/examples/sdpa_mlu590_test_report.md](../reports/examples/sdpa_mlu590_test_report.md)：SDPA MLU590 测试报告

备注：torch 为 CPU 轮子 + torch_mlu 插件加速；MLU590-M9 / MLU_VISIBLE_DEVICES=7。 本环境未安装 vLLM（profile quirks.no_vllm_runtime=true 时跳过框架层入口）。

<a id="p800-kunlunxin-gems5"></a>
### Kunlunxin P800 FlagGems 5.0 复测容器实测版本

| 项 | 声明 |
|---|---|
| profile | [`p800-kunlunxin-gems5`](../configs/devices/p800-kunlunxin-gems5.yaml) |
| 平台家族 | `kunlunxin` |
| 硬件 | P800 OAM ×8 · 98304 MiB/卡 · 驱动: 515.58 · 固件: ALL 1.0.2.43.1.86 · 运行时: XPU-RT 10.2 |
| 系统 / 内核 | Linux / 6.8.0-87-generic |
| Python | `3.10.18` |
| 声明采集时间 | 软件栈 2026-10-16 · 硬件快照 2026-10-16 |
| 环境锁 | [`p800-kunlunxin-gems5.lock.yaml`](../configs/env/p800-kunlunxin-gems5.lock.yaml) |
| 基座镜像 | 未登记 |

适用范围：

- ops/sdpa / sdpa_math / embedding 在 FlagGems 5.0 栈的兼容性复测
- vllm-plugin-FL 0.2.0 dispatch 行为（known-issues #20 / #23）
- 框架级冒烟（冒烟模型缺失，framework 矩阵未复测，见复测页 §5.5）

| 软件包 / 组件 | 锁定版本 |
|---|---|
| `torch` | `2.9.0+cu129` |
| `torch_xmlir` | `1.0.0.1` |
| `xtorch_ops` | `0.1.2935+50a5d6a4` |
| `triton` | `3.0.0+f69a5406` |
| `flag_gems` | `5.0.0` |
| `vllm` | `0.20.2+empty` |
| `vllm_fl` | `0.2.0+g38e7dbc` |
| `transformers` | `4.57.6` |

证据入口：

- [docs/flaggems5-p800-retest.md](../docs/flaggems5-p800-retest.md)：FlagGems 5.0 P800 复测总入口（黄金、A1、mini-consumer、dispatch 与性能证据）

备注：dev 容器 flagos-p800-gems5-vllm0202-dev（非共享镜像正式发布通道）； 尚未登记对应镜像 tag/digest 或构建配方，本锁用于核对已有容器的软件版本， 不作为完整安装清单。中央入口须显式选择 `p800-kunlunxin-gems5`， 独立算子使用各自的本地硬件别名，见 [昆仑芯环境配置](installation.md#kunlunxin)。 flag_gems / vllm-plugin-FL 为 pip 安装，与 4.2 共享镜像的本地路径安装 （/env/FlagGems、/env/xvllm-plugin-FL）不同。框架冒烟模型 /workspace/random-llm-models/llama-3.1-8b-like 在该容器缺失， `--require-model` 会失败，framework 矩阵未复测 （见 docs/flaggems5-p800-retest.md）。硬件快照取自主机 xpu-smi； torch_xmlir 捆绑的 xpu-smi 报告另一组数值（Driver 5.0.21.47 / XPU-RT 5.13.0），以主机工具为准。 `torch_xmlir` 对应发行包名 `xmlir`；`xtorch_ops` 对应发行包名 `XTorch-Ops`；`vllm_fl` 对应发行包名 `vllm-plugin-fl`； `flag_gems` 对应发行包名 `flag-gems`。

<a id="p800-kunlunxin"></a>
### Kunlunxin P800 FlagGems 4.2 基线共享镜像实测版本

| 项 | 声明 |
|---|---|
| profile | [`p800-kunlunxin`](../configs/devices/p800-kunlunxin.yaml) |
| 平台家族 | `kunlunxin` |
| 硬件 | P800 OAM ×8 · 98304 MiB/卡 · 驱动: 515.58 · 固件: ALL 1.0.2.43.1.86 · 运行时: XPU-RT 未暴露版本 |
| 系统 / 内核 | Linux / 6.8.0-87-generic |
| Python | `3.10.18` |
| 声明采集时间 | 软件栈 2026-10-26 · 硬件快照 2026-10-16 |
| 环境锁 | [`p800-kunlunxin.lock.yaml`](../configs/env/p800-kunlunxin.lock.yaml) |
| 基座镜像 | [`kunlunxin-p800-operator-runtime-v2`](base-images.md#kunlunxin-p800-operator-runtime-v2)（版本对齐） |

适用范围：

- ops/sdpa P800 backend
- ops/sdpa_math P800 backend
- ops/embedding P800 backend
- 13 个基础 aten 算子（clone/contiguous/copy_/detach/detach_/dropout/empty/empty_like/empty_strided/item/_local_scalar_dense/result_type/type_as）的 P800 三层验证、黄金与性能门禁
- FlagGems / vLLM 组合使用验证

| 软件包 / 组件 | 锁定版本 |
|---|---|
| `torch` | `2.9.0+cu129` |
| `torch_xmlir` | `1.0.0.1` |
| `xtorch_ops` | `0.1.2640+9bc36a18` |
| `triton` | `3.6.0` |
| `flag_gems` | `4.2.1rc0` |
| `vllm` | `0.13.0` |
| `vllm_fl` | `0.1.0` |
| `transformers` | `4.57.1` |

证据入口：

- [ops/sdpa/reports/p800-kunlunxin.md](../ops/sdpa/reports/p800-kunlunxin.md)：SDPA 平台实现与性能证据
- [ops/sdpa_math/REPORT.md](../ops/sdpa_math/REPORT.md)：sdpa_math 三层验证与 P800 性能证据
- [ops/embedding/REPORT.md](../ops/embedding/REPORT.md)：embedding 三层验证与 P800 性能证据
- [ops/README.md](../ops/README.md)：13 个基础算子已登记 p800-kunlunxin（累计 16 个）
- [ops/type_as/PLATFORM.md](../ops/type_as/PLATFORM.md)：13 算子 P800 绑定/黄金/性能矩阵入口（各算子目录同构）

备注：本锁对应 FlagGems 4.2 基线；5.0 复测栈使用独立的 `p800-kunlunxin-gems5` profile 与环境锁。选择与运行步骤见 [昆仑芯环境配置](installation.md#kunlunxin)。 flag_gems 与 vllm-plugin-fl 在共享镜像中为本地路径安装 （/env/FlagGems、/env/xvllm-plugin-FL），核对版本即可， 不建议 pip 覆盖安装。硬件快照来自 2026-10-16 的 `xpu-smi -q` 与 `scripts/env_snapshot.py --device p800-kunlunxin`。 `torch_xmlir` 对应发行包名 `xmlir`；`xtorch_ops` 对应发行包名 `XTorch-Ops`。 Triton 3.6.0 来自镜像内置的 FlagTree 0.6.1 构建，未安装 dist-info； `scripts/check_env.py` 在缺少发行元数据时回退读取模块 `__version__`。

<a id="thead-ppu-c2"></a>
### T-Head PPU ZW810E 真实 vLLM C2 栈（flag_gems + vllm-plugin-FL）实测版本

| 项 | 声明 |
|---|---|
| profile | [`thead-ppu-c2`](../configs/devices/thead-ppu-c2.yaml) |
| 平台家族 | `thead` |
| 硬件 | PPU-ZW810E ×16 · 98304 MiB/卡 · 驱动: 1.5.1-1d747a · 固件: 未暴露版本 · 运行时: HGGC 13.0（容器内 PPU-SMI 1.28） · 监控工具: ppu-smi · 可见设备: 容器 --privileged 16卡全见；C2 用卡 10,14（实测 10MiB/0% 空闲） |
| 系统 / 内核 | Linux / 5.10.134-17.2.al8.x86_64 |
| Python | `3.12.3` |
| 声明采集时间 | 软件栈 2026-10-28 · 硬件快照 2026-10-28 |
| 环境锁 | [`thead-ppu-c2.lock.yaml`](../configs/env/thead-ppu-c2.lock.yaml) |
| 基座镜像 | 未登记 |

适用范围：

- ops/embedding C2（真实 vLLM EngineCore 消费 aten::embedding）
- ops/sdpa_math C2（真实 vLLM EngineCore 经显式 MATH 路径消费 aten::_scaled_dot_product_attention_math）
- framework 级冒烟（tiny Llama / FLAGOS_MODEL_PATH 覆盖）

| 软件包 / 组件 | 锁定版本 |
|---|---|
| `torch` | `2.11.0+v0.1.0.ppu2.1.0.oe` |
| `triton` | `3.5.0+gitcc5446cf` |
| `flagtree` | `0.7.0+ppu.gitf51dde96` |
| `flag_gems` | `5.4.0+g0d95c2092` |
| `vllm` | `0.24.0+empty` |
| `vllm-plugin-fl` | `0.3.0rc2.post3` |
| `transformers` | `5.19.0` |

证据入口：

- [ops/embedding/reports/e2e_vllm_c2_thead-ppu-c2.json](../ops/embedding/reports/e2e_vllm_c2_thead-ppu-c2.json)：embedding C2 真实 vLLM EngineCore 双跑（hits=10，token 与基线一致）
- [ops/sdpa_math/reports/e2e_vllm_c2_thead-ppu-c2.json](../ops/sdpa_math/reports/e2e_vllm_c2_thead-ppu-c2.json)：sdpa_math C2 真实 vLLM EngineCore 双跑（hits=20，token 与 MATH 基线一致）
- [ops/embedding/reports/thead-ppu-c2.md](../ops/embedding/reports/thead-ppu-c2.md)：embedding C2 叙述报告
- [ops/sdpa_math/reports/thead-ppu-c2.md](../ops/sdpa_math/reports/thead-ppu-c2.md)：sdpa_math C2 叙述报告
- [docs/base-images.md](../docs/base-images.md)：本锁对应容器镜像登记（flagrelease_ppu_vllm024_gems_540:1012）

备注：镜像 flagrelease-public/flagrelease_ppu_vllm024_gems_540:1012（容器 flagos-ppu-c2）。 vllm 与 vllm-plugin-FL 为 editable 安装（/opt/vllm、/opt/vllm-plugin-FL）。 FL 插件激活后默认 attention_backend=flagos→vendor→reference（FA3 wheel）， **默认路径不消费** aten::_scaled_dot_product_attention_math；sdpa_math C2 用 sitecustomize 把 transformers backend 的 ALL_ATTENTION_FUNCTIONS["vllm"] 换成 F.sdpa+MATH 作为显式消费路径（见 ops/sdpa_math/script/e2e_vllm_c2.py）。 模型权重不在镜像内；C2 证据用 _aten_harness.ensure_model() 生成的 tiny Llama /tmp/flagos_tiny_llama（LlamaForCausalLM，hidden 64、2 层）。 卡况（2026-10-28 快照）: 空闲 2,6,7,10,14（10MiB/0%）；4/5/12/13/15 高占用勿用。torch import 版本 2.11.0+v0.1.0.ppu2.1.0.oe（与 pip metadata 一致）。

<a id="thead-ppu"></a>
### T-Head PPU ZW810E dev 基座（未装 FlagGems）实测版本

| 项 | 声明 |
|---|---|
| profile | [`thead-ppu`](../configs/devices/thead-ppu.yaml) |
| 平台家族 | `thead` |
| 硬件 | PPU-ZW810E ×16 · 98304 MiB/卡 · 驱动: 1.5.1-1d747a · 固件: 未暴露版本 · 运行时: HGGC 13.0（容器内 PPU-SMI 1.28；主机 1.17） · 监控工具: ppu-smi（主机不响应 ICMP；连通用 /dev/tcp） · 可见设备: 容器 --privileged 16卡全见（device_count()=16）；PPUS=6,7 仅 env 标记 |
| 系统 / 内核 | Linux / 5.10.134-17.2.al8.x86_64 |
| Python | `3.12.3` |
| 声明采集时间 | 软件栈 2026-10-23 · 硬件快照 2026-10-23 |
| 环境锁 | [`thead-ppu.lock.yaml`](../configs/env/thead-ppu.lock.yaml) |
| 基座镜像 | [`thead-ppu-runtime-v1`](base-images.md#thead-ppu-runtime-v1)（版本对齐） |

适用范围：

- ops/sdpa thead backend（kernel/A1）
- ops/sdpa_math thead backend（kernel/A1）
- ops/embedding thead backend（kernel/A1）

| 软件包 / 组件 | 锁定版本 |
|---|---|
| `torch` | `2.10.0` |
| `triton` | `3.6.0` |
| `flagtree` | `0.6.1+ppu3.6` |
| `vllm` | `0.19.0+cu130` |

证据入口：

- [docs/base-images.md](../docs/base-images.md)：thead-ppu-runtime 登记基座说明（本锁 dev 基座；registry_reference 同机）
- [configs/images/catalog.yaml](../configs/images/catalog.yaml)：thead-ppu-runtime-v1 登记（sdpa 后 version-aligned + related_profiles [thead-ppu]）
- [ops/sdpa/reports/thead-ppu.md](../ops/sdpa/reports/thead-ppu.md)：sdpa thead-ppu 移植与验证报告（三层/黄金/perf 基线证据）
- [ops/embedding/reports/thead-ppu.md](../ops/embedding/reports/thead-ppu.md)：embedding thead-ppu 移植与验证报告（三层/黄金/perf 基线证据）
- [ops/sdpa_math/reports/thead-ppu.md](../ops/sdpa_math/reports/thead-ppu.md)：sdpa_math thead-ppu 移植与验证报告（三层/黄金/perf 基线证据）

备注：本栈无 flag_gems、无 vllm-plugin-fl；A2/framework 级证据由 **thead-ppu-c2** 第二 profile/锁承接（2026-10-28 已交付，镜像 flagrelease_ppu_vllm024_gems_540:1012，含 flag_gems 5.4.0 + 插件 0.3.0rc2.post3 + vllm 0.24.0；仿 p800-kunlunxin 与 p800-kunlunxin-gems5 双 profile 先例， 见 thead-ppu-c2.lock.yaml）。 ⚠️ 基座缺陷：flagtree 0.6.1+ppu3.6 wheel 损坏——pip show 在册但 import ModuleNotFoundError；pip show -f 可见条目仅 dist-info 与 proton 两 bin； torch/triton 不受影响。 卡况（2026-10-23 快照）：4/11 100% 占用、5 占 76G 勿用； 空闲 6,7,8,9,10,13,14,15（2MiB/0%）。 登记基座干净实例（registry_reference 镜像）无 triton/flagtree——本锁钉的是 dev 基座实测实例；sdpa 交付后 thead-ppu-runtime-v1 已改为 version-aligned + related_profiles: [thead-ppu]（主版本线 python 3.12.3/torch 2.10.0/ triton 3.6.0 对齐），锁内 base_images 段同步补关联。 登记基座干净实例组件集与 dev 栈不同（缺 triton/flagtree）， 未验同镜像实例，不作 validated-in-image 声明。

<a id="not-validated"></a>
## 未登记为实测环境

| 对象 | 当前状态 | 边界 |
|---|---|---|
| `cpu` profile | 语义参考与 CPU CI | 无固定硬件环境锁；GitHub runner 版本随 CI 变化，不能转化为加速器结论 |
| `nvidia` profile | profile 就绪，未建立实测环境锁 | 未声明已验证的 NVIDIA 硬件、驱动或 CUDA 版本 |
| A100 80GB SXM | 外部论文参照 | 数据来自 FlashAttention-2 论文图表读取，不是本仓库运行环境；见 [A100 对标](../ops/sdpa/reports/perf_a100.md) |

规划中的厂商家族不预填环境声明。目标范围和当前覆盖以
[算子 / 平台矩阵](operator-catalog.md) 为准；只有拿到实际硬件并完成
设备 profile、环境锁和算子验证后，才加入本页的实测环境列表。

<!-- AUTO-GENERATED FROM configs/env/*.lock.yaml; do not edit. -->
