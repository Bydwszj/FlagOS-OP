# 算子总体报告: embedding（aten::embedding）

> **当前汇总**：本页当前交付为 P800/Kunlunxin、Cambricon MLU590、
> T-head PPU ZW810E（dev + **C2 真实 vLLM** 双栈）与
> **ascend910-c2-vllm（C2 证据已注册：真实 vLLM-Ascend 前向命中 `aten::embedding`
> 97 次，2026-10-31 复跑 @ `1dc1d8c`，C1 恒等 6/6、性能门禁 0.94 OK，
> 验收状态 complete-pending-review（待审阅，不写「C2 成立」）；2026-11-01
> 判定器修复后离线重判仍 exit 0（reports/c2_ascend910/acceptance_redecide.json）——见
> [reports/c2_ascend910.md](reports/c2_ascend910.md)）**。
> 旧 dev profile `ascend910` 保留 C0/C1 历史实验范围，不持有 C2 声明。
> Ascend910 的三层/黄金（C0/C1）行是 2026-09-30 历史实验记录，保留追溯。
> 当前验证以本页各平台表和对应 `reports/*.json` 为准；`README.md` 负责
> 使用入口，`PLATFORM.md` 负责平台绑定，详细实验保留在 `reports/`。

| 项 | 值 |
|---|---|
| 算子名称 | `aten::embedding`（稠密查表 + dense 反向） |
| 语义 | [reference.py](reference.py)（`weight[index]` 任意 index 形状；padding/scale_grad/sparse 不改前向） |
| 当前硬件平台 | **4 个**：p800-kunlunxin（厂商委托）· cambricon / MLU590（厂商委托）· thead-ppu / PPU ZW810E（厂商委托 + C2）· ascend910-c2-vllm（**C2 证据已注册，验收状态 complete-pending-review（待审阅）**；dev profile `ascend910` 为历史范围） |
| 路线 | **A1**（aten 拦截） |
| 日期 / 状态 | 2026-10-23 三平台交付；2026-10-28 thead-ppu C2 PASS；Ascend910 C2 证据 2026-10-31（**complete-pending-review**，2026-11-01 离线重判 exit 0，三层历史记录见下方） |

## Ascend910 记录（C0/C1 历史实验 + C2 当前证据）

Ascend910 的 C2 当前证据在 [reports/c2_ascend910.md](reports/c2_ascend910.md)
（profile `ascend910-c2-vllm`，真实 vLLM-Ascend 前向命中 97 次、
输出逐位一致、性能门禁 OK，complete-pending-review；2026-11-01
离线重判 exit 0）。以下表格为 2026-09-30
的 C0/C1 实验记录（dev profile `ascend910` 范围），保留用于追溯。

| 算子 | `aten::embedding` |
| 范围 | 普通查表 forward + dense backward；`sparse=True` forward 可用，稀疏 backward 不覆盖 |
| 硬件平台 | **Ascend910**（历史实验，`torch_npu`） |
| 路线 | A1：`AutogradPrivateUse1` |
| 日期 / 状态 | 2026-09-30 / 历史记录 |

## 实现矩阵（当前 P800/MLU 与历史 Ascend）

表中的 P800 和 Cambricon 行属于当前交付；Ascend910 行仅用于历史追溯。

| 平台 | 级别 | 文件 | 生产路径 | 状态 |
|---|---|---|---|---|
| p800-kunlunxin | 厂商委托 / Triton 探针 | kernel/p800_kunlunxin.py · kernel/triton_level.py | `aten::index_select` 委托 | ✅ |
| **ascend910** | 厂商委托 | **kernel/ascend910.py** | `aten::index_select`（CANN 行采集，实测 0.185ms@131k×4k×16k 与 F.embedding 同源同速）· dense 反向委托 `aten::embedding_dense_backward` + scale_grad_by_freq 逐出现逆频率缩放 | ✅ |

Triton gather 在两平台均慢数十倍（P800 4.6-69x、Ascend 探针见
reports/performance.md），保留为可复现平台探针——embedding 的本质是
gather，行采集是厂商库强项，**生产委托是两平台一致的工程结论**。

## 验证矩阵（历史 Ascend 与 P800 复验记录）

本节的 Ascend910 行属于历史实验，P800 行属于当前复验；Cambricon 结果见下方独立矩阵。

| 层级 | 入口 | 结果 |
|---|---|---|
| 层 | 命令 | 结果 |
|---|---|---|
| kernel 层（ascend910） | `EMBEDDING_PROFILE=ascend910 python3 test/kernel_level.py` | ✅ 20 前向 + 6 反向 case，err=0.0（委托原生本征精确） |
| op 层（ascend910） | `test/op_level.py` | ✅ 拦截 5 次 · 注册=直调逐位 · dense/scale_freq 梯度 err=0.0 · sparse bwd 守卫 |
| 应用层（ascend910） | `test/framework_level.py` | ✅ nn.Embedding+TokenMLP 消费: logits diff=0 · top1=1.0 · 续写一致率 1.0 |
| 黄金（ascend910） | `gen_golden.py` + `check_accuracy.py --impl ascend` | ✅ 350 PASS（data 本地生成） |
| 一键（ascend910） | `EMBEDDING_PROFILE=ascend910 python3 example.py` | ✅ 三层全绿 |
| kernel forward（p800） | `test/kernel_level.py` | 20/20，FP32/FP16/BF16 全部 0 error |
| kernel backward（p800） | 同上 | 6/6；重复 index、padding、`scale_grad_by_freq` 全部 0 error |
| 黄金（p800） | `script/check_accuracy.py --impl p800` | 174/174，worst=0 |
| A1 op（p800） | `test/op_level.py` | 拦截 5 次；hooked=direct 逐位；dense/scale/sparse 边界全绿 |
| 应用层（p800） | `test/framework_level.py`（FlagGems enabled） | `nn.Embedding` + MLP；logits/梯度 0 diff；贪心一致率 1.00 |
| 性能（p800） | `script/bench_perf.py` | 大 shape native parity；Triton 显著慢 |
| dispatch（p800） | `script/bench_dispatch.py` | **作废，待复测**（旧脚本未持有 `torch.library.Library`，测得未注册状态） |
| gatherFIX 对照（p800） | `script/probe_gatherfix.py` | 正确性 0 error；P800 特化比 native 慢 10.9-177.9x |

| 平台 / 级别 | 文件 | 状态 | 说明 |
|---|---|---|---|
| P800 厂商委托 | [kernel/p800_kunlunxin.py](kernel/p800_kunlunxin.py) | ✅ | forward 走 native `index_select`；dense backward 走 native `embedding_backward`；XPU 缺 `scale_grad_by_freq` → 自实现 inverse-frequency 补偿 |
| Cambricon MLU 委托 | [kernel/cambricon.py](kernel/cambricon.py) | ✅ | forward 走 native `index_select`；backward 走 native `embedding_backward`（`padding_idx`/`scale_grad_by_freq` 完整，无需补偿）；`sparse=True` backward 显式拒绝 |
| T-head PPU 委托 | [kernel/thead_ppu.py](kernel/thead_ppu.py) | ✅ | forward `index_select`；backward 走 native `embedding_backward`（原生完整、零补偿）；sparse 拒绝 |
| 多平台门面 | [kernel/platform.py](kernel/platform.py) | ✅ | 按 `tensor.device.type` 路由后端（cuda 双候选按栈身份，进程级缓存），附 `dispatch_key_for` / `synchronize` |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ✅ 探针 | 正确但比 native gather 慢，未接入生产 |
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ | 独立 index-select 参考/对照 |
| 硬件级 | [kernel/hardware_level/README.md](kernel/hardware_level/README.md) | ⬜ 置空 | native row-gather 已达到生产水位，暂无重写收益 |

## 验证矩阵（三层，P800）

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel forward | `test/kernel_level.py p800-kunlunxin` | 20/20，FP32/FP16/BF16 全部 0 error |
| kernel backward | 同上 | 6/6；重复 index、padding、`scale_grad_by_freq` 全部 0 error |
| 黄金 | `script/check_accuracy.py --impl p800` | 174/174，worst=0 |
| A1 op | `test/op_level.py` | 拦截 5 次；hooked=direct 逐位；dense/scale/sparse 边界全绿 |
| 应用层 | `test/framework_level.py`（FlagGems enabled） | `nn.Embedding` + MLP；logits/梯度 0 diff；贪心一致率 1.00 |
| 性能 | `script/bench_perf.py` | 大 shape native parity；Triton 显著慢 |
| dispatch | `script/bench_dispatch.py` | native / direct 行有效；**A1 行作废待复测**（见遗留 4） |

## 验证矩阵（三层，cambricon / MLU590）

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel forward | `test/kernel_level.py cambricon` | 20/20，FP32/FP16/BF16 全部 0 error |
| kernel backward | 同上 | 6/6 全部 `err=0`（含 `scale_grad_by_freq`，原生实现）；守卫 2 项通过 |
| 黄金 | `script/check_accuracy.py --impl cambricon --device mlu:0` | 174/174，worst=0（逐位） |
| A1 op | `test/op_level.py cambricon` | 拦截 5 次；hooked=direct 逐位；dense/scale grad 0 error；sparse fwd 接受 / bwd 拒绝 |
| 应用层 | `test/framework_level.py cambricon` | 拦截 8 次；logits 0 diff；贪心一致率 1.00；FlagGems gelu 生效 |
| 性能 | `script/bench_perf.py --device mlu:0` | 见下表；gate `FAIL 0 · WARN 0 · NEW 0` |
| dispatch | `script/bench_dispatch.py --device mlu:0` | A1 − native ≈ **+0.059ms**（3 轮 0.059/0.060/0.061） |
| 单测 | `pytest tests/unit -q` | 20 passed |

## 验证矩阵（三层，thead-ppu / PPU ZW810E）

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel forward | `test/kernel_level.py thead-ppu` | 20/20，FP32/FP16/BF16 全部 0 error |
| kernel backward | 同上 | 6/6 全部 `err=0`（原生直通, 含 `scale_grad_by_freq`）；守卫 2 项通过 |
| tail 补充（§2d） | `test/kernel_level_tail_thead_ppu.py thead-ppu` | 37/37：3072/5000/5120 × 3 dtype × 3 seed + zeros/large/boundary 全 err=0 |
| 黄金 | `script/check_accuracy.py --impl thead-ppu --device cuda:6` | 174/174，worst=0（逐位） |
| A1 op | `test/op_level.py thead-ppu` | 拦截 5 次；hooked=direct 逐位；dense/scale grad 0 error；sparse fwd 接受 / bwd 拒绝 |
| 应用层 | `test/framework_level.py thead-ppu` | 拦截 8 次；logits 0 diff；贪心一致率 1.00（flag_gems 缺证, 纯 A1 拦截口径） |
| **C2 真实 vLLM** | `script/e2e_vllm_c2.py`（profile `thead-ppu-c2`） | PASS：EngineCore 内 A1 命中 10；token 与基线一致（[reports/e2e_vllm_c2_thead-ppu-c2.json](reports/e2e_vllm_c2_thead-ppu-c2.json)） |
| 性能 | `script/bench_perf.py --device cuda:6` | 见下表；gate `FAIL 0 · WARN 0` |
| dispatch | `script/bench_dispatch.py --device cuda:6` | A1 − native ≈ **+0.016ms** |
| 单测 | `pytest tests/unit -q` | 46 passed |
| 平台报告 | [reports/thead-ppu.md](reports/thead-ppu.md) | 首测/验证/性能全记录 |

## 性能速览（fp16）

| shape | ours（index_select） | native F.embedding | Triton 探针 |
|---|---|---|---|
| prefill_16k_d128 | 0.081ms | 0.060ms | 1.45ms（18x 慢） |
| long_131k_d128 | 0.152ms | 0.095ms | 10.4ms（69x 慢） |
| backward 16k | 0.197ms | 0.192ms | —（0.98x） |

完整数据: reports/perf_fp16_ascend910.json。

P800 forward（fp16）：

| shape | ours | native | Triton | ours/native |
|---|---:|---:|---:|---:|
| 1k×D128 | 0.0680ms | 0.0358ms | 0.2403ms | 0.526x |
| 16k×D128 | 0.0514ms | 0.0494ms | 2.2206ms | 0.963x |
| 131k×D128 | 0.1665ms | 0.1530ms | 16.9373ms | 0.919x |
| 16k×D512 | 0.0616ms | 0.0548ms | 8.7184ms | 0.889x |
| vocab128k 16k×D128 | 0.0514ms | 0.0515ms | 2.2458ms | 1.001x |

P800 backward 16k×D128：

## A1 注册点（平台差异，sdpa 开发报告 §3.1 证据链复用）

torch_npu 栈拦截点为 **AutogradPrivateUse1**（PrivateUse1 永不命中）；
register_a1 已按 dispatch_key 自动选择 backend（p800→AutogradCUDA /
ascend→AutogradPrivateUse1），或显式 `platform="ascend910"`。

cambricon forward（fp16，[perf_fp16_cambricon.json](reports/perf_fp16_cambricon.json)）：

| shape | ours | native | Triton | ours/native |
|---|---:|---:|---:|---:|
| 1k×D128 | 0.0681ms | 0.0445ms | 0.2995ms | 0.654x |
| 16k×D128 | 0.0612ms | 0.0451ms | 2.3029ms | 0.736x |
| 131k×D128 | 0.0819ms | 0.0662ms | 失败（grid 65536>65535） | 0.809x |
| 16k×D512 | 0.0550ms | 0.0617ms | 10.5459ms | **1.123x** |
| vocab128k 16k×D128 | 0.0525ms | 0.0404ms | 2.3306ms | 0.769x |

cambricon backward 16k×D128：

| mode | ours | native | ours/native |
|---|---:|---:|---:|
| dense | 0.2651ms | 0.2497ms | 0.942x |
| scale_grad_by_freq | 0.2944ms | 0.2796ms | 0.950x |

性能门禁（cambricon）：`ops.embedding.cambricon.forward` 0.060ms / 142.8 GB/s，
`ops.embedding.native.forward` 0.045ms / 187.9 GB/s；`perf/baselines/cambricon.json`
共 33 条 case。

thead-ppu forward（fp16，[perf_fp16_thead-ppu.json](reports/perf_fp16_thead-ppu.json)）：

| shape | ours | native | ours/native |
|---|---:|---:|---:|
| 1k×D128 | 0.0286ms | 0.0239ms | 0.833x |
| 16k×D128 | 0.0359ms | 0.0307ms | 0.855x |
| 131k×D128 | 0.1033ms | 0.0982ms | 0.951x |
| 16k×D512 | 0.0359ms | 0.0307ms | 0.855x |
| vocab128k 16k×D128 | 0.0360ms | 0.0275ms | 0.764x |

thead-ppu backward 16k×D128：

| mode | ours | native | ours/native |
|---|---:|---:|---:|
| dense | 0.1669ms | 0.1597ms | 0.957x |
| scale_grad_by_freq | 0.1842ms | 0.1771ms | 0.962x |

性能门禁（thead-ppu）：`ops.embedding.thead-ppu.forward` 0.033ms / 254.97 GB/s，
`ops.embedding.native.forward` 0.030ms / 286.90 GB/s；`perf/baselines/thead-ppu.json`
共 6 条 case（含 T1 sdpa 4 条）。

## 结论与遗留

## 结论

当前三平台按 MERGE 流程合入：生产路径与 P800 同构（厂商委托），
三层 + 黄金 + 性能全绿，委托实现 err=0.0。Triton 探针数据进一步
佐证各平台一致的结论——embedding 峰值属厂商库，Triton 价值在
长尾与可移植（若未来需要自定义 padding 语义/量化查表再启用）。

第二平台（cambricon）与第三平台（thead-ppu）进一步印证：**原生算子能力
决定实现复杂度**——MLU/PPU 的 `embedding_backward` 完整，后端因此是纯
委托（零补偿代码）；XPU 缺 `scale_grad_by_freq`，P800 后端才需要
inverse-frequency 补偿。三平台共用同一套三层测试与 174 条黄金数据，
逐位判据一致。thead-ppu 的接入同时验证了 cuda 双候选路由（XMLIR/PPU
按栈身份探测）——未来再有伪装 CUDA 的栈按 MERGE.md 探测链扩展即可。

## 交付物清单

| 内容 | 位置 |
|---|---|
| 需求 | [requirement.md](requirement.md) |
| 语义参考 | [reference.py](reference.py) |
| P800 backend | [kernel/p800_kunlunxin.py](kernel/p800_kunlunxin.py) |
| Cambricon backend | [kernel/cambricon.py](kernel/cambricon.py) |
| T-head PPU backend | [kernel/thead_ppu.py](kernel/thead_ppu.py) |
| 多平台门面 | [kernel/platform.py](kernel/platform.py) |
| A1 注册 | [register.py](register.py) |
| 一键三层 | [example.py](example.py) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度报告 | [reports/accuracy.md](reports/accuracy.md) |
| 性能报告 | [reports/performance.md](reports/performance.md) |
| 开发报告 | [reports/development.md](reports/development.md) |
| cambricon 平台报告 | [reports/cambricon.md](reports/cambricon.md) |
| thead-ppu 平台报告 | [reports/thead-ppu.md](reports/thead-ppu.md) |
| cambricon 性能原始数据 | [reports/perf_fp16_cambricon.json](reports/perf_fp16_cambricon.json) |
| cambricon dispatch 原始数据 | [reports/dispatch_cambricon.json](reports/dispatch_cambricon.json) |
| thead-ppu 性能原始数据 | [reports/perf_fp16_thead-ppu.json](reports/perf_fp16_thead-ppu.json) |
| thead-ppu dispatch 原始数据 | [reports/dispatch_thead-ppu.json](reports/dispatch_thead-ppu.json) |
| perf 门禁基线 | `perf/baselines/cambricon.json`（33 case）、`perf/baselines/thead-ppu.json`（6 case） |

## 遗留

1. `sparse=True` backward / sparse weight gradient 不在本期范围（MLU 侧显式
   `NotImplementedError`，避免 torch_mlu 返回 COO 稀疏梯度）；
2. `scale_grad_by_freq=True` 是能力补齐路径：P800 比 dense backward 慢约 47%，
   MLU 慢约 11%；
3. Triton gather 仅保留为平台证据，不作为生产实现（MLU 上 131k indices 还会
   撞 grid 65535 上限）；
4. **P800 的 A1 dispatch 数字作废待复测**：历史 `+0.0241ms` 出自
   `bench_dispatch.py` 未持有 `torch.library.Library` 的缺陷（析构即反注册，
   实测的是未注册状态），本机无 `torch_xmlir` 无法重测；native / direct 两行
   仍然有效。详见 [reports/cambricon.md](reports/cambricon.md) §7。
