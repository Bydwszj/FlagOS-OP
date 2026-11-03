# 算子测试报告: sdpa_math（`aten::_scaled_dot_product_attention_math`）

| 项 | 值 |
|---|---|
| 算子名称 / 路线 / 级别 | `aten::_scaled_dot_product_attention_math` / **A1** / Triton（NPU）+ P800 fast（P800）+ thead fast（thead-ppu）+ torch（CPU/fallback） |
| 目标设备 / 测试日期 | `ascend910` · `p800-kunlunxin` · `thead-ppu` · `cpu` / 2026-09-30~10-02 · C2 补测 10-28 · thead 接入 10-24 · C2 复测 10-30/31 · 验收判定器二轮复审 11-01（离线重判） |
| 结论 | **通过（Must 清单全勾）**；应用证据按 profile 拆分见 §6——p800 FlagOS E2E ☑、thead C1 ☑、thead-ppu-c2 真实 vLLM C2 ☑（#37）；**Ascend910 C2（profile `ascend910-c2-vllm`）：c2-hit（验收状态 complete-pending-review（待审阅）**——同路径三跑 C1 恒等 6/6、性能门禁 1.06 OK，2026-11-01 判定器修复后离线重判仍 exit 0（[c2_ascend910/acceptance_redecide.json](c2_ascend910/acceptance_redecide.json)），见 [c2_ascend910.md](c2_ascend910.md)，待 #38 审阅闭环，未审阅前不写「C2 验收通过」） |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ☑ | `test/kernel_level.py [ascend910\|cpu\|p800-kunlunxin\|thead-ppu]` | PASS：52 组（16 shape × 3 dtype + 全遮蔽/哨兵/dropout/错误路径）×4 profile |
| thead tail 直测 | ☑ | `test/kernel_level_tail_thead_ppu.py` | PASS：47 组（3072/5000/5120/5000×5001 tails × fp32/bf16/fp16 × seeds 11/22/33 + zeros/large/boundary + 4096 floatmask） |
| 框架层 | ☑ | `test/op_level.py [ascend910\|cpu\|p800-kunlunxin\|thead-ppu]` | PASS：22 项（四模式拦截 · 注册=直调逐位 · P800 fused backward · gradcheck · F.sdpa(MATH)） |
| 应用层 | ☑ | `test/framework_level.py [ascend910\|cpu\|p800-kunlunxin\|thead-ppu]` | PASS：8-9 项（mini-decoder + 概率图消费者双跑） |
| 黄金回归 | ☑ | `script/gen_golden.py` + `check_accuracy.py` | **175/175** ×5 实现（含 thead）；原生对照 131/131（44 bool 跳过） |
| 性能门禁 | ☑ | `scripts/perf_run.py --pattern sdpa_math` + `perf_compare.py` | OK：FAIL 0 · WARN 0（triton 0.504ms / 8.524 TFLOPS 入库基线） |
| 语义证据 | ☑ | `probes/native_semantics.py [ascend910\|cpu]` | 6 节全过（schema / dropout 双规则表 / bool 怪癖 / 冲突边界 / F.sdpa 参数形态 / 注册键） |
| 跨层一致性 | ☑ | 黄金生成的三重互验（§3） | 参考 vs `F.sdpa(MATH)` vs 原生直调 vs 自洽式，全一致 |
| FlagOS E2E（C1，p800-kunlunxin） | ☑ | `script/e2e_flagos.py` | PASS：4 层 mini-LLM / 前向 / 反向 / 生成 / 概率图熵正则（§4.1） |
| FlagOS E2E（thead-ppu） | ✗ | `script/e2e_flagos.py` | 缺证：dev 栈无 flag_gems，结构性缺证（同 T1/T2，跟踪 #37） |
| **C2 真实引擎（thead-ppu-c2）** | ☑ | `script/e2e_vllm_c2.py --model ... --model-impl transformers` | PASS：EngineCore 内 MATH 路径 A1 命中 20；token 与同路径基线一致（[e2e_vllm_c2_thead-ppu-c2.json](e2e_vllm_c2_thead-ppu-c2.json)、[thead-ppu-c2.md](thead-ppu-c2.md)） |
| **C2 真实引擎（profile `ascend910-c2-vllm`）** | ☑ | `script/c2_real_vllm_ascend.py --mode accept --profile ascend910-c2-vllm` | **c2-hit（验收状态 complete-pending-review（待审阅））**（2026-10-31 复跑 @ `1dc1d8c`；2026-11-01 判定器修复后离线重判仍 exit 0）：三跑 OK、accept 退出码 0；EngineCore 真实前向 forward_hits=**2880**（correctness 1440 + warmup 240 + measured 1200，阶段归因自洽）+ 主进程 probe 1 独立列示；**同路径（baseline_math vs inject）C1 恒等 6/6 位级一致**、prompt logprob max\|Δ\|=0.170 / mean=0.022 nats（93 token，<0.5 失效线）、性能门禁 inject/baseline_math 中位数比 1.06 OK；native 路径差 5/6 描述性单列。旧版（10-28）无引擎侧补丁为 routing boundary、hits=0。见 [c2_ascend910.md](c2_ascend910.md) |
| 一键 | ☑ | `python3 example.py [ascend910\|cpu\|p800-kunlunxin\|thead-ppu]` | 三层全绿 |

## 2. 环境

Ascend910_9382 · torch 2.10.0+cpu · torch_npu 2.10.0 · triton 3.5.1 ·
flag_gems 5.3.5 · python 3.11.15 · profile `configs/devices/ascend910.yaml`
（环境锁 [configs/env/ascend910.lock.yaml](../../../configs/env/ascend910.lock.yaml)，
`scripts/check_env.py --device ascend910`；历史 development §1 表保留追溯）。
C2 真实引擎栈另有双 profile 锁
[configs/env/ascend910-c2-vllm.lock.yaml](../../../configs/env/ascend910-c2-vllm.lock.yaml)
（editable vllm + vllm-ascend 源码树，profile `ascend910-c2-vllm`）。
P800 复验环境为 torch `2.9.0+cu129` + torch_xmlir
`XMLIR--bc1b1dc6f-dev+2026032411` + FlagGems `4.2.1rc0`，设备 `cuda:1`
（详见 development §1.1）。thead-ppu 复验环境为 torch `2.10.0` +
triton `3.6.0`（flagtree wheel 损坏），dev 容器 `flagos-ppu-wt-dev`，
设备 `cuda:6`（详见 [configs/env/thead-ppu.lock.yaml](../../../configs/env/thead-ppu.lock.yaml)）。
C2 真实引擎环境为镜像 `flagrelease_ppu_vllm024_gems_540:1012`
（vllm 0.24.0 + flag_gems 5.4.0 + vllm-plugin-FL 0.3.0rc2.post3，
profile `thead-ppu-c2`，见 [configs/env/thead-ppu-c2.lock.yaml](../../../configs/env/thead-ppu-c2.lock.yaml)）。

## 3. 精度结果

### 3.1 三方（实为四实现）对照——黄金 175 组

容差：fp32 `1e-5` abs · fp16/bf16 `2e-2` abs · extreme `1e-3` rel。

| 实现 | 结果 | 最差 abs | 备注 |
|---|---|---|---|
| 自研 triton（NPU） | **175/175** | 3.906e-3（bf16） | [reports/accuracy.md](accuracy.md) |
| 自研 torch（CPU） | **175/175** | 1.953e-3（bf16） | |
| P800 fast（cuda:1） | **175/175** | 3.906e-3（bf16） | fp16/bf16 走 vendor O/LSE + exact-P；fp32/mask/dropout 回退 |
| thead fast（cuda:6） | **175/175** | 1.050e-2（bf16） | maskless flash、masked efficient；LSE pad 切片；fp32 回退 torch |
| 参考 reference | **175/175** | 0 | 判卷标准本体 |
| 原生（未注册进程直调） | **131/131** | 9.766e-4（bf16，NPU） | 44 组 bool mask 按规则跳过（§5 #1） |
| FlagGems | 不适用 | — | 无"返回概率图"的同语义实现；其 SDPA 为融合算法、不产出 P（Should 项已注明原因） |

黄金生成时**四重互验**：①`F.sdpa(MATH)` ②原生直调（out+P）③
`out == (P/(1-p))@V` 自洽式（dropout 时）④参考实现，任一不一致即报错
——175 组全部通过后才落盘。

### 3.2 kernel 层直测（52 组/profile，16 shape × 3 dtype + 边界）

| dtype | NPU（triton）max\|Δout\| | CPU（torch）max\|Δout\| | max\|ΔP\|（两 profile） | 容差 | 判定 |
|---|---:|---:|---:|---:|---|
| fp32 | 1.79e-7 | 2.38e-7 | 1.22e-4 | 1e-5（out）/ 同档（P） | ✓ |
| fp16 | 1.95e-3 | 1.22e-4 | 1.22e-4 | 2e-2 | ✓ |
| bf16 | 7.81e-3 | 4.88e-4 | 1.22e-4 | 2e-2 | ✓ |

注：`ΔP` 最差 1.22e-4 出现在 bf16 概率图（量化级），远低于 2e-2。
P800 fast 直测同样 52/52：fp32 回退路径 max `7.15e-7`，fp16
`4.88e-4`，bf16 `3.91e-3`，均低于容差。

**哨兵**：同输入两次逐位一致（确定性）；输入改动输出必变（敏感）；
随机 dropout 同种子 fwd/bwd 完全可复现。

**边界与错误路径**：全遮蔽行 `P=O=0` 无 NaN；`Sq≠Skv`（6×4/4×6）、
`decode Sq=1`、`Hkv=1` 广播、`D=80` 非 2 幂、`S=100/197` 非整倍数；
`causal+mask`（bool/float 皆然）与 `GQA 头不整除` 均按 native 报错。

**dropout 统计判定**：显式掩码下与参考逐位一致（fp32 1e-5 内）且满足
native 怪癖 `O = (P/(1-p))@V`；随机路径 keep 率 0.499（期望 0.5）、
kept 上缩放 2.000（期望 `1/(1-p)`）。

### 3.3 框架层（22 项）

| 检查 | 结果 |
|---|---|
| 拦截计数：grad 开启 / no_grad / **inference_mode** / 无 requires_grad | 4/4 命中（+1 各）；inference_mode 为成对注册的回归项 |
| 注册路径 vs 直接调 impl | **逐位相等** |
| 注册路径 vs 原生基线 | NPU 3.58e-7 / CPU 4.17e-7 |
| gradcheck（fp64，6 组） | base / causal / float-mask ±grad / gqa / explicit-dropout / **概率图消费者 dP** 全过 |
| out-消费者 / P-消费者梯度 vs 参考反传 | 4.3e-7 / 7.5e-8（NPU） |
| P800 fused backward（fp16 causal GQA，dO+dP） | vs native max **3.91e-3** |
| `F.sdpa(MATH)` 拦截 + 输出=参考 | CPU 键命中，err 2.98e-7 |
| 错误路径 | causal+mask、GQA 不整除均抛出 native 同款异常 |
| 拦截总次数 | 3822（gradcheck 逐点调用计入） |

### 3.4 应用层（8 项）

| 指标 | NPU（triton） | CPU（torch） |
|---|---:|---:|
| 消费方命中次数 | 21（前向 3 + 每次反向若干） | 21 |
| logits L∞ vs 原生 | 1.86e-8 | 0 |
| top-1 一致率 | 1.000 | 1.000 |
| 概率图 L∞ | 2.98e-8 | 0 |
| 全权重梯度 L∞（含 dprobs） | 1.75e-10 | 5.82e-11 |
| 贪心序列逐步一致率 | 1.000（8/8） | 1.000（8/8） |
| inference_mode 消费方 | 命中 +2 | 命中 +2 |

## 4. 性能结果

| 实现 | NPU fp16 1k D128 | 基线对比 | 精度代价 |
|---|---:|---|---|
| 自研 triton | 0.515-0.561ms | 基线 | 0（同过 175/175） |
| 自研 A1 包装 | 0.572ms | +0.034ms（autograd.Function） | 0 |
| 原生 math（同返回 out+P） | 0.695ms | 1.29x 慢于自研 | 0（131/131） |
| 原生 F.sdpa（融合，**不返回 P**） | 参考值见 [performance.md](performance.md) | 输出契约不同，不可直接判卷 | 不适用 |

门禁：`perf_compare --device ascend910` → **OK，FAIL 0 · WARN 0**
（triton 0.543ms vs 基线 0.504ms，+7.8%，阈值内）。
本算子延迟 >100µs，不触发微算子 `bench_dispatch` 附加项。

### 2026-10-01 P800 复验

| 层级 | 结果 |
|---|---|
| kernel | 52/52；`p800_fast_level` direct 与 CPU reference 全部对齐 |
| 黄金 | `--impl p800 --device cuda:1` **175/175**，worst 3.906e-3（bf16 容差内） |
| A1 | `AutogradCUDA+CUDA` 成对注册；22/22 项通过（含 P800 fused backward 与 6 组 fp64 gradcheck） |
| 应用层 | `flag_gems.only_enable(['gelu'])` 后 9/9；logits / 概率图 / 梯度 diff 均为 0 |
| perf gate | `ops.sdpa_math.p800` / `.native` 入库；FAIL 0 · WARN 0 |

P800 direct 性能 **1.46-2.21x** native；A1 **1.31-2.17x**。
复现：

```bash
python3 example.py p800-kunlunxin
python3 script/check_accuracy.py --impl p800 --device cuda:1
python3 script/bench_perf.py --device cuda:1 --register \
  --json-out reports/perf_p800-kunlunxin.json
```

### 2026-10-02 FlagOS 端到端复验（p800-kunlunxin · C1 mini consumer）

以下结果仅覆盖 p800-kunlunxin；thead-ppu 因 dev 栈无 flag_gems 未能运行
本入口（结构性缺证，同 T1/T2；其 C1 证据由 framework 层 mini-decoder +
概率图消费者覆盖）。C2 真实引擎（EngineCore/worker）证据本报告无。

入口: [script/e2e_flagos.py](../script/e2e_flagos.py)。原生与 A1 两条链
均运行在 FlagOS 栈内：`flag_gems.only_enable(['gelu'])` 提供模型 MLP 的
GELU，目标 `sdpa_math` 由 A1 拦截。模型为 4 层 Llama 风格 mini-LLM
（D128 · GQA 8/2 · FFN256 · vocab1024），每层直调 private math op，并把
第二输出 `P` 送入注意力熵正则；覆盖 no-grad forward、CE+entropy
backward、8 步贪心生成。

| 检查 | 结果 |
|---|---|
| A1 拦截 | 936 次 |
| logits L∞ | **0.015625**（bf16 阈值 0.02 内） |
| top-1 一致率 | **99.38%**（160 个位置，bf16 语义噪声阈值 ≥98%） |
| 注意力熵差 | **8.82e-4** |
| 8 步贪心序列 | **完全一致** |
| embedding / q0 / head 梯度 L∞ | **1.53e-5 / 3.81e-6 / 1.83e-4** |

原始数据: [e2e_flagos_p800-kunlunxin.json](e2e_flagos_p800-kunlunxin.json)。
端到端延迟见 [performance.md](performance.md) §3.4。训练链分别覆盖
“仅 CE（不消费 P 梯度）”与“CE+概率图熵正则”两条真实路径：
**1.45x / 1.48x**；fused backward off/on A/B 为 **1.50x / 1.45x**。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | `bool attn_mask` 直调的 0/1 加性怪癖 vs 本实现的 `-inf` 遮蔽 | 44 组黄金在 `--impl native` 下跳过；`F.sdpa` 调用方不受影响 | 有意分歧（[development.md](development.md) §6 #1） |
| 2 | 只注册 `Autograd*` 时 inference_mode 不命中 | 推理路径绕过自研实现 | 已修（成对注册），有前后对照证据 |
| 3 | NPU decode/tail 融合前 0.92-0.97x 慢于原生 | Ascend 小形状启动占比高；A1 包装另有 ~0.078ms | **已优化**：单 kernel 融合后直调 1.16-1.17x，A1 仍 0.86x；P800 fast 的 A1 小形状为 1.31-1.53x |
| 4 | `configs/env/ascend910.lock.yaml` 缺失 | `check_env` 无法核对 | 记录在 development §1；黄金/性能数字均注明环境 |
| 5 | `torch.library` 注册不可撤销 | 同进程测试必须先采原生基线 | bench/test/probe 均已按"先基线后注册"实现 |

## 6. 结论

对照 [Must 清单](../../../docs/acceptance.md#levels)：
精度必测矩阵 ☑（非整倍数 shape、三 dtype、特殊用例、≥2 seed/组合）·
哨兵 ☑ · 黄金 100% ☑ · 三层全绿 ☑ · perf 门禁 0 FAIL ☑ ·
报告四件套 ☑ → **可交付**。

应用证据按 profile 拆分（C1/C2 口径见 [docs/testing.md](../../../docs/testing.md#framework)）：

| profile | C1 mini consumer | C2 真实引擎 |
|---|---|---|
| p800-kunlunxin | ☑ framework 层 8-9 项 + FlagOS E2E（4 层 mini-LLM，§4.1） | ✗ 本报告无 |
| thead-ppu | ☑ framework 层 8-9 项（mini-decoder + 概率图消费者）；e2e_flagos.py / run.py --consistency 缺证（dev 栈无 flag_gems） | ✗ 跟踪 #37 |
| **thead-ppu-c2** | —（同硬件 C1 已由 thead-ppu 覆盖） | ☑ 真实 vLLM EngineCore：MATH 路径 A1 命中 20；token 一致（[e2e_vllm_c2_thead-ppu-c2.json](e2e_vllm_c2_thead-ppu-c2.json)） |

Should 项（三方对比中 FlagGems 缺席）已给出
原因：无同语义（返回概率图）实现。
