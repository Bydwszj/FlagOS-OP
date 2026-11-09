# 算子总体报告: sdpa_math（`aten::_scaled_dot_product_attention_math`）

> **职责与证据等级**：本页是当前验证矩阵；`README.md` 负责使用入口，
> `PLATFORM.md` 负责平台绑定，`reports/` 保存带日期的实验和原始数据。
> 当前为 Ascend910/P800/thead-ppu 三个硬件交付，CPU 仅作 ATen 对照。FlagOS E2E 使用
> mini-LLM harness（C1），不等同于真实 vLLM；**已补三处真实 vLLM C2**：
> thead-ppu-c2 显式 MATH 路径（见 [reports/thead-ppu-c2.md](reports/thead-ppu-c2.md)）、
> **2026-10-29 P800 5.0 栈**（下表「真实 vLLM C2」行，issue #39；4.2 栈待复验）
> 与 **2026-10-31 Ascend vLLM-Ascend 栈**（profile `ascend910-c2-vllm`）：
> 同路径三跑 @ `1dc1d8c`、EngineCore 真实前向命中 math 后端
> 2880 次（correctness/warmup/measured 阶段分解），`baseline = 同 MATH 路径
> 无 A1` 的 C1 恒等 6/6 位级一致、prompt logprob max|Δ|=0.17 nats（<0.5
> 失效线）、性能门禁 inject/baseline_math 中位数比 1.06 OK（相对原生
> 注意力 1.2876，描述性）——
> **验收状态 complete-pending-review（证据已采集、待审阅，不写「C2 成立」）**，
> 2026-11-01 判定器修复后离线重判仍 exit 0
> （[reports/c2_ascend910/acceptance_redecide.json](reports/c2_ascend910/acceptance_redecide.json)），
> 详见 [reports/c2_ascend910.md](reports/c2_ascend910.md)。
> 与 native 路径的输出差异为路径差（数值精度差），按预注册口径记录
> （5/6 描述性对照、teacher-forced prompt logprob、首分歧位置），
> 不豁免任何判据。
> 无补丁时的真实路由仍为 `npu_fusion_attention`（known-issues #19）。

> 定位: 一页看全一个算子——实现矩阵、验证矩阵、关键数字。
> 分册在 [reports/](reports/)。

| 项 | 值 |
|---|---|
| 算子名称 | `aten::_scaled_dot_product_attention_math`（SDPA **math 后端**，双输出 `(out, attn_probs)`） |
| 语义 | [reference.py](reference.py)（GQA / causal / bool·float mask / dropout 双规则 / fp32 内部；P800 半精度 fast path 有保守回退） |
| 目标设备 / 路线 | `ascend910` + `p800-kunlunxin` + `thead-ppu` + `cpu` / **A1** ☑（aten 拦截，成对注册 `Autograd*` + 纯设备键） |
| 日期 / 状态 | 2026-10-01 定稿 / 2026-10-12 文档复核 / 2026-10-24 thead-ppu / 2026-10-28 thead-ppu-**c2** 真实 vLLM PASS |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 | 备注 |
|---|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ☑ | 独立 ATen 组合；CPU 交付 + P800 fallback/第二判卷人 |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ☑ | 两段式 `_probs_kernel`(pass1/2 共用 `_score_block`) + `_pv_kernel`；小形状（≤56 score tiles，e2e 实测定标见 [§9](reports/performance.md)）单 kernel 融合；dropout 用 ATen 收口 |
| P800 fast | [kernel/p800_fast_level.py](kernel/p800_fast_level.py) | ☑ | vendor efficient `O/LSE` + `P=exp(scale·QKᵀ-LSE)`；保留双输出契约 |
| thead fast | [kernel/thead_fast_level.py](kernel/thead_fast_level.py) | ☑ | flash(maskless 方阵)/efficient(masked 或非方阵 causal) `O/LSE` + exact-P；efficient LSE 按 32 pad 须切片 |
| 硬件级 | `kernel/hardware_level/` | — | 未做：CANN/厂商绑定不在本交付范围（torch 级已覆盖 CPU） |

`reference.py` 为独立判卷标准；reference / torch / triton / P800 fast /
thead fast 均过同一 175 组黄金。

## 验证矩阵（三层）

| 层级 | 入口 | 结果 | 详细报告 |
|---|---|---|---|
| 一键三层 | `python3 example.py ascend910` / `cpu` / `p800-kunlunxin` / `thead-ppu` | ☑ 全绿（52+21+8/9 项） | — |
| kernel 层 | `test/kernel_level.py` | ☑ 52 组 ×4 profile，哨兵/dropout/错误路径全过 | [reports/accuracy.md](reports/accuracy.md) |
| 框架层 | `test/op_level.py` | ☑ 22 项：四模式拦截、注册=直调（逐位）、P800 fused backward、6 组 gradcheck、F.sdpa(MATH) 拦截 | [reports/test-report.md](reports/test-report.md) |
| 应用层 | `test/framework_level.py` | ☑ 8-9 项：mini-decoder + 概率图消费者；P800 在 FlagGems 栈内运行 | [reports/test-report.md](reports/test-report.md) |
| 黄金回归 | `gen_golden.py` + `check_accuracy.py` | ☑ **175/175**（五实现，含 thead）；原生对照 **131/131**（44 bool 跳过） | [reports/accuracy.md](reports/accuracy.md) |
| 性能 | `script/bench_perf.py`（含 `--register`） | ☑ NPU 1.16-2.03x；CPU 1.05-1.40x；P800 1.46-2.21x；thead **2.38x** native composite | [reports/performance.md](reports/performance.md) |
| 语义证据 | `probes/native_semantics.py` | ☑ 6 节全过（两 profile） | [reports/development.md](reports/development.md) |
| perf 门禁 | `common/perf_registry.py` → `scripts/perf_run.py --pattern sdpa_math` | ☑ 已登记 `ops.sdpa_math`（含 thead-ppu） | [reports/performance.md](reports/performance.md) |
| P800 | `python3 example.py p800-kunlunxin` | ☑ kernel 52 · 黄金 175/175 · A1 22 · FlagGems 应用层 9 | [reports/perf_p800-kunlunxin.json](reports/perf_p800-kunlunxin.json) |
| thead-ppu | `python3 example.py thead-ppu` | ☑ kernel 52 · tail 47 · 黄金 175/175 · A1 21 · framework 8 · perf gate OK | [reports/thead-ppu.md](reports/thead-ppu.md) |
| FlagOS E2E | `python3 script/e2e_flagos.py --device cuda:1` | ☑ 4 层 mini-LLM · forward/backward/generate · A1 拦截 856 | [reports/e2e_flagos_p800-kunlunxin.json](reports/e2e_flagos_p800-kunlunxin.json) |
| **真实 vLLM C2** | `python3 script/e2e_vllm_c2.py --model ... --model-impl transformers` | ☑ EngineCore 内 MATH 路径命中 20 · token 与同路径基线一致 · profile `thead-ppu-c2` | [reports/e2e_vllm_c2_thead-ppu-c2.json](reports/e2e_vllm_c2_thead-ppu-c2.json) · [reports/thead-ppu-c2.md](reports/thead-ppu-c2.md) |
| **真实 vLLM C2（p800 5.0，2026-10-29）** | `tests/framework_level/test_a1_aten_ops.py`（`FLAGOS_A1_OPS=embedding,sdpa_math`） | ☑ 1 passed；等价消费者（真实 prefill 注意力→A1 private math）命中 9（真实消费 8）· 注入前后 token 逐位一致 · 概率图行和偏差 9.9e-3 · 梯度 probe 对 CPU fp64 参照 max_abs≤0.016（issue #39，4.2 侧待复验） | [evidence json](../../tests/framework_level/evidence/p800-kunlunxin-gems5-2026-10-29.json) · [框架级 README](../../tests/framework_level/README.md) |
| **真实 vLLM C2（ascend910，2026-10-31）** | `ops/sdpa_math/script/c2_real_vllm_ascend.py --mode accept`（profile `ascend910-c2-vllm`） | ☑ 同路径三跑 token 6/6 位级一致 · MATH 路径命中 2880 · prompt logprob max\|Δ\|=0.17 nats · 性能比 1.06 OK · **验收状态 complete-pending-review（证据已采集、待审阅，不写「C2 成立」）** | [reports/c2_ascend910.md](reports/c2_ascend910.md) · [acceptance_redecide.json](reports/c2_ascend910/acceptance_redecide.json) |

## 关键数字

| 指标 | 值 | 对照 |
|---|---|---|
| 黄金最差 abs | 3.91e-3（triton 与 P800 fast/bf16）· 1.95e-3（torch/native/bf16） | 容差 bf16=2e-2；fp32 ≤1.2e-7 |
| 原生一致性 | 131/131 PASS，worst 9.77e-4（NPU） | `--impl native`，bool 用例 44 组按规则跳过 |
| 梯度 | 6 组 fp64 gradcheck（base/causal/float-mask±grad/gqa/explicit-dropout/dP 消费者） | 数值差分 |
| 延迟（NPU fp16 vs 原生 math） | 2k D128 **1.369ms / 2.776ms = 2.03x**；1k D128 1.29x；GQA 1.67x；decode **1.17x**；tail **1.16x** | 同为"返回 out+概率图"的公平口径 |
| A1 包装开销 | 1k D64 直调 0.515ms → A1 0.555ms；decode 直调 0.217ms → A1 0.295ms（≈0.078ms） | 1k 仍 1.13x 于原生；decode A1 0.86x（小形状包装占比高） |
| P800 fp16 | direct **1.46-2.21x** native；A1 **1.31-2.17x** | 同为返回 out+P 的公平口径 |
| thead-ppu fp16 | direct **2.38x** native composite（0.328 vs 0.782ms）；A1 ~1.00x（注册后 native 即本实现） | 同为返回 out+P；native 列为未注册 composite |
| P800 no-P 参考 | exact-contract math 比 `F.sdpa` 慢 **1.98-4.46x** | **非同输出契约**：`F.sdpa` 不返回/物化 P，只作融合上限参考 |
| A100 绝对性能（外部参照） | 同契约 private-math 未测；非等价 no-P 口径下 P800 vendor efficient 为 A100·FA2 的 **36-53%** | 论文读数、非同机复测；不能折算成 exact-P 加速比 |
| FlagOS E2E | forward **1.07x**；train-out **1.45x**；train+probs **1.48x**；generate **1.09x** | fused backward off/on **1.45-1.50x**，覆盖 dO+dP |
| 哨兵 | 确定性 ☑ 敏感 ☑（dropout 随机路径同种子可复现 ☑） | |

## 结论与遗留

**可交付**: 三层全绿、黄金与原生对照双过、A1 拦截在 grad/no_grad/
inference_mode/无 requires_grad 四种模式全部命中，梯度（含 `dprobs`
通路）通过 gradcheck 与参考反传双验证。

遗留（有意为之，均有注释与证据）:
1. `bool attn_mask` **直调**语义按 `-inf` 遮蔽实现（对齐 `F.sdpa`），
   与 native 直调的 0/1 加性怪癖有意分歧——44 组黄金在 `--impl native`
   下跳过；走 `F.sdpa` 的消费方行为完全一致。
2. decode/tail 的 **A1 包装路径** 0.86x（直调经单 kernel 融合已
   1.16-1.17x 反超）：`autograd.Function` 包装 ~0.078ms 对 0.29ms
   级调用占比高，属 torch 图节点开销，非 kernel 问题；P800 fast path
   的 A1 小形状为 1.31-1.53x。
3. `dropout_p>0` 且走 A1 路径时，`out=(P)@V` 用 ATen matmul 而非
   fused PV（dropout 不是本算子的性能路径，`register.py` 已注明）。

## 交付物清单

| 交付物 | 位置 |
|---|---|
| 开发报告（7 章） | [reports/development.md](reports/development.md) |
| 测试报告（范围矩阵） | [reports/test-report.md](reports/test-report.md) |
| 精度分册 | [reports/accuracy.md](reports/accuracy.md) |
| 性能分册 | [reports/performance.md](reports/performance.md) |
| native 语义证据 | [probes/native_semantics.py](probes/native_semantics.py) |
| 性能原始数据 | [reports/perf_ascend910.json](reports/perf_ascend910.json) · [reports/perf_cpu.json](reports/perf_cpu.json) |
| P800 性能数据 | [reports/perf_p800-kunlunxin.json](reports/perf_p800-kunlunxin.json) · [reports/perf_f_context_p800-kunlunxin.json](reports/perf_f_context_p800-kunlunxin.json) |
| FlagOS E2E 数据 | [reports/e2e_flagos_p800-kunlunxin.json](reports/e2e_flagos_p800-kunlunxin.json) |
| 仓库已知问题登记（七类语义/注册/平台分化） | [docs/known-issues.md #19](../../docs/known-issues.md) |
