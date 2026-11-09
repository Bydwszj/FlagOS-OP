# FlagGems 5.0 兼容性复测（P800 / 昆仑芯）

[← 返回文档中心](index.md)

本页记录仓库 P800 算子成果在 **FlagGems 5.0 栈**上的兼容性复测：范围、
数据、暴露的问题与修复状态。问题条目与检测方法见
[已知问题清单](known-issues.md)，性能口径见
[性能回归追踪](performance-regression.md)。

首次配置环境与选择命令入口见[昆仑芯环境配置](installation.md#kunlunxin)；
本页保留复测时的结果，后续文档修正不代表新增真机验证。

| 项 | 值 |
|---|---|
| 日期 | 2026-10-16 |
| 容器 | `flagos-p800-gems5-vllm0202-dev` |
| 被测仓库 | FLAG-OP/FlagOS-OP；`baseline_commit: 4d42567`（基线对照点），`tested_commit: cb84eef`（复测在该提交树上完成；其后为 review 整改提交——文档口径、auto_detect 探测边界与单测，数值结论不变，受影响入口在 `9133a93` 定向核验） |
| 复测分支 | `docs/flaggems5-p800-retest` |
| 对照基线 | 仓库内 4.2.1rc0 环境锁的报告 / 黄金 / `perf/baselines` |
| 硬件 | Kunlunxin P800 OAM ×8（测试用空闲卡，`xpu-smi` 确认） |
| 环境登记 | [`p800-kunlunxin-gems5`](../configs/devices/p800-kunlunxin-gems5.yaml) profile + [环境锁](../configs/env/p800-kunlunxin-gems5.lock.yaml)（4.2 基线锁 `p800-kunlunxin` 保留） |
| 总结论 | **已完成的 kernel/A1/mini-consumer 证据与 4.2 基线一致；推理/生成性能一致；训练收益收窄（xtorch_ops/plugin 变化，非 FlagGems）；#20/#21 已修、#22 样例侧规避（根因未修）、#23 仅记录（A2 BLOCKED）、framework 矩阵未复测（§5.5）；A1-aten 真实 vLLM C2（embedding/sdpa_math）已用随机权重 random-qwen3-tiny 补齐（issue #39，见 §5.5 备注）** |

## 1. 环境对照

本复测栈已正式登记为 `p800-kunlunxin-gems5`（profile + 环境锁）：
`python3 scripts/check_env.py --device p800-kunlunxin-gems5` 当前 9/9 全绿。
下表为与 4.2 共享镜像基线锁（`p800-kunlunxin`）的对照：

| 包 | 4.2 锁定 | 当前 5.0 | 判定 |
|---|---|---|---|
| torch | 2.9.0+cu129 | 2.9.0+cu129 | ✓ |
| torch_xmlir | 1.0.0.1 | 1.0.0.1 | ✓ |
| python | 3.10.18 | 3.10.18 | ✓ |
| **flag_gems** | 4.2.1rc0 | **5.0.0** | ✗ 预期 |
| **vllm** | 0.13.0 | **0.20.2+empty** | ✗ 预期 |
| **vllm_fl** | 0.1.0 | **0.2.0+g38e7dbc** | ✗ 预期 |
| **xtorch_ops** | 0.1.2640+9bc36a18 | **0.1.2935+50a5d6a4** | ✗ 需注意 |
| triton | 3.0.0+03c4c9be | 3.0.0+f69a5406 | ✗ 构建号 |
| transformers | 4.57.1 | 4.57.6 | ✗ 轻微 |

FlagOS-OP 的 P800 算子（embedding / sdpa / sdpa_math）生产路径为
**A1 aten 拦截（`AutogradCUDA`）+ 厂商委托**（`index_select` /
`aten._scaled_dot_product_efficient_attention`）；应用层测试仅启用
`flag_gems.only_enable(include=["gelu"])`，不接管 sdpa/embedding 本体。
FlagGems 4.2→5.0 对这三个算子的数值路径几乎无直接影响，主要风险在
**vllm-plugin-FL 0.2.0 分发策略**与 **xtorch_ops 版本**——复测证实这两处，
见 §4。

## 2. 正确性：已完成范围与 4.2 一致

### 2.1 三层测试（`example.py` 一键）

| 算子 | 层级 | 结果 |
|---|---|---|
| embedding | kernel | 20 前向 + 6 反向，FP32/FP16/BF16 全部 err=0 |
| embedding | A1 op | 拦截 5 次；hooked=direct 逐位；dense/scale_freq grad err=0 |
| embedding | 应用层 | nn.Embedding+MLP；logits/grad diff=0；贪心一致率 1.00 |
| sdpa | kernel | 37 case，max_err=3.9e-3（bf16 容差内） |
| sdpa | A1 op | 注册=直调逐位；vs native max_diff=0；grad 检查通过 |
| sdpa | 应用层 | mini-decoder 4L；logits diff=0；top1=1.00 |
| sdpa_math | kernel | 52 case，max_err=1.95e-3 / P 3.91e-3；哨兵/dropout/错误路径全过 |
| sdpa_math | A1 op | 22 项（含 6 组 gradcheck、F.sdpa(MATH) 拦截、fused backward） |
| sdpa_math | 应用层 | intercept=21；logits/∇ L∞=0；贪心一致率 1.00 |

### 2.2 黄金精度

| 算子 | 组数 | 5.0 结果 | 4.2 报告 | 对齐 |
|---|---|---|---|---|
| embedding | 174 | **174/174, worst=0** | 174/174, worst=0 | ✓ 逐位 |
| sdpa_math | 175 | **175/175, worst=3.906e-3** | 175/175, worst=3.91e-3 | ✓ |
| sdpa | 397 | **397/397, worst=4.25e-3** | （同容差口径） | ✓ |

### 2.3 FlagGems 5.0 API 兼容与单测

- `flag_gems.only_enable(include=["gelu"])` ✓；enable 后
  `flag_gems.current_work_registrar.include_ops` 非 None ✓
- FlagGems gelu（bf16）vs native：L∞=0
- 5.0 打印 `Overriding a previously registered kernel ... aten::gelu`
  警告——属正常覆盖，非错误
- `pytest tests/unit -q`：`cb84eef` 数值复测时 **44 passed**；
  `9133a93` 增加两条自动探测单测后 **46 passed**，最终文档提交
  `9ed1aa6` 的 CI 也通过 46 项（见 [PR32 核验记录](https://github.com/FLAG-OP/FlagOS-OP/pull/32#issuecomment-6230636011)）

### 2.4 e2e 数值（vLLM 栈 mini-LLM，与 4.2 同种子逐项相同）

| 指标 | 4.2 报告 | 5.0 复测 |
|---|---|---|
| logits_linf | 0.015625 | **0.015625** |
| top1_rate | 0.99375 | **0.99375** |
| entropy_abs | 8.816719055175781e-4 | **相同** |
| greedy_sequence_match | True | **True** |
| embedding_grad | 1.526e-05 | **1.526e-05** |
| q0_grad / head_grad | 3.81e-6 / 1.8311e-4 | **相同** |
| A1 拦截次数 | 936 | **936** |

## 3. 性能

### 3.1 微基准（空闲卡）

| 基准 | 5.0 实测 | 4.2 基线/报告 | 判定 |
|---|---|---|---|
| sdpa_math direct | 1.34–2.17x native | 1.46–2.21x | ✓ |
| sdpa_math A1 | 1.16–2.12x | 1.31–2.17x | ✓ |
| embedding 大 shape | 与 native parity；Triton 慢数十倍 | 同结论 | ✓ |
| embedding backward 16k | ours≈native 1.001x | 0.976x | ✓ |
| sdpa triton 探针 | 仍慢于 native | 同结论 | ✓ |

### 3.2 e2e 性能与归因（空闲卡，两次采样）

| case | 4.2 speedup | 5.0 speedup | 判定 |
|---|---|---|---|
| forward_b1_s256 | 1.077x | 1.06–1.08x | ✓ 一致 |
| forward_b2_s128 | 1.074x | 1.07–1.10x | ✓ 一致 |
| generate8_b1_s96 | 1.088x | 1.04–1.15x | ✓ 一致 |
| train_out_b1_s128 | 1.452x | 0.94–1.01x | ↓ 收窄 |
| train_probs_b1_s128 | 1.484x | 0.75–0.92x | ↓ 收窄 |
| fused bwd A/B (out) | 1.499x | 1.03–1.20x | ↓ 收窄 |

**归因**：sdpa_math P800 fast path 走厂商
`aten._scaled_dot_product_efficient_attention`（torch_xmlir /
xtorch_ops），与 FlagGems 无关；训练路径偏移随 **xtorch_ops
0.1.2640→0.1.2935** 出现。建议锁定 xtorch_ops 版本后重跑
`e2e_flagos.py` 保存同口径训练报告；中央性能基线须用
`perf_run.py --device p800-kunlunxin-gems5 --update-baseline` 在成功采集后
单独建立并审阅，不覆盖 4.2 的 `p800-kunlunxin.json`。`e2e_flagos.py`
本身不更新中央基线，详见[性能回归追踪](performance-regression.md)。

**卡状态警告**：占用卡（如当时 100% util 的物理 XPU2）上 forward 掉到
0.78–0.85x、fused backward 倒挂——性能结论必须以空闲卡数据为准；物理
XPU1 另见 [known-issues #5](known-issues.md) 的 Xid 补注，复测避开
`cuda:0`。

## 4. 路线样例与矩阵复测

| 项 | 结果 | 说明 |
|---|---|---|
| `examples/a1-op` | **PASS** | 移除模块级 `from __future__ import annotations` 后（#22） |
| `examples/a2-op` | 精度 9/9 PASS；**stage2 FAIL** | #23（本轮仅记录，不改代码） |
| `examples/b-op` | **PASS** | #20 修复：`VLLM_FL_PER_OP` 钉 `vendor:myvendor` 实测命中 |
| `examples/b-fullstack` | stage1/2 **PASS**；stage3 未复测 | stage2 按 #20 钉 `vendor:my-cpp`；stage3 受冒烟模型缺口（§5.5） |
| `tests/op_level/test_b` | **PASS** | #20 修复后选中 `vendor:audit` |
| `tests/op_level/test_a2` | **FAIL** | #23：内置 `gelu_and_mul` 冲突，TypeError |
| `run.py --consistency` | **FAIL** | #23（跨层一致性收尾格） |
| `run.py --route {a1,a2,b} --level framework` 各格 | 未复测 | 冒烟模型缺口 → HFValidationError（§5.5） |
| `ops/embedding/script/bench_perf.py` | **PASS**（含 scale 行） | #21 修复后 `scale_grad_by_freq=true` 行记 `native_error` 不再崩溃 |
| `pytest tests/unit` | 44 → 46 passed | 数值复测时 44；review 新增两条探测单测后 46（§2.3） |

## 5. 发现与修复

### 5.1 插件平台配置钉序覆盖 `with_preference`（已修）

vllm-plugin-FL 0.2.0+ 的 `dispatch/config/kunlunxin.yaml` 含
`op_backends.silu_and_mul: [flagos, vendor, reference]`，在
`_resolve_impl` 中**优先于** `with_preference("vendor")`——B 路线钉
vendor 的旧写法失效，选中 `default.flagos`。修复：
`VLLM_FL_PER_OP=silu_and_mul=vendor:<name>|reference` +
`reset_global_policy()`，已落地 `tests/op_level/test_b.py`、
`examples/b-op`、`examples/b-fullstack`（stage2）——三处均在导入
`vllm_fl` 前清掉 `VLLM_FL_OP_CONFIG`（JSON 文件路径，优先级更高且在
import 期缓存）、断言最终 impl_id，并在结束时恢复环境变量（`--all`
不污染）。详见 [known-issues #20](known-issues.md)。

### 5.2 embedding bench 容错缺口（已修）

`scale_grad_by_freq=true` 时原生 `aten.embedding_backward` 抛
`ValueError`（非 RuntimeError），脚本原 `except RuntimeError` 漏接导致
崩溃。已改为 `except (RuntimeError, ValueError)`，该行记
`native_error`。4.2 报告该行 native 空值同因。详见
[known-issues #21](known-issues.md)。

### 5.3 FlagGems 5.0 pointwise codegen 回归（样例侧规避）

5.0 的 `_collect_jit_deps` 把模块级 `from __future__ import annotations`
拷进生成文件中部 → `SyntaxError`。`examples/a1-op`、`examples/a2-op`
已移除模块级 future import——本仓库侧已规避，**FlagGems codegen 根因
未修**（根治需在 FlagGems 侧过滤 `__future__` 条目）。详见
[known-issues #22](known-issues.md)。

### 5.4 A2 插件与内置 backend impl_id 冲突（仅记录）

插件 `v0.1.0+` 内置注册 `gelu_and_mul` 的 `default.flagos` /
`reference.torch`（packed 单张量协议）；A2 演示插件同名双张量注册被
`Duplicate impl_id` 整体拒绝。受影响：`tests/op_level/test_a2`、
`run.py --consistency`、`examples/a2-op` stage2（及模型缺口解封后的
`tests/framework_level/test_a2` / `examples/a2-framework`）。按本轮决策
**仅文档记录**，修复方向见 [known-issues #23](known-issues.md)。

### 5.5 框架层冒烟模型缺口（环境，待补）

设备 profile 声明的 `model_path`
（`/workspace/random-llm-models/llama-3.1-8b-like`）**在本容器不存在**
——`run.py --route <route> --level framework --device p800-kunlunxin-gems5`
各格与 `examples/b-fullstack` stage3 均在
vLLM 启动时 `HFValidationError`，本轮无法复测。仓内仅有
`/models/random-qwen3-tiny`（Qwen3 4L，`max_position_embeddings=512`）：
架构与 profile `engine_args`（`max_model_len=4096`）不兼容，且黄金按
llama-like 模型生成，**不能替代**。可用 `FLAGOS_MODEL_PATH` 换模型路径，
但黄金需在同模型上重新生成后再比对。

**2026-10-29 备注（C2 已部分补齐，黄金矩阵缺口不变）**：issue #39 要求的
A1-aten 真实 vLLM C2 证据（`embedding` + `sdpa_math`）已用
`FLAGOS_MODEL_PATH=/models/random-qwen3-tiny` 在本容器跑通——该路径走
`tests/framework_level/test_a1_aten_ops.py` 的专用 runner 口径
（`max_model_len=128`、TP=1、不做黄金比对），因此不受上述 profile
`engine_args` / llama-like 黄金约束；命令、命中计数、概率图与梯度证据见
[框架级 README「P800 5.0 真实 vLLM C2 实测」](../tests/framework_level/README.md)
与 [evidence 存档](../tests/framework_level/evidence/)。本页 §2/§5.5 所指的
**框架矩阵 + 黄金**（`run.py --route … --level framework`）仍待模型与
黄金回填，两项口径互不替代。

当前也未入库 `golden/p800-kunlunxin-gems5_golden.json`；A1/B framework
按 profile 名查找该文件，缺失时历史黄金检查会跳过。因此补回模型后还需
建立并审阅 5.0 框架黄金，完成相应验证后再回填本页。操作入口见
[安装指南](installation.md#kunlunxin)，这里的框架黄金与 §2.2 独立算子的
黄金精度记录不同。

## 6. 复测命令清单

```bash
# 环境核对（5.0 栈；4.2 基线对照用 --device p800-kunlunxin）
python3 scripts/check_env.py --device p800-kunlunxin-gems5

# 先 xpu-smi 确认空闲卡（本复测用 5,6 号卡）
# 此变量控制当前进程；framework 子进程的卡号另由 profile.visible_devices 决定
export CUDA_VISIBLE_DEVICES=5,6

# 算子三层一键（embedding / sdpa / sdpa_math 同构）
# standalone 参数是本地硬件别名（ops/*/_profile.py 内置 p800-kunlunxin 等），
# 不读取中央 configs/devices/*.yaml；别名不代表选择/切换 FlagGems 4.2，
# 软件栈以上方 check_env 的 5.0 环境锁核对为准
cd ops/sdpa_math && python3 example.py p800-kunlunxin

# 黄金
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl p800 --device cuda:1   # 双卡可见时
# 单卡可见（CUDA_VISIBLE_DEVICES=5）时用 --device cuda:0

# 微基准（脚本无 --register 参数）
python3 script/bench_perf.py --device cuda:1 --json-out reports/perf_p800-kunlunxin.json

# vLLM 栈 E2E
python3 script/e2e_flagos.py --device cuda:0 --dtype bfloat16

# 单测与矩阵
cd ../.. && python3 -m pytest tests/unit -q
python3 run.py --consistency --device p800-kunlunxin-gems5

# A1-aten 真实 vLLM C2（issue #39；随机权重 tiny 模型，runner 口径不比黄金）
# 卡号以 xpu-smi 为准；卡 5 内核态挂起勿用（见框架级 README 限制）
FLAGOS_DEVICE=p800-kunlunxin-gems5 \
FLAGOS_MODEL_PATH=/models/random-qwen3-tiny \
FLAGOS_A1_OPS=embedding,sdpa_math \
USE_FLAGGEMS=0 CUDA_VISIBLE_DEVICES=0,3 \
python -m pytest tests/framework_level/test_a1_aten_ops.py -q -s

# framework 矩阵（待 §5.5 冒烟模型就绪后执行；模型缺失时 vLLM 启动即 HFValidationError）
# 运行前核对所用 profile 的 device.visible_devices；该字段会覆盖上面的 5,6
for route in a1 a2 b; do
  python3 run.py --route "$route" --level framework \
    --device p800-kunlunxin-gems5
done

# 路线样例
python3 examples/a1-op/example.py p800-kunlunxin-gems5
python3 examples/a2-op/example.py p800-kunlunxin-gems5   # stage2 现 FAIL（#23）
python3 examples/b-op/example.py p800-kunlunxin-gems5
python3 examples/b-fullstack/example.py p800-kunlunxin-gems5
python3 tests/op_level/test_b.py p800-kunlunxin-gems5
```

**执行状态与参数口径**，本清单分两类入口：

- **standalone 算子入口**（`ops/*/_profile.py` 本地加载器）：参数是**本地硬件
  别名**，不读取中央 `configs/devices/*.yaml`，仅接受其内置值（如
  `p800-kunlunxin`）；别名不代表选择/切换 FlagGems 版本，实际软件栈由已安装
  包决定，用 `check_env.py --device p800-kunlunxin-gems5` 按 5.0 环境锁核对
- **中央入口**（`run.py`、路线样例、`test_b.py`）：读取中央 profile，显式传
  已登记的 `p800-kunlunxin-gems5`

framework 矩阵待 §5.5 冒烟模型就绪后按本清单执行（复测当日该项在 vLLM 启动
阶段因模型缺失止于 `HFValidationError`，未进入测试，非参数错误）。其余命令为
复测当日实际执行项，当时两类入口均以 `p800-kunlunxin` 参数运行（同硬件同包
版本；中央入口本清单已改传 `p800-kunlunxin-gems5` 对齐环境锁，standalone 保持
其本地别名）。

**设备可见性**：两个 P800 中央 profile 默认 `torch_device: cuda:1`、
`visible_devices: "1,2"`。framework harness 与 b-fullstack stage3 会将
子进程 `CUDA_VISIBLE_DEVICES` 写为 profile 的 `visible_devices`，因此上面的
`export ...=5,6` 不代表框架子进程也使用卡 5,6；运行前须同步检查该字段。
`SDPA_TEST_DEVICE` / `SDPA_MATH_TEST_DEVICE` / `EMBEDDING_TEST_DEVICE`
仅覆盖对应独立算子的逻辑设备，不覆盖中央入口，也不替代框架 TP=2 的配置。
具体设置见[昆仑芯卡号与入口选择](device-profiles.md#kunlunxin-devices)。
复测原始 JSON 留档于复测容器 `/work/logs/`（未入库）。

## 7. 结论

| 维度 | 4.2 → 5.0 |
|---|---|
| 算子数值 / 黄金 / A1 拦截（§2 已完成范围） | **一致** |
| 应用层贪心序列 / 梯度（§2 已完成范围） | **逐位一致** |
| 推理 / 生成 E2E | **一致** |
| 训练 E2E / fused backward | 收益收窄（xtorch_ops/plugin 变化，非 FlagGems） |
| FlagGems API（only_enable / gelu） | **兼容** |
| A2 dispatch（插件路线） | **BLOCKED**（#23，本轮仅记录） |
| 仓库样例 / 测试 / 脚本 | #20/#21 已修；#22 本仓库已规避（FlagGems 根因未修）；#23 仅记录 |
| 框架层矩阵 | 未复测（冒烟模型缺口，§5.5） |
| 真实 vLLM C2（A1-aten embedding/sdpa_math） | **已补齐**（2026-10-29，random-qwen3-tiny，issue #39；与矩阵/黄金口径分开） |

**结论仅适用于 §2–§4 已完成的验证范围**：kernel、A1 与 mini-consumer
证据在 FlagGems 5.0 栈上与 4.2 基线一致，可继续作为这些范围内 P800
算子开发的参考；A2 插件路线仍为 BLOCKED（#23），framework 矩阵因冒烟
模型缺失尚未验证（§5.5）——两项回填前不扩大基线声明；A1-aten 的真实 vLLM C2（embedding/sdpa_math）已于 2026-10-29 单独补齐（issue #39），只覆盖该两个算子的 C2 判定、不扩大到矩阵/黄金声明。性能基线建议在
锁定 xtorch_ops 与空闲卡条件下重标。

---

**下一步**: [已知问题](known-issues.md)（#20–#23 排障细节）·
[全链路指南](fullstack-guide.md)（七步流水线）·
[性能回归追踪](performance-regression.md)（基线更新规范）
