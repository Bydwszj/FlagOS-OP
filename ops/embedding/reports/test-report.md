# embedding 测试报告

| 项 | 值 |
|---|---|
| 日期 | 2026-09-23（cambricon 接入复测 2026-09-24；thead-ppu 接入 2026-10-23）· Ascend910 C2 复跑 2026-10-31 · 验收判定器二轮复审 11-01（离线重判） |
| 设备 | p800-kunlunxin `cuda:1` · cambricon MLU590 `mlu:0` · thead-ppu PPU-ZW810E `cuda:6` |
| 环境 | P800: Torch 2.9.0+cu129 / torch_xmlir / Triton 3.0；MLU: Torch 2.11.0+cpu / torch_mlu 1.33.1 / Triton 3.4.0+mlu2.1.1；PPU: Torch 2.10.0 CUDA 兼容层 / Triton 3.6.0 PPUDriver |
| 结论 | 通过（p800 / MLU / PPU 三平台）；Ascend910 **C2 命中（profile `ascend910-c2-vllm`，验收状态 complete-pending-review（待审阅），2026-11-01 离线重判仍 exit 0）**，见 §1 C2 行与 [c2_ascend910.md](c2_ascend910.md) |

## 1. 测试范围

| 层级 | 覆盖 | 结果 |
|---|---|---|
| kernel forward | 1D/2D/3D、D8/80/512、empty、int32、padding | ✅ |
| kernel backward | FP32/FP16/BF16 × dense/scale_freq × duplicate/padding | ✅ |
| 黄金 | 174 组 | ✅ |
| A1 | 拦截、direct bitwise、autograd | ✅ |
| 应用层 | `nn.Embedding` + MLP forward/backward/greedy | ✅ |
| FlagOS 栈 | `flag_gems.only_enable(['gelu'])` 后运行应用层 | ✅ |
| 边界 | sparse forward 接受、sparse backward / invalid padding 拒绝 | ✅ |
| 第二平台 cambricon | kernel / op / framework / guard 同判据复跑 | ✅ |
| 第三平台 thead-ppu | kernel / tail / op / framework / guard / 黄金同判据复跑 | ✅ |
| C2 真实 vLLM-Ascend（profile `ascend910-c2-vllm`） | `script/c2_real_vllm_ascend.py --mode accept --profile ascend910-c2-vllm` | **c2-hit（验收状态 complete-pending-review（待审阅））**（2026-10-31 复跑 @ `1dc1d8c`；2026-11-01 判定器修复后离线重判仍 exit 0）：双跑 OK、accept 退出码 0、C1 恒等 6/6 位级一致（prompt logprob 0.0）；EngineCore 前向 `aten::embedding` 命中 **97** 次（init 1 + correctness 48 + warmup 8 + measured 40，阶段归因自洽）+ 主进程 0；性能门禁 inject/baseline 中位数比 0.94 OK；按 pid 角色拆分防串算。见 [c2_ascend910.md](c2_ascend910.md) |

## 2. 环境

| 项 | 值 |
|---|---|
| Python | 3.10.18 |
| Torch | 2.9.0+cu129 |
| torch_xmlir | XMLIR--bc1b1dc6f-dev+2026032411 |
| Triton | 3.0.0+03c4c9be |
| 设备 / 卡 | p800-kunlunxin / `cuda:1` |

## 3. 精度结果

复现命令：

```bash
python3 example.py p800-kunlunxin
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl p800 --device cuda:1
python3 script/check_accuracy.py --impl triton --device cuda:1
python3 script/check_accuracy.py --impl native --device cuda:1
pytest -q tests/unit

# 第二平台 cambricon（MLU）
MLU_VISIBLE_DEVICES=7 python3 example.py cambricon
MLU_VISIBLE_DEVICES=7 python3 script/check_accuracy.py --impl cambricon --device mlu:0
```

结果摘要：

- kernel forward: 20/20，max error 0；
- kernel backward: 6/6，max error 0；
- golden: 174/174，max error 0；
- A1: interception count 5，registered path equals direct path bitwise；
- dense grad error 0；
- scale-grad error 0；
- application logits/gradient diff 0；
- greedy continuation match 1.00。
- sparse=True forward 与 dense forward bitwise equal；
- sparse backward显式拒绝。

## 4. 性能结果

```bash
python3 script/bench_perf.py --device cuda:1 --dtype float16 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
python3 scripts/perf_run.py --device p800-kunlunxin --pattern ops.embedding
python3 scripts/perf_compare.py --device p800-kunlunxin
```

perf gate 结果：`FAIL 0 · WARN 0 · NEW 0`。入库基线中
`ops.embedding.p800.forward` 为 0.051ms，native 对照为 0.052ms。
A1 dispatch 独立子进程测量（P800）：native 0.0489ms，A1 0.0730ms，
附加约 0.0241ms——**该 a1 数字已作废**：当时 `bench_dispatch.py` 裸调
`register_a1`，返回的 `Library` 被 GC 即反注册，实测的是未注册状态
（修复与说明见 [cambricon.md](cambricon.md) §7）；native/direct 行与
精度结论不受影响。P800 需在有 `torch_xmlir` 的环境复跑。

## 5. 问题与风险

| 风险 | 状态 |
|---|---|
| `scale_grad_by_freq=True` fallback 有额外中间 tensor | 已量化：约 1.96ms vs dense 1.33ms（MLU 走原生，+11%） |
| Triton gather 慢 | 仅保留探针，不接生产（MLU 131k 还撞 grid 65535 上限） |
| sparse backward 未覆盖 | 显式拒绝，符合需求边界 |
| MLU `sparse=True` 反向返回 COO 而非报错 | `kernel/cambricon.py` 显式 `NotImplementedError`，守卫用例覆盖 |
| P800 A1 dispatch 数字作废（Library GC） | 脚本已修；MLU 复测 +0.059ms，P800 待有 `torch_xmlir` 环境复跑 |
| `check_env --device cambricon` 6 项不符 | 既有锁漂移（`d5c7a3d` 写锁的容器镜像 ≠ 本机 torch2.11 栈），与本改动无关；`--device mlu590` 为 OK |

## 6. cambricon（MLU590）复跑结果

```bash
MLU_VISIBLE_DEVICES=7 python3 test/kernel_level.py cambricon
MLU_VISIBLE_DEVICES=7 python3 test/op_level.py cambricon
MLU_VISIBLE_DEVICES=7 python3 test/framework_level.py cambricon
MLU_VISIBLE_DEVICES=7 python3 example.py cambricon
python3 -m pytest tests/unit -q
```

| 层级 | 结果 |
|---|---|
| kernel forward | ✅ 20/20，`max_forward_err=0.0` |
| kernel backward | ✅ 6/6，`max_backward_err=0.0`（3 dtype × scale 0/1） |
| 守卫 | ✅ sparse backward 拒绝 / invalid padding 拒绝 / sparse forward 接受 |
| 黄金 | ✅ 174/174，worst=0.000e+00（`--impl cambricon`，逐位判据） |
| A1 op | ✅ 拦截 5 次；注册=直调（逐位）；dense/scale grad err=0；sparse fwd OK / bwd rejected |
| 应用层 | ✅ 拦截 8 次；logits diff=0.000e+00；greedy 1.00；embedding/首层 linear 梯度 diff=0；`flag_gems.only_enable(['gelu'])` 生效 |
| 一键三层 | ✅ `example.py cambricon` 1/3 kernel · 2/3 A1 op · 3/3 application 全 PASS |
| 单测 | ✅ `pytest tests/unit -q` 20 passed |
| perf gate | ✅ `FAIL 0 · WARN 0 · NEW 0`（`ops.embedding.cambricon.forward` 0.060ms） |

## 7. thead-ppu（PPU ZW810E）复跑结果

```bash
python3 example.py thead-ppu
python3 test/kernel_level_tail_thead_ppu.py thead-ppu
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl thead-ppu --device cuda:6
python3 scripts/perf_run.py --device thead-ppu --pattern ops.embedding
python3 scripts/perf_compare.py --device thead-ppu
python3 -m pytest tests/unit -q
```

| 层级 | 结果 |
|---|---|
| kernel forward | ✅ 20/20，`max_forward_err=0.0` |
| kernel backward | ✅ 6/6，`max_backward_err=0.0`（3 dtype × scale 0/1，原生直通） |
| tail 补充（§2d） | ✅ 37/37：3072/5000/5120 × 3 dtype × 3 seed + zeros/large/boundary 全 err=0；哨兵通过 |
| 守卫 | ✅ sparse backward 拒绝 / invalid padding 拒绝 / sparse forward 接受 |
| 黄金 | ✅ 174/174，worst=0.000e+00（`--impl thead-ppu`，逐位判据） |
| A1 op | ✅ 拦截 5 次；注册=直调（逐位）；dense/scale grad err=0；sparse fwd OK / bwd rejected |
| 应用层 | ✅ 拦截 8 次；logits diff=0.000e+00；greedy 1.00；两级梯度 diff=0（flag_gems 缺证, 纯 A1 拦截口径） |
| 一键三层 | ✅ `example.py thead-ppu` 1/3 kernel · 2/3 A1 op · 3/3 application 全 PASS |
| 单测 | ✅ `pytest tests/unit -q` 46 passed |
| perf gate | ✅ `FAIL 0 · WARN 0`（`ops.embedding.thead-ppu.forward` 0.033ms / 255.0 GB/s） |
| 仓库一致性 | ◐ `run.py --consistency` FAIL：`ModuleNotFoundError: flag_gems`（dev 栈结构性缺证，同 T1 sdpa） |
| C2 真实 vLLM（thead-ppu-c2） | ✅ EngineCore 内 A1 命中 10；token 与基线一致（[e2e_vllm_c2_thead-ppu-c2.json](e2e_vllm_c2_thead-ppu-c2.json)、[thead-ppu-c2.md](thead-ppu-c2.md)） |

## 8. 结论

kernel 层、op 层、应用层、黄金与性能门禁在 **p800-kunlunxin、cambricon 与
thead-ppu 三个平台上全部通过**；thead 另有 **thead-ppu-c2 真实 vLLM C2** 证据。
建议合入。
