# C2 真实引擎验证记录: embedding × vLLM-Ascend（ascend910-c2-vllm）

| 项 | 值 |
|---|---|
| 日期 | 2026-10-31（复跑；证据与代码同提交） |
| profile | `ascend910-c2-vllm`（Ascend 910 / torch_npu 2.10.0，真实 vLLM-Ascend 栈；环境锁 [configs/env/ascend910-c2-vllm.lock.yaml](../../../configs/env/ascend910-c2-vllm.lock.yaml)） |
| 引擎 | **vLLM 0.20.2**（editable `/vllm-workspace/vllm` @ `bc150f5` clean）+ **vllm-ascend** 源码树 @ `367b8e62`（`VLLM_PLUGINS=ascend`，NPUPlatform 激活） |
| 模型 | HuggingFaceTB/SmolLM2-135M（snapshot `93efa2f0`，bf16，max_model_len=192） |
| harness | `ops/embedding/script/c2_real_vllm_ascend.py`（`--mode accept`） |
| consumer_type | `real-engine`（vLLM-Ascend 推理引擎，非 mini consumer） |
| 证据等级 | **C2**（真实引擎命中目标算子） |
| run_id | `20261031T013206Z-2663709` |
| commit | `1dc1d8c`（`summary.commit` == 证据产出提交，见 [c2_ascend910/summary.json](c2_ascend910/summary.json)） |
| **验收状态** | `complete-pending-review`（同路径验收必需项全过、待审阅；**不写「C2 成立/验收通过」**——#38 意见 6） |
| 二轮复审处置 | 2026-11-01 判定器修复后**离线重判**（原始材料未动、未复跑 NPU）——见 [c2_ascend910/acceptance_redecide.json](c2_ascend910/acceptance_redecide.json) 与 §9 |

## 1. 命令（可复现）

```bash
cd /root/FlagOS-OP
python3 ops/embedding/script/c2_real_vllm_ascend.py \
  --out ops/embedding/reports/c2_ascend910
# 默认 --mode accept: 命中/阶段命中/证据完整性/正确性/性能/归因任一未过 → exit ≠ 0
# 默认 --profile ascend910-c2-vllm（缺对应环境锁直接 exit 2）
# --mode collect: 显式采集模式（采集完成 exit 0，不代表验收通过）
```

环境要点（脚本已内置并落盘到 summary.controls）：

- `PYTHONPATH=/vllm-workspace/vllm-ascend` + CANN python site-packages（含 `acl`）
- `VLLM_PLUGINS=ascend`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`（规避 NPU fork 再初始化）
- `FLAGOS_A1_ATEN_INJECT=1` + `FLAGOS_A1_OPS=embedding` + `A1_DISPATCH_KEY=PrivateUse1`
- `FLAGOS_A1_IMPL` 显式清理（embedding 的 `register_a1` 不接受 impl——
  能力分发事实落盘 `injection.a1_reg_info.impl_applied=false`，意见 5）
- 模型路径可用 `FLAGOS_MODEL_PATH` 覆盖

## 2. 双跑结果（预注册对照组）

| 跑 | ok | token_ids（贪心 8 token × 6 prompt，表列 prompt0） | text | init_s | wall_s |
|---|---|---|---|---:|---:|
| baseline（无注入） | ☑ | `[260, 2240, 282, 7042, 30, 657, 314, 260]` | ` the city of Paris. It is the` | 24.02 | 43.74 |
| inject（A1 embedding） | ☑ | `[260, 2240, 282, 7042, 30, 657, 314, 260]` | ` the city of Paris. It is the` | 28.76 | 60.37 |

- **C1 恒等 PASS 6/6 prompt 位级一致**（8/8 token × 6）；调用链与
  baseline 完全同构，唯一变量 = 是否注册 FlagOS A1 embedding 实现。
- teacher-forced prompt logprob（93 个可比 token）:
  `max_abs_diff=0.0` / `mean_abs_diff=0.0`——位级等价，与 embedding
  索引查找语义一致（黄金 20/20 PASS）。
- `--mode accept` 退出码 0，`failed_checks` 为空。

## 3. 调用计数与阶段归因（意见 3/4）

三份计数文件（`embedding_hits_<pid>.json`，run_id 不符即剔除——本次
stale=0），角色按 pid/ppid 判定：

| pid | ppid | 角色 | hits | 阶段分解 |
|---:|---:|---|---:|---|
| 2680213 | 2663709 | main（child runner） | 0 | —（主进程不做模型前向，预期） |
| 2685667 | 2680213（=main） | engine/worker | 0 | —（非前向辅助进程） |
| 2685670 | 2680213（=main） | engine/worker（EngineCore） | **97** | init 1 + correctness 48 + warmup 8 + measured 40 |

- **97 分解（交叉核对）**: init profile/dummy 前向 1（`skip_attn=True`
  但 `embed_tokens` 照常执行）+ correctness 6 prompt × 8 前向 = 48
  + warmup 1 × 8 = 8 + measured 5 × 8 = 40 = 97；每前向恰好 1 次
  `aten::embedding`。`outside_windows=0`，归因自洽检查通过。
- 主进程命中 0——真实前向全部在 EngineCore，无 probe 混入。
- `a1_counter_retrieved = true`；
  `a1_reg_info = {impl_requested: null, impl_applied: false,
  counter_requested: true, counter_passed: true}`——register_a1 无 impl
  参数时不再经 TypeError 回退丢 counter（意见 5 的旧 bug 已修）。

## 4. dispatch key 选择与 vllm_ascend 冲突规避

首次尝试用 `AutogradPrivateUse1` 注册时被拒绝：

```text
RuntimeError: This is not allowed since there's already a kernel
registered from python overriding embedding's behavior for
AutogradPrivateUse1 dispatch key and aten namespace.
```

根因：`vllm_ascend` 在 import 时已注册 `aten::embedding` 的
`AutogradPrivateUse1` kernel。推理（no_grad）路径 dispatch 到
`PrivateUse1`（非 autograd key），因此：

- harness 显式设 `A1_DISPATCH_KEY=PrivateUse1`；
- embedding `register.py` 的 `_backend_for` 已映射
  `PrivateUse1 → _ascend`，无需改动 kernel。

## 5. 性能（意见 4/7）

预声明阈值（跑前固化）: inject/baseline 中位数比 ≤1.2 OK、≤1.3 WARN、
>1.3 FAIL。

| 对比 | 中位数比 | 判定 |
|---|---:|---|
| **inject / baseline（同调用链门禁）** | **0.9436** | **OK** |

中位数（每次 generate 墙钟，warmup 1 + repeats 5）: baseline 0.2271s /
inject 0.2143s；同步口径如实记录（见 summary.perf.sync_note）。

**算子级性能回归（意见 7）**: `scripts/perf_run.py --device ascend910
--pattern embedding` + `perf_compare.py`（未使用 `--update-baseline`），
报告 [perf_compare_ascend910_20261030.md](perf_compare_ascend910_20261030.md):
`ops.embedding.ascend910.forward` 0.081ms / `ops.embedding.native.forward`
0.061ms——基线文件未含此二用例，标记 **NEW（描述性，不入门禁）**；
入库基线更新需单独走 `--update-baseline` 流程（本记录不触发）。

## 6. 版本指纹

| 组件 | 版本/来源 |
|---|---|
| python | 3.11.15 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| triton | 3.5.1（triton-ascend 源码树，无 dist-info——check_env 元数据探测报未安装为已知限制） |
| flag_gems | 5.3.5（已安装，C2 harness 未启用） |
| vllm | 0.20.2 editable `/vllm-workspace/vllm` @ `bc150f5` clean |
| vllm_ascend | 源码树 `/vllm-workspace/vllm-ascend` @ `367b8e62`（M `csrc/utils/inc/kernel/moe_distribute_base.h`、`?? csrc/build_out/`——采集时如实记录） |
| CANN | 9.0.0（`/usr/local/Ascend/cann-9.0.0/python/site-packages`，`acl` 可导入） |

## 7. 对 issue #38 的口径

| 范围项 | 本记录结论 |
|---|---|
| embedding 真实引擎验证 | **c2-hit / complete-pending-review（待审阅）**: 双跑 C1 恒等 6/6、logprob 0.0、性能门禁 OK 0.9436、engine 命中 97（init/correctness/warmup/measured 分解如上），accept 模式 exit 0；**不写「C2 成立」，待 #38 审阅 + CI 闭环** |
| sdpa_math Ascend | 不在本记录范围——见 `ops/sdpa_math/reports/c2_ascend910.md` |
| 版本/模型/profile/commit/注入方式/调用计数 | §1–§6 已记录（证据提交 = 代码提交 `1dc1d8c`，意见 7 可追溯性） |
| 性能 | 同调用链门禁 + 算子级回归双口径（§5） |

## 8. 与前几版记录的差异

- 2026-10-29/30 首测/复测: engine 命中 9（1 init + 8 generate 前向，
  单 prompt）。
- 本版（2026-10-31, commit `1dc1d8c`）: 预注册 6-prompt 集 + 阶段归因
  （97 = init 1 + correctness 48 + warmup 8 + measured 40）+ 性能门禁 +
  验收取口（accept/collect 模式）+ 注册能力分发事实落盘。旧 run_id
  命中文件全部剔除（stale=0 残留）。

## 9. 二轮复审处置（2026-11-01，PR #44 comment 6559530331）

复审在 `04c2229` 上用纯 Python 复现了两条**验收器假通过路径**
（复审明确: 现有实测材料**未触发**这两条漏洞，硬件结果不因此无效）：

| 复审条目 | 处置 | 证据 |
|---|---|---|
| 1 [P1] 双方缺失输出 → `[] == []` 计成“6/6 PASS” | `correctness_compare` 前置 `_validate_evidence`（预期集非空 / 两跑完整唯一覆盖 prompt_idx / 身份对应 / token 非空；缺失·重复·错配直接 FAIL，不进入数值比较）；`decide` 强制 `evidence_complete=True` | 代码 `ops/embedding/script/c2_real_vllm_ascend.py`；负例 `tests/unit/test_c2_evidence_phase.py`（**实际调用 harness 比较函数**，含双方空输出/空 token/缺 prompt/空预期集 + 合法短输出正例） |
| 2 [P1] `forward_hits` 取生命周期累计，仅 init 命中也过 | 去掉主进程累计回退；`decide` 新增必需项: engine/worker **correctness 阶段命中 > 0** 与 **measured 阶段命中 > 0**；init 1（§3）/warmup/probe/outside 独立保留；`hit_attribution`/`hit_attribution_check` 缺失不放行 | 同上；负例用**真实 `attribute_hits()`** 构造 init-only / probe-only / outside-only / correctness 有命中但 measured 为零，全部 accept 非零（本 run 实测: correctness 48、measured 40 均 > 0） |
| 2 附带 status 矛盾 | `acceptance.status` 与 `acceptance_decision` 由同一次 `decide` 同源派生 | harness main（先判定、后写 status） |
| 3 [P2] profile 误标 / catalog 共用 C2 | harness `--profile`（默认 `ascend910-c2-vllm`）+ 缺环境锁 exit 2；summary 落 `profile`/`env_lock`；catalog evidence 改逐 profile 列表，C2 只归属 `ascend910-c2-vllm`，`ascend910` 保留历史范围；README 按 `level@profile·status` 展示 | `ops/catalog.yaml`、`scripts/build_ops_index.py`、`ops/README.md` |

**离线重判（复审指引: 保留原始测量材料，另存派生结果）**:

- 原始 [c2_ascend910/summary.json](c2_ascend910/summary.json)（run_id
  `20261031T013206Z-2663709`，采集 commit `1dc1d8c`）**未改动**，
  sha256 记录在派生文件内；
- 派生 [c2_ascend910/acceptance_redecide.json](c2_ascend910/acceptance_redecide.json):
  新比较函数重算 correctness（证据完整、6/6 PASS、logprob 0.0）+
  新 `decide` 重判 → `complete-pending-review` / `failed_checks=[]` /
  exit 0；`redecide.commit` = 重判代码提交；
- **profile 误标更正（如实记录）**: 原 summary `profile: ascend910`
  系旧 harness 硬编码，实际验证环境为 `ascend910-c2-vllm`——
  corrections 字段记录 was→now，不覆写原始文件；
- 采集协议（执行/注入/计数落盘）未变 → **未复跑 NPU**。

复现:

```bash
python3 scripts/c2_redecide.py \
  --harness ops/embedding/script/c2_real_vllm_ascend.py \
  --summary ops/embedding/reports/c2_ascend910/summary.json \
  --out ops/embedding/reports/c2_ascend910/acceptance_redecide.json
python3 -m pytest tests/unit/test_c2_evidence_phase.py -q
```

**已知边界（结论限定）**: 本 C2 结论限定为**已测单请求配置**；多请求 /
cached / chunked prefill 的真实引擎覆盖**尚未完成**（CPU 回归不等于
该场景 NPU 已验收），后续矩阵另列，未跑场景不写成已通过。算子级性能
**NEW**（§5）如实保留为描述性，不称“基线回归通过”。
