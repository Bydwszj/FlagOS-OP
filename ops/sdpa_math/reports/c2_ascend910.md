# C2 真实引擎验证记录: sdpa_math × vLLM-Ascend（ascend910-c2-vllm）

| 项 | 值 |
|---|---|
| 日期 | 2026-10-31（复跑；证据与代码同提交） |
| profile | `ascend910-c2-vllm`（Ascend 910 / torch_npu 2.10.0，真实 vLLM-Ascend 栈；环境锁 [configs/env/ascend910-c2-vllm.lock.yaml](../../../configs/env/ascend910-c2-vllm.lock.yaml)） |
| 引擎 | **vLLM 0.20.2**（editable `/vllm-workspace/vllm` @ `bc150f5` clean）+ **vllm-ascend** 源码树 @ `367b8e62`（`VLLM_PLUGINS=ascend`，NPUPlatform 激活） |
| 模型 | HuggingFaceTB/SmolLM2-135M（snapshot `93efa2f0`，bf16，max_model_len=192） |
| harness | `ops/sdpa_math/script/c2_real_vllm_ascend.py`（`--mode accept`） |
| consumer_type | `real-engine`（vLLM-Ascend 推理引擎，非 mini consumer） |
| 证据等级 | **C2**（真实引擎命中目标算子） |
| run_id | `20261031T012918Z-2603520` |
| commit | `1dc1d8c`（`summary.commit` == 证据产出提交，见 [c2_ascend910/summary.json](c2_ascend910/summary.json)） |
| **验收状态** | `complete-pending-review`（同路径验收必需项全过、待审阅；**不写「C2 成立/验收通过」**——#38 意见 6） |
| 二轮复审处置 | 2026-11-01 判定器修复后**离线重判**（原始材料未动、未复跑 NPU）——见 [c2_ascend910/acceptance_redecide.json](c2_ascend910/acceptance_redecide.json) 与 §10 |

## 1. 命令（可复现）

```bash
cd /root/FlagOS-OP
python3 ops/sdpa_math/script/c2_real_vllm_ascend.py \
  --out ops/sdpa_math/reports/c2_ascend910
# 默认 --mode accept: 命中/阶段命中/证据完整性/正确性/性能/归因任一未过 → exit ≠ 0
# 默认 --profile ascend910-c2-vllm（缺对应环境锁直接 exit 2）
# --mode collect: 显式采集模式（采集完成 exit 0，不代表验收通过）
```

环境要点（脚本已内置并落盘到 summary.controls）：

- `PYTHONPATH=/vllm-workspace/vllm-ascend` + CANN python site-packages（含 `acl`）
- `VLLM_PLUGINS=ascend`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`（规避 NPU fork 再初始化）
- `FLAGOS_SDPA_MATH_ATTN=1`（引擎侧 attention backend 补丁）
- inject 组 `FLAGOS_A1_IMPL=torch`（显式设置、子进程亲测落盘
  `injection.a1_impl`；不再使用 setdefault 的隐式口径——意见 5）
- 模型路径可用 `FLAGOS_MODEL_PATH` 覆盖

## 2. 三跑结果（预注册对照组）

| 跑 | ok | token_ids（贪心 8 token × 6 prompt，表列 prompt0） | text | init_s | wall_s |
|---|---|---|---|---:|---:|
| baseline（原生注意力路径，无注入） | ☑ | `[260, 2240, 282, 7042, 30, 657, 314, 260]` | ` the city of Paris. It is the` | 23.87 | 43.42 |
| baseline_math（同 MATH 路径，无 A1） | ☑ | `[260, 3575, 282, 260, 1798, 30, 198, 198]` | ` the capital of the country.\n\n` | 22.40 | 44.58 |
| inject（同 MATH 路径 + A1 torch 实现） | ☑ | `[260, 3575, 282, 260, 1798, 30, 198, 198]` | ` the capital of the country.\n\n` | 23.86 | 50.19 |

- **同路径（baseline_math vs inject，唯一变量 = A1 实现）: C1 恒等 PASS
  6/6 prompt 位级一致**（8/8 token × 6）。
- teacher-forced prompt logprob（同路径，93 个可比 token）:
  `max_abs_diff=0.169592` / `mean_abs_diff=0.021703` nats——在预声明
  失效线 0.5 nats 之内（`scripts/c2_acceptance.py`，超限即 FAIL 不豁免）。
  生成 token 位级一致但 prompt logprob 有低幅差：A1 torch 组合实现与
  aten math 原实现的浮点序不同，贪心 argmax 未翻转。
- **路径差（baseline vs inject，math vs `npu_fusion_attention`，仅
  描述性、不参与同路径判据）**: 5/6 prompt 一致（仅 prompt0 第 2 token
  起分叉），共同前缀与首分歧位置见
  `correctness_native_path_descriptive.per_prompt`——路径差是预期的
  数值精度差，不再作为「例外」豁免任何判据。
- 三组均在预注册 prompt 集（6 条，源码固化）+ 温度 0 + 同一权重快照上
  运行；`--mode accept` 退出码 0，`failed_checks` 为空。

## 3. 调用计数与阶段归因（意见 3/4）

三份计数文件（`sdpa_math_hits_<pid>.json`，run_id 不符即剔除——本次
stale=0），角色按 pid/ppid 判定，阶段按 hit_times 落入窗口归因：

| pid | ppid | 角色 | hits | 阶段分解 |
|---:|---:|---|---:|---|
| 2634344 | 2603520 | main（child runner） | 1 | probe 1 |
| 2636629 | 1395859 | engine/worker | 0 | — |
| 2636630 | 2634344（=main） | engine/worker（EngineCore） | **2880** | correctness 1440 + warmup 240 + measured 1200 |

- **2880 分解（交叉核对）**: 6 prompt × 8 前向 × 30 层 = 1440（correctness）
  + 1 prompt × 8 × 30 = 240（warmup）+ 5 次 × 8 × 30 = 1200（measured）
  = 2880；`outside_windows=0`，归因自洽检查
  `hits_total == phase_attributed_total` 通过。
- 主进程 probe 1（generate 后直调 math 后端，形状 `[1,4,8,16]` fp16）
  **独立列示，不计入** engine 计数。
- `attn_patched = true`；`a1_counter_retrieved = true`；
  `a1_reg_info = {impl_requested: torch, impl_applied: true,
  counter_requested: true, counter_passed: true}`——注册参数事实由
  `inspect.signature` 能力分发产生并如实落盘（意见 5）。

## 4. attention 几何修复（意见 1/2，CPU 回归见 tests/unit）

审阅意见 1/2 针对 `flagos_math_attn.py` 的两处几何错误，本提交已修，
并有无引擎依赖的 CPU 单测钉死（`tests/unit/test_flagos_math_attn_geometry.py`）：

1. **query 切分（意见 1）**: vllm-ascend（`attention_v1.py:323`）给出的
   `actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()` 已是**累计
   端点**；旧代码再逐段累加会把多请求 query 切错、混用相邻请求 KV
   （审阅 CPU 复现: 端点 `[1,2,3]` → 错误 `[10,20,20]`）。现改为
   `attn_geometry.resolve_q_boundaries`（端点/逐段长度双格式识别，
   歧义布局显式 `RuntimeError` 拒绝，不静默算错）；回归断言正确输出
   `[10,20,30]`。
2. **causal mask（意见 2）**: 旧代码 `1<Sq<Skv`（cached/chunked prefill、
   decode）统一 `is_causal=False, attn_mask=None`，chunk 内较早 query
   会读取后续 token（CPU 复现: Sq=2 Skv=4, V=[0,0,0,100] → 错误
   `[25,25]`）。现用带历史偏移的加性 mask（允许 `k <= Skv-Sq+q`），
   回归断言正确输出 `[0,25]`；普通 prefill 走 `is_causal=True`。
   注意: math op 的 `attn_mask` 是加性语义——bool mask 会被当
   1.0/0.0 相加（同场景 bool 得 `[10.92,25]`），必须 0.0/-inf 浮点 mask。

说明: 审阅意见已确认这两处**不是**当前 2nd-token 分歧（路径差）的
根因——路径差源于 math 与 `npu_fusion_attention` 的数值精度差
（§2 描述性对照），本记录如实区分两者。

## 5. 验收判定（意见 4）

判定收敛为纯函数 `scripts/c2_acceptance.decide`（负例回归
`tests/unit/test_c2_acceptance.py`：输出失配/零命中/缺证据/子进程
失败在 accept 模式下全部非零退出）。本次 `acceptance_decision`：

```json
{"mode": "accept", "collection_status": "completed",
 "acceptance_status": "complete-pending-review",
 "failed_checks": [], "warnings": [], "exit_code": 0}
```

必需项（全部通过）: ① 真实前向命中（engine 2880 > 0）；② 同路径
正确性 C1 恒等 PASS；③ logprob 失效线 ≤0.5 nats；④ 同路径性能门禁
`OK`；⑤ 计数归因自洽。**`rc/ok` 仅表示采集成功，不等于验收通过**；
collect 模式的退出码语义在 summary 与文档中显式标注。
二轮复审（§10）在此之上追加 ⑥ 输入证据完整性 `evidence_complete`
与 ⑦ engine correctness/measured **阶段命中**两项必需项，本记录经
离线重判后仍全部通过（`acceptance_redecide.json`）。

## 6. 性能（意见 4/7）

预声明阈值（跑前固化）: inject/baseline_math 中位数比 ≤1.2 OK、
≤1.3 WARN、>1.3 FAIL（对齐 `docs/performance-regression.md`）。

| 对比 | 中位数比 | 判定 |
|---|---:|---|
| **inject / baseline_math（门禁）** | **1.0634** | **OK** |
| baseline_math / baseline（路径差，描述性） | 1.2109 | （不入门禁） |
| inject / baseline（路径差，描述性） | 1.2876 | （不入门禁） |

中位数（每次 generate 墙钟，warmup 1 + repeats 5）: baseline 0.2267s /
baseline_math 0.2745s / inject 0.2919s。同步口径如实记录
（`llm.generate` 同步阻塞，前向在 EngineCore 子进程，主进程无法对
引擎 `torch.npu.synchronize()`，三组同构——见 summary.perf.sync_note）。

**口径提醒（复审要求保留）**: 门禁比 1.0634 只是 **inject / baseline_math
（同 MATH 路径）**；inject / baseline（相对原生 `npu_fusion_attention`
注意力的整体时延比）为 **1.2876**，仅描述性、不入门禁——不得把
1.0634 解读成“引擎整体只增加 6.34%”。

**计数落盘 I/O 干扰的发现与处置（如实披露）**: 首轮复跑（2026-10-30，
run_id `20261030T032536Z-3886431`）每次命中全量重写计数文件，落盘
耗时 `dump_s≈6.12s` 摊进 measured 窗口，把门禁比抬到 4.32（量级失真）。
处置: 计数落盘加 0.25s 节流（`FLAGOS_C2_DUMP_INTERVAL_S`，仅 atexit
全量落盘保留完整计时轴），阈值不变；本轮 `dump_s≤0.029`，门禁比
1.0634。该迭代过程不隐藏：两轮数字与根因均在此记录。

**算子级性能回归（意见 7）**: `scripts/perf_run.py --device ascend910
--pattern sdpa_math` + `perf_compare.py`（未使用 `--update-baseline`），
报告 [perf_compare_ascend910_20261030.md](perf_compare_ascend910_20261030.md):
`ops.sdpa_math.{triton,torch,reference}` 对入库基线 +5.6%/+1.2%/-0.5%
全 OK（阈值 WARN +20% / FAIL +30%）。

## 7. 版本指纹

| 组件 | 版本/来源 |
|---|---|
| python | 3.11.15 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| triton | 3.5.1（triton-ascend 源码树，无 dist-info——check_env 元数据探测报未安装为已知限制） |
| flag_gems | 5.3.5（已安装，C2 harness 未启用——范围 = torch 组合实现） |
| vllm | 0.20.2 editable `/vllm-workspace/vllm` @ `bc150f5` clean |
| vllm_ascend | 源码树 `/vllm-workspace/vllm-ascend` @ `367b8e62`（M `csrc/utils/inc/kernel/moe_distribute_base.h`、`?? csrc/build_out/`——采集时如实记录） |
| CANN | 9.0.0（`/usr/local/Ascend/cann-9.0.0/python/site-packages`，`acl` 可导入） |

## 8. 对 issue #38 的口径

| 范围项 | 本记录结论 |
|---|---|
| sdpa_math 真实引擎验证 | **c2-hit / complete-pending-review（待审阅）**: 同路径三跑，C1 恒等 6/6、logprob 0.17 nats（<0.5 线）、性能门禁 OK 1.0634、engine 命中 2880（correctness/warmup/measured 分解如上），accept 模式 exit 0；**不写「C2 成立」，待 #38 审阅 + CI 闭环** |
| embedding Ascend | 不在本记录范围——见 `ops/embedding/reports/c2_ascend910.md` |
| 版本/模型/profile/commit/注入方式/调用计数 | §1–§7 已记录（证据提交 = 代码提交 `1dc1d8c`，意见 7 可追溯性） |
| 性能 | 同路径门禁 + 算子级回归双口径（§6），落盘 I/O 干扰已披露 |

## 9. 与前几版记录的差异

- 2026-10-28 无引擎侧补丁: `c2-routing-boundary`（forward_hits=0）。
- 2026-10-29/30 引入 `FLAGOS_SDPA_MATH_ATTN=1`: `c2-hit`，engine 240
  （30 层 × 8 前向，仅 prompt0 correctness 单跑）；旧口径 241 = 240 +
  probe 1（未按 pid 拆分）。
- 本版（2026-10-31, commit `1dc1d8c`）: 同路径三跑 + 预注册 6-prompt
  集 + 阶段归因（2880 = correctness+warmup+measured）+ 性能门禁 +
  验收取口（accept/collect 模式）+ attention 几何修复回归。旧 run_id
  命中文件全部剔除（stale=0 残留）。

## 10. 二轮复审处置（2026-11-01，PR #44 comment 6559530331）

复审在 `04c2229` 上用纯 Python 复现了两条**验收器假通过路径**
（复审明确: 现有实测材料**未触发**这两条漏洞，硬件结果不因此无效）：

| 复审条目 | 处置 | 证据 |
|---|---|---|
| 1 [P1] 双方缺失输出 → `[] == []` 计成“6/6 PASS” | `correctness_compare` 前置 `_validate_evidence`（预期集非空 / 两跑完整唯一覆盖 prompt_idx / 身份对应 / token 非空；缺失·重复·错配直接 FAIL，不进入数值比较）；`decide` 强制 `evidence_complete=True` | 代码 `ops/sdpa_math/script/c2_real_vllm_ascend.py`；负例 `tests/unit/test_c2_evidence_phase.py`（**实际调用 harness 比较函数**，含双方空输出/空 token/缺 prompt/空预期集 + 合法短输出正例） |
| 2 [P1] `forward_hits` 取生命周期累计，仅 init 命中也过 | 去掉主进程累计回退；`decide` 新增必需项: engine/worker **correctness 阶段命中 > 0**（支撑正确性）与 **measured 阶段命中 > 0**（支撑性能）；init/warmup/probe/outside 独立保留；`hit_attribution`/`hit_attribution_check` 缺失不放行 | 同上；负例用**真实 `attribute_hits()`** 构造 init-only / probe-only / outside-only / correctness 有命中但 measured 为零，全部 accept 非零 |
| 2 附带 status 矛盾 | `acceptance.status` 与 `acceptance_decision` 由同一次 `decide` 同源派生 | harness main（先判定、后写 status） |
| 3 [P2] profile 误标 / catalog 共用 C2 | harness `--profile`（默认 `ascend910-c2-vllm`）+ 缺环境锁 exit 2；summary 落 `profile`/`env_lock`；catalog evidence 改逐 profile 列表，C2 只归属 `ascend910-c2-vllm`，`ascend910` 保留历史范围；README 按 `level@profile·status` 展示 | `ops/catalog.yaml`、`scripts/build_ops_index.py`、`ops/README.md` |

**离线重判（复审指引: 保留原始测量材料，另存派生结果）**:

- 原始 [c2_ascend910/summary.json](c2_ascend910/summary.json)（run_id
  `20261031T012918Z-2603520`，采集 commit `1dc1d8c`）**未改动**，
  sha256 记录在派生文件内；
- 派生 [c2_ascend910/acceptance_redecide.json](c2_ascend910/acceptance_redecide.json):
  新比较函数重算 correctness（证据完整、6/6 PASS）+ 新 `decide` 重判
  → `complete-pending-review` / `failed_checks=[]` / exit 0；
  `redecide.commit` = 重判代码提交；
- **profile 误标更正（如实记录）**: 原 summary `profile: ascend910`
  系旧 harness 硬编码，实际验证环境为 `ascend910-c2-vllm`（环境锁
  §1 已绑定）——corrections 字段记录 was→now，不覆写原始文件；
- 采集协议（执行/注入/计数落盘）未变 → **未复跑 NPU**。

复现:

```bash
python3 scripts/c2_redecide.py \
  --harness ops/sdpa_math/script/c2_real_vllm_ascend.py \
  --summary ops/sdpa_math/reports/c2_ascend910/summary.json \
  --out ops/sdpa_math/reports/c2_ascend910/acceptance_redecide.json
python3 -m pytest tests/unit/test_c2_evidence_phase.py -q
```

**已知边界（结论限定）**: 本 C2 结论限定为**已测单请求配置**
（`max_num_seqs` 默认、无多请求混合调度）；多请求 / cached / chunked
prefill 的真实引擎覆盖**尚未完成**——CPU 回归（§4 几何修复）不等于该
场景 NPU 已验收，后续矩阵另列，未跑场景不写成已通过。
