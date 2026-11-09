# 应用层（framework 验证）测试

验证两件事:
1. **算子真的被调用**: 自定义实现被应用层消费者选中执行；消费者可能是
   真实 vLLM 前向，也可能是设备 profile 明确声明的 mini consumer。
2. **端到端不破坏**: 启用后消费者输出符合预期。mini consumer 结果不能
   代替真实 vLLM 证据，报告必须标注实际消费者类型。

## 统一流程（_harness.py 编排）

- 输入: `inputs/prompts.txt`（每行一个 prompt；`PROMPTS_FILE`/`PROMPTS_LIMIT` 可覆盖/限量）
- 基线跑: 无插件，贪心解码，记录输出 token ids
- 插件跑: 按 route 注入对应实现，同 prompt 同 seed
- 断言: 调用计数 > 0 + 输出比对

## 三条路线的注入方式与断言

| 路线 | 注入 | 输出断言 |
|---|---|---|
| A1 | 数值恒等 aten::silu 覆盖 + `PER_OP=silu_and_mul=reference` | 前 8 token 完全一致（恒等数学） |
| A2 | Triton silu_and_mul 以 `vendor:triton-template` 注册 + PER_OP 钉住 | 前 2 token 一致 ≥2/3（自定义数值允许混沌分叉） |
| B | audit vendor + PER_OP 钉住，基线与插件均走 reference（纯拦截） | 前 8 token 完全一致（数值恒等） |

引擎参数全部来自设备 profile（TP/显存/怪癖），跨芯片无需改代码。

## 黄金输出（golden outputs）回归检测

```bash
# 跨设备黄金锚点（多快照 → 多数票共识前缀）
python3 scripts/build_golden.py --device cpu                 # transformers CPU 权威参考
python3 scripts/build_golden.py --device <加速卡profile>    # 示例: p800-kunlunxin
python3 scripts/compare_golden.py --devices <加速卡profile>,cpu
```

- 黄金 = N 次独立运行的快照集合 + 逐位置多数票共识前缀
- CPU 设备走 HuggingFace transformers 直接推理（最权威语义参考、
  确定性、绕过 vLLM/厂商栈）；GPU/XPU 设备走 vLLM reference 路径
- 没有 vLLM runtime 的 profile（例如 MLU590）使用 mini-decoder；这只证明
  应用层消费、注册和数值链路，真实 vLLM 验证须在有 runtime 的环境补跑
- 保存于 `golden/<device>_golden.json`（含引擎参数指纹、prompts 哈希，应提交入库）
- A1/B 框架级测试自动比对黄金共识前缀（跨会话回归检测），exact
  断言前缀按黄金实测稳定前缀自适应
- `scripts/drift_study.py` 可量化本栈跨进程非确定性，校准断言前缀长度
- 输入统一为 tokenize+词表钳制的 token id（`PROMPTS_AS_TOKENS=1`），
  规避 vendor 栈 embedding 静默越界，保证跨设备输入严格一致

## P800 真实 vLLM C2 实测（2026-10-26）

在自建容器 `daizijian-p800-flagosop`（vllm 0.13.0 + vllm_fl 0.1.0 +
torch_xmlir）上，用现成 **Qwen2.5-0.5B-Instruct** 跑通 8 个可注入算子：

```bash
FLAGOS_DEVICE=p800-kunlunxin \
FLAGOS_MODEL_PATH=/workspace/.cache/models/qwen2.5-0.5b \
USE_FLAGGEMS=0 \
CUDA_VISIBLE_DEVICES=0 \
FLAGOS_A1_OPS=type_as,clone,contiguous,dropout,empty_like,empty_strided,item,_local_scalar_dense \
python -m pytest tests/framework_level/test_a1_aten_ops.py -q -s
```

结果：`1 passed in 72.04s`；真实 vLLM 前向命中
`empty_strided=6299`、`empty_like=5917`、`contiguous=2208`（total=14424），
注入前后输出 token ids 逐位一致。模型来源为
`/data2/evalx-workspace/models/Qwen__Qwen2.5-0.5B-Instruct`，复制到自建容器
workspace 内使用，不占用任何其他人的容器。

限制（均在本机实测复现）：

- `copy_` 必须从 `FLAGOS_A1_OPS` 排除：P800 bf16 兜底会走
  `aten::type_as` / `Tensor.to`，而两者内部重新进入已注册的 `copy_`，
  形成 `copy_ ↔ type_as/to` 递归；需改为不经过 `copy_` 的 cast。
- `USE_FLAGGEMS=0`：FlagGems 4.2 的 P800 `gather` 在 vLLM 采样阶段报
  `xpuLaunchKernel ... Operation not permitted(err_code: 1)`；开启
  FlagGems 时 C2 不通过。
- 更大的 Qwen3-8B 在本栈 EngineCore 初始化阶段挂起（GPU 100%、显存仅
  166MiB），因此当前 C2 证据使用 0.5B 现成模型；8B 需要另查栈/模型兼容性。

## P800 5.0（gems5）真实 vLLM C2 实测（2026-10-29）

在开发容器 `flagos-p800-gems5-vllm0202-dev`（vllm 0.20.2+empty ·
vllm-plugin-fl 0.2.0+g38e7dbc · torch_xmlir 1.0.0.1 · xtorch_ops
0.1.2935+50a5d6a4 · FlagGems 5.0.0）上，用随机权重 Qwen3 小模型
`/models/random-qwen3-tiny`（4L · hidden 256 · head 64 · 4Q/2KV · vocab
151669）补齐 issue #39 的 embedding 与 sdpa_math C2 证据（与 4.2 段的
Qwen2.5-0.5B 结果分开记录，互不复用）：

```bash
FLAGOS_DEVICE=p800-kunlunxin-gems5 \
FLAGOS_MODEL_PATH=/models/random-qwen3-tiny \
FLAGOS_A1_OPS=embedding,sdpa_math \
USE_FLAGGEMS=0 \
CUDA_VISIBLE_DEVICES=0,3 \
python -m pytest tests/framework_level/test_a1_aten_ops.py -q -s
```

结果：`1 passed in 219.68s`；真实 vLLM EngineCore 基线/注入双跑，
命中 `embedding=19`、`sdpa_math=9`（total=28），注入前后输出 token ids
逐位一致；生成延迟 base 22.99s / inj 26.58s（注入侧含逐请求私有 math
与全量 P 物化，属证据口径非性能口径）。完整证据存档：
[evidence/p800-kunlunxin-gems5-2026-10-29.json](evidence/p800-kunlunxin-gems5-2026-10-29.json)。

sdpa_math 消费机制（issue #39「等价目标消费者」口径）：

- vLLM 0.20 的 kunlunxin vendor `prefill_attention` 不经过 `F.sdpa`，
  真实文本模型前向里没有任何原生 aten SDPA 消费点。
- 注入侧（`injection/a1_aten_ops.py`，环境变量 `A1_SDPA_CONSUMER=prefill`
  启用）把 `KunlunxinAttentionBackendImpl.context_forward` 的无前缀缓存
  分支改为逐请求调用 A1 的
  `torch.ops.aten._scaled_dot_product_attention_math`，双输出 `(O, P)`
  的 `O` 直接写回模型输出缓冲继续参与前向；decode、前缀缓存命中与
  profiling fake-meta（`slot_mapping=None`）分支保持 vendor 原路径。
- 概率图证据：本次真实 prompt（6 token）`probs_shape=[1,4,6,6]`、
  行和最大偏差 0.0099、对角均值 0.441、熵均值 0.822；引擎内每次消费
  追加 JSONL 到 `A1_SDPA_EVIDENCE_FILE`。
- 梯度证据：注入运行结束后在同进程做 bf16 前向 + 双输出反向
  （`dout`/`dprobs` 全 1，覆盖 dP 消费者路径），对照 CPU fp64 原生
  math：out/probs/dq/dk/dv `max_abs ≤ 0.016`、`max_rel ≤ 0.029`
  （`sdpa_grad.ok=true`）。

限制：

- 单 chunk prompt（长度 ≤ `max_num_batched_tokens`）；多 chunk 前缀
  与前缀缓存命中会记录 `fallback`，不计入消费证据。
- profiling fake-meta 的 16k dummy 调用不进消费统计（否则 16384² P
  会污染概率图样本）。
- 本用例沿用 runner 的 TP=1 单引擎口径（与 4.2 段一致）；profile 的
  TP=2 / framework 参数不在该路径。
- 卡 5 在本容器内核态挂起（异常中止的引擎进程残留所致），本实测改用
  空闲卡 0/3；卡 5 未恢复前不要选它跑引擎。
