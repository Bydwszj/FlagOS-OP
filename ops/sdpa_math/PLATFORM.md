# 平台绑定清单（Platform Binding）— sdpa_math

> 当前交付包含三个平台：**ascend910**（自研 Triton：两段式 probs+PV，
> 小形状单 kernel 融合）、**p800-kunlunxin**（vendor O/LSE + exact-P
> + A1）与 **thead-ppu**（平头哥 ZW810E：flash/efficient 分流 + exact-P）。
> cpu 为 ATen 组合对照。
> 绑定结论移植时逐项重验；详细实验数据见 `reports/`。

## 1. 可直接复用（平台无关）

| 内容 | 位置 |
|---|---|
| fp32 语义参考 | `reference.py` |
| 黄金规格与数据 | `goldendata/`（CPU 生成） |
| 三层测试、性能脚本 | `test/`、`script/`（位置参数传平台） |
| A1 注册（含平台分发） | `register.py` |

## 2. ascend910 绑定（自研 Triton）

- A1 拦截 key **AutogradPrivateUse1**（sdpa 同结论）
- Triton tile 上限 64×64（单核 192KB UB——见 sdpa
  reports/perf_analysis.md，本算子同栈约束）
- 与 sdpa 全量版的关系：本算子是 `_scaled_dot_product_math` 的
  exact-P 分解路线；形状路由建议沿用 sdpa `kernel/auto_dispatch.py`
  的截流范式（大 S 走原生）
- **C2（vLLM-Ascend，profile `ascend910-c2-vllm`）**（issue #38，
  2026-10-31 复跑 @ `1dc1d8c`；2026-11-01 判定器修复后离线重判
  仍 `complete-pending-review` / exit 0，见
  [reports/c2_ascend910/acceptance_redecide.json](reports/c2_ascend910/acceptance_redecide.json)）:
  **c2-hit / complete-pending-review（待审阅，不写「C2 成立」）**——
  引擎侧 `FLAGOS_SDPA_MATH_ATTN=1` 补丁
  `NPUPlatform.get_attn_backend_cls` 返回 `FlagOSMathAttentionBackend` 后，
  EngineCore 真实前向命中 `aten::_scaled_dot_product_attention_math`
  2880 次（correctness 1440 + warmup 240 + measured 1200，阶段归因自洽）；
  同路径（baseline_math vs inject）C1 恒等 6/6 位级一致、prompt logprob
  max|Δ|=0.17 nats（<0.5 失效线）、性能门禁 inject/baseline_math 中位数比
  1.06 OK。与原生 `npu_fusion_attention` 的路径差（5/6 描述性对照）为
  数值精度差，如实单列、不豁免任何判据。
  **无补丁时默认路由仍为 `npu_fusion_attention`（forward_hits=0，
  known-issues #19）**。详见 [reports/c2_ascend910.md](reports/c2_ascend910.md)
  （含 attention 几何修复: query 累计端点切分 + 历史偏移 causal mask，
  CPU 回归 `tests/unit/test_flagos_math_attn_geometry.py`）。

## 3. p800-kunlunxin 绑定

- XMLIR 呈现 CUDA 张量（AutogradCUDA）
- vendor O/LSE 委托 + exact-P 补偿；`kernel/p800_fast_level.py`
- probes/ 归档（7 个探针）为该平台的证据链

## 3a. thead-ppu 绑定（平头哥 ZW810E，flash/efficient 分流 + exact-P）

环境锁定：开发容器 `flagos-ppu-wt-dev`（torch **2.10.0** CUDA 兼容层 /
triton **3.6.0**），16 卡均为 **PPU-ZW810E**，用卡 **6,7**（避 4/5/11）。
profile 显式 `--device thead-ppu`，测试设备默认 `cuda:6`，
`SDPA_MATH_TEST_DEVICE` 可覆盖。

### 分叉判定（MERGE.md §0 → **结构级**）

sdpa_math 必须物化第二输出 P，不能整体替换为 flash/efficient——与 p800
同构为 vendor O/LSE + exact-P 补偿。故新增 `kernel/thead_fast_level.py`
（非参数级常数调整）。与 p800 的差异是**路由与 LSE 布局**（见下表），
写入本节防止后来者拿 p800 结论误用。

| 绑定项 | 结论 |
|---|---|
| 设备表现 | PPU 经 CUDA 兼容层直曝 `device.type="cuda"`；守卫以设备名含 `PPU` 为准（triton `PPUDriver` 为二路证据） |
| 注册 key | **`AutogradCUDA`**。`probes/thead_first_probe.py`：6/7 命中（零命中仅 `inference_mode`，known-issues #19）；与 p800/T1-sdpa/T2-embedding 一致选它 |
| 中央 YAML `dispatch_key` | **`CUDA`（基键，有意分层）**：中央 A1 消费点读它注入；`AutogradCUDA` inference_mode 零命中（#19.3）故不用。取证见 sdpa PLATFORM §3c-1 |
| 设备编号口径（PR36 条目 3） | yaml `torch_device: cuda:6` 与 `visible_devices "6,7"` 同取物理卡号，三层都落物理卡 6；换算与实测见 sdpa PLATFORM §3c-2 |
| 主实现（maskless 且非 causal 或 Sq==Skv） | flash 直调 `aten::_scaled_dot_product_flash_attention`。**LSE 精确 (B,H,Sq)，无 pad**。T1-sdpa 证原生 F.sdpa 内部即此 kernel |
| 主实现（带 mask 或非方阵 causal） | efficient 委托。**LSE 按 32 倍数 pad**——物化 P 前须 `lse[..., :Sq]` 切片（thead 专属，见下）；16B 不齐走 `_aligned_bias` |
| flash 非方阵 causal 缺陷 | PPU flash 对 Sq≠Skv 的 causal **不生效**（首测 flash O err≈2.0 vs efficient err≈0）——非方阵 causal 强制走 efficient |
| flash dtype 限制 | flash 仅 fp16/bf16（fp32 直调报 `FlashAttention only support fp16 and bf16 data type`） |
| fp32 路径 | 全量走 torch_level ATen 组合。mild 委托 err 3.6e-7 达标但 extreme scale=8 abs≈2e-5 超 1e-5 判卷线（T1-sdpa/p800 同判）；无 p800 bmm JIT 缺陷 |
| GQA | flash/efficient schema 均无 `enable_gqa` → `repeat_interleave` 扩 KV |
| bool_b1 头广播 mask | efficient 要求 bias 头维与 Hq 对齐——`(B,1,Sq,Skv)` 先 `expand` 到 Hq 再进 kernel |
| 全遮蔽行 | vendor LSE=0、out=0；naive exp(score−LSE) 出垃圾 P——mask/causal 零填充后 P=0（native 语义） |
| bf16 boundary | V≈dtype_max/2 时厂商 kernel O 中间量溢出（原生同溢出）；fp32 torch 路径保持有限。tail 判定: 有限子集 rel + inf parity |
| 性能 | `perf/baselines/thead-ppu.json`：0.328ms / 13.10 TFLOPS vs native math 0.782ms = **2.38x**；bench 六形状 0.97-1.06x；A1 包装 ~1.00x native |
| 跨层一致性缺证 | `run.py --consistency` 锚定 gelu_and_mul（A2）硬依赖 `flag_gems`——dev 容器无，结构性缺证（T1/T2 同根因）。op 层拦截 + 注册=直调逐位覆盖 |

表行拆出的补充细节：

- 注册 key 探针每 key 起独立进程；纯 `CUDA` 虽 7/7 全命中但反传带
  “未注册 Autograd kernel”警告，故选 `AutogradCUDA`。
- 中央 A1 消费点：`tests/framework_level/_harness.py` →
  `A1_DISPATCH_KEY` → `injection/sitecustomize.py`；另有
  `scripts/intake_validate.py`、`bench_dispatch.py`、
  `tests/op_level/test_a1.py` 同读。先例 `p800-kunlunxin{,-gems5}`=CUDA、
  `cambricon`=PrivateUse1 同型；复现 `python3 scripts/smoke_dispatch_key.py`。
- 设备编号：全仓仅三处写 `CUDA_VISIBLE_DEVICES`（`_harness.py:62`、
  `drift_study.py:31`、`b-fullstack/example.py:208`）且只写 vLLM 子进程
  env；被重映射的 `_vllm_runner.py` 只读 `engine_args()`/`name`、不读
  `torch_device`——无「重映射后 `cuda:6` 越界」路径。
  `SDPA_MATH_TEST_DEVICE` 覆盖值同样按物理卡号理解。
- LSE pad 实测：Sq=16→32、100→128、197→224；p800 XMLIR 栈无此 pad。
  efficient 的 bool mask 转 additive `-inf` bias；16B stride 不齐走
  `_aligned_bias` 列 pad（报错原文与 bf16 验收见 sdpa PLATFORM §3c-4）。

## 4. 新平台接入

参照 `ops/sdpa/MERGE.md` 七步流程（本算子暂无独立 MERGE.md，结构
对齐后可裁剪复用）；kernel 本体 + register 平台分发 + 三层验证 +
包化 `__init__.py`（已有）四件套不可省。
