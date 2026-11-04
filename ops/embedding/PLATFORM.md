# 平台绑定清单（Platform Binding）— embedding

> 当前交付包含四个平台：**p800-kunlunxin**（厂商委托同构，
> gatherFIX 探针见 reports/）、**cambricon**（MLU590，Autograd key
> 注册——见 reports/cambricon.md §4）、**thead-ppu**（平头哥 PPU
> ZW810E，CUDA 兼容层——见 reports/thead-ppu.md）与 **ascend910**
> （C2：真实 vLLM-Ascend 前向命中 `aten::embedding` 97 次，2026-10-31
> 复跑 @ `1dc1d8c`，验收状态 complete-pending-review（待审阅）——见 §2
> 与 `reports/c2_ascend910.md`；C0/C1 三层/黄金为 2026-09-30 历史记录）。
> 口径与 `ops/catalog.yaml` / `ops/README.md` / `REPORT.md` 一致（issue #38）。
> 本文件记录不可凭直觉搬运的绑定结论；跨平台移植时逐项重验，不要把
> 单平台实验结果当成普适行为。

## 1. 可直接复用（平台无关）

| 内容 | 位置 |
|---|---|
| fp32 语义参考 | `reference.py` |
| 黄金规格与数据 | `goldendata/`（CPU 生成，天然跨平台） |
| 平台 facade（按张量设备路由） | `kernel/platform.py` |
| 三层测试、性能脚本 | `test/`、`script/`（`EMBEDDING_PROFILE` 切平台） |
| A1 autograd.Function 包装 | `register.py`（`_BACKENDS` 选择器 + `_DEVICE_PROBE` 守卫） |

## 2. ascend910 绑定（C2 归属 profile `ascend910-c2-vllm`，C0/C1 为历史实验）

> **C2 证据（issue #38，2026-10-31 复跑 @ `1dc1d8c`，profile
> `ascend910-c2-vllm`；2026-11-01 判定器修复后离线重判仍
> complete-pending-review / exit 0，见
> [reports/c2_ascend910/acceptance_redecide.json](reports/c2_ascend910/acceptance_redecide.json)）**：真实 vLLM-Ascend
> 双跑，EngineCore 前向命中 `aten::embedding` **97 次** = 1 次引擎初始化
> dummy/profile 前向（`skip_attn`，仍执行 embed_tokens）+ 48 次 correctness
> 前向（6 prompt × 8）+ 8 次 warmup + 40 次 measured（阶段归因自洽），
> token_ids 与基线**逐位一致**（C1 恒等 6/6、prompt logprob 0.0），
> 性能门禁 inject/baseline 中位数比 0.94 OK，主进程/辅助进程 0 命中
> （按 pid 角色标注，防串算）；验收状态 complete-pending-review（待审阅）。
> 见 [reports/c2_ascend910.md](reports/c2_ascend910.md)。
> 下表绑定结论来自 2026-09-30 的 C0/C1 实验（dev profile `ascend910`
> 历史范围），移植时逐项重验。

| 绑定项 | 实测结论 |
|---|---|
| A1 拦截 key | 训练/autograd 路径 **AutogradPrivateUse1**（`PrivateUse1` 在 autograd 路径被 vllm_ascend 注册冲突拒绝）；**推理（no_grad）路径 dispatch 到 PrivateUse1**——C2 引擎证据即用此 key 实证命中（两者按路径区分，见 reports/c2_ascend910.md §4） |
| aten 签名坑 | `aten::embedding_dense_backward` **5 参无 sparse**（与 Python 层 6 参不同） |
| autograd.Function | staticmethod 读模块全局 → `register_a1` 里 `global` 全部三变量（PLATFORM/embedding/embedding_backward） |
| 同步 API | `torch.npu.synchronize`（非 cuda）；`kernel/platform.py::synchronize` 已平台化 |
| 性能 | 委托=包装开销 0.62-0.99x native；Triton gather 探针 18-69x 慢（P800 结论在 Ascend 复现——embedding 是带宽型，自研 Triton 无优势） |

## 3. p800-kunlunxin 绑定

- XMLIR 将 XPU 呈现为 **CUDA 张量**（device.type=="cuda"），dispatch
  key 为 AutogradCUDA
- gatherFIX 探针结论见 `reports/`（正确性 0 error；特化比 native 慢
  10.9-177.9x——与 ascend910 的 Triton 探针结论一致）

## 4. cambricon（MLU590）绑定

- **Autograd key** 注册（与 configs/devices/cambricon.yaml 的
  PrivateUse1 不同）：embedding 带 autograd 的 A1 交付须注册 Autograd
  key 才能携带自带 backward；两个 key 实测都能命中，取 Autograd key
  以免覆盖 functorch 的 PrivateUse1 batch rule（reports/cambricon.md §4）
- 与 ascend910 同为 AutogradPrivateUse1 时：单机双 NPU+MLU 场景须
  `EMBEDDING_PROFILE` 显式指定平台（`_backend_for` 的 platform 参数
  优先于 dispatch_key 推断）

## 5. thead-ppu（PPU ZW810E）绑定

- PPU torch 经 **CUDA 兼容层**直曝 `cuda:N` 张量——与 XMLIR 伪装 CUDA
  同 device.type，`kernel/platform.py::_cuda_impl` 按栈身份路由：
  `torch_xmlir` 可导入 → p800；设备名含 `PPU` → thead；进程级缓存
  （T1 教训: 无缓存 find_spec 路由 80-100µs/调用）。`register.py`
  `_cuda_backend` 同款。
- A1 拦截 key = **AutogradCUDA**（probes/thead_first_probe.py 实测
  6/7 命中; 唯一零命中 inference_mode = known-issues #19 Autograd 键
  通病, 与 p800/T1-sdpa 同结论）。
- 中央 YAML 分层（PR36 条目 2 取证, 2026-10-25）:
  `configs/devices/thead-ppu.yaml` 的 `dispatch_key: CUDA` 是**中央 A1 基键**,
  与本节 `AutogradCUDA` 分层并存（同 cambricon.yaml=`PrivateUse1` /
  本地 `AutogradPrivateUse1` 的分层, 见 reports/cambricon.md §4）。中央消费点
  （`tests/framework_level/_harness.py` → `A1_DISPATCH_KEY` →
  `injection/sitecustomize.py`, 另有 `scripts/intake_validate.py`、
  `scripts/bench_dispatch.py`、`tests/op_level/test_a1.py`）实测基键在
  grad/no_grad/inference_mode 三口径各拦 1 次, 而 `AutogradCUDA` 在
  inference_mode 零命中（known-issues #19.3）——复现
  `python3 scripts/smoke_dispatch_key.py`; 取证表见 `ops/sdpa/PLATFORM.md` §3c-1。
- 设备编号口径（PR36 条目 3 取证, 2026-10-25）: 本地 `_profile.py` 的 `cuda:6`
  与中央 YAML `device.torch_device: cuda:6`、`visible_devices: "6,7"` 同取
  **物理卡号**——kernel / op / framework 三层都落在物理卡 6（中央 vLLM harness
  用 6,7, 其子进程逻辑 0/1 = 物理 6/7）。全仓只有三处写 `CUDA_VISIBLE_DEVICES`,
  且被重映射的 `_vllm_runner.py` 子进程不读 `torch_device`；换算规则与实测见
  `ops/sdpa/PLATFORM.md` §3c-2。`EMBEDDING_TEST_DEVICE` 的覆盖值同样按物理卡号理解。
- **原生 `embedding_backward` 完整**（`padding_idx` 与
  `scale_grad_by_freq=True` 直通, fp16 err 6.1e-5~1.2e-4 累加序噪声）
  ——与 cambricon 同型零补偿；若未来栈升级出现 p800 式
  `Check scale_grad_by_freq == false failed`，回退 p800 的
  inverse-frequency 补偿（p800_kunlunxin.py 已有）。
- forward native `embedding`/`index_select` vs CPU 官方**逐位 err=0**
  （3 dtype × padding 全组合）。
- 同步 API：`torch.cuda.synchronize`（CUDA 兼容层提供），
  `kernel/platform.py::synchronize` cuda 分支直接覆盖。
- 测试尾块维口径：§2d 的 3072/5000/5120 对 embedding 取 gather 长度维
  （#15a 尾块 masked-load 归约陷阱不适用——gather 无归约）。
- dev 栈无 flag_gems：framework_level 跳过 FlagOS 栈门禁（纯 A1 拦截
  口径）；`run.py --consistency`/Triton gather 对比结构性缺证（同 T1
  sdpa 报告 §6.11）。

## 6. 新平台接入

按 `MERGE.md` 七步走；kernel 层实现 `embedding`/`embedding_backward`
+ `PLATFORM` + `_DEVICE_PROBE` 四件即可并入 `_BACKENDS`。
