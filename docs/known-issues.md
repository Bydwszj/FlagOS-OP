# 已知问题清单

[← 返回文档中心](index.md)

## 关于本文件

本文件记录**参考实例（P800 / Kunlunxin 栈）与 MLU590 栈**的实测问题，用作:

1. 该设备栈使用者的排障参考
2. 其他芯片接入时的检查清单模板——每接入一种新芯片，建议用同样的
   方法（哨兵检查 / drift_study / 黄金比对）建立该设备自己的记录

问题分类: 数值正确性 / 稳定性 / 接口与文档。

## 数值正确性类

### 1. VLLM_FL_PREFER 设为 flagos 会触发 rms_norm 失败（该栈）
FlagGems 的 rms_norm Triton dispatch 实现在真实 vLLM 前向中会失败。
容器默认值（非法值 `flagos|vendor`）恰好让所有算子回落
厂商算子注册——是生产可用路径。测试自定义算子可用
`VLLM_FL_PER_OP` 精确钉住目标算子（避免全局改动）。

### 2. 厂商 kernel 不写输出: xtorch_ops.swiglu（严重）
独立 eager 调用下完全不写输出张量（哨兵测试: torch.full 预填后
调用，全部元素保持原值；返回 int 0）。厂商算子注册 的
silu_and_mul 在此栈上不可靠——生产 vLLM 表面正常疑似依赖 allocator
复用旧激活内存。建议向芯片厂商反馈。

### 3. 同类: xtorch_ops.silu 不写输出
out 参数模式调用后输出全零未写入（sentinel 0/65536）。

### 4. embedding 静默越界
随机模型 tokenizer 词表(151669) > embedding(128256)，文本输入产生
越界 token id。该栈的 embedding 查找不做边界检查（静默读越界
内存）；CPU transformers 则正确报 IndexError。模板已统一
tokenize+词表钳制输入。

### 5. 厂商 kernel 按物理设备分化: gelu_tanh_and_mul
XPU1 上正常（vs 参考 err≈0.125、确定）；XPU2 上非确定且输出 inf。
kernel 层哨兵检查稳定抓出——**逐设备直测的必要性实证**。

5.0 复测补注: 物理 XPU1 另在 e2e/矩阵乘中触发
`KL_XID_KERNEL_EXCEPTION` / `error 299 noc idle timeout`（dmesg 当日多次
历史 Xid，非测试独有）。逐卡复测前先 `xpu-smi` 确认空闲，避开 `cuda:0`；
复现与数据见 [FlagGems 5.0 复测页](flaggems5-p800-retest.md)。

## 稳定性类

### 6. 跨进程后期 token 偶发非确定性
历史观察到 16+ token 偶发漂移；受控漂移实验（reference 8 连跑 +
vendor 4 连跑）全部 100% 确定——属低概率条件相关事件。
防御: 逐 prompt 自适应断言前缀 + 黄金多数票共识。

### 7. 单卡路径通道异常
TP=1 曾触发厂商 reshape_and_cache 通道级异常，设备 profile 固定 TP=2。

## 接口/文档类

### 8. 插件入口函数名不一致
dispatch 文档写 `register_builtins`，代码实际查找
`register` / `vllm_fl_register`——外挂插件入口函数名需与代码查找一致。

### 9. vLLM v1 前向在子进程
主进程 torch.library 注册不传播到 EngineCore/Worker 子进程。
aten 路线框架级需 sitecustomize 注入桥（本库已内置）。

### 10. 性能基准的分配器陷阱
逐次新分配大输出的长循环基准（如 500 次 × 128MB）触发分配器
池增长，均值被抬高一两个数量级（实测同 kernel 0.047ms vs 16ms）。
两层防护: ① 短采样（≤100 次）；② 多用例不共进程——
[perf_run](performance-regression.md) 对每个用例起独立子进程。
短采样单独不够: 同一 C++ kernel 单独跑 0.311ms，与其他用例同进程
顺序混跑（仍 20+100 短采样）被抬到 1.688ms；A2 参考实现
3.69ms → 317ms。分配器状态污染跨用例传播，隔离后恢复健康态。

### 11. ⚠️ 裸 Triton 启动缺 device 上下文 → 静默 no-op（严重）
裸 `@triton.jit` kernel 启动若不包
`flag_gems.runtime.torch_device_fn.device(...)` 上下文，**首次编译启动
正常，之后所有启动静默 no-op**（输出为未初始化内存，不报任何错）。
FlagGems 算子内部都包了上下文故从未暴露；自写 kernel 必踩。
表现极具迷惑性：首次结果正确、性能基准测到假数据（实测同一 kernel
假 0.19ms vs 真 0.030ms）。修复（bmm-fullstack 发现，见其 README）:

```python
from flag_gems.runtime import torch_device_fn
with torch_device_fn.device(x.device):
    my_kernel[grid](...)
```

该缺陷已进入 [intake 负例](ai-intake.md): AI 生成 kernel 若漏包此
上下文，[intake 验证](ai-intake.md)的哨兵检查会在 kernel 层直接
拦截（确定性 ✗），返回机器可读的 BLOCKED 原因供生成侧迭代。

### 16. A2 `call_op` 长循环偶发挂起
测量分发开销时发现: 对同一 op 连续 `call_op` 约 300 次的循环在
本栈多次挂起（Ctrl-C 也难中断，需 kill 进程）；同一调用 100 次
以内稳定。挂起出现在 `with_preference` 上下文内的纯 Python 循环，
不涉及新 kernel 编译，复现条件尚不稳定（有时 300 次也能跑完）。
缓解: `scripts/bench_dispatch.py` 默认 100 次短循环；如需复现
排查，从 n=200 起逐步加长。

<a id="method"></a>
### 12. CUDA C++ 设备码（NVIDIA）无法在 P800/XPU 执行
nvcc 编译 NVIDIA CUDA C++ 通过，但 P800 的 XPU 硬件无法识别
NVIDIA PTX 指令集（报 `invalid device function`）。XMLIR 兼容层
只翻译 ATen/Triton 中间表示，不翻译 nvcc 二进制。

P800 真正的硬件级开发需昆仑芯 SDK（本容器未提供）。当前最接近
硬件级的方式是用 xtorch_ops 厂商原语组合（见 hw-kernel-example）
或 Triton 级开发（→ XMLIR → XPU 指令）。

### 13. FlagGems `gelu(approximate="tanh")` 在本栈链接失败
FlagGems 4.2.1rc0 的 `gelu_tanh` kernel 使用 `tanh` 内建函数，而本栈
xtriton 的 XPU libdevice 表中该 fp32 重载映射到占位符符号
`"Unsupported"`，`xpu*-elfconv` 链接时报
`ld.lld: error: undefined symbol: Unsupported`。同栈实测:
`gelu(none)` / `silu` / `softmax` / `bmm` 均正常，仅 tanh 路径损坏。

影响: A1（gelu_tanh）与 A2（gelu_and_mul）**没有同语义的 FlagGems
生产基线**；自研 kernel 用 `1 - 2/(exp(2x)+1)` 手写 tanh 故不受影响。

责任边界：该问题位于算子实现 / FlagGems 与框架集成边界，当前不属于
FlagOS-OP 的修复范围。本仓库保留复现条件、影响和文档说明；后续跟踪见
[issue #30](https://github.com/FLAG-OP/FlagOS-OP/issues/30)。

**2026-10-27 更新（#30 三条 done criteria）**：

1. **栈与版本**：复现于 P800（XMLIR）+ FlagGems **4.2.1rc0**（xtriton 随包，
   XPU libdevice 表）。同一 P800 硬件、FlagGems **5.0.0**（5.0 复测栈
   `flagos-p800-gems5-vllm0202-dev`，torch 2.9.0+cu129 / torch_xmlir
   1.0.0.1）实测 `gelu(none)` 与 `gelu(approximate="tanh")` 均通过
   （`max_diff=0.000e+00`），tanh 路径在 5.0 已修复：

   ```bash
   python3 -c "import torch,flag_gems;flag_gems.enable();x=torch.randn(64,128,device='cuda:0');y=flag_gems.gelu(x,approximate='tanh');r=torch.nn.functional.gelu(x,approximate='tanh');print((y-r).abs().max())"
   ```

2. **组件归属**：FlagGems 4.2 捆绑 xtriton 的 **XPU libdevice 符号表**——
   fp32 `tanh` 重载映射到占位符 `"Unsupported"`，`xpu*-elfconv` 链接报
   `undefined symbol: Unsupported`（另有 known-issues #14 会把该报错掩盖为
   `NameError: sys`）。属 FlagGems/xtriton XPU backend 侧，非 FlagOS-OP。
3. **修复/规避**：升级 FlagGems 5.0（上文实测）即外部修复；4.2 栈保留
   自研 kernel 手写 tanh（`1 - 2/(exp(2x)+1)`）规避，A1/A2 在 4.2 上仍无
   同语义 FlagGems 基线的结论不变。

### 14. XPU Triton 编译错误被 `NameError: sys` 掩盖
两因叠加: ① xtriton `backends/xpu/compiler.py` 的 `run_cmd()` 使用了
未 import 的 `sys`；② torch_xmlir 符号改写会重新导入 xpu backend，
产生一份**不在 `sys.modules`** 的 compiler 实例——对
`triton.backends.xpu.compiler` 打补丁无效。结果: 任何 elfconv 报错
（含 #13）都被 NameError 吞掉。

修复: `common/xpu_compat.py` 经 `triton.backends` 注册表直达第二实例
注入 `sys`（`common/device.py` 导入时自动生效），此后真实错误可见。

### 15. ⚠️ XPU 栈两个 Triton 正确性陷阱（softmax 实测发现）

**a) 尾块 masked load + `tl.sum` 归约错误**: N 非 BLOCK_N 整数倍时，
尾块中 masked load 的结果污染归约——`other=0.0`、`other=-1e30`、
`tl.where` 显式清洗**三种防护全部无效**；无 load 的纯归约正常，
`tl.max` 归约正常，仅 `tl.sum` 中招。实测 N=3072/BN=2048 输出 inf，
N=5120 误差 0.6。原 softmax 样例的测试形状恰好全是整倍数，漏测尾块。

**b) autotuner 选出非法 `num_warps=5`**: 配置表只定义了 4/8/16，
`best_config` 却显示 `num_warps: 5`（非 2 的幂），导致错误执行且
**每次调优结果不同**——同一 kernel 时对时错，极具迷惑性。

修复（softmax-fullstack 已落地）: wrapper 把输入 pad 到 2048 整数倍
（实体填充 -60000，exp 后贡献为 0），kernel 固定 `BLOCK_N=2048`——
全程无 mask、无调优，确定性路径；整数倍 N 零开销。测试形状补充
3072/5000/5120 非整倍数回归。

### 17. P800 SDPA 的七类平台分化（sdpa-op 实测发现）

1. **Ascend Triton 不能零改动平移**: causal 循环边界中的 runtime select
   触发 XMLIR Triton rewrite 失败（`Could not find PtrState returned by
   the loop` / `Unsupported select predicate`）。
2. **ATen 组合 bmm JIT 缺陷**: `S∈(320,640]` 可触发
   `bmm_one_loop` 模板实例化编译错误；SDPA fp32 精确路径将序列 pad 到
   768（更长边保持原长）并用 mask 还原语义。
3. **厂商 efficient attention 全遮蔽行为分化**: 全 False 行返回有限值，
   CPU/参考语义为 NaN；SDPA backend 显式 `masked_fill` 恢复。
4. **efficient backward 参数分化**: q/k/v 反向要求 forward 计算
   log-sumexp；float bias 反向暂不支持 `bias_requires_grad`。SDPA
   direct 路径按需开启 log-sumexp，可微 mask 自动复用 A1 数学 backward。
5. **丢弃输出计时会虚高**: 仅调用 kernel 后丢弃 tensor、再
   `torch.cuda.synchronize()`，可能只测到 launch/异步提交时间，出现
   9000+ TFLOPS 的物理不可能结果。基准必须读取
   `output[0,0,0,0].item()` 强制完成。
6. **FlagGems `_kunlunxin` SDPA Triton 前向性能低**: 固定 tile 后，
   FA2 协议仍比厂商 efficient attention 慢约 5.4-7.0x；raw-pointer 改写
   触发 XMLIR `PtrState returned by the loop`。复现见
   [ops/sdpa/reports/p800-triton-probe.md](../ops/sdpa/reports/p800-triton-probe.md)。
7. **自研固定调度无法突破 dot/loop lowering 平台**: no-mask causal、
   GQA 与尾块可正确，但 mask launch 失败且比厂商路径慢 2.1-6.5x；
   多组 tile/stages/exp/diagonal-order 变体无实质变化。复现见
   [ops/sdpa/reports/p800-custom-schedule.md](../ops/sdpa/reports/p800-custom-schedule.md)。

平台结论与复现命令见
[ops/sdpa/reports/p800-kunlunxin.md](../ops/sdpa/reports/p800-kunlunxin.md)。

### 18. MLU590 SDPA 的五类平台分化（sdpa-op 实测发现）

1. **`efficient/flash/cudnn` attention aten 私有 op 不可用**:
   MLU 上均 CPU fallback 失败（`NotImplementedError: ... from the 'CPU'
   backend`）。**原生 `F.sdpa` 实际走 `_scaled_dot_product_fused_attention_
   overrideable`（CNNL FA v2）**——直调 math private 只能拿到未融合
   分解，早期实现 0.11-0.55x 原生。
2. **bool mask 与 math private 分化**: math 路径直收 bool vs 参考
   err≈0.19；必须先转 additive `-inf` bias（`F.sdpa`/reference 对
   bool 正确）。
3. **FlagGems `_cambricon` attention 慢 10-40x 且无 autograd**: 精度
   397/397 全过，但 1k D64 达 8.4ms（原生 0.31ms）；不可作生产主路径，
   仅兜底/对照。direct 可微调用复用 A1 数学 backward。
4. **全遮蔽行返回有限值**: fused/TMO/math/FlagGems 均不返回 NaN，SDPA
   backend 显式 `masked_fill` 恢复 CPU/aten 语义（同 #17.3）。
5. **A1 注册 key 与 torch_npu 同为 `AutogradPrivateUse1`**:
   `AutogradMLU`/`MLU` 亦命中，但 PrivateUse1 是 torch_mlu 后端 key；
   `register_a1` 守卫已放宽为 torch_npu **或** torch_mlu 可用。

修订后主路径 = TMO FA（半精度）→ fused overrideable；实测 **1.00-1.33x**
原生。`SDPA_MLU_TMO=0` 可强制只走 overrideable。

平台结论与复现命令见
[ops/sdpa/reports/mlu590.md](../ops/sdpa/reports/mlu590.md)。

### 19. SDPA math 后端的七类语义/注册/平台分化（sdpa_math 实测发现）

1. **torch_npu 的 `F.sdpa` 从不调用 math 后端**: NPU 上
   `F.sdpa(sdpa_kernel(MATH))` 对本 op 命中 **0 次**（整条走
   `npu_fusion_attention`），CPU 上才命中 → NPU 消费方只能直调
   `torch.ops.aten._scaled_dot_product_attention_math`。算子层/应用层
   测试据此设计两套入口。证据
   [probes/native_semantics.py](../ops/sdpa_math/probes/native_semantics.py) §5。
2. **math op 直调的 bool mask 是 0/1 加性怪癖**: 直调 `bool` attn_mask
   按加法（遮蔽位 e_add=0.0；全 True 段 e_fill=0.514），而
   `F.sdpa`/reference 语义是 `-inf` 遮蔽 → 本实现取 `-inf`（有意分歧，
   `check_accuracy --impl native` 对 44 组 bool 用例 SKIP 并注明）。
   MLU 侧同类见 [#18.2](#18-mlu590-sdpa-的五类平台分化sdpa-op-实测发现)。
3. **必须成对注册 `Autograd*` + 纯设备 key**: 只注册 `Autograd*` →
   `torch.inference_mode()` 下不命中（counter 不增），反向时 torch 报
   "an autograd kernel was not registered" 警告；只注册纯 key →
   requires_grad 调用先落 `Autograd*`（仍是原生 composite）。两键缺一不可。
4. **同进程多 impl 互相覆盖 + lib 被 GC**: 共享 autograd.Function 类属性
   会互相覆盖（npu=triton、CPU=torch 同进程注册），每次注册必须绑定独立
   子类闭包；返回的 lib 列表**必须持有引用**（GC 后 counter 恒 0，表现为
   "注册了但不拦截"）。
5. **dropout 有两条不同的 native 规则**: 显式 `dropout_mask` →
   `keep=(mask!=0)`、返回的 P **不缩放**、`O=(P/(1-p))@V`（即 `O≠P@V`）；
   随机路径 P/O 同乘 `keep/(1-p)`（自洽）。反向必须用**预 dropout** 的
   P0 雅可比，否则 gradcheck 与黄金都会暴露。
6. **`causal` 与 `attn_mask` 互斥**: 同时给（bool/float 皆然）直接
   raise `Explicit attn_mask should not be set when is_causal=True`；
   全 `-inf` 行返回 P=O=0 而非 NaN。
7. **P800 不能直接移植 Ascend exact-P Triton 双 pass**: 源码级
   row-stat + P-write 结构会触发 XMLIR pointer-state rewrite 失败；
   既有 no-P custom schedule 也慢于 vendor efficient attention。生产路径
   改为 vendor `O/LSE` + `P=exp(scale·QKᵀ-LSE)`，保持双输出契约。

复现: `python3 ops/sdpa_math/probes/native_semantics.py`（6 节证据表）·
黄金互验: `ops/sdpa_math/script/gen_golden.py`。

### 20. 插件平台配置的 per_op_order 覆盖 `with_preference` 钉选
vllm-plugin-FL 自 `38e7dbc`（2026-08-19，"Support kunlunxin backend"；
当前复测构建 `0.2.0+g38e7dbc` 含之，`v0.2.0` tag 尚无此文件）起内置
`dispatch/config/kunlunxin.yaml`，其中
`op_backends.silu_and_mul: [flagos, vendor, reference]` 成为该 op 的
per-op 钉选顺序；`OpManager._resolve_impl` 用
`policy.per_op_order_dict[op] or 默认顺序`——**平台 per_op_order 优先，
`with_preference("vendor")` 上下文被整体绕过**。0.1.0 及更早无此平台
配置，同写法在旧栈有效、新栈会选中 `default.flagos`。
钉选修复（env 优先级 `VLLM_FL_OP_CONFIG`（JSON 文件路径，
`{"op": "a|b"}`）> `VLLM_FL_PER_OP`（内联 `op=a|b;op2=x|y`）>
平台 yaml；`OP_CONFIG` 在 `vllm_fl.utils` import 期被读入模块缓存，
清 env 后需刷新该缓存；policy 单例缓存需重置）:

```python
os.environ["VLLM_FL_PER_OP"] = "silu_and_mul=vendor:<name>|reference"
from vllm_fl.dispatch.policy import reset_global_policy
reset_global_policy()
```

本库已落地: `tests/op_level/test_b.py`、`examples/b-op`、
`examples/b-fullstack`（stage2）——三处均先保存并清掉
`VLLM_FL_OP_CONFIG`（优先级高于 `VLLM_FL_PER_OP`，会覆盖钉选），
`reset_global_policy()` 后断言最终选中 impl_id，运行结束恢复原环境
变量，`--all` 同进程串跑不残留（框架层 harness 与 b-fullstack stage3
的子进程 env 同步 pop）。实测见
[FlagGems 5.0 复测页](flaggems5-p800-retest.md)。

### 21. embedding 反向原生对照 `scale_grad_by_freq=True` 抛 ValueError
XPU native `aten.embedding_backward(scale_grad_by_freq=True)` 抛
`ValueError: Check scale_grad_by_freq == false failed...`（不是
RuntimeError）；`ops/embedding/script/bench_perf.py` 原
`except RuntimeError` 漏接 → 整个基准脚本崩溃。已修:
`except (RuntimeError, ValueError)`——该组合记为 `native_error`、
`native_ms=null`，其余行照常。4.2 报告中该行 native 空值同因；
kernel 层 `scale_freq` 组测自研实现、不走原生对照，不受影响。

### 22. FlagGems 5.0 pointwise_dynamic 把模块级 `__future__` import 拷进生成文件
5.0 新增 `_collect_jit_deps`: 扫描标量函数所在模块顶层
`from ... import ...` 并复制进生成的 codegen 文件**中部**；模块级
`from __future__ import annotations` 被排到固定 import 头之后，每次
`instantiate()` 即
`SyntaxError: from __future__ imports must occur at the beginning of the file`。
4.2.1rc0 无该 codegen（源码无 `_collect_jit_deps`），属 5.0 侧回归。
规避: 使用 `@pointwise_dynamic` 的模块不放模块级 future import
（`examples/a1-op`、`examples/a2-op` 已移除）；根治在 FlagGems 侧——
生成时过滤 `__future__` 条目。

### 23. A2 演示插件与内置 backend 的 `gelu_and_mul` impl_id 冲突（插件注册整体被拒）
vllm-plugin-FL 自 `2d9ae03`（2026-03-23，`v0.1.0+` 全部含）起由内置
flaggems/reference backend 注册 `gelu_and_mul` 的 `default.flagos` /
`reference.torch`，协议为 **packed 单张量** `fn(obj, x)`（内部对半拆
`[..., 2d]`）。本库 A2 插件 `routes/a2_dispatch/plugin` 以**同名
impl_id + 双张量** `fn(obj, x, gate)` 注册:

- `OpRegistry.register_many` 首项即
  `ValueError: Duplicate impl_id 'default.flagos' for op='gelu_and_mul'`，
  批次中止——同批的 `silu_and_mul vendor.triton-template` 一并丢失；
  `discovery` 捕获记 error → **插件注册整体无效**
- 测试按双张量调用落到内置实现 →
  `TypeError: ...gelu_and_mul() takes 3 positional arguments but 4 were given`

实测（5.0 栈）: `tests/op_level/test_a2` FAIL、`run.py --consistency`
收尾格 FAIL、`examples/a2-op` stage2 FAIL。同一冲突在 4.2 环境锁声明的
`vllm_fl 0.1.0`（tag `v0.1.0+vllm0.13.0`）源码上离体注册实验同样复现
——非 5.0 独有；`examples/a2-op/README` 记载的 dispatch PASS 对应内置
注册加入**之前**的更早插件构建。附带影响: `tests/framework_level/test_a2`
与 `examples/a2-framework` 钉选的 `vendor:triton-template` 因同批中止
不会注册（当前另受冒烟模型缺口遮蔽，见
[FlagGems 5.0 复测页](flaggems5-p800-retest.md)）。

修复方向（本轮仅记录，未改代码）: ① 插件改唯一 impl_id 并把 priority
提到 >150（同 token 匹配内按 `(priority, impl_id)` 降序压过内置
DEFAULT=150）；② `fn` 改 packed 单张量协议；③ 上游允许插件覆盖同名
内置注册；④ A2 演示改用内置未占用的算子。

### 24. FlagGems 4.2 P800 `gather` 在 vLLM 采样阶段报 err_code 1
4.2 栈开 FlagGems 跑真实 vLLM 生成时，采样阶段
`xpuLaunchKernel ... Operation not permitted (err_code: 1)`，C2 直接不过；
`USE_FLAGGEMS=0`（走 torch/vendor 兜底）后通过（4.2 与 5.0 两轮 C2 实测均
用该口径）。同硬件的 `gatherFIX` 探针
（`ops/embedding/script/probe_gatherfix.py`）正确性 0 error、但比 native
慢 10.9-177.9x，尚未合入默认路径。5.0 栈开 FlagGems 的 C2 未复测
（本轮同口径 `USE_FLAGGEMS=0`）——FlagGems 侧修复验证前，C2 建议保持
该开关并在此记录。

### 25. Qwen3-8B 在 P800 4.2 栈 EngineCore 初始化阶段挂起
vllm 0.13.0 + vllm_fl 0.1.0（4.2 共享镜像）加载 Qwen3-8B 时 EngineCore
初始化挂死：GPU 100%、显存仅占 166MiB，无 traceback（4.2 C2 实测记录，
见[框架级 README](../tests/framework_level/README.md)）。规避: 4.2 侧 C2
改用现成 Qwen2.5-0.5B。5.0 栈（vllm 0.20.2）未复测 8B；5.0 侧 C2 用
随机权重 random-qwen3-tiny（4L）跑通。根因未查（疑与该栈 8B 权重/配置
路径相关），8B 级验证前需先复现定位。

### 26. P800 bf16 下 `copy_` 与 `type_as`/`Tensor.to` 的 A1 注入递归
P800 bf16 兜底路径会经 `aten::type_as` / `Tensor.to` 转换，而两者内部
重新进入已注册的 `copy_`，`FLAGOS_A1_OPS` 同时含 `copy_` 与
`type_as` 时形成 `copy_ ↔ type_as/to` 无限递归（4.2 C2 实测）。规避:
`copy_` 从 `FLAGOS_A1_OPS` 排除（框架级默认列表不含；需要时改用不经过
`copy_` 的 cast）。5.0 侧本轮未注入 `copy_`，未触发。设计层面
`copy_` 的可注入边界见其 op 报告。

## 通用检测方法

| 问题类型 | 检测工具 |
|---|---|
| kernel 不写输出 / 非确定 | kernel 层哨兵检查（`sentinel_check`） |
| 按设备分化的正确性 | 逐物理设备跑 `--level kernel` |
| 栈升级数值漂移 | 黄金回归（`build_golden` + 框架测试自动比对） |
| 跨进程非确定性 | `scripts/drift_study.py` |
| 裸 Triton 启动静默 no-op | 哨兵检查（zeros 预填看零占比）+ 对照 `flag_gems` 同算子 |
| embedding 越界 | tokenize 后比对词表上限；CPU transformers 交叉验证 |
| FlagGems 某算子链接失败 | `common/xpu_compat` 补丁后看 elfconv 真实 stderr；对照 #13 |
| 尾块归约静默错误 | 精度探针含非整倍数 N（`accuracy_report.py`）；对照 #15 |
| call_op 长循环挂起 | 短循环（≤100 次）规避；排查见 #16 |
| SDPA 平台语义/JIT 分化 | 397 组黄金 + S=333/D=80 尾块用例；对照 #17 / #18 |
| A1 注册漏键 / lib 被 GC | counter dict 逐次递增 + `inference_mode`/消费方两路 intercept 计数；对照 #19 |
| 钉选落空（选中预期外 impl） | 打印/断言 `call_op` 选中 impl_id，`VLLM_FL_PER_OP` 钉选对照；对照 #20 |
| 插件注册整体无效 | 日志搜 `Error calling plugin` / `Duplicate impl_id`；对照 #23 |
| 栈升级后 codegen / 脚本回归 | 路线样例与 bench 脚本全量重跑；对照 #21 / #22 |

---

**下一步**: [样例索引](../examples/README.md)（可运行范本）·
[AI 生成算子接入](ai-intake.md)（缺陷自动拦截）。
