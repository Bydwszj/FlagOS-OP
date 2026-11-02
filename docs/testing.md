# 测试体系

[← 返回文档中心](index.md)

> 这一页汇总所有验证机制，并解释断言为什么这样设计：三个层级各
> 验证什么、哨兵检查如何抓住"不写输出"的坏 kernel、黄金输出和
> 漂移实验如何守住端到端行为。开发步骤本身见
> [全链路指南](fullstack-guide.md)。

## 目录

- [三层验证](#levels) · [kernel 层](#kernel-level) · [framework 层](#framework)
- [断言策略](#assertions)
- [测试输入模板](#inputs)
- [黄金输出](#golden) · [漂移实验](#drift) · [跨层一致性](#consistency)
- 性能维度单独成体系: [性能回归追踪](performance-regression.md)

## 总览

| 路线 \ 验证层级 | 算子库层 · kernel 直测 | 框架层 · op 注册/分发 | 应用消费者层 · framework 验证 |
|---|---|---|---|
| **A1** torch 算子替换 | Triton kernel 直测 | aten 注册+拦截 | aten 恒等计数注入 |
| **A2** FlagOS 融合算子 | Triton kernel 直测 | dispatch 注册 | vendor 身份注入 |
| **B** 厂商算子注册 | 厂商 kernel 直测+哨兵 | vendor 注册/选择 | audit vendor 拦截 |

<a id="levels"></a>
## 三层验证
```mermaid
flowchart TD
    K["算子库层 · kernel 直测<br/>精度·哨兵·性能"] -->|"注册<br/>(aten impl / OpImpl / vendor)"| O["框架层 · op 注册/分发<br/>注册·分发·拦截"]
    O -->|"注入<br/>(PLUGIN_MODULES / PER_OP / sitecustomize)"| F["应用消费者层 · framework<br/>调用计数·输出比对"]
    F --> G["黄金回归<br/>跨会话漂移检测"]
    K -.->|"同算子同输入"| C["跨层一致性<br/>算子库层↔框架层"]
```


| 层级 | 物理栈位置 | 验证什么 | 形式 | 入口 |
|---|---|---|---|---|
| [kernel 层](#kernel-level) | 算子库层 | kernel 本体: 精度/哨兵/性能 | 直测，秒级 | `--level kernel` |
| op 层 | 框架层 | 注册/分发/拦截正确 | 策略切换验证 | `--level op` |
| [framework 层](#framework) | 应用消费者层 | 被测消费者调用且不破坏输出；若使用真实 vLLM 需单独标注 | 基线/插件双跑 | `--level framework` |

<a id="kernel-level"></a>
## kernel 层（硬件语言直测）

不经任何注册/分发，直接验证 kernel——**厂商算子注册路线的核心层**
（详见[路线 B](route-b-vendor.md)）。

这一层回答的问题是: **抛开一切框架机制，kernel 本身的输出对不对、
行为正不正常**。它看似简单，实际是最容易埋雷的一层——本库实测
发现的 15 个问题里，一大半只有在这层才能暴露，因为注册和分发会把
错误稀释或掩盖。三个子项各管一类风险:

1. **精度**: vs PyTorch 语义参考（`common/kernel_spec.py` 的
   `SEMANTIC_REFS`），多 shape×dtype。容差随 dtype 分档
   （fp32=1e-5，bf16/fp16=1e-2），依据见[验收标准](acceptance.md#accuracy)
   的容差权威表
2. **哨兵检查**: 见下
3. **性能**: 短采样（≤100 次）+ synchronize；可重复的结构化记录与
   回归门禁见[性能回归追踪](performance-regression.md)

精度测试还有一个必须写进清单的坑: shape 要包含**非 BLOCK 整数倍的
维度**。softmax 尾块 bug（#15a）长期漏网，就是因为最早的测试形状
恰好全是 2 的幂——N=5120 误差 0.6、N=3072 直接 inf，而
1024/2048/14336 全对。现在样板与 softmax 样例都固定包含
3072/5000/5120 这类尾块维度。

<a id="sentinel"></a>
### 哨兵健全性检查（`sentinel_check`）

哨兵的动机来自一个真实故障: `xtorch_ops.swiglu` 独立调用时**不报
任何错，但输出缓冲区一个字节都没写**——返回值是形状正常的张量，
内容全是未初始化内存。这种缺陷精度测试抓不到（它根本没写，谈不上
对错）、性能测试也抓不到（反而"快"得可疑）。哨兵的思路是主动构造
可判定的证据:

| out_mode | 检查方法 | 能抓的问题 |
|---|---|---|
| `out_param` | `torch.full` 哨兵预填 out → 调用 → 检查哨兵被覆写 | kernel 不写输出（参考 case 检出 2 个） |
| `return` | 同输入两次调用一致（确定性）+ 输入变化输出变化（敏感性） | 非确定 kernel / 按设备分化的垃圾输出（检出 1 个） |

厂商 kernel 清单在[设备 profile](device-profiles.md) 的 `vendor_kernels`
段声明，`expected_ok: false` 的条目作为**负例**——验证哨兵的检测能力。
负例判定有一条实测学来的规则: 概率性缺陷（如 #5 按设备分化）可能
单次不复现，此时降级 WARN 而非 FAIL——否则整格通过依赖"坏 kernel
每次都坏"，测试天然 flaky。

<a id="framework"></a>
## framework 层（应用消费者 harness 编排）

这一层回答两个独立的问题: **应用消费者到底用没用上你的算子，用上
之后输出还对不对**。消费者可以是真实 vLLM，也可以是仓库内的
mini-attention、mini-decoder 或其他最小复现 harness；报告必须写明具体类型。
分开验证是因为失败模式完全不同——"没被调用"
是注入链路断了（vLLM v1 前向在子进程，主进程注册不传播，#9）；
"调用了但输出坏了"是数值语义变了。所以编排成基线/插件双跑: 同一批
prompt、同一个 seed，唯一变量是你的实现，输出差异就能归因到算子。

- 输入: [inputs/prompts.txt](#inputs)（统一 tokenize+词表钳制）
- 基线跑: 无插件，贪心解码，记录输出 token ids
- 插件跑: 按 route 注入对应实现，同 prompt 同 seed
- 断言: 调用计数（pid 分片汇总）> 0 + 输出比对 + [黄金回归](#golden)

验收判定与退出码（#38 意见 4 + 二轮复审意见 1/2）收敛在纯函数
`scripts/c2_acceptance.py::decide`：**采集成功 ≠ 验收通过**。C2 harness
默认 `--mode accept`——以下必需项任一未过即非零退出：
- 真实前向命中（engine/worker 生命周期累计 > 0，不回退主进程累计值）；
- **阶段命中**：correctness 阶段命中支撑正确性验收、measured 阶段命中
  支撑性能验收（init/warmup/probe 独立保留；归因数据缺失不静默放行）；
- **输入证据完整性** `evidence_complete=True`（预期 prompt 集非空、两跑
  完整唯一覆盖、身份对应、token 有效——先验证再比较，双方空输出不得
  产出“6/6 PASS”）；
- 同路径正确性 PASS；
- prompt logprob 失效线（超 0.5 nats 即 FAIL 不豁免）；
- 性能门禁 OK；计数归因自洽（check 缺失也不放行）。
缺证据/子进程失败退出码更高。`--mode collect` 是显式采集模式（采集完成
退出 0，不代表验收通过，报告须标明所用模式）。

三条路线注入方式见各自文档（[A1](route-a1-aten.md) /
[A2](route-a2-dispatch.md) / [B](route-b-vendor.md)）。

### 证据等级：应用消费者与 C0 契约基线

| 证据等级 | 运行对象 | 可以支持的结论 |
|---|---|---|
| **C0 契约** | API、参考实现、直接算子调用 | 接口、数值语义或注册契约满足 |
| **C1 最小消费者** | mini-attention、mini-decoder、专项 consumer | 目标消费者能调用算子并完成局部输出比对 |
| **C2 真实引擎** | 真实 vLLM/目标推理引擎及其 worker 进程 | 指定引擎、版本和配置下的端到端调用与输出结论 |

“framework 通过”只表示测试脚本通过，不能自动等同于 C2。报告、README
和交付索引应同时记录 `consumer`、引擎版本、profile、commit 和日期；缺少
真实引擎时使用“mini consumer 验证”或“真实 vLLM 待验证”等措辞。

一个工程细节值得强调: 调用计数文件必须按 **pid 分片**。tensor
parallel 拉起的多个 worker 会并发写同一计数文件，实测不加处理的
计数会**低估约 35 倍**——不是少几个调用，是数量级错误。

<a id="assertions"></a>
## 断言策略（两种）

为什么不能对所有实现都用"输出完全一致"？因为随机权重模型 + 贪心
解码构成一个**混沌放大器**: 两个数学等价的 kernel，只要在某 token
的 logit 上有 1e-3 微差，解码路径就可能从那里分叉，之后输出面目
全非——这不是 bug，是精度与解码敏感性的正常相互作用。本库实测中
自研 kernel 与参考实现端到端就是这样分叉的，而逐层比对中间张量
却完全一致。所以断言必须按实现类型分档:

| 场景 | 断言 | 原因 |
|---|---|---|
| 数值恒等路径（aten 恒等覆盖 / audit reference 委托） | 逐 prompt 前缀完全一致 | 数学恒等，位级可比 |
| 自定义数值实现（Triton / C++） | 前 2 token 一致率 ≥ 2/3×N | 随机权重+贪心解码把数值微差混沌放大 |

恒等断言前缀**逐 prompt 自适应**: `max(1, min(8, 黄金稳定长度-1))`
——黄金稳定长度是"已观测到分叉的位置"，取 -1 避免踩线。

<a id="inputs"></a>
## 测试输入模板

- 声明式定义: [inputs/spec.yaml](../inputs/spec.yaml)
  （fixed / length_sweep / edge_cases 文本类 + token_based 算子类）
- 生成: `scripts/gen_inputs.py`（`--check` 校验产物与 spec 一致）
- 同一 spec 在任何硬件生成完全相同的输入——[黄金](#golden)可比的前提
- 框架级统一 `tokenize + 词表钳制`（`PROMPTS_AS_TOKENS=1`），
  规避 embedding 静默越界（[已知问题](known-issues.md)）

<a id="golden"></a>
## 黄金输出（golden outputs）

```bash
python3 scripts/build_golden.py --device cpu        # transformers CPU 权威参考
python3 scripts/build_golden.py --device <加速卡>   # vLLM reference 路径 ×N 快照
python3 scripts/compare_golden.py --devices <加速卡>,cpu
```

- 黄金 = N 次独立运行快照集合 + 逐位置多数票共识前缀
- 落盘 `golden/<device>_golden.json`（含引擎指纹/prompts 哈希，应入库）
- 用途: 跨会话回归检测——栈升级改变数值行为时报警
- CPU 走 HuggingFace transformers 直接推理（最权威语义参考，确定性）
- framework 测试自动比对同设备黄金

<a id="drift"></a>
## 漂移实验

```bash
python3 scripts/drift_study.py --device <profile> --runs 8 --mode reference
python3 scripts/drift_study.py --device <profile> --runs 4 --mode vendor
```

量化跨进程非确定性（逐位置一致率/两两首分歧/全员一致前缀），
校准断言前缀。参考实例结论: 受控条件下两模式连跑全部 100% 确定；
偶发后期 token 漂移为低概率条件相关事件。**新芯片接入建议复跑建立基线**。

<a id="consistency"></a>
## 跨层一致性

```bash
python3 run.py --consistency --device <profile>
```

锚定算子 gelu_and_mul，同一输入张量在四处输出两两比对
（张量级，bf16 容差 1e-2）:

- 算子库层直调 Triton/C++ · 框架层 dispatch `call_op` · PyTorch 参考

参考实例结果: 4×4 全零误差矩阵。这是"同一算子在各层验证一致"的
自动化形态；人工跟读版见[全链路指南](fullstack-guide.md)。

---

**下一步**: [性能回归追踪](performance-regression.md)——正确性之外，
同一套用例的可持续性能门禁。
