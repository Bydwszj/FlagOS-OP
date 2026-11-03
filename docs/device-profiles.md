# 设备 Profile

[← 返回文档中心](index.md)

> 中央测试入口通过 YAML profile 选择设备、框架参数和结果归档名称。
> `ops/` 下独立算子入口还使用各自的本地 profile 加载器，参数范围以该
> 算子的说明为准。profile 可加载不代表所有测试层级已经在该环境验证。

## 设计

中央运行配置声明于 `configs/devices/<profile名>.yaml`，软件版本由
`configs/env/<profile名>.lock.yaml` 记录。同一硬件可对应多个软件环境
profile；选择 profile 不会安装或切换软件包。昆仑芯 4.2 / 5.0 的选择与
核验命令见[安装指南](installation.md#kunlunxin)。

<a id="fields"></a>
## 字段速查

| 字段 | 作用 |
|---|---|
| `vendor` | 与 flag_gems DeviceDetector().vendor_name 匹配（自动探测用） |
| `auto_detect.module` | 可 import 即识别该设备 |
| `auto_detect.enabled` | 默认 true；false 时不进入自动探测候选（仅 `--device` / `load_profile()` 显式选择） |
| `device.torch_device` | 算子层测试设备串（cuda:1 / xpu:0 / npu:0 / cpu） |
| `device.visible_devices` | framework harness / b-fullstack stage3 写入子进程 `CUDA_VISIBLE_DEVICES` 的卡号，会覆盖外层同名环境变量 |
| `device.dispatch_key` | A1 aten 注册 key（CUDA / PrivateUse1 / CPU ...） |
| `golden_engine` | vllm 或 transformers（CPU 用后者） |
| `framework.*` | vLLM 引擎参数（TP/显存/序列数/模型路径/seed） |
| `framework.quirks.*` | 设备怪癖开关（见下） |
| `vendor_delegate` | audit 委托的厂商 kernel（"pkg.func"）；null→reference |
| `vendor_kernels` | [kernel 层](testing.md#kernel-level)直测的厂商 kernel 清单（含 `expected_ok: false` 负例与 `build: csrc` JIT 项） |

## quirks 说明

- `keep_default_prefer: true` — 保持 `VLLM_FL_PREFER` 继承值（避免覆盖）
  （参考实例 P800: 环境默认非法值恰好让算子回落 vendor kernel，是保护路径）
- `require_two_visible_devices: true` — 记录单卡路径的厂商 reshape_and_cache
  异常约束；当前代码未自动检查该字段。P800 的 TP=2 由
  `framework.tensor_parallel_size: 2` 设置，运行前仍需确认两张卡可见

<a id="kunlunxin-devices"></a>
## 昆仑芯卡号与入口选择

两个 P800 中央 profile 默认 `device.torch_device: cuda:1`、
`device.visible_devices: "1,2"`，框架 TP=2。`cuda:N` 是进程可见设备的
逻辑编号；例如 `CUDA_VISIBLE_DEVICES=5,6` 下 `cuda:1` 指向所选第二张卡，
并非物理卡 1。先用主机 `xpu-smi` 确认空闲卡及容器映射，再执行测试。

| 入口 | profile / 设备选择 | 改卡时注意 |
|---|---|---|
| `ops/{sdpa,sdpa_math,embedding}/example.py` | 本地别名 `p800-kunlunxin`；默认 `cuda:1` | 分别用 `SDPA_TEST_DEVICE`、`SDPA_MATH_TEST_DEVICE`、`EMBEDDING_TEST_DEVICE` 覆盖逻辑设备；这些变量不修改中央 YAML |
| `run.py` 的 kernel/op/consistency、`examples/a1-op`、`examples/b-op` 等中央入口 | 5.0 显式选择 `p800-kunlunxin-gems5`，读取 YAML `device.torch_device` | 不读取上述 `*_TEST_DEVICE`；调整所用 profile 的设备字段并记录配置差异 |
| framework harness、`examples/b-fullstack` stage3 | 从 YAML `device.visible_devices` 设置子进程卡号 | 会覆盖外层 `CUDA_VISIBLE_DEVICES`；框架测空闲卡 5,6 时，还需将所用 profile 的该字段改为 `"5,6"` |

独立算子仅暴露一张卡时，可用对应变量选择 `cuda:0`。这不等于支持 P800
框架单卡运行；框架测试仍按双卡/TP=2 的已记录条件配置。采集报告时记录
实际使用的 profile、卡号和覆盖变量，避免把本地调整写成入库配置的实测结果。

## 模型路径覆盖

profile 里的 `model_path` 是参考实例所在共享镜像的路径。换机器或换模型时
不用改 YAML，设环境变量即可（`load_profile` 统一生效）:

```bash
export FLAGOS_MODEL_PATH=/path/to/compatible-llama-model
python3 scripts/check_env.py --device p800-kunlunxin --require-model
python3 run.py --all --device p800-kunlunxin
```

以上为 4.2 基线示例；5.0 使用 `--device p800-kunlunxin-gems5`，已知 A2 /
consistency 问题和 framework 未验证范围见[复测页](flaggems5-p800-retest.md)。
`scripts/check_env.py --require-model` 会按同一优先级解析并检查目录存在，
不会验证权重、tokenizer 或模型架构。5.0 复测容器缺少默认模型目录，需先补齐。

**对冒烟模型的要求**（参考实例为 llama-3.1-8b-like 随机权重）:

- 结构: Llama 系列 decoder-only，config 与所用 vLLM 版本兼容
- 权重: safetensors 格式，随机权重即可（本库验证的是注入链路，不是生成质量）
- 词表: 常规 Llama 词表大小（tokenizer 一并放入目录）；
  [输入模板](testing.md#inputs)会在 tokenize 后钳制 token 上限
- 引擎参数: 模型结构和上下文长度须与 profile 的 `max_model_len=4096` 等参数匹配；
  更换模型后须在同一模型和软件栈上重新建立黄金，不能沿用另一模型的输出

<a id="onboard"></a>
## 新芯片接入（4 步）

1. 复制 `_template.yaml` 为 `<芯片名>.yaml` 并填写字段
2. `python3 run.py --all --device <芯片名>` 跑 9 格矩阵 + consistency
3. 有厂商 C++ kernel 时按 [BUILD.md](../routes/b_vendor/csrc/BUILD.md) 编译接入，
   并在 `vendor_kernels` 声明（含[哨兵负例](testing.md#sentinel)）
4. 建议立即: 生成[黄金](testing.md#golden) + 跑一次[漂移实验](testing.md#drift)
   + 用[哨兵检查](testing.md#sentinel)建立你自己的[已知问题清单](known-issues.md)

## 自动探测顺序

1. 各 profile 的 `auto_detect.module` 可 import → 选中
2. flag_gems `DeviceDetector().vendor_name` 匹配 `vendor` 字段 → 选中
3. 失败则报错并列出可用 profile（提示 `--device` 显式指定）

`auto_detect.enabled: false` 的 profile 不进入以上两条路径的候选，仅可
显式 `--device` / `load_profile()` 选择。自动探测只选择**默认硬件配置**，
**不校验软件栈版本**——同硬件多软件栈（如 P800 4.2 / 5.0）必须显式指定
profile，软件版本用 `scripts/check_env.py` 核对。

## 参考实例

- `p800-kunlunxin`: 4.2.1rc0 共享镜像基线，历史曾完成 9/9 矩阵与跨层
  一致性验证；后续插件已记录同名实现冲突，不能仅凭版本锁推断当前 A2
  全绿（见[已知问题](known-issues.md)条目 23）
- `p800-kunlunxin-gems5`: FlagGems 5.0 复测栈——已完成的 kernel / A1 /
  mini-consumer 证据与 4.2 基线一致；A2 插件路线 BLOCKED
  （known-issues #23），framework 矩阵因冒烟模型缺失未复测
  （见 [5.0 复测页](flaggems5-p800-retest.md)）
- `mlu590`: SDPA 算子三层 + 黄金 397/397 全绿（A1 `AutogradPrivateUse1`；
  fused/TMO 委托修订后 1.00-1.33x 原生，问题见 known-issues.md #18）
- `nvidia`: profile 就绪；曾借 CUDA 兼容层在加速卡环境验证过
  profile 切换机制（真实 N 卡的 vLLM 引擎路径已备好）
- `cpu`: transformers 黄金引擎已实测（确定性语义参考）
- `ascend910`: 历史 SDPA 交付栈（vllm 空壳，kernel/op 层收口）；
  `ascend910-c2-vllm`: 同硬件真实 vLLM-Ascend C2 栈双 profile
  （editable vllm + vllm-ascend 源码树，`auto_detect.enabled: false`
  仅显式选择；C2 证据与环境锁见
  [environments](environments.md) 与 `ops/*/reports/c2_ascend910.md`）

其他芯片: 复制 `_template.yaml` 填写后即可进入同一体.

---

**下一步**: [报告指南](reporting.md)——环境配置章节直接复用
profile 全文与环境快照。
