# 贡献指南

## 快速开始

```bash
git clone https://github.com/FLAG-OP/FlagOS-OP.git
cd FlagOS-OP
python3 run.py --list                    # 查看设备与矩阵
python3 run.py --route a1 --level op     # 跑最轻的一格
```

## 贡献类型与检查清单

### 新增算子

- [ ] 从 [templates/operator](templates/operator/) 样板起步，目录放
      [ops/](ops/README.md)（`cp -r templates/operator ops/<算子名>/`）
      并在 [ops/catalog.yaml](ops/catalog.yaml) 登记算子、平台和证据
- [ ] **包化入口必带**: 目录含 `__init__.py`（模板已带，改名后更新
      导出）；合并前 `python3 ops/test_packaging.py` 全算子导入扫描
      绿（多算子同进程冲突的硬门，见 #13/#14 实证）
- [ ] **交付自查清单**: 按模板 README「交付自查清单」逐项打勾
      （三层测试/性能对照/黄金/署名等七项）
- [ ] **多平台时**: 补 PLATFORM.md（绑定清单）+ MERGE.md（七步
      流程）——范式见 ops/embedding 双份
- [ ] 补 `goldendata/inputs_spec.yaml` 并生成黄金（含非整倍数维度与
      zeros/large/boundary 特殊用例）
- [ ] `example.py` 含 `main()` 入口 + 设备 profile 参数（不硬编码设备）
- [ ] README 含: 定位 / 运行方式 / 预期输出 / 关键点
- [ ] 若含新 kernel: 确认启动包了 `torch_device_fn.device` 上下文（[known-issues #11](docs/known-issues.md)）
- [ ] 归约类 kernel: 确认尾块安全（pad 或两阶段，[known-issues #15](docs/known-issues.md)）
- [ ] 标注[开发层级](docs/architecture.md#levels)（Triton 级/torch 级/硬件级）
- [ ] 更新 `examples/README.md` 索引表与定位图
- [ ] 产出[测试报告](templates/op-test-report.md)（样例见 [reports/examples](reports/examples/)）

`ops/README.md` 由 `ops/catalog.yaml` 生成，不要手工编辑。登记 catalog 后运行：

```bash
python3 scripts/build_ops_index.py
python3 scripts/build_ops_index.py --check
```

### 新增设备 profile

- [ ] 复制 `configs/devices/_template.yaml`
- [ ] 填写 `vendor`（与 flag_gems `DeviceDetector().vendor_name` 一致）
- [ ] 声明 `vendor_kernels`（含哨兵负例）
- [ ] 生成[黄金](docs/testing.md#golden) + 跑[漂移实验](docs/testing.md#drift)建立基线
- [ ] 用[哨兵检查](docs/testing.md#sentinel)建立该设备的已知问题清单
- [ ] 新增 `configs/env/<profile>.lock.yaml`，运行
      `python3 scripts/build_env_index.py` 更新
      [软硬件环境版本汇总](docs/environments.md)
- [ ] 有协作基座镜像时，在 `configs/images/catalog.yaml` 登记平台、
      系列、版本、分发状态和外部来源，并运行
      `python3 scripts/build_base_image_index.py` 更新
      [基座镜像索引](docs/base-images.md)

### 修改核心测试

- [ ] 保持 `run(profile)` 签名；返回 `bool` 或
      `{"ok": bool, ...指标}`（指标自动落盘到结果 JSON 的 `metrics`）
- [ ] 不引入硬编码设备串/卡号/厂商库名
- [ ] 修改后跑通该路线全部三层

## 提交前检查

```bash
python3 -m pytest tests/unit -q                                            # 基础设施单测（无 GPU）
python3 scripts/check_env.py --device <profile>                            # 匹配所用软件栈的环境锁
python3 scripts/build_ops_index.py --check                                 # catalog 证据等级 / 路径（issue #41）
```

昆仑芯 4.2 基线使用 `p800-kunlunxin`，5.0 复测栈使用
`p800-kunlunxin-gems5`（见[环境配置](docs/installation.md#kunlunxin)）。
真实框架测试前另加 `--require-model` 检查模型目录；5.0 复测容器的默认
模型缺失，不能把软件版本核对通过当作 framework 验证通过。

```bash
# 文档链接与 mermaid 完整性（0 错误才过）
# Python 语法（0 错误才过）
# 相关矩阵格全绿
```

## 提交规范

```
<类型>: <描述>

类型: feat / fix / docs / audit / refactor / test
```

## 已知重要约束

1. **不要设 `VLLM_FL_PREFER=flagos`**（参考实例 P800）——见
   [known-issues #1](docs/known-issues.md)
2. 裸 Triton 启动必须包 `torch_device_fn.device` 上下文——见
   [known-issues #11](docs/known-issues.md)
3. 插件入口必须是 `register` / `vllm_fl_register`——见
   [known-issues #8](docs/known-issues.md)

## 路线图（欢迎贡献）

| 方向 | 说明 | 参考 |
---|---|---|
| ~~reduction 类算子样例~~ | ✅ softmax-fullstack 已完成 | — |
| ~~autotune 集成~~ | ⚠ 本栈 autotuner 选出非法 num_warps 导致错误结果，softmax 已改固定配置（known-issues #15b） | — |
| ~~backward / autograd~~ | ✅ backward-example 已完成 | — |
| ~~多芯片 CI~~ | 部分完成（链接/语法/YAML 检查已有，多设备运行待硬件） | GitHub Actions |
| 昆仑芯 SDK 集成 | SDK 到位后激活 sdk_template 中的 XPU kernel | hw-kernel-example |
