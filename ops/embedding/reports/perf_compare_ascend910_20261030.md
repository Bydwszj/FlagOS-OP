# 性能回归对比: ascend910
生成时间: 2026-10-31T01:35:48.390286+00:00

- 阈值: WARN ≥ 20% · FAIL ≥ 30%（仅对 latency_ms 门禁）
- 基线: /root/FlagOS-OP/perf/baselines/ascend910.json（commit `13396c997008`，更新于 2026-09-30T06:54:07）
- 环境: 一致（仅提示，不阻断）

| case | 基线 ms | 本次 ms | Δ | 判定 | 附加指标 |
|---|---|---|---|---|---|
| ops.embedding.ascend910.forward | — | 0.081 | — | NEW | GBps=105.529 |
| ops.embedding.native.forward | — | 0.061 | — | NEW | GBps=140.3 |
| ops.sdpa_math.reference | 0.892 | 0.887 | -0.5% | **OK** | TFLOPS=4.84 |
| ops.sdpa_math.torch | 0.569 | 0.576 | +1.2% | **OK** | TFLOPS=7.457 |
| ops.sdpa_math.triton | 0.504 | 0.532 | +5.6% | **OK** | TFLOPS=8.072 |

## 结论: OK
FAIL 0 · WARN 0 · NEW 2 · 共 5 条