#!/usr/bin/env python3
"""真实 vLLM 前向 runner（供 A1 aten 多算子框架级验证使用）。

用法:
  _aten_runner.py <out.json> <model_path>

在真实 vLLM v1 引擎上跑一次最小生成，输出 output_token_ids。
A1 注入由 injection/sitecustomize.py 在本进程（含 EngineCore 子进程）
启动时完成，故本脚本不显式注册任何算子。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("model")
    args = ap.parse_args()

    import torch  # noqa: F401
    # 先于 vllm 导入绑定后端：MLU 走 torch_mlu，P800/XPU 走 torch_xmlir。
    try:
        import torch_mlu  # noqa: F401
    except ImportError:
        import torch_xmlir  # noqa: F401
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, enforce_eager=True, max_model_len=128,
              tensor_parallel_size=1, gpu_memory_utilization=0.2,
              trust_remote_code=True, disable_log_stats=True)
    t0 = time.perf_counter()
    outs = llm.generate(
        ["the future of AI is", "hello world"],
        SamplingParams(max_tokens=16, temperature=0.0, detokenize=False),
    )
    latency = time.perf_counter() - t0
    result = {
        "output_token_ids": [list(o.outputs[0].token_ids) for o in outs],
        "latency_s": round(latency, 4),
    }
    # sdpa_math 框架级梯度证据（仅注入运行；A1 已由 sitecustomize 注册）
    if (os.environ.get("FLAGOS_A1_ATEN_INJECT") == "1"
            and os.environ.get("A1_SDPA_GRADPROBE") == "1"):
        try:
            import a1_aten_ops
            result["sdpa_grad"] = a1_aten_ops.grad_probe()
        except Exception as e:
            result["sdpa_grad"] = {"ok": False, "error": repr(e)}
    Path(args.out_json).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
