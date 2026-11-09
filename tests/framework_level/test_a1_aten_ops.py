# 真实 vLLM 的 A1 aten 多算子框架级验证（第 4 步落地）。
#
# 与 test_a1.py（锚定 silu_and_mul 的官方 harness）互补: 本用例验证
# 「本仓库自研的 9 个 A1 aten 算子」在真实 vLLM 前向中被消费，且注入
# 前后输出逐位一致。
#
# 运行（需 vLLM + 模型；无 vLLM 自动 skip）:
#   FLAGOS_DEVICE=cambricon pytest tests/framework_level/test_a1_aten_ops.py -q
#   # 指定模型: FLAGOS_MODEL_PATH=/path/to/model
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

pytest.importorskip("vllm")

from common.device import load_profile  # noqa: E402
from _aten_harness import run_real_vllm  # noqa: E402


def test_a1_aten_real_vllm():
    profile = load_profile(os.environ.get("FLAGOS_DEVICE", "cambricon"))
    r = run_real_vllm(profile)
    assert r["output_match"], f"注入前后输出不一致: {r}"
    assert r["total_calls"] > 0, f"没有 A1 aten 算子被真实前向消费: {r}"
    print(f"[a1-aten] {r['counts']} total={r['total_calls']}")
    ops = [o.strip() for o in
           os.environ.get("FLAGOS_A1_OPS", "").split(",") if o.strip()]
    if "sdpa_math" in ops:
        sdpa = r.get("sdpa") or {}
        assert r["counts"].get("sdpa_math", 0) > 0, \
            f"sdpa_math 未被真实前向消费: {r}"
        assert sdpa.get("calls", 0) > 0, \
            f"sdpa_math 等价消费者(prefill)未命中: {sdpa}"
        assert sdpa.get("prob_sample", {}).get("probs_rowsum_max_dev", 1) < 1e-2, \
            f"概率图行和偏差异常: {sdpa.get('prob_sample')}"
        grad = sdpa.get("grad") or {}
        assert grad.get("ok"), f"sdpa_math 梯度 probe 失败: {grad}"
        print(f"[a1-aten] sdpa calls={sdpa['calls']} "
              f"prob={sdpa['prob_sample']} grad={grad}")
