# C2 引擎侧 attention 几何回归（#38 审阅意见 1/2 的 CPU 复现与修复验证）。
#
# 意见 1 场景: 三个单 token decode 请求，KV 值 10/20/30——上游给出的
# actual_seq_lengths_q 是累计端点 [1,2,3]；若再累加切分会混用相邻请求
# KV（实测错误输出 [10,20,20]），正确输出 [10,20,30]。
# 意见 2 场景: cached/chunked prefill Sq=2 Skv=4，Q/K 全零、V=[0,0,0,100]；
# 带历史偏移 causal 的正确输出 [0,25]，旧实现（is_causal=False 无 mask）
# 输出 [25,25]（较早 query 读到后续 token）。
from __future__ import annotations

import pytest
import torch

from ops.sdpa_math.engine.attn_geometry import (
    causal_mask,
    resolve_q_boundaries,
)


def _math(q, k, v, mask=None, is_causal=False):
    out, _ = torch.ops.aten._scaled_dot_product_attention_math(
        q, k, v, mask, 0.0, is_causal, None)
    return out


class TestResolveQBoundaries:
    def test_cumulative_endpoints_single_token_requests(self):
        # 意见 1: 累计端点不得再累加
        assert resolve_q_boundaries([1, 2, 3], 3, num_seqs=3) == [0, 1, 2, 3]

    def test_per_seq_lengths_fallback(self):
        assert resolve_q_boundaries([1, 2, 3], 6, num_seqs=3) == [0, 1, 3, 6]

    def test_single_request(self):
        assert resolve_q_boundaries([8], 8, num_seqs=1) == [0, 8]
        assert resolve_q_boundaries(None, 8, num_seqs=1) == [0, 8]
        assert resolve_q_boundaries([], 8, num_seqs=1) == [0, 8]

    def test_multi_prompt_batch(self):
        assert resolve_q_boundaries([5, 13], 13, num_seqs=2) == [0, 5, 13]

    def test_reject_uninterpretable(self):
        # [1,2] 与 T=4: 既非累计端点（末项≠4）也非逐段长度（和≠4）
        with pytest.raises(RuntimeError):
            resolve_q_boundaries([1, 2], 4, num_seqs=2)
        with pytest.raises(RuntimeError):
            resolve_q_boundaries([1, 2], 10, num_seqs=3)


def test_multi_request_decode_keeps_own_kv():
    """意见 1 端到端 CPU 复现: 三请求各 1 token，KV 值 10/20/30。"""
    boundaries = resolve_q_boundaries([1, 2, 3], 3, num_seqs=3)
    assert boundaries == [0, 1, 2, 3]
    kv_values = [10.0, 20.0, 30.0]
    # query [T=3,H=1,D=1] 全零; key/value 每 token 一个标量
    q_all = torch.zeros(3, 1, 1)
    k_all = torch.tensor([[x] for x in kv_values])
    v_all = torch.tensor([[x] for x in kv_values])
    outs = []
    for i in range(3):
        q0, q1 = boundaries[i], boundaries[i + 1]
        q_i = q_all[q0:q1].transpose(0, 1).unsqueeze(0)   # [1,1,1,1]
        k_i = k_all[q0:q1].transpose(0, 1).unsqueeze(0)   # [1,1,1,1]
        v_i = v_all[q0:q1].transpose(0, 1).unsqueeze(0)
        out_b, _ = torch.ops.aten._scaled_dot_product_attention_math(
            q_i, k_i, v_i, None, 0.0, True, None)
        outs.append(float(out_b.flatten()[0]))
    assert outs == pytest.approx([10.0, 20.0, 30.0])


def test_causal_mask_cached_prefill_reads_no_future():
    """意见 2 端到端 CPU 复现: Sq=2 Skv=4, V=[0,0,0,100] → [0,25]。

    加性 mask（0.0/-inf）；若误用 bool mask（math op 当 1.0/0.0 相加）
    会得到 [10.92, 25]——见 attn_geometry.causal_mask 注释。
    """
    mask = causal_mask(2, 4)
    assert torch.isfinite(mask).tolist() == [[True, True, True, False],
                                             [True, True, True, True]]
    q = torch.zeros(1, 1, 2, 1)
    k = torch.zeros(1, 1, 4, 1)
    v = torch.tensor([0.0, 0.0, 0.0, 100.0]).view(1, 1, 4, 1)
    out = _math(q, k, v, mask=mask, is_causal=False)
    assert out.flatten().tolist() == pytest.approx([0.0, 25.0])


def test_causal_mask_decode_sees_full_history():
    mask = causal_mask(1, 3)
    assert torch.isfinite(mask).all()
    q = torch.zeros(1, 1, 1, 1)
    k = torch.zeros(1, 1, 3, 1)
    v = torch.tensor([3.0, 6.0, 9.0]).view(1, 1, 3, 1)
    out = _math(q, k, v, mask=mask, is_causal=False)
    assert float(out.flatten()[0]) == pytest.approx(6.0)


def test_causal_mask_full_prefill_matches_is_causal():
    mask = causal_mask(3, 3)
    allowed = torch.isfinite(mask)
    assert allowed.tril().tolist() == allowed.tolist() and allowed.sum() == 6
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    out_mask = _math(q, k, v, mask=mask, is_causal=False)
    out_causal = _math(q, k, v, mask=None, is_causal=True)
    assert torch.allclose(out_mask, out_causal, atol=1e-6)


def test_causal_mask_rejects_sq_gt_skv():
    with pytest.raises(RuntimeError):
        causal_mask(5, 3)
