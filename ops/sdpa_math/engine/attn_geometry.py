# 引擎侧注意力几何工具（纯函数，无 vllm-ascend 依赖，CPU 可单测）。
#
# 供 flagos_math_attn.py 使用。拆出的原因（#38 审阅意见 1/2）:
#   - query 切分必须按上游 metadata 的**累计端点**构造——vllm-ascend
#     @367b8e62 的 builder 给出的 actual_seq_lengths_q =
#     query_start_loc_cpu[1:].tolist() 是累计端点（attention_v1.py:323），
#     再次累加会把多请求的 query 切错、混用相邻请求的 KV；
#   - cached/chunked prefill 的 causal 语义是**带历史位置偏移**的下三角
#     （允许 k <= Skv - Sq + q），不是左上对齐下三角；Sq=1 的 decode
#     只是它的特例。此前 Sq<Skv 统一 is_causal=False 且无 mask，会让
#     chunk 内较早 query 读到后续 token。
# wt-2026-10-30-fix
from __future__ import annotations

import torch


def resolve_q_boundaries(
    actual_seq_lengths_q: list[int] | None,
    num_tokens: int,
    num_seqs: int | None = None,
) -> list[int]:
    """返回长度 num_seqs+1 的累计 query 边界 [0, e1, ..., eN=num_tokens]。

    识别两种上游形态:
      1. 累计端点（vllm-ascend 实测形态）: 严格递增且最后一项 == num_tokens；
      2. 逐段长度: 求和 == num_tokens（兼容其它 builder）。
    单请求两者重合（[T]）。歧义/非法输入一律 RuntimeError——宁可显式
    拒绝，也不能静默切错导致相邻请求 KV 串用（#38 意见 1）。
    """
    if not actual_seq_lengths_q:
        return [0, int(num_tokens)]
    a = [int(x) for x in actual_seq_lengths_q]
    n = int(num_seqs) if num_seqs is not None else len(a)
    if len(a) != n:
        raise RuntimeError(
            f"actual_seq_lengths_q 长度 {len(a)} 与序列数 {n} 不一致，"
            "拒绝在未知布局下切分 query（#38 意见 1）")
    cumulative = (
        a[-1] == int(num_tokens)
        and all(a[i] < a[i + 1] for i in range(len(a) - 1))
    )
    if cumulative:
        return [0, *a]
    if sum(a) == int(num_tokens):
        out = [0]
        for length in a:
            out.append(out[-1] + length)
        return out
    raise RuntimeError(
        f"无法解释 actual_seq_lengths_q={a}（num_tokens={num_tokens}）: "
        "既非累计端点也非逐段长度，拒绝静默计算（#38 意见 1）")


def causal_mask(
    sq: int,
    skv: int,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """带历史偏移的 causal **加性** float mask: 允许位置 (q, k) 满足
    k <= skv - sq + q（值 0.0），其余 -inf。

    覆盖: 普通 prefill（sq==skv → 左下三角）、单 token decode（sq=1 →
    全部可见）、带缓存/chunked prefill 多 token（sq<skv → 右下对齐三角）。

    注意: torch.ops.aten._scaled_dot_product_attention_math 的 attn_mask
    是**加性**语义——bool mask 会被当成 1.0/0.0 相加而不是 select
    （CPU 实测: Sq=2 Skv=4 V=[0,0,0,100] 若传 bool 得 [10.92,25] 而非
    [0,25]），因此必须用 0.0 / -inf 浮点加性 mask。
    """
    if sq > skv:
        raise RuntimeError(
            f"causal 语义要求 Sq<=Skv，得到 sq={sq} skv={skv}——拒绝计算")
    allowed = torch.ones(
        (sq, skv), dtype=torch.bool, device=device
    ).tril(diagonal=skv - sq)
    return torch.zeros((sq, skv), dtype=dtype, device=device).masked_fill(
        ~allowed, float("-inf"))
