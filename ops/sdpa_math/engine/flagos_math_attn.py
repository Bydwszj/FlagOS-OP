# FlagOS C2 引擎侧 attention 后端: 强制 vLLM-Ascend 走
# aten::_scaled_dot_product_attention_math（A1 拦截目标算子）。
#
# 背景（docs/known-issues.md #19）: vLLM-Ascend 默认注意力走
# npu_fused_infer_attention_score / npu_fusion_attention，真实前向对
# aten math 后端命中为 0——即使 A1 注册链可用，也无法形成 C2。
#
# 本模块提供一条自定义后端路径（注册名带 FLAGOS_SDPA_MATH 标记避免冲突）:
#   1. 复用 AscendAttentionMetadataBuilder / KV cache 布局（block_size=128），
#      保证 vLLM 的 KV 更新、slot mapping、mask 语义不变；
#   2. 只替换注意力计算本体: gather paged KV → 调
#      torch.ops.aten._scaled_dot_product_attention_math → 写回 output；
#   3. 通过 NPUPlatform.get_attn_backend_cls 的运行时补丁选择本后端
#      （env FLAGOS_SDPA_MATH_ATTN=1），不修改 vllm-ascend 源码。
#
# 设计约束:
#   - enforce_eager=True（禁 ACL/CUDA graph），否则 graph replay 绕过
#     Python 路径（见 c2 harness 注入说明）;
#   - 多序列: 按累计端点切分 query（engine/attn_geometry.py），逐序列
#     调 math op；无法解释的 metadata 布局显式 RuntimeError 拒绝；
#   - causal: 普通 prefill 走 is_causal；cached/chunked prefill / decode
#     用带历史偏移 mask（k <= Skv - Sq + q，见 attn_geometry.causal_mask）；
#   - math op 消化 O(L^2) 内存，仅作 C2 证据路径，非生产默认。
#
# wt-2026-10-28-fix  wt-2026-10-30-fix
from __future__ import annotations

import torch

from ops.sdpa_math.engine.attn_geometry import (
    causal_mask as _causal_mask,
    resolve_q_boundaries as _resolve_q_boundaries,
)
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)

# 注册名: 自定义别名，避免与 AscendAttentionBackend 的 CUSTOM 冲突
FLAGOS_BACKEND_NAME = "FLAGOS_SDPA_MATH"


class FlagOSMathAttentionBackendImpl(AscendAttentionBackendImpl):
    """AscendImpl 子类: KV 更新复用父类，注意力本体改走 aten math。"""

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        # 绕过 paged_attention decode 快路径与 FIA，统一走 math。
        return self.forward_math_attention(
            query, key, value, attn_metadata, output, kv_cache
        )

    def forward_math_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ) -> torch.Tensor:
        """按序列 gather KV 并调 aten math 后端（A1 拦截目标）。"""
        if attn_metadata is None:
            return output.fill_(0)

        # 确保 cache 已绑定（DecodeOnly 时 key/value 为 None）
        if self.key_cache is None and kv_cache is not None:
            if isinstance(kv_cache, torch.Tensor) and kv_cache.dim() > 0 \
                    and kv_cache.shape[0] == 2:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            elif isinstance(kv_cache, (list, tuple)) and len(kv_cache) >= 2:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        state = attn_metadata.attn_state
        seq_lens_list = attn_metadata.seq_lens_list
        actual_q = attn_metadata.actual_seq_lengths_q
        num_seqs = len(seq_lens_list) if seq_lens_list else 1

        # query: [T, H, Dh] → 按累计端点切分为各序列（#38 意见 1）:
        # vllm-ascend builder 的 actual_seq_lengths_q =
        # query_start_loc_cpu[1:].tolist() 已是累计端点，不得再累加——
        # 再累加会把多请求 query 切错、混用相邻请求的 KV。
        # 非法/歧义布局由 resolve_q_boundaries 显式 raise，拒绝静默算错。
        q_offsets = _resolve_q_boundaries(
            actual_q, query.shape[0], num_seqs=num_seqs)

        outs = []
        for i in range(num_seqs):
            q0, q1 = q_offsets[i], q_offsets[i + 1]
            q_i = query[q0:q1]  # [Sq, H, Dh]
            sq = q_i.shape[0]
            skv = seq_lens_list[i] if seq_lens_list else sq

            if state == AscendAttentionState.PrefillNoCache and key is not None \
                    and value is not None:
                # 无缓存 prefill: key/value 即本序列新 token
                k_i = key[q0:q1]
                v_i = value[q0:q1]
            else:
                # 有缓存: 从 paged cache gather 完整历史
                k_i, v_i = self._gather_paged_kv(
                    i, skv, attn_metadata, key, value, q0, q1
                )

            # [S, H, D] → [1, H, S, D] 喂 math op
            q_b = q_i.transpose(0, 1).unsqueeze(0)   # [1, H, Sq, Dh]
            k_b = k_i.transpose(0, 1).unsqueeze(0)   # [1, Hkv, Skv, Dh]
            v_b = v_i.transpose(0, 1).unsqueeze(0)

            # causal 语义（#38 意见 2）:
            #   普通 prefill (Sq==Skv): 左上对齐下三角即正确 → is_causal=True。
            #   decode (Sq=1, Skv=N) 与 cached/chunked prefill (1<Sq<Skv):
            #     query 是序列**尾部**连续段，允许位置 k <= Skv - Sq + q
            #     （带历史位置偏移的下三角）——用显式 bool mask 构造。
            #     此前 Sq<Skv 统一 is_causal=False 且 attn_mask=None，会让
            #     chunk 内较早 query 读取后续 token（CPU 复现: Sq=2 Skv=4,
            #     V=[0,0,0,100] → 错误 [25,25]，正确 [0,25]）。
            is_causal = False
            attn_mask = None
            if getattr(attn_metadata, "causal", False):
                if sq == skv:
                    is_causal = True
                else:
                    # 加性 float mask（0.0/-inf）——math op 的 attn_mask
                    # 不是 select 语义，bool 会被当 1.0/0.0 相加（见
                    # attn_geometry.causal_mask 注释）。
                    attn_mask = _causal_mask(
                        sq, skv, device=q_b.device, dtype=q_b.dtype)

            # math op: 返回 (out, attn_probs)
            out_b, _probs = torch.ops.aten._scaled_dot_product_attention_math(
                q_b, k_b, v_b,
                attn_mask,      # causal 非左上对齐时的显式 mask
                0.0,           # dropout_p
                is_causal,
                None,          # dropout_mask
                scale=self.scale,
                enable_gqa=(self.num_heads != self.num_kv_heads),
            )
            # [1, H, Sq, Dh] → [Sq, H, Dh]
            outs.append(out_b.squeeze(0).transpose(0, 1))

        attn_out = torch.cat(outs, dim=0)
        num_tokens = attn_out.shape[0]
        output[:num_tokens] = attn_out
        return output

    def _gather_paged_kv(
        self,
        seq_idx: int,
        skv: int,
        attn_metadata: AscendMetadata,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        q0: int,
        q1: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从 paged cache 按 block_table gather 单序列完整 KV。

        PrefillNoCache 且 key/value 可用时优先直接切片（避免多余 gather）；
        否则从 self.key_cache / self.value_cache 读取。
        注意: reshape_and_cache 已由父类 forward 写入 cache。
        """
        block_size = self.key_cache.shape[1]
        num_blocks_needed = (skv + block_size - 1) // block_size
        block_row = attn_metadata.block_tables[seq_idx][:num_blocks_needed]

        k_blocks = self.key_cache[block_row]      # [Nb, Bz, Hkv, Dh]
        v_blocks = self.value_cache[block_row]
        k_full = k_blocks.reshape(-1, k_blocks.shape[2], k_blocks.shape[3])
        v_full = v_blocks.reshape(-1, v_blocks.shape[2], v_blocks.shape[3])
        return k_full[:skv], v_full[:skv]


class FlagOSMathAttentionBackend(AscendAttentionBackend):
    """FlagOS 自定义 attention 后端（注册名 FLAGOS_SDPA_MATH）。

    仅替换 impl；metadata builder / KV 布局与 Ascend 一致。
    """

    @staticmethod
    def get_name() -> str:
        # Attention.__init__ 会做 AttentionBackendEnum[self.attn_backend.get_name()]，
        # 必须返回合法 enum 成员名。复用 Ascend 的 HACK: CUSTOM。
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[FlagOSMathAttentionBackendImpl]:
        return FlagOSMathAttentionBackendImpl

    @staticmethod
    def get_builder_cls():
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # 与 AscendAttentionBackend 相同布局
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


_BACKEND_PATH = (
    "ops.sdpa_math.engine.flagos_math_attn.FlagOSMathAttentionBackend"
)


def patch_npu_platform_get_attn_backend_cls():
    """运行时补丁 NPUPlatform.get_attn_backend_cls，返回 FlagOS 后端。

    在 sitecustomize / child runner 的注入路径中调用（spawn 各进程内执行）。
    不修改 vllm-ascend 源文件。返回 True 表示补丁成功。
    """
    try:
        from vllm_ascend.platform import NPUPlatform
    except Exception:
        return False

    # 幂等: 已补丁则跳过
    if getattr(NPUPlatform, "_flagos_math_attn_patched", False):
        return True

    _orig = NPUPlatform.get_attn_backend_cls.__func__

    def _patched(cls, selected_backend, attn_selector_config, num_heads=None):
        use_mla = getattr(attn_selector_config, "use_mla", False)
        use_sparse = getattr(attn_selector_config, "use_sparse", False)
        if not use_mla and not use_sparse:
            return _BACKEND_PATH
        return _orig(cls, selected_backend, attn_selector_config, num_heads)

    NPUPlatform.get_attn_backend_cls = classmethod(_patched)
    NPUPlatform._flagos_math_attn_patched = True
    return True
