# A1 框架级测试的跨进程 aten 注入桥。
#
# vLLM v1 的模型前向跑在 EngineCore/Worker 子进程中，主进程的
# torch.library 注册不会传播。把本目录加入 PYTHONPATH 后，每个
# Python 进程（含 spawn 子进程）启动时都会执行 sitecustomize，
# 从而在每个进程内完成 aten::silu 注册。
import os

if os.environ.get("FLAGOS_TEMPLATES_A1_INJECT") == "1":
    try:
        import sys

        _root = os.environ.get("FLAGOS_TEMPLATES_ROOT", "")
        if _root and _root not in sys.path:
            sys.path.insert(0, _root)

        from routes.a1_aten import register_aten

        register_aten.register_silu_aten(
            os.environ.get("A1_DISPATCH_KEY", "CUDA")
        )
    except Exception:
        # 注入失败不能影响宿主进程启动（如 tokenizer 等轻量子进程）
        pass

# FlagOS-OP 自研 aten 算子的多算子注入（真实 vLLM 框架级验证用）。
# 与上面的 silu 注入相互独立，可同时开启。
if os.environ.get("FLAGOS_A1_ATEN_INJECT") == "1":
    try:
        import sys

        _root = os.environ.get("FLAGOS_TEMPLATES_ROOT", "")
        if _root and _root not in sys.path:
            sys.path.insert(0, _root)
        # sitecustomize 启动时 torch_npu 尚未 import，部分算子的
        # register_a1 会探测 torch.npu.is_available()（如 sdpa_math）。
        # 此处先确保 torch_npu 已加载，再注册——spawn 子进程启动时
        # NPU 尚未被 vLLM 初始化，import torch_npu 本身是安全的。
        try:
            import torch  # noqa: F401
            import torch_npu  # noqa: F401
        except Exception:
            pass
        import a1_aten_ops

        a1_aten_ops.register_all(
            os.environ.get("A1_DISPATCH_KEY", "PrivateUse1"),
            root=_root or None,
        )
    except Exception:
        pass

# FlagOS C2 引擎侧 attention 后端补丁（sdpa_math × vLLM-Ascend）。
# FLAGOS_SDPA_MATH_ATTN=1 时，把 NPUPlatform.get_attn_backend_cls 补丁为
# 返回 FlagOSMathAttentionBackend——真实前向走 aten math 后端（A1 拦截目标），
# 突破 known-issues #19 的 routing boundary。
# spawn 子进程启动时 sitecustomize 先于 torch_npu 执行；此处直接尝试
# import vllm_ascend.platform（PYTHONPATH 已含源码树）并补丁。
# 若 import 失败则静默，child runner 的兜底补丁会在主进程重试。
if os.environ.get("FLAGOS_SDPA_MATH_ATTN") == "1":
    try:
        import sys as _sys

        _root = os.environ.get("FLAGOS_TEMPLATES_ROOT", "")
        if _root and _root not in _sys.path:
            _sys.path.insert(0, _root)
        from ops.sdpa_math.engine.flagos_math_attn import (
            patch_npu_platform_get_attn_backend_cls,
        )

        patch_npu_platform_get_attn_backend_cls()
    except Exception:
        pass


# ---- 以下两段为 thead-ppu C2 显式路径注入（origin/main a158d05 引入）----

# C2 embedding 注入: 真实 vLLM EngineCore 子进程内注册 aten::embedding。
# 计数写 {A1_EMBEDDING_COUNT_FILE}.{pid}（ops/embedding/register.py）。
if os.environ.get("FLAGOS_C2_EMB_INJECT") == "1":
    try:
        import sys
        import importlib
        import importlib.util
        from pathlib import Path as _P

        _root = os.environ.get("FLAGOS_TEMPLATES_ROOT", "")
        if _root and _root not in sys.path:
            sys.path.insert(0, _root)
        for _m in [m for m in list(sys.modules)
                   if m == "kernel" or m.startswith("kernel.")]:
            del sys.modules[_m]
        _opd = str(_P(_root) / "ops" / "embedding") if _root else None
        if _opd:
            sys.path.insert(0, _opd)
            _spec = importlib.util.spec_from_file_location(
                "_c2embreg", str(_P(_opd) / "register.py"))
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_spec.name] = _mod
            _spec.loader.exec_module(_mod)
            for _sub in ("kernel", "kernel.torch_level", "kernel.triton_level",
                         "kernel.platform", "kernel.thead_ppu",
                         "kernel._native_shim"):
                try:
                    importlib.import_module(_sub)
                except Exception:
                    pass
            _key = os.environ.get("A1_DISPATCH_KEY", "CUDA")
            _plat = os.environ.get("A1_EMB_PLATFORM", "thead-ppu")
            _lib = _mod.register_a1(_key, platform=_plat)
            _C2_EMB_LIB = _lib  # noqa: F841  保持 Library 存活
    except Exception:
        pass

# C2 sdpa_math 注入: 真实 vLLM EngineCore 子进程内注册
# aten::_scaled_dot_product_attention_math，并把 transformers backend 的
# 注意力路径换成 F.sdpa+MATH（显式消费路径，默认 flash 不消费 math 算子）。
# FLAGOS_C2_FORCE_MATH_PATH=1: 仅换路径不注册 A1（baseline 对照用，
# 与注入组同走 MATH，保证 token 可比）。
if os.environ.get("FLAGOS_C2_SDPA_MATH_INJECT") == "1" or \
        os.environ.get("FLAGOS_C2_FORCE_MATH_PATH") == "1":
    try:
        import sys
        import importlib
        import importlib.util
        from pathlib import Path as _P

        _root = os.environ.get("FLAGOS_TEMPLATES_ROOT", "")
        if _root and _root not in sys.path:
            sys.path.insert(0, _root)

        if os.environ.get("FLAGOS_C2_SDPA_MATH_INJECT") == "1":
            for _m in [m for m in list(sys.modules)
                       if m == "kernel" or m.startswith("kernel.")]:
                del sys.modules[_m]
            _opd = str(_P(_root) / "ops" / "sdpa_math") if _root else None
            if _opd:
                sys.path.insert(0, _opd)
                _spec = importlib.util.spec_from_file_location(
                    "_c2sdpareg", str(_P(_opd) / "register.py"))
                _mod = importlib.util.module_from_spec(_spec)
                sys.modules[_spec.name] = _mod
                _spec.loader.exec_module(_mod)
                for _sub in ("kernel", "kernel.torch_level",
                             "kernel.triton_level",
                             "kernel.thead_fast_level",
                             "kernel.p800_fast_level",
                             "reference"):
                    try:
                        importlib.import_module(_sub)
                    except Exception:
                        pass
                _key = os.environ.get("A1_DISPATCH_KEY", "CUDA")
                _impl = os.environ.get("SDPA_MATH_IMPL", "torch")
                _libs = _mod.register_a1(_key, impl=_impl)
                _C2_SDPA_LIBS = _libs  # noqa: F841  保持 Library 存活

        # 显式模型路径: transformers backend 的 ALL_ATTENTION_FUNCTIONS["vllm"]
        # 改为 F.sdpa + MATH backend（会落到 aten::_scaled_dot_product_attention_math）。
        # 不覆盖其他进程/路径；默认 flash 路径不消费该算子（实测）。
        try:
            import torch
            import torch.nn.functional as _F
            from vllm.model_executor.models.transformers.base import (
                ALL_ATTENTION_FUNCTIONS as _AAF,
            )

            def _c2_math_sdpa_forward(
                module, query, key, value, attention_mask,
                scaling=None, **kwargs,
            ):
                # HF 形状 (B,S,H,D) → SDPA (B,H,S,D)
                _q = query.transpose(1, 2)
                _k = key.transpose(1, 2)
                _v = value.transpose(1, 2)
                _scale = scaling
                if _scale is None:
                    _scale = _q.shape[-1] ** -0.5
                _gqa = _k.shape[1] < _q.shape[1]
                with torch.nn.attention.sdpa_kernel(
                    backends=[torch.nn.attention.SDPBackend.MATH],
                ):
                    _out = _F.scaled_dot_product_attention(
                        _q, _k, _v, scale=_scale, enable_gqa=_gqa,
                    )
                return _out.transpose(1, 2), None

            _AAF["vllm"] = _c2_math_sdpa_forward
        except Exception:
            pass
    except Exception:
        pass
