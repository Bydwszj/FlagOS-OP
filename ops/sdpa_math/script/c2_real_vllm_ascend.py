#!/usr/bin/env python3
"""C2 真实 vLLM-Ascend 三跑 harness（issue #38，同路径对照版）。

对照组（预注册，唯一变量 = 被测 A1 实现，其余控制变量全部固定）:
  baseline      — 原生注意力路径（npu_fusion_attention），无 FlagOS 注入
  baseline_math — **同 MATH 路径无 A1**: FLAGOS_SDPA_MATH_ATTN=1 走
                  FlagOSMathAttentionBackend → aten math 后端（PyTorch 原生
                  实现），不注册 A1。与 inject 的注意力适配器/路径/权重/
                  prompt/seed/dtype/engine 配置完全一致，仅算子实现不同。
  inject        — 同 MATH 路径 + A1 注册 aten::_scaled_dot_product_attention_math
                  （FLAGOS_A1_IMPL=torch，FlagOS torch 级实现）。

baseline vs 其余两组 = **路径差**（math vs npu_fusion_attention），只作描述性
对照单列，不参与同路径正确性/性能判据（thead-ppu-c2 先例:
ops/sdpa_math/reports/thead-ppu-c2.md）。

断言（预注册于本文件，跑前已固化，不得事后缩短前缀或放宽阈值；
依据 docs/testing.md#assertions）:
  C1 恒等（首选）: baseline_math vs inject 逐 prompt 生成前缀完全一致
      （全部 6 个 prompt，位级可比）。
  C2 自定义数值实现下限: 前 2 token 一致率 ≥ 2/3 × N=6。
      未过 C1 但过 C2 → correctness=INCONCLUSIVE（补首分歧/中间张量/logits
      排查，不得直接判过）；C1 C2 都未过 → correctness=FAIL。
  中间证据: teacher-forced prompt logprob 逐 token 差异 + 首分歧位置 +
  黄金回归（script/check_accuracy.py，torch 实现 vs reference）。

阶段计数（意见 3）: FLAGOS_C2_RUN_ID 贯穿 + hit_times 时间轴把引擎/主进程
命中拆到 init/correctness/warmup/measured/probe 阶段窗口；旧文件（run_id
不符）剔除；pid/ppid 角色判定，probe 单列。

性能（意见 4）: 每 mode 预热 1 次后连续 K 次 generate 计时（中位数/均值/
p95/std），同步方式如实记录（llm.generate 同步阻塞；前向在 EngineCore 子
进程，主进程无法对引擎 torch.npu.synchronize()，故为墙钟时延且各组同构）。
同路径性能门禁（预声明）: inject/baseline_math 中位数比 ≤1.2 OK、
≤1.3 WARN、>1.3 FAIL（阈值对齐 docs/performance-regression.md +20%/+30%）；
路径比、合并比仅展示不作门禁。

用法:
  python3 ops/sdpa_math/script/c2_real_vllm_ascend.py --out reports/c2_ascend910
  # 默认 --mode accept: 命中/阶段命中/证据完整性/正确性/性能/归因任一
  # 未过 → 非零退出；--mode collect: 采集完成即 0（不得被当作验收通过）
  # 默认 --profile ascend910-c2-vllm（缺 configs/env/<profile>.lock.yaml → exit 2）
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from c2_acceptance import decide  # noqa: E402  纯函数验收判定（意见 4）
MODEL_DEFAULT = (
    "/root/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M"
    "/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
)

# 预注册 prompt 集（跑前固化于源码，三组共用同一份、同一 tokenizer 输入）。
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(10)) #",
    "Once upon a time, in a land far away, there lived a",
    "Water boils at 100 degrees",
    "2 + 2 = 4, 3 + 3 =",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
]
PROMPT = PROMPTS[0]
MAX_TOKENS = 8
DEFAULT_WARMUP = 1
DEFAULT_REPEATS = 5
# 预声明断言阈值（docs/testing.md#assertions）。
IDENTITY_ALL_PROMPTS = True          # C1: 逐 prompt 前缀完全一致
CUSTOM_MIN_FIRST2_RATE = 2.0 / 3.0   # C2: 前 2 token 一致率下限
# 预声明性能门禁（同路径 inject/baseline_math 中位数比）。
PERF_OK = 1.2
PERF_WARN = 1.3
SYNC_NOTE = (
    "llm.generate 为同步阻塞调用；模型前向在 EngineCore 子进程执行，"
    "主进程无法对引擎进程调用 torch.npu.synchronize()，故计时为 "
    "generate 墙钟时延（含调度/排队，非纯 kernel 时延）；三组同步方式"
    "完全一致，且每组独立进程运行。注入组计数落盘开销见 "
    "hit_files[].dump_s（相对 generate 时延为微秒级）。"
)
_CANN = [
    "/usr/local/Ascend/cann-9.0.0/python/site-packages",
    "/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe",
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
]
MODES = ("baseline", "baseline_math", "inject")


def _child_env(mode: str, workdir: pathlib.Path, run_id: str,
               a1_impl: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    # spawn 模式: sitecustomize 在每个子进程启动时执行（NPU 未初始化），
    # 无 fork 再初始化冲突。injection/ 放 PYTHONPATH 首位。
    parts = [str(ROOT / "injection"), "/vllm-workspace/vllm-ascend", *_CANN]
    env["PYTHONPATH"] = ":".join(parts + [p for p in [env.get("PYTHONPATH")] if p])
    env["VLLM_PLUGINS"] = "ascend"
    env["ASCEND_HOME_PATH"] = "/usr/local/Ascend/cann-9.0.0"
    env["HCCL_VERIFY"] = "0"
    # NPU 必须 spawn，否则 fork 后 torch.npu 再初始化失败
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["FLAGOS_C2_RUN_ID"] = run_id
    # 所有组共用的 FlagOS env 清理（唯一变量 = 是否 A1 注入）
    for k in ("FLAGOS_A1_ATEN_INJECT", "FLAGOS_A1_OPS", "FLAGOS_C2_COUNT_DIR",
              "A1_DISPATCH_KEY", "FLAGOS_SDPA_MATH_ATTN", "FLAGOS_A1_IMPL"):
        env.pop(k, None)
    if mode in ("baseline_math", "inject"):
        env["FLAGOS_SDPA_MATH_ATTN"] = "1"   # 同 MATH 注意力路径
    if mode == "inject":
        env["FLAGOS_A1_ATEN_INJECT"] = "1"
        env["FLAGOS_TEMPLATES_ROOT"] = str(ROOT)
        env["FLAGOS_A1_OPS"] = "sdpa_math"
        env["A1_DISPATCH_KEY"] = "AutogradPrivateUse1"
        env["FLAGOS_C2_COUNT_DIR"] = str(workdir)
        # C2 引擎侧路径显式指定 impl（意见 5: 不再用 setdefault 的
        # 「隐式 torch」表述——此处为显式设置、且落盘到 summary）。
        # triton 在 bf16 大 GQA shape 首轮可能出现异常输出（见
        # reports/c2_ascend910.md 调试记录）。
        if a1_impl:
            env["FLAGOS_A1_IMPL"] = a1_impl
    return env


# 子进程: mode ∈ {baseline, baseline_math, inject}
_CHILD_RUNNER = r"""
import json, math, os, pathlib, statistics, sys, time, traceback

mode, model, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = json.loads(os.environ["FLAGOS_C2_PROMPTS_JSON"])
max_tokens = int(os.environ.get("FLAGOS_C2_MAX_TOKENS", "8"))
warmup = int(os.environ.get("FLAGOS_C2_WARMUP", "1"))
repeats = int(os.environ.get("FLAGOS_C2_REPEATS", "5"))
run_id = os.environ.get("FLAGOS_C2_RUN_ID")
result = {"mode": mode, "ok": False, "pid": os.getpid(), "ppid": os.getppid(),
          "run_id": run_id}
phases = {}
main_snaps = {}
counter = None


def _snap(name):
    if counter is not None:
        main_snaps[name] = int(counter.get("n", 0))


class _Phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        phases[self.name] = {"start": round(self.t0, 6),
                             "end": round(time.time(), 6)}


try:
    with _Phase("setup"):
        import torch
        import torch_npu  # noqa: F401
        result["torch"] = torch.__version__
        result["torch_npu"] = getattr(__import__("torch_npu"), "__version__", "?")
        result["npu_available"] = bool(hasattr(torch, "npu") and torch.npu.is_available())
        result["npu_count"] = torch.npu.device_count() if result["npu_available"] else 0
        result["mp_method"] = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork")
        result["flagos_attn"] = os.environ.get("FLAGOS_SDPA_MATH_ATTN", "0")
        result["a1_inject"] = os.environ.get("FLAGOS_A1_ATEN_INJECT", "0")
        result["a1_impl_env"] = os.environ.get("FLAGOS_A1_IMPL")

        if mode == "inject":
            root = pathlib.Path(os.environ["FLAGOS_TEMPLATES_ROOT"])
            sys.path.insert(0, str(root / "injection"))
            from a1_aten_ops import get_counter, get_reg_info
            counter = get_counter("sdpa_math")
            result["a1_counter_retrieved"] = counter is not None
            # 意见 5: 注册参数事实（impl/counter 是否真正传入 register_a1）
            result["a1_reg_info"] = get_reg_info("sdpa_math")
            try:
                from vllm_ascend.platform import NPUPlatform
                patched = bool(getattr(NPUPlatform, "_flagos_math_attn_patched", False))
                result["attn_patched"] = patched
                if not patched and os.environ.get("FLAGOS_SDPA_MATH_ATTN") == "1":
                    from ops.sdpa_math.engine.flagos_math_attn import (
                        patch_npu_platform_get_attn_backend_cls,
                    )
                    ok = patch_npu_platform_get_attn_backend_cls()
                    result["attn_patched_fallback"] = ok
                    result["attn_patched"] = bool(
                        getattr(NPUPlatform, "_flagos_math_attn_patched", False))
            except Exception as e:
                result["attn_patched_err"] = repr(e)

    from vllm import LLM, SamplingParams
    import vllm
    result["vllm"] = getattr(vllm, "__version__", "?")
    try:
        import vllm_ascend
        result["vllm_ascend"] = str(getattr(vllm_ascend, "__version__", "source"))
        result["vllm_ascend_file"] = getattr(vllm_ascend, "__file__", None)
    except Exception as e:
        result["vllm_ascend_err"] = repr(e)

    with _Phase("init"):
        t0 = time.time()
        llm = LLM(model=model, enforce_eager=True, max_model_len=192,
                  gpu_memory_utilization=0.2, max_num_seqs=1,
                  trust_remote_code=True, disable_log_stats=True)
        result["init_s"] = round(time.time() - t0, 2)
    _snap("after_init")

    # correctness: 全 prompt 集一次 generate，同时取 teacher-forced
    # prompt logprobs——分两次 generate 会把第二次前向的命中混进阶段计数。
    with _Phase("correctness"):
        t1 = time.time()
        outs = None
        try:
            sp = SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                prompt_logprobs=0)
            outs = llm.generate(prompts, sp)
        except Exception:
            result["prompt_logprobs_err"] = traceback.format_exc().splitlines()[-1]
            sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
            outs = llm.generate(prompts, sp)
        result["correctness_s"] = round(time.time() - t1, 2)
        result["outputs"] = []
        result["prompt_logprobs"] = {}
        for idx, o in enumerate(outs):
            result["outputs"].append({
                "prompt_idx": idx,
                "prompt": prompts[idx],
                "token_ids": list(o.outputs[0].token_ids[:max_tokens]),
                "text": o.outputs[0].text,
            })
            plmap = {}
            for pos, d in enumerate(o.prompt_logprobs or []):
                if not d:
                    continue
                lp = next(iter(d.values()))
                plmap[str(pos)] = float(getattr(lp, "logprob", lp))
            if plmap:
                result["prompt_logprobs"][str(idx)] = plmap
    _snap("after_correctness")

    # perf: 预热 + K 次计时（同 prompt / 同采样参数 / 温度 0 确定性输出）
    sp_perf = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with _Phase("warmup"):
        for _ in range(max(warmup, 0)):
            llm.generate([prompts[0]], sp_perf)
    _snap("after_warmup")
    samples = []
    with _Phase("measured"):
        for _ in range(max(repeats, 1)):
            tp = time.perf_counter()
            llm.generate([prompts[0]], sp_perf)
            samples.append(time.perf_counter() - tp)
    _snap("after_measured")
    if samples:
        _s = sorted(samples)
        _p95_idx = max(0, math.ceil(0.95 * len(_s)) - 1)
        result["perf"] = {
            "warmup": warmup,
            "repeats": len(samples),
            "samples_s": [round(x, 4) for x in samples],
            "stats_s": {
                "mean": round(statistics.fmean(samples), 4),
                "median": round(statistics.median(samples), 4),
                "p95": round(_s[_p95_idx], 4),
                "min": round(min(samples), 4),
                "max": round(max(samples), 4),
                "stdev": round(statistics.pstdev(samples), 4)
                         if len(samples) > 1 else 0.0,
            },
        }

    if mode == "inject" and counter is not None:
        with _Phase("probe"):
            # generate/probe 之后: 验证 A1 拦截链在本进程仍生效
            q = torch.randn(1, 4, 8, 16, dtype=torch.float16, device="npu")
            k = torch.randn(1, 4, 8, 16, dtype=torch.float16, device="npu")
            v = torch.randn(1, 4, 8, 16, dtype=torch.float16, device="npu")
            with torch.no_grad():
                out = torch.ops.aten._scaled_dot_product_attention_math(
                    q, k, v, None, 0.0, False, None)
            torch.npu.synchronize()
            result["probe_hits_after_direct_call"] = int(
                counter.get("n", 0)) - main_snaps.get("after_measured", 0)
            result["probe_out_shape"] = list(out[0].shape)
            result["probe_out_dtype"] = str(out[0].dtype)
            if hasattr(counter, "_dump"):
                counter._dump()
        _snap("after_probe")

    result["ok"] = True
except Exception:
    result["error"] = traceback.format_exc()

result["phases"] = phases
result["main_counter_snapshots"] = main_snaps
result["main_counter_scope"] = (
    "仅本进程（child runner）A1 计数快照；引擎子进程计数须读 "
    "FLAGOS_C2_COUNT_DIR 下的 *_hits_<pid>.json 并按 hit_times 归阶段")
pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False))
"""


def run_mode(mode: str, model: str, out_dir: pathlib.Path, run_id: str,
             warmup: int, repeats: int, a1_impl: str | None = None,
             timeout_s: int = 900) -> dict:
    out_path = out_dir / f"{mode}.json"
    # 每组开跑前清理旧 hits 文件 + 本组再以 run_id 复核，双保险防旧文件混入。
    for old in out_dir.glob("sdpa_math_hits_*.json"):
        old.unlink()
    env = _child_env(mode, workdir=out_dir, run_id=run_id, a1_impl=a1_impl)
    env["FLAGOS_C2_PROMPTS_JSON"] = json.dumps(PROMPTS, ensure_ascii=False)
    env["FLAGOS_C2_MAX_TOKENS"] = str(MAX_TOKENS)
    env["FLAGOS_C2_WARMUP"] = str(warmup)
    env["FLAGOS_C2_REPEATS"] = str(repeats)
    cmd = [sys.executable, "-c", _CHILD_RUNNER, mode, model, str(out_path)]
    print(f"[{mode}] start run_id={run_id}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), timeout=timeout_s,
                          capture_output=True, text=True)
    elapsed = round(time.time() - t0, 2)
    (out_dir / f"{mode}.stdout.log").write_text(proc.stdout or "")
    (out_dir / f"{mode}.stderr.log").write_text(proc.stderr or "")
    data: dict = {}
    if out_path.is_file():
        data = json.loads(out_path.read_text())
    data.setdefault("mode", mode)
    data["rc"] = proc.returncode
    data["wall_s"] = elapsed
    data["ok"] = bool(data.get("ok")) and proc.returncode == 0
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"[{mode}] rc={proc.returncode} ok={data.get('ok')} wall={elapsed}s",
          flush=True)
    return data


def merge_hits(out_dir: pathlib.Path, run_id: str) -> tuple[list[dict], list[dict]]:
    """返回 (本次 run_id 的 hits, 被剔除的旧文件/stale 记录)。"""
    keep, stale = [], []
    for p in sorted(out_dir.glob("sdpa_math_hits_*.json")):
        try:
            h = json.loads(p.read_text())
        except Exception:
            stale.append({"path": str(p), "reason": "unreadable"})
            continue
        h["path"] = str(p)
        if h.get("run_id") != run_id:
            h["reason"] = f"stale run_id={h.get('run_id')!r}"
            stale.append(h)
            continue
        keep.append(h)
    return keep, stale


def attribute_hits(hits: list[dict], inject: dict, phases: dict) -> dict:
    """pid/ppid 角色 + 阶段窗口归因（意见 3）。"""
    main_pid = inject.get("pid")
    role_split = {"main": {}, "engine/worker": {}, "unknown": {}}
    outside = 0
    for h in hits:
        pid = h.get("pid")
        if main_pid is None or pid is None:
            h["role"] = "unknown"
        elif pid == main_pid:
            h["role"] = "main"
        else:
            h["role"] = "engine/worker"
        h["ppid_match_main"] = (h.get("ppid") == main_pid)
        buckets = role_split.setdefault(h["role"], {})
        buckets["hits"] = buckets.get("hits", 0) + int(h.get("hits") or 0)
        # hit_times → 阶段窗口
        phase_counts: dict[str, int] = {}
        windows = [(name, w["start"], w["end"])
                   for name, w in (phases or {}).items()]
        for t in (h.get("hit_times") or []):
            for name, s, e in windows:
                if s <= t <= e:
                    phase_counts[name] = phase_counts.get(name, 0) + 1
                    break
            else:
                phase_counts["outside_windows"] = phase_counts.get(
                    "outside_windows", 0) + 1
        h["phase_counts"] = phase_counts
        for k, v in phase_counts.items():
            buckets[k] = buckets.get(k, 0) + v
            if k == "outside_windows":
                outside += v
    return {"by_role": role_split, "outside_windows": outside}


def _prefix(a: list, b: list) -> tuple[int, int | None]:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            return n, n
        n += 1
    if len(a) != len(b):
        return n, n
    return n, None


def _validate_evidence(base: dict, inject: dict) -> list[str]:
    """比较前的输入证据完整性检查（复审意见 1）。

    双方缺失输出/空 token 时 `[] == []` 会被旧逻辑计成恒等成功——
    必须先验证: 预期 prompt 集非空；两跑完整、唯一地覆盖预期 prompt_idx；
    prompt 身份对应；每项 token 输出有效且非空（本 harness 不记录
    finish 元数据，零 token 不能当作合法 EOS 放行）。
    """
    errors: list[str] = []
    expected = list(range(len(PROMPTS)))
    if not expected:
        errors.append("expected prompt set is empty")
        return errors
    for side, run in (("baseline", base), ("inject", inject)):
        outs = run.get("outputs")
        if outs is None:
            errors.append(f"{side}: outputs missing")
            continue
        if not isinstance(outs, list):
            errors.append(f"{side}: outputs is not a list")
            continue
        idxs = [o.get("prompt_idx") if isinstance(o, dict) else None
                for o in outs]
        seen: list = []
        duplicates: list = []
        for i in idxs:
            if i in seen and i not in duplicates:
                duplicates.append(i)
            seen.append(i)
        if duplicates:
            errors.append(f"{side}: duplicate prompt_idx {duplicates}")
        missing = [i for i in expected if i not in seen]
        if missing:
            errors.append(f"{side}: missing prompt_idx {missing}")
        extra = [i for i in seen if i not in expected]
        if extra:
            errors.append(f"{side}: unexpected prompt_idx {extra}")
        by_idx = {o.get("prompt_idx"): o for o in outs
                  if isinstance(o, dict)}
        for i in expected:
            o = by_idx.get(i)
            if o is None:
                continue  # already reported as missing
            if o.get("prompt") != PROMPTS[i]:
                errors.append(f"{side}: prompt_idx {i} identity mismatch")
            t = o.get("token_ids")
            if t is None:
                errors.append(f"{side}: prompt_idx {i} token_ids missing")
            elif len(t) == 0:
                errors.append(
                    f"{side}: prompt_idx {i} token_ids empty"
                    "（无 finish 元数据，不作合法 EOS 放行）")
    return errors


def correctness_compare(base: dict, inject: dict) -> dict:
    """预注册判据的执行（C1 恒等 / C2 自定义数值实现下限）。"""
    if not (base.get("ok") and inject.get("ok")):
        return {"applicable": False, "reason": "base/inject 未全部成功",
                "evidence_complete": False}
    errors = _validate_evidence(base, inject)
    if errors:
        # 证据不完整 → 不进入数值比较（复审意见 1: 先验证、再比较）
        return {
            "applicable": True,
            "evidence_complete": False,
            "evidence_errors": errors,
            "expected_prompt_count": len(PROMPTS),
            "correctness_verdict": "FAIL",
            "identity_pass": False,
            "identity_pass_count": None,
            "prompt_logprob_same_path": None,
        }
    b_outs = {o["prompt_idx"]: o for o in base.get("outputs") or []}
    i_outs = {o["prompt_idx"]: o for o in inject.get("outputs") or []}
    per_prompt, n_identity, n_first2 = [], 0, 0
    for idx in range(len(PROMPTS)):
        b = list((b_outs.get(idx) or {}).get("token_ids") or [])
        i = list((i_outs.get(idx) or {}).get("token_ids") or [])
        pre, div = _prefix(b, i)
        first2_ok = b[:2] == i[:2]
        bitwise = b == i
        n_identity += int(bitwise)
        n_first2 += int(first2_ok)
        per_prompt.append({
            "prompt_idx": idx,
            "prompt": (b_outs.get(idx) or {}).get("prompt")
                      or (i_outs.get(idx) or {}).get("prompt"),
            "baseline_token_ids": b,
            "inject_token_ids": i,
            "common_prefix_len": pre,
            "first_divergence_token_index": div,
            "bitwise_equal": bitwise,
            "first2_equal": first2_ok,
        })
    n = len(PROMPTS)
    rate = n_first2 / n if n else 0.0
    c1 = (n_identity == n) if IDENTITY_ALL_PROMPTS else None
    c2 = rate >= CUSTOM_MIN_FIRST2_RATE
    if c1:
        verdict = "PASS"
    elif c2:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"

    def _lp(b: dict, i: dict) -> dict:
        bd, idd = b.get("prompt_logprobs") or {}, i.get("prompt_logprobs") or {}
        diffs, tokens = [], 0
        per_prompt_out = {}
        for k in sorted(set(bd) & set(idd), key=int):
            common = sorted(set(bd[k]) & set(idd[k]), key=int)
            if not common:
                continue
            ds = [abs(float(bd[k][p]) - float(idd[k][p])) for p in common]
            diffs.extend(ds)
            tokens += len(common)
            per_prompt_out[k] = {"tokens": len(common),
                                 "max_abs_diff": round(max(ds), 6),
                                 "mean_abs_diff": round(sum(ds) / len(ds), 6)}
        return {"prompts_compared": len(per_prompt_out), "tokens": tokens,
                "max_abs_diff": round(max(diffs), 6) if diffs else None,
                "mean_abs_diff": round(sum(diffs) / len(diffs), 6) if diffs else None,
                "per_prompt": per_prompt_out}

    return {
        "applicable": True,
        "evidence_complete": True,
        "evidence_errors": [],
        "evidence_policy": "预期 prompt 集非空；两跑完整、唯一覆盖预期 "
                           "prompt_idx；prompt 身份一致；每项 token_ids "
                           "有效且非空；缺失/重复/错配直接 FAIL（不进入"
                           "数值比较）",
        "criterion_predeclared": {
            "C1_identity": "baseline_math vs inject 逐 prompt 生成前缀完全一致"
                            "（6/6 prompt，位级）",
            "C2_custom_min": f"前 2 token 一致率 ≥ {CUSTOM_MIN_FIRST2_RATE:.3f}"
                              f"（≥{int(CUSTOM_MIN_FIRST2_RATE * n)}/{n} prompt）",
            "policy": "未过 C1 但过 C2 → INCONCLUSIVE（补首分歧/中间张量/"
                      "logits 排查）；两皆未过 → FAIL；不得事后缩短前缀或放宽阈值"
                      "（docs/testing.md#assertions）",
        },
        "per_prompt": per_prompt,
        "identity_pass": c1,
        "identity_pass_count": f"{n_identity}/{n}",
        "first2_match_rate": round(rate, 4),
        "first2_match": f"{n_first2}/{n}",
        "custom_min_pass": c2,
        "correctness_verdict": verdict,
        "prompt_logprob_same_path": _lp(base, inject),
    }


def perf_compare(perfs: dict[str, dict | None]) -> dict:
    def med(mode: str):
        p = perfs.get(mode) or {}
        return ((p.get("stats_s") or {}).get("median"))

    comparisons = [
        {"name": "inject_vs_baseline_math", "gate": True,
         "a": "inject", "b": "baseline_math"},
        {"name": "baseline_math_vs_baseline", "gate": False,
         "a": "baseline_math", "b": "baseline"},
        {"name": "inject_vs_baseline", "gate": False,
         "a": "inject", "b": "baseline"},
    ]
    out = []
    for c in comparisons:
        a, b = med(c["a"]), med(c["b"])
        if not a or not b:
            out.append({**c, "available": False})
            continue
        ratio = round(a / b, 4)
        verdict = ("OK" if ratio <= PERF_OK
                   else "WARN" if ratio <= PERF_WARN else "FAIL")
        out.append({**c, "available": True, "a_median_s": a, "b_median_s": b,
                    "ratio_median": ratio, "verdict": verdict})
    gate = next((x for x in out if x["gate"]), {})
    return {
        "thresholds_predeclared": {"ok": PERF_OK, "warn": PERF_WARN,
                                   "note": "同路径 inject/baseline_math 中位数比；"
                                           "对齐 docs/performance-regression.md "
                                           "+20%/+30%"},
        "sync_note": SYNC_NOTE,
        "per_mode": {m: p for m, p in perfs.items() if p},
        "comparisons": out,
        "gate_verdict": gate.get("verdict", "MISSING"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FLAGOS_MODEL_PATH",
                                                      MODEL_DEFAULT))
    ap.add_argument("--out", default=str(
        pathlib.Path(__file__).resolve().parent.parent / "reports" / "c2_ascend910"))
    ap.add_argument("--skip-baseline", action="store_true",
                    help="跳过 baseline/baseline_math（仅补 inject）")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--mode", choices=("accept", "collect"), default="accept",
                    help="accept: 全部验收必需项通过才 exit 0（CI/验收入口）；"
                         "collect: 显式采集模式，采集完成 exit 0（非验收）")
    ap.add_argument("--a1-impl", default="torch",
                    help="inject 组显式 FLAGOS_A1_IMPL（意见 5，如实落盘；"
                         "空串=不设置，由算子默认实现）")
    ap.add_argument("--profile", default="ascend910-c2-vllm",
                    help="实际验证环境 profile（复审意见 3，必须有对应 "
                         "configs/env/<profile>.lock.yaml 环境锁）")
    args = ap.parse_args()
    model = args.model
    if not pathlib.Path(model).is_dir():
        print(f"model path missing: {model}", file=sys.stderr)
        return 2
    env_lock = ROOT / "configs" / "env" / f"{args.profile}.lock.yaml"
    if not env_lock.is_file():
        print(f"env lock missing for profile {args.profile}: {env_lock}",
              file=sys.stderr)
        return 2
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    run_id = os.environ.get("FLAGOS_C2_RUN_ID") or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{os.getpid()}")
    (out_dir / "run_id.txt").write_text(run_id + "\n")

    a1_impl = args.a1_impl or None
    results: dict[str, dict] = {}
    for mode in MODES:
        if args.skip_baseline and mode != "inject":
            results[mode] = {"mode": mode, "ok": False, "skipped": True}
            continue
        results[mode] = run_mode(mode, model, out_dir, run_id,
                                 args.warmup, args.repeats, a1_impl=a1_impl)
    hits, stale = merge_hits(out_dir, run_id)
    inject = results["inject"]
    phases = inject.get("phases") or {}
    for h in hits:
        if inject.get("pid") is None or h.get("pid") is None:
            h["role"] = "unknown"
        else:
            h["role"] = "main" if h["pid"] == inject["pid"] else "engine/worker"
    attr = attribute_hits(hits, inject, phases)
    engine_bucket = attr["by_role"].get("engine/worker") or {}
    main_bucket = attr["by_role"].get("main") or {}
    engine_hits = int(engine_bucket.get("hits") or 0)
    main_hits = int(main_bucket.get("hits") or 0)
    # 复审意见 2: forward_hits 只取 engine/worker 生命周期累计（不回退
    # 主进程累计值）；正确性/性能验收分别由 decide 强制 correctness/
    # measured 阶段命中（engine 角色），probe/init/warmup 独立保留。
    forward_hits = engine_hits
    engine_phase_hits = {k: int(v) for k, v in engine_bucket.items()
                         if k != "hits"}
    correctness_phase_hits = int(engine_bucket.get("correctness") or 0)
    measured_phase_hits = int(engine_bucket.get("measured") or 0)
    probe_hits = inject.get("probe_hits_after_direct_call")
    hits_total = sum(int(h.get("hits") or 0) for h in hits)
    phase_total = sum(
        int(v or 0) for role, bucket in attr["by_role"].items()
        for k, v in bucket.items() if k != "hits")

    baseline, bmath = results["baseline"], results["baseline_math"]
    correctness = correctness_compare(bmath, inject)   # 同路径（首选）
    correctness_native = correctness_compare(baseline, inject)  # 路径差（描述性）
    perfs = {m: (results[m].get("perf")) for m in MODES}
    perf = perf_compare(perfs)

    # acceptance.status 不在此处单独派生——由下方 decide 同一次判定
    # 统一给出（复审意见 2: 避免 summary 同时留两个互相矛盾的状态）。
    acceptance = {
        "hit": bool((forward_hits or 0) > 0),
        "correctness_phase_hits": correctness_phase_hits,
        "measured_phase_hits": measured_phase_hits,
        "correctness_same_path": correctness.get("correctness_verdict"),
        "perf_same_path_gate": perf["gate_verdict"],
        "rc_all_zero": all(results[m].get("rc") == 0 for m in MODES
                           if not results[m].get("skipped")),
    }

    summary = {
        "schema": 4,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # 复审意见 3: 显式记录实际验证 profile 并核对环境锁（不再硬编码旧名）
        "profile": args.profile,
        "env_lock": f"configs/env/{args.profile}.lock.yaml",
        "engine": "vllm-ascend",
        "commit": commit,
        "run_id": run_id,
        "model": model,
        "prompts": PROMPTS,
        "max_tokens": MAX_TOKENS,
        "python": platform.python_version(),
        "injection": {
            "mp_method": inject.get("mp_method"),
            "attn_backend_patch": inject.get("attn_patched"),
            "flagos_attn_env": inject.get("flagos_attn"),
            "a1_ops": "sdpa_math",
            # 实际值取 inject 子进程亲测 env（意见 5: 显式设置、如实落盘）
            "a1_impl": inject.get("a1_impl_env"),
            "a1_reg_info": inject.get("a1_reg_info"),
        },
        "controls": {
            "note": "三组共用同一模型/权重快照/prompt 集/采样参数(温度0)"
                    "/dtype/enforce_eager/engine 配置；唯一变量 = 是否注册 A1 "
                    "实现（baseline_math vs inject 同走 MATH 注意力路径）",
            "modes": list(MODES),
            "warmup": args.warmup, "repeats": args.repeats,
            "env_overrides_inject": [
                "FLAGOS_A1_ATEN_INJECT=1",
                "FLAGOS_SDPA_MATH_ATTN=1",
                "FLAGOS_A1_OPS=sdpa_math",
                "A1_DISPATCH_KEY=AutogradPrivateUse1",
                "FLAGOS_C2_RUN_ID=<run_id>",
                f"FLAGOS_A1_IMPL={inject.get('a1_impl_env')!r}",
            ],
            "env_cleared_all_modes": [
                "FLAGOS_A1_ATEN_INJECT", "FLAGOS_A1_OPS", "FLAGOS_C2_COUNT_DIR",
                "A1_DISPATCH_KEY", "FLAGOS_SDPA_MATH_ATTN", "FLAGOS_A1_IMPL",
            ],
        },
        "baseline": baseline,
        "baseline_math": bmath,
        "inject": inject,
        "hit_files": hits,
        "hit_files_stale_excluded": stale,
        "hit_attribution": attr,
        "hit_attribution_check": {
            "hits_total": hits_total,
            "phase_attributed_total": phase_total,
            "consistent": hits_total == phase_total,
            "note": "phase_attributed 含 outside_windows 桶（未落入任何阶段"
                    "窗口的命中，如实计入，不删除）",
        },
        "engine_core_hits": engine_hits,
        "main_process_hits": main_hits,
        "forward_hits": forward_hits,
        "engine_phase_hits": engine_phase_hits,
        "correctness_phase_hits": correctness_phase_hits,
        "measured_phase_hits": measured_phase_hits,
        "total_hits_all_incl_probe": hits_total,
        "probe_hits_after_direct_call": probe_hits,
        "outputs_equal": (
            [o["token_ids"] for o in baseline.get("outputs") or []]
            == [o["token_ids"] for o in inject.get("outputs") or []]
            if baseline.get("ok") and inject.get("ok") else None),
        "correctness_same_path": correctness,
        "correctness_native_path_descriptive": correctness_native,
        "perf": perf,
        "acceptance": acceptance,
        "c2_status": None,
        "notes": [],
    }
    # 意见 4 + 复审意见 2: 验收判定与退出码收敛为纯函数
    # （scripts/c2_acceptance.decide）。accept 模式: 命中/阶段命中/证据
    # 完整性/正确性 PASS/logprob 失效线/性能 OK/归因自洽任一未过 → 1；
    # 采集失败/缺证据 → 2。collect 模式: 仅采集完成 exit 0（不代表验收
    # 通过）。acceptance.status 与 acceptance_decision 由同一次判定派生，
    # 避免 summary 留两个互相矛盾的状态。
    decision = decide(summary, mode=args.mode)
    summary["acceptance_decision"] = decision
    summary["run_mode"] = args.mode
    acceptance["status"] = decision["acceptance_status"]

    ok_run = all(results[m].get("ok") for m in MODES
                 if not results[m].get("skipped"))
    if ok_run:
        if (forward_hits or 0) > 0:
            summary["c2_status"] = "c2-hit"
            summary["notes"].append(
                "真实 vLLM-Ascend 前向命中 aten math 后端（引擎侧 "
                "FlagOSMathAttentionBackend 补丁生效）。rc/ok 仅表示运行"
                "成功，不等于正确性通过或 C2 验收通过。")
        elif (probe_hits or 0) > 0:
            summary["c2_status"] = "c2-routing-boundary"
            summary["notes"].append(
                "A1 注册链在 NPU 进程内可用（probe 命中），但真实前向 "
                "forward_hits=0：注意力未走 FlagOS math backend 路径。")
        else:
            summary["c2_status"] = "c2-inject-unverified"
            summary["notes"].append("注入进程 probe 未命中，注册链存疑")
        if acceptance["status"] in ("PARTIAL", "FAILED"):
            summary["notes"].append(
                f"验收状态 {acceptance['status']}: 真实引擎已尝试但必需项"
                "未全部通过（明细见 acceptance_decision.failed_checks）"
                "——按 #38 意见 6 不写「C2 验收通过」。")
        if correctness.get("correctness_verdict") in ("INCONCLUSIVE", "FAIL"):
            summary["notes"].append(
                f"同路径正确性 {correctness['correctness_verdict']}: 首分歧"
                "位置与 prompt logprob 差异见 correctness_same_path，需继续"
                "中间张量排查（意见 2，不放行也不改口径）。")
    else:
        summary["c2_status"] = "c2-failed"
        summary["notes"].append("至少一组跑失败，见 *.stderr.log")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({
        "run_id": run_id,
        "run_mode": args.mode,
        "c2_status": summary["c2_status"],
        "acceptance": acceptance,
        "acceptance_decision": {
            "mode": decision["mode"],
            "collection_status": decision["collection_status"],
            "acceptance_status": decision["acceptance_status"],
            "failed_checks": decision["failed_checks"],
            "warnings": decision["warnings"],
            "exit_code": decision["exit_code"],
        },
        "forward_hits": forward_hits,
        "engine_core_hits": engine_hits,
        "main_process_hits": main_hits,
        "engine_phase_hits": engine_phase_hits,
        "correctness_phase_hits": correctness_phase_hits,
        "measured_phase_hits": measured_phase_hits,
        "phase_attribution": attr["by_role"],
        "stale_excluded": len(stale),
        "correctness_same_path": {
            "verdict": correctness.get("correctness_verdict"),
            "evidence_complete": correctness.get("evidence_complete"),
            "identity": correctness.get("identity_pass_count"),
            "first2": correctness.get("first2_match"),
        },
        "perf_gate": perf["gate_verdict"],
        "perf_ratios": {c["name"]: c.get("ratio_median")
                        for c in perf["comparisons"]},
        "attn_patched": inject.get("attn_patched"),
        "a1_reg_info": inject.get("a1_reg_info"),
        "out": str(out_dir / "summary.json"),
    }, ensure_ascii=False, indent=2))
    return decision["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
