# A1 多算子注入: 在单个进程（含 vLLM EngineCore 子进程）内注册 FlagOS-OP
# 的多个 aten 算子。
#
# 背景: ops/<op>/register.py 的实现用 `from kernel.torch_level import ...`
# 这一「裸 kernel 包名」导入；当多个算子同时注册时，各自的 ops/<op>/kernel/
# 同名包会互相覆盖（先被导入者胜出），导致 A 算子执行到 B 算子的 kernel。
# 本模块为每个算子隔离导入其 kernel 模块，并在该算子的 aten 包装函数调用
# 期间把 sys.modules['kernel*'] 切回它自己的包，从而多算子可同时注册、
# 互不串味（详见 tests/framework_level/test_a1_aten_ops.py）。
#
# 由 injection/sitecustomize.py 在 FLAGOS_A1_ATEN_INJECT=1 时调用。
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[1]

# 可在真实 vLLM 进程内安全注册的算子（实测）。
# 排除项与原因:
#   empty / detach_ / result_type —— 设计上不可 A1 拦截（见各 op 的 register.py）。
#   detach —— 可独立注册，但在真实 vLLM 建模时注册会破坏
#             torch.Tensor._make_subclass(nn.Parameter 构造)，报
#             "raw Tensor object is already associated to a python object
#             of type Tensor"。与 detach_ 类似，属于进程内不可共存的算子。
_DEFAULT_OPS = [
    "type_as", "clone", "contiguous", "copy_", "dropout",
    "empty_like", "empty_strided", "item", "_local_scalar_dense",
]


def _ops() -> list[str]:
    raw = os.environ.get("FLAGOS_A1_OPS")
    if raw is None:
        return list(_DEFAULT_OPS)
    return [x for x in raw.split(",") if x]


def _purge_kernel() -> None:
    for m in [m for m in list(sys.modules)
              if m == "kernel" or m.startswith("kernel.")]:
        del sys.modules[m]


def _load_isolated(root: Path, op: str):
    opd = root / "ops" / op
    reg = opd / "register.py"
    if not reg.is_file():
        return None
    _purge_kernel()
    sys.path.insert(0, str(opd))
    try:
        spec = importlib.util.spec_from_file_location(f"_a1reg_{op}", reg)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        for sub in ("kernel", "kernel.torch_level", "kernel.triton_level"):
            try:
                importlib.import_module(sub)
            except Exception:
                pass
        saved = {n: sys.modules[n] for n in list(sys.modules)
                 if n == "kernel" or n.startswith("kernel.")}
    except Exception:
        return None
    finally:
        try:
            sys.path.remove(str(opd))
        except ValueError:
            pass
    return mod, saved


def _wrap(fn, saved):
    def wrapper(*args, **kwargs):
        old = {}
        for n in saved:
            old[n] = sys.modules.get(n)
            sys.modules[n] = saved[n]
        try:
            return fn(*args, **kwargs)
        finally:
            for n, v in old.items():
                if v is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = v

    wrapper.__name__ = getattr(fn, "__name__", "wrapper")
    return wrapper


def _maybe_counter(op: str):
    """若设置了 FLAGOS_C2_COUNT_DIR，则返回 (counter, dump_fn)。

    计数文件按 pid 分片（docs/testing.md: TP worker 并发写同一文件会
    低估约 35 倍）。dump 在 atexit 与每次递增后触发（FLAGOS_C2_DUMP_MODE
    可切 atexit），进程崩溃也能留痕。

    计数归因（issue #38 意见 3）:
      - run_id 取自 FLAGOS_C2_RUN_ID，父进程用它剔除旧文件/他人运行的
        残留（防串算）；
      - hit_times 记录每次命中的 epoch 秒（__setitem__ 钩子，register.py
        只做 counter["n"] += 1，无需改它）——父进程据此把命中归到
        init/warmup/measured/correctness/probe 各阶段窗口；
      - dump_s 累计本 counter 花在落盘上的秒数，供性能对照披露
        写文件开销（与 baseline 无注入的差异项）。
    """
    count_dir = os.environ.get("FLAGOS_C2_COUNT_DIR")
    if not count_dir:
        return None, None
    import atexit
    import json as _json
    import time as _time
    from datetime import datetime, timezone

    run_id = os.environ.get("FLAGOS_C2_RUN_ID")
    cpath = Path(count_dir) / f"{op}_hits_{os.getpid()}.json"
    Path(count_dir).mkdir(parents=True, exist_ok=True)

    class _Counter(dict):
        def __init__(self) -> None:
            super().__init__(n=0)
            self.hit_times: list[float] = []
            self.registered_at: float = _time.time()
            self.dump_s: float = 0.0

        def __setitem__(self, key, value):
            if key == "n":
                try:
                    if int(value) > int(self.get("n", 0)):
                        self.hit_times.append(_time.time())
                except Exception:
                    pass
            super().__setitem__(key, value)
            # 每次递增后增量落盘（受 FLAGOS_C2_DUMP_MODE 控制）。
            _fn = getattr(self, "_dump_fn", None)
            if _fn is not None:
                try:
                    _fn()
                except Exception:
                    pass

    counter = _Counter()

    def dump():
        _aten_label = {
            "sdpa_math": "aten::_scaled_dot_product_attention_math",
            "embedding": "aten::embedding",
        }.get(op, op)
        _t0 = _time.time()
        cpath.write_text(_json.dumps({
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "op": op,
            "aten": _aten_label,
            "run_id": run_id,
            "hits": counter["n"],
            "hit_times": [round(t, 6) for t in counter.hit_times],
            "registered_at": round(counter.registered_at, 6),
            "dump_s": round(counter.dump_s, 6),
            "dumped_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
        }))
        counter.dump_s += _time.time() - _t0

    # 增量落盘节流（意见 4 性能对照公平性）: 每次递增都全量重写
    # （含不断增长的 hit_times 数组）会在被测时间窗内引入随命中数
    # 二次增长的写盘开销（实测 6.1s/2881 hits，会把 inject/baseline
    # 中位数比虚高 ~0.5s/次 generate）。默认 0.25s 节流一次，
    # atexit 仍写全量（进程崩溃前的落盘间隔 = 节流窗口）。
    _throttle_s = float(os.environ.get("FLAGOS_C2_DUMP_INTERVAL_S", "0.25"))
    _last = [0.0]

    def dump_if_enabled():
        if os.environ.get("FLAGOS_C2_DUMP_MODE", "every") == "atexit":
            return
        now = _time.time()
        if now - _last[0] < _throttle_s:
            return
        _last[0] = now
        dump()

    atexit.register(dump)
    # 挂到 counter 上: register_all 存档 counter 后，harness 可调
    # counter._dump() 强制落盘（阶段边界快照）。
    counter._dump = dump  # type: ignore[attr-defined]
    counter._dump_fn = dump_if_enabled  # type: ignore[attr-defined]
    return counter, dump


# 进程内已注册算子的 counter 存档: {op: counter_dict}。
# sitecustomize 注册后，child runner 可经 get_counter(op) 取回同一份
# counter 做主进程 probe 验证（避免二次注册产生独立 counter）。
_COUNTERS: dict[str, dict] = {}

# 每个算子的注册参数事实: {op: {impl_requested, impl_applied,
# counter_requested, counter_passed}}（#38 意见 5——实现选择如实落盘）。
_REG_INFO: dict[str, dict] = {}

# 持有 register_a1 返回的 lib 列表——GC 后注册失效（见 register.py 注释）。
_HELD_LIBS: list = []


def get_counter(op: str):
    """返回 register_all 存档的 counter dict，未注册或未接线则 None。"""
    return _COUNTERS.get(op)


def get_reg_info(op: str) -> dict | None:
    """返回该算子最近一次 register_all 的参数事实（意见 5 证据）。"""
    return _REG_INFO.get(op)


def _param_acceptor(fn):
    """按函数签名判断是否接受指定关键字参数（不用宽泛 TypeError 猜测）。"""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return lambda name: False
    var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())

    def accepts(name: str) -> bool:
        return name in params or var_kw

    return accepts


def register_all(dispatch_key: str = "PrivateUse1",
                 root: str | None = None) -> list[str]:
    """注册本仓库的 A1 aten 算子，返回成功注册的算子名列表。"""
    base = Path(root or os.environ.get("FLAGOS_TEMPLATES_ROOT") or _DEFAULT_ROOT)
    registered: list[str] = []
    for op in _ops():
        r = _load_isolated(base, op)
        if r is None:
            continue
        mod, saved = r
        for name in list(vars(mod)):
            fn = getattr(mod, name, None)
            if callable(fn) and name.endswith("_aten"):
                setattr(mod, name, _wrap(fn, saved))
        try:
            counter, _dump = _maybe_counter(op)
            # FLAGOS_A1_IMPL 指定实现（如 torch），用于 C2 调试/规避
            # triton 在特定 shape 下的首轮异常。
            _impl = os.environ.get("FLAGOS_A1_IMPL") or None
            # 按 register_a1 的签名能力传参（#38 意见 5）: 不再用
            # try/TypeError 回退——宽泛 TypeError 会吞掉算子内部错误，
            # 且回退丢 counter 仍存档假象（embedding 不接受 impl，
            # 旧代码回退到 register_a1(dispatch_key) 把 counter 一起丢了）。
            accepts = _param_acceptor(mod.register_a1)
            kwargs: dict = {}
            counter_passed = False
            impl_applied = False
            if counter is not None and accepts("counter"):
                kwargs["counter"] = counter
                counter_passed = True
            if _impl and accepts("impl"):
                kwargs["impl"] = _impl
                impl_applied = True
            libs = mod.register_a1(dispatch_key, **kwargs)
            _REG_INFO[op] = {
                "impl_requested": _impl,
                "impl_applied": impl_applied,
                "counter_requested": counter is not None,
                "counter_passed": counter_passed,
            }
            # 仅当 counter 真正传入 register_a1 时存档——否则 get_counter
            # 返回 None，harness 如实报 a1_counter_retrieved=false，
            # 不再出现「存档了没接线的 counter」假象。
            if counter is not None and counter_passed:
                _COUNTERS[op] = counter
            if libs is not None:
                _HELD_LIBS.append(libs)
            registered.append(op)
        except Exception:
            pass
    return registered
