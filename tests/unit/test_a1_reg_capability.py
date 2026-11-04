# a1_aten_ops 注册参数能力分发回归（#38 意见 5）。
#
# 旧实现: FLAGOS_A1_IMPL 非空时对不接受 impl 的 register_a1（如
# embedding）触发 TypeError → 回退 register_a1(dispatch_key)，把 counter
# 一起丢掉，却仍存档 counter 并报告注册成功——counter 假象。
# 新实现: 按签名能力传参；counter 只有真正传入才存档；不吞内部 TypeError。
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_a1_ops():
    spec = importlib.util.spec_from_file_location(
        "_a1_aten_ops_under_test", ROOT / "injection" / "a1_aten_ops.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FAKE_WITH_COUNTER = '''
def register_a1(dispatch_key, counter=None, impl=None, platform=None):
    if counter is not None:
        counter["n"] = counter.get("n", 0) + 1
    return [dispatch_key, impl, platform]
'''

_FAKE_NO_COUNTER_PARAM = '''
def register_a1(dispatch_key, platform=None):
    return [dispatch_key, platform]
'''

_FAKE_INTERNAL_TYPEERROR = '''
def register_a1(dispatch_key, counter=None):
    raise TypeError("boom: internal failure, not a signature mismatch")
'''


def _make_root(tmp_path: Path, op: str, src: str) -> Path:
    op_dir = tmp_path / "ops" / op
    op_dir.mkdir(parents=True)
    (op_dir / "register.py").write_text(src)
    return tmp_path


def _run(tmp_path: Path, ops: str, monkeypatch):
    monkeypatch.setenv("FLAGOS_A1_OPS", ops)
    monkeypatch.setenv("FLAGOS_C2_COUNT_DIR", str(tmp_path / "counts"))
    monkeypatch.setenv("FLAGOS_A1_IMPL", "torch")
    a1 = _load_a1_ops()
    registered = a1.register_all("TestKey", root=str(tmp_path))
    return a1, registered


def test_counter_kept_when_register_accepts_it(tmp_path, monkeypatch):
    root = _make_root(tmp_path, "fakeop", _FAKE_WITH_COUNTER)
    a1, registered = _run(root, "fakeop", monkeypatch)
    assert "fakeop" in registered
    counter = a1.get_counter("fakeop")
    assert counter is not None and counter["n"] == 1
    info = a1.get_reg_info("fakeop")
    assert info["counter_requested"] and info["counter_passed"]
    assert info["impl_requested"] == "torch" and info["impl_applied"]


def test_counter_not_archived_when_register_cannot_take_it(tmp_path, monkeypatch):
    root = _make_root(tmp_path, "fakeop2", _FAKE_NO_COUNTER_PARAM)
    a1, registered = _run(root, "fakeop2", monkeypatch)
    assert "fakeop2" in registered
    # counter 无法传入 → 不存档，harness 会如实报 a1_counter_retrieved=false
    assert a1.get_counter("fakeop2") is None
    info = a1.get_reg_info("fakeop2")
    assert info["counter_requested"] and not info["counter_passed"]
    assert info["impl_requested"] == "torch" and not info["impl_applied"]


def test_internal_typeerror_not_treated_as_signature_mismatch(tmp_path, monkeypatch):
    root = _make_root(tmp_path, "fakeop3", _FAKE_INTERNAL_TYPEERROR)
    a1, registered = _run(root, "fakeop3", monkeypatch)
    # 内部 TypeError → 注册失败，不进 registered，也不存档 counter
    assert "fakeop3" not in registered
    assert a1.get_counter("fakeop3") is None
    assert a1.get_reg_info("fakeop3") is None
