# C2 验收器二轮复审负例回归（#38 意见 1/2）。
#
# 复审用纯 Python 复现了两条假通过路径:
#   1) 双方缺失输出时 `[] == []` 被计成恒等成功 → “6/6 PASS”；
#      将该 verdict 塞进 summary，decide() 仍 exit 0。
#   2) forward_hits 取 engine 生命周期累计 → 只有 init 命中、
#      correctness/measured 均为零也能通过。
# 本文件按审阅要求**实际调用 harness 的比较函数 / attribute_hits()**
# 构造负例（不手工改 verdict），覆盖: 双方空输出、双方同 prompt 空 token、
# 缺单个 prompt、空预期集合、重复 prompt_idx、身份错配，以及
# init-only / probe-only / outside-only / correctness 有命中但 measured
# 零命中的阶段负例；并保留合法短输出正例。
from __future__ import annotations

import importlib.util
import sys

import pytest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from c2_acceptance import decide  # noqa: E402


def _load(rel: str, name: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SDPA = _load("ops/sdpa_math/script/c2_real_vllm_ascend.py",
             "c2_harness_sdpa_under_test")
EMB = _load("ops/embedding/script/c2_real_vllm_ascend.py",
            "c2_harness_emb_under_test")

HARNESSES = pytest.mark.parametrize("harness", [SDPA, EMB], ids=["sdpa_math",
                                                                "embedding"])


def _run(harness, outputs, ok=True, rc=0):
    return {"ok": ok, "rc": rc, "outputs": outputs}


def _outputs(harness, drop=(), tokens=(7,)):
    """完整、合法的逐 prompt 输出（默认单 token 短输出正例素材）。"""
    return [
        {"prompt_idx": i, "prompt": p, "token_ids": list(tokens)}
        for i, p in enumerate(harness.PROMPTS)
        if i not in drop
    ]


# ---------------------------------------------------------------------------
# 意见 1: 比较函数必须先验证输入证据完整（真实 harness 函数）
# ---------------------------------------------------------------------------

@HARNESSES
def test_both_empty_outputs_is_fail_not_pass(harness):
    # 复审复现: 旧逻辑 `[] == []` → 6/6 PASS；新逻辑必须 FAIL
    r = harness.correctness_compare(_run(harness, []),
                                    _run(harness, []))
    assert r["applicable"] is True
    assert r["evidence_complete"] is False
    assert r["correctness_verdict"] != "PASS"
    assert r.get("identity_pass_count") != f"6/6"
    assert r["evidence_errors"]


@HARNESSES
def test_both_same_prompt_empty_tokens_fails(harness):
    outs = [
        {"prompt_idx": i, "prompt": p, "token_ids": []}
        for i, p in enumerate(harness.PROMPTS)
    ]
    r = harness.correctness_compare(_run(harness, outs),
                                    _run(harness, [dict(o) for o in outs]))
    assert r["evidence_complete"] is False
    assert r["correctness_verdict"] == "FAIL"
    assert any("token_ids empty" in e for e in r["evidence_errors"])


@HARNESSES
def test_missing_single_prompt_fails(harness):
    base = _outputs(harness)
    inject = _outputs(harness, drop={3})
    r = harness.correctness_compare(_run(harness, base),
                                    _run(harness, inject))
    assert r["evidence_complete"] is False
    assert r["correctness_verdict"] == "FAIL"
    assert any("missing prompt_idx [3]" in e for e in r["evidence_errors"])


@HARNESSES
def test_empty_expected_prompt_set_fails(harness, monkeypatch):
    # 内存里清空预期 prompt 集: 旧逻辑 n=0 → (0==0) PASS；新逻辑必须 FAIL
    monkeypatch.setattr(harness, "PROMPTS", [])
    r = harness.correctness_compare(_run(harness, []),
                                    _run(harness, []))
    assert r["evidence_complete"] is False
    assert r["correctness_verdict"] == "FAIL"
    assert any("expected prompt set is empty" in e for e in r["evidence_errors"])


@HARNESSES
def test_duplicate_prompt_idx_fails(harness):
    outs = _outputs(harness) + [dict(_outputs(harness)[0])]
    r = harness.correctness_compare(_run(harness, outs),
                                    _run(harness, _outputs(harness)))
    assert r["evidence_complete"] is False
    assert any("duplicate prompt_idx" in e for e in r["evidence_errors"])


@HARNESSES
def test_prompt_identity_mismatch_fails(harness):
    inject = _outputs(harness)
    inject[0] = dict(inject[0], prompt="not the registered prompt")
    r = harness.correctness_compare(_run(harness, _outputs(harness)),
                                    _run(harness, inject))
    assert r["evidence_complete"] is False
    assert any("identity mismatch" in e for e in r["evidence_errors"])


@HARNESSES
def test_legal_short_output_positive_passes(harness):
    # 合法短输出（非恰好 8 token）: 证据完整且逐位一致 → PASS
    r = harness.correctness_compare(_run(harness, _outputs(harness, tokens=(1,))),
                                    _run(harness, _outputs(harness, tokens=(1,))))
    assert r["evidence_complete"] is True
    assert r["evidence_errors"] == []
    assert r["correctness_verdict"] == "PASS"
    assert r["identity_pass_count"] == f"{len(harness.PROMPTS)}/{len(harness.PROMPTS)}"


@HARNESSES
def test_missing_outputs_key_fails(harness):
    r = harness.correctness_compare({"ok": True}, {"ok": True})
    assert r["evidence_complete"] is False
    assert r["correctness_verdict"] == "FAIL"
    assert any("outputs missing" in e for e in r["evidence_errors"])


# ---------------------------------------------------------------------------
# 意见 1: decide 不得只信任 verdict（复审复现路径）
# ---------------------------------------------------------------------------

def _good_summary(**overrides) -> dict:
    s = {
        "baseline": {"rc": 0, "ok": True},
        "baseline_math": {"rc": 0, "ok": True},
        "inject": {"rc": 0, "ok": True},
        "forward_hits": 2880,
        "correctness_same_path": {
            "evidence_complete": True,
            "correctness_verdict": "PASS",
            "prompt_logprob_same_path": {"max_abs_diff": 0.0},
        },
        "perf": {"gate_verdict": "OK"},
        "hit_attribution": {
            "by_role": {
                "engine/worker": {"hits": 2880, "correctness": 1440,
                                  "warmup": 240, "measured": 1200},
                "main": {"hits": 1, "probe": 1},
            },
            "outside_windows": 0,
        },
        "hit_attribution_check": {
            "hits_total": 2881, "phase_attributed_total": 2881,
            "consistent": True,
        },
    }
    s.update(overrides)
    return s


def test_decide_rejects_empty_output_compare_result():
    # 将新比较函数对双方空输出的结果（若旧逻辑则为 6/6 PASS）放进 summary
    for harness in (SDPA, EMB):
        block = harness.correctness_compare(_run(harness, []),
                                            _run(harness, []))
        assert block["correctness_verdict"] == "FAIL"
        s = _good_summary(correctness_same_path=block)
        d = decide(s, mode="accept")
        assert d["exit_code"] != 0
        assert any("correctness" in c for c in d["failed_checks"])


def test_decide_rejects_verdict_pass_without_evidence_complete():
    # 手工 verdict=PASS 但缺 evidence_complete → 不放行（意见 1）
    s = _good_summary(correctness_same_path={
        "correctness_verdict": "PASS",
        "identity_pass_count": "6/6",
        "prompt_logprob_same_path": {"max_abs_diff": 0.0},
    })
    d = decide(s, mode="accept")
    assert d["exit_code"] != 0
    assert any("evidence_complete" in c for c in d["failed_checks"])


def test_decide_rejects_evidence_complete_false_even_if_pass():
    s = _good_summary(correctness_same_path={
        "evidence_complete": False,
        "correctness_verdict": "PASS",
        "prompt_logprob_same_path": {"max_abs_diff": 0.0},
    })
    d = decide(s, mode="accept")
    assert d["exit_code"] != 0
    assert any("evidence_complete" in c for c in d["failed_checks"])


# ---------------------------------------------------------------------------
# 意见 2: 阶段命中断言（真实 attribute_hits() 构造）
# ---------------------------------------------------------------------------

_PHASES = {
    "init": {"start": 0, "end": 10},
    "correctness": {"start": 20, "end": 30},
    "warmup": {"start": 35, "end": 38},
    "measured": {"start": 40, "end": 50},
    "probe": {"start": 60, "end": 70},
}
MAIN_PID = 1000
ENGINE_PID = 2000


def _attr_summary(harness, entries, forward_hits):
    """entries: [(pid, [hit_times...])] → 用真实 attribute_hits 构造 summary。"""
    hits = [
        {"pid": pid, "ppid": 1, "hits": len(times), "hit_times": list(times)}
        for pid, times in entries
    ]
    inject = {"pid": MAIN_PID}
    attr = harness.attribute_hits(hits, inject, dict(_PHASES))
    phase_total = sum(
        int(v or 0) for _, bucket in attr["by_role"].items()
        for k, v in bucket.items() if k != "hits")
    hits_total = sum(int(h.get("hits") or 0) for h in hits)
    return _good_summary(
        forward_hits=forward_hits,
        hit_attribution=attr,
        hit_attribution_check={
            "hits_total": hits_total,
            "phase_attributed_total": phase_total,
            "consistent": hits_total == phase_total,
        },
    )


def test_init_only_hits_fails_accept():
    for harness in (SDPA, EMB):
        s = _attr_summary(harness, [(ENGINE_PID, [5])], forward_hits=1)
        d = decide(s, mode="accept")
        assert d["exit_code"] != 0
        assert any("correctness" in c for c in d["failed_checks"])


def test_probe_only_hits_fails_accept():
    for harness in (SDPA, EMB):
        s = _attr_summary(harness, [(MAIN_PID, [65])], forward_hits=0)
        d = decide(s, mode="accept")
        assert d["exit_code"] != 0
        assert any("forward_hits" in c for c in d["failed_checks"])


def test_outside_only_hits_fails_accept():
    for harness in (SDPA, EMB):
        s = _attr_summary(harness, [(ENGINE_PID, [999])], forward_hits=1)
        d = decide(s, mode="accept")
        assert d["exit_code"] != 0
        assert any("measured" in c for c in d["failed_checks"])


def test_correctness_hits_but_zero_measured_fails_accept():
    for harness in (SDPA, EMB):
        s = _attr_summary(harness, [(ENGINE_PID, [25, 26])], forward_hits=2)
        d = decide(s, mode="accept")
        assert d["exit_code"] != 0
        assert any("measured" in c for c in d["failed_checks"])
        assert not any("correctness 阶段命中=0" in c for c in d["failed_checks"])


def test_full_phase_hits_passes_accept():
    for harness in (SDPA, EMB):
        s = _attr_summary(harness, [(ENGINE_PID, [25, 45])], forward_hits=2)
        d = decide(s, mode="accept")
        assert d["exit_code"] == 0
        assert d["failed_checks"] == []


def test_missing_hit_attribution_fails_accept():
    # 归因数据缺失不得静默放行
    s = _good_summary()
    del s["hit_attribution"]
    d = decide(s, mode="accept")
    assert d["exit_code"] != 0
    assert any("hit_attribution 缺失" in c for c in d["failed_checks"])


def test_missing_hit_attribution_check_fails_accept():
    s = _good_summary()
    del s["hit_attribution_check"]
    d = decide(s, mode="accept")
    assert d["exit_code"] != 0
    assert any("hit_attribution_check 缺失" in c for c in d["failed_checks"])
