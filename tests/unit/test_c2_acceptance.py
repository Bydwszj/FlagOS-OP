# C2 验收判定负例回归（#38 意见 4）。
#
# 审阅意见用控制流 mock 复现了旧 harness 的假通过路径:
#   输出失配 + 有命中 → 旧 exit 0；零命中 → 旧 exit 0（status c2-hit 分支）；
#   缺证据/子进程失败 → 仅 "c2-failed"。
# scripts/c2_acceptance.decide 把验收收敛为纯函数，accept 全部必需项
# 通过才 exit 0；--collect 显式采集模式除外。
from __future__ import annotations

import sys

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from c2_acceptance import decide  # noqa: E402


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


def test_all_green_accept_exit0():
    d = decide(_good_summary(), mode="accept")
    assert d["exit_code"] == 0
    assert d["collection_status"] == "completed"
    assert d["acceptance_status"] == "complete-pending-review"
    assert d["failed_checks"] == []


def test_output_mismatch_with_hits_fails_accept():
    # 控制流 mock 场景 1: 有命中但正确性失配——旧 exit 0 假通过
    s = _good_summary()
    s["correctness_same_path"] = {"evidence_complete": True,
                                  "correctness_verdict": "FAIL",
                                  "prompt_logprob_same_path": {"max_abs_diff": 0.2}}
    d = decide(s, mode="accept")
    assert d["exit_code"] == 1
    assert any("correctness" in c for c in d["failed_checks"])


def test_zero_hits_fails_accept():
    # 控制流 mock 场景 2: 命中为 0——旧 harness 仍可能 exit 0
    s = _good_summary(forward_hits=0)
    d = decide(s, mode="accept")
    assert d["exit_code"] == 1
    assert any("hit" in c for c in d["failed_checks"])


def test_missing_evidence_child_failure_nonzero_both_modes():
    # 控制流 mock 场景 3/4: 子进程失败（rc=1/缺 ok）→ accept 与 collect
    # 都必须非零
    s = _good_summary(inject={"rc": 1, "ok": False})
    d_accept = decide(s, mode="accept")
    d_collect = decide(s, mode="collect")
    assert d_accept["exit_code"] == 2
    assert d_collect["exit_code"] == 2
    assert d_accept["collection_status"] == "failed"
    assert d_collect["collection_status"] == "failed"


def test_perf_gate_not_ok_fails_accept():
    s = _good_summary()
    s["perf"] = {"gate_verdict": "FAIL"}
    d = decide(s, mode="accept")
    assert d["exit_code"] == 1
    assert any("perf" in c for c in d["failed_checks"])


def test_logprob_over_limit_fails_accept():
    # logprob 失效线: 超过 0.5 nats → 不豁免
    s = _good_summary()
    s["correctness_same_path"] = {
        "evidence_complete": True,
        "correctness_verdict": "PASS",
        "prompt_logprob_same_path": {"max_abs_diff": 0.9},
    }
    d = decide(s, mode="accept")
    assert d["exit_code"] == 1
    assert any("logprob" in c for c in d["failed_checks"])


def test_attribution_inconsistent_fails_accept():
    s = _good_summary()
    s["hit_attribution_check"] = {
        "hits_total": 2880, "phase_attributed_total": 2700, "consistent": False}
    d = decide(s, mode="accept")
    assert d["exit_code"] == 1
    assert any("attribution" in c for c in d["failed_checks"])


def test_collect_mode_ok_with_mismatch_but_flagged_not_evaluated():
    # 显式采集模式: 采集完成 exit 0，但 acceptance 明确标 not-evaluated
    s = _good_summary()
    s["correctness_same_path"] = {"correctness_verdict": "FAIL",
                                  "prompt_logprob_same_path": {"max_abs_diff": 0.2}}
    d = decide(s, mode="collect")
    assert d["exit_code"] == 0
    assert d["acceptance_status"] == "not-evaluated-collect-mode"


def test_embedding_summary_shape_correctness_key():
    # embedding harness 用 correctness 键（非 correctness_same_path）
    s = _good_summary()
    del s["correctness_same_path"]
    s["correctness"] = {"evidence_complete": True,
                        "correctness_verdict": "PASS",
                        "prompt_logprob_same_path": {"max_abs_diff": 0.0}}
    del s["baseline_math"]
    d = decide(s, mode="accept")
    assert d["exit_code"] == 0


def test_no_runs_collection_failed():
    d = decide({}, mode="accept")
    assert d["exit_code"] == 2
    assert d["collection_status"] == "failed"


def test_unknown_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        decide(_good_summary(), mode="review")
