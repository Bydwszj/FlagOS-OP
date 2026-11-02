#!/usr/bin/env python3
"""C2 证据验收判定（#38 审阅意见 4）。

把「采集成功」与「验收通过」分开:
  - collection_status: 双跑/三跑是否采集完成（rc/ok/子进程存活）；
  - acceptance_status: 在采集完成之上，命中、同路径正确性、性能门禁、
    计数归因自洽等**必需项**是否全部通过。

旧 harness 只要程序跑完且有命中就 exit 0（输出失配/零命中/缺证据均可
假通过，审阅意见已用控制流 mock 复现）。本模块把判定收敛为纯函数，
accept 入口遇失败/缺证返回非零；--collect 显式采集模式除外（不得被
CI 当作验收通过——文档与 PR 描述须标明本次跑的是哪个模式）。

正例/负例回归: tests/unit/test_c2_acceptance.py、
tests/unit/test_c2_evidence_phase.py（复审 1/2: 证据完整性 + 阶段命中断言）。
wt-2026-10-30-fix
"""
from __future__ import annotations

# 同路径 teacher-forced prompt logprob 差异的失效线（沿用既有报告口径:
# 超出数值精度差量级即视为正确性证据失效，转 FAIL 而非豁免）。
LOGPROB_MAX_ABS_DIFF_LIMIT = 0.5


def _correctness_block(summary: dict) -> dict:
    """sdpa_math 用 correctness_same_path；embedding 用 correctness。"""
    return summary.get("correctness_same_path") or summary.get("correctness") or {}


def _perf_gate(summary: dict) -> str | None:
    perf = summary.get("perf") or {}
    return perf.get("gate_verdict")


def decide(summary: dict, mode: str = "accept") -> dict:
    """mode ∈ {accept, collect}; 返回判定与退出码。

    accept: 需要全部必需项通过才 exit 0（CI/验收入口用）。
    collect: 仅要求采集完成 exit 0（显式采集模式，不代表验收通过）。
    """
    if mode not in ("accept", "collect"):
        raise ValueError(f"unknown mode: {mode!r}")

    failed: list[str] = []
    warnings: list[str] = []

    # --- 采集状态（两种模式都先判） ---
    runs = {k: v for k, v in summary.items()
            if k in ("baseline", "baseline_math", "inject")
            and isinstance(v, dict)}
    required_runs = list(runs)
    collection_failed = False
    for name, run in runs.items():
        if run.get("skipped"):
            continue
        if run.get("rc") not in (0,) or not run.get("ok"):
            collection_failed = True
            failed.append(
                f"collection: {name} rc={run.get('rc')} ok={run.get('ok')}")
    if not runs or all(r.get("skipped") for r in runs.values()):
        collection_failed = True
        failed.append("collection: no completed runs")

    collection_status = "failed" if collection_failed else "completed"

    if mode == "collect":
        return {
            "mode": mode,
            "collection_status": collection_status,
            "acceptance_status": "not-evaluated-collect-mode",
            "failed_checks": failed,
            "warnings": warnings,
            "exit_code": 2 if collection_failed else 0,
            "required_runs": required_runs,
        }

    # --- 验收必需项（仅 accept 模式） ---
    # 1) 真实前向命中（probe/主进程计数不算）
    if not (int(summary.get("forward_hits") or 0) > 0):
        failed.append("hit: forward_hits==0（真实前向未命中目标算子）")

    # 1b) 阶段命中（复审意见 2: 生命周期累计只证明注册存在；
    #     正确性验收要求推理进程在 correctness 阶段命中，性能验收要求
    #     measured 阶段命中；归因数据缺失不得静默放行）
    attribution = summary.get("hit_attribution") or {}
    by_role = attribution.get("by_role") or {}
    engine = by_role.get("engine/worker") or {}
    if not attribution:
        failed.append(
            "attribution: hit_attribution 缺失（阶段归因数据缺失，不得静默放行）")
    else:
        corr_phase = int(engine.get("correctness") or 0)
        meas_phase = int(engine.get("measured") or 0)
        if corr_phase <= 0:
            failed.append(
                f"hit: engine/worker correctness 阶段命中={corr_phase}"
                "（正确性验收要求推理进程在 correctness 阶段命中目标算子）")
        if meas_phase <= 0:
            failed.append(
                f"hit: engine/worker measured 阶段命中={meas_phase}"
                "（性能验收要求 measured 阶段命中目标算子）")

    # 2) 输入证据完整性（复审意见 1: 先验证证据完整，再比较数值——
    #     双方缺失/空 token/缺 prompt 时旧比较函数会产出假 6/6 PASS）
    cb = _correctness_block(summary)
    if cb.get("evidence_complete") is not True:
        failed.append(
            "correctness: evidence_complete 非 True"
            f"（值={cb.get('evidence_complete')!r}；预期 prompt 集非空、"
            "两跑完整唯一覆盖、身份对应、token 有效是数值比较的前置条件）")

    # 3) 同路径正确性（预注册判据）必须 PASS
    verdict = cb.get("correctness_verdict")
    if verdict != "PASS":
        failed.append(f"correctness: verdict={verdict!r}（必需 PASS）")

    # 4) logprob 失效线: 差异超限 → FAIL（不豁免）
    lp = cb.get("prompt_logprob_same_path") or {}
    lp_max = lp.get("max_abs_diff")
    if lp_max is None:
        warnings.append("logprob: max_abs_diff 缺失（prompt_logprobs 不可用）")
    elif float(lp_max) > LOGPROB_MAX_ABS_DIFF_LIMIT:
        failed.append(
            f"logprob: max_abs_diff={lp_max} > {LOGPROB_MAX_ABS_DIFF_LIMIT}"
            "（超出数值精度差量级，正确性证据失效）")

    # 5) 同路径/同调用链性能门禁
    gate = _perf_gate(summary)
    if gate != "OK":
        failed.append(f"perf: gate_verdict={gate!r}（必需 OK）")

    # 6) 计数归因自洽（阶段归因总和 == hits 总和；check 缺失也不放行）
    check = summary.get("hit_attribution_check")
    if check is None:
        failed.append(
            "attribution: hit_attribution_check 缺失"
            "（归因一致性校验缺失，不得静默放行）")
    elif not check.get("consistent"):
        failed.append(
            f"attribution: hits_total={check.get('hits_total')} != "
            f"phase_attributed_total={check.get('phase_attributed_total')}")

    acceptance_status = "complete-pending-review" if not failed else (
        "FAILED" if collection_failed else "PARTIAL")
    return {
        "mode": mode,
        "collection_status": collection_status,
        "acceptance_status": acceptance_status,
        "failed_checks": failed,
        "warnings": warnings,
        "exit_code": 2 if collection_failed else (1 if failed else 0),
        "required_runs": required_runs,
    }
