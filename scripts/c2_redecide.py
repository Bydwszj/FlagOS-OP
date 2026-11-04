#!/usr/bin/env python3
"""C2 验收离线重判（#38 二轮复审意见 1/2/3 的收尾步骤）。

背景: 复审在 HEAD 04c2229 上用纯 Python 复现了两条假通过路径——
  1) 双方缺失输出时 `[] == []` 被比较函数计成恒等成功 → “6/6 PASS”；
  2) forward_hits 取 engine 生命周期累计 → 仅 init 命中也能过验收。
已提交的实测材料（run_id 20261031T*，采集 commit 1dc1d8c）**未触发**
这两条漏洞（双方各 6/6 prompt 输出完整、engine correctness/measured
阶段命中充足）。按复审指引处理:

  - 原始测量材料（summary.json / hit 文件 / logs）保持不动、不覆写；
  - 用修复后的比较函数（证据完整性前置）与判定器（阶段命中断言）
    对原始 run 块离线重判，结果**另存**派生文件；
  - 派生文件记录: 原始 run_id、采集 commit、原始文件 sha256、
    重判代码 commit、profile 误标更正说明——不冒充“当时已由新代码产出”。
  - 采集协议（执行/注入/计数落盘）未变，故不复跑 NPU 项。

用法:
  python3 scripts/c2_redecide.py \\
      --harness ops/sdpa_math/script/c2_real_vllm_ascend.py \\
      --summary ops/sdpa_math/reports/c2_ascend910/summary.json \\
      --out ops/sdpa_math/reports/c2_ascend910/acceptance_redecide.json

退出码 = 重判 decide() 的退出码（accept 模式；collect 模式同 decide）。
wt-2026-11-01-fix
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from c2_acceptance import decide  # noqa: E402

CORRECTED_PROFILE = "ascend910-c2-vllm"


def _load_harness(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(
        "c2_redecide_harness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def redecide(harness_path: pathlib.Path, summary_path: pathlib.Path,
             out_path: pathlib.Path, mode: str) -> dict:
    harness = _load_harness(harness_path)
    raw_text = summary_path.read_text()
    raw = json.loads(raw_text)
    raw_sha = hashlib.sha256(raw_text.encode()).hexdigest()
    collection_commit = raw.get("commit")
    raw_profile = raw.get("profile")

    derived = copy.deepcopy(raw)

    # 1) 用修复后的比较函数重算正确性块（从原始 run 块；不改原始文件）
    if "baseline_math" in raw and "correctness_same_path" in raw:
        correctness_key = "correctness_same_path"
        native_key = "correctness_native_path_descriptive"
        derived[correctness_key] = harness.correctness_compare(
            raw["baseline_math"], raw["inject"])
        if native_key in raw:
            derived[native_key] = harness.correctness_compare(
                raw["baseline"], raw["inject"])
    elif "correctness" in raw:
        correctness_key = "correctness"
        derived[correctness_key] = harness.correctness_compare(
            raw["baseline"], raw["inject"])
    else:
        raise SystemExit(f"unrecognized summary shape: {summary_path}")

    # 2) profile / 环境锁更正（复审意见 3: harness 曾硬编码旧名）
    corrections: list[dict] = []
    if raw_profile != CORRECTED_PROFILE:
        corrections.append({
            "field": "profile",
            "was": raw_profile,
            "now": CORRECTED_PROFILE,
            "reason": "旧 harness 硬编码 ascend910（dev profile）；"
                      "本次真实验证环境为 ascend910-c2-vllm（复审意见 3）",
        })
        derived["profile"] = CORRECTED_PROFILE
        derived["env_lock"] = f"configs/env/{CORRECTED_PROFILE}.lock.yaml"

    # 3) 同一判定器离线重判（acceptance.status 与 decision 同源）
    decision = decide(derived, mode=mode)
    derived["acceptance_decision"] = decision
    if isinstance(derived.get("acceptance"), dict):
        derived["acceptance"]["status"] = decision["acceptance_status"]

    redecide_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    rel_summary = str(summary_path.relative_to(ROOT))

    result = {
        "schema": 1,
        "kind": "c2-acceptance-redecide",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "raw_material": {
            "summary_path": rel_summary,
            "summary_sha256": raw_sha,
            "run_id": raw.get("run_id"),
            "collection_commit": collection_commit,
            "collection_date": raw.get("date"),
            "profile_field_original": raw_profile,
            "preserved_unmodified": True,
            "rerun_npu": False,
            "rerun_reason": "采集协议（执行/注入/计数落盘）未变；"
                            "复审确认原实测未触发假通过漏洞，"
                            "仅判定器修复 → 离线重判（复审指引）",
        },
        "redecide": {
            "commit": redecide_commit,
            "correctness_key": correctness_key,
            "correctness": derived[correctness_key],
            "decision": decision,
            "acceptance_status": decision["acceptance_status"],
            "exit_code": decision["exit_code"],
        },
        "corrections": corrections,
        "notes": [
            "派生结果，不替代原始 summary.json；两者并存可追溯。",
            "acceptance.status 与 acceptance_decision 由同一次 decide 派生。",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harness", required=True, type=pathlib.Path)
    ap.add_argument("--summary", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--mode", choices=("accept", "collect"), default="accept")
    args = ap.parse_args()
    harness_path = args.harness.resolve()
    summary_path = args.summary.resolve()
    out_path = args.out.resolve()
    result = redecide(harness_path, summary_path, out_path, args.mode)
    print(json.dumps({
        "out": str(out_path),
        "run_id": result["raw_material"]["run_id"],
        "collection_commit": result["raw_material"]["collection_commit"],
        "redecide_commit": result["redecide"]["commit"],
        "acceptance_status": result["redecide"]["acceptance_status"],
        "failed_checks": result["redecide"]["decision"]["failed_checks"],
        "warnings": result["redecide"]["decision"]["warnings"],
        "corrections": result["corrections"],
        "exit_code": result["redecide"]["exit_code"],
    }, ensure_ascii=False, indent=2))
    return result["redecide"]["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
