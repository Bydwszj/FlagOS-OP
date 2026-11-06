from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import yaml

from scripts.evidence_check import check

_C1_BASIS = "framework_level 基线/注入双跑（mini consumer 对拍）"


def _make_repo(
    root: Path,
    *,
    evidence: str | None = "C2",
    basis: str | None = "demo 引擎双跑 2026-10-29",
    paths: list[str] | None = None,
    with_framework: bool = True,
    updated: str = "2026-10-29",
    guide_status: str = "partial",
    catalog_status: str = "partial",
    readme: bool = True,
    report: str | None = None,
    lock: dict | None = None,
    profiles: list[str] | None = None,
    declared: list[str] | None = None,
    records: dict | None = None,
) -> Path:
    """搭一个最小可过的 demo 仓。

    - `declared`：`platforms.demo.profiles`（默认 `["demo-1"]`）；
      每个 declared profile 默认生成一条与顶层 `evidence` 同级的记录，
      C1/C2 记录自动带 `basis`，C2 记录 `report` 指向 `evidence.json`
      （内容声明该 profile 与 PASS 状态）；
    - `records`：整体替换 `platforms.demo.evidence`（**逐 profile
      记录列表**，每条带 `profile`；不写报告文件，需要时测试自行
      落盘）；
    - `profiles`：额外创建的 `configs/devices` profile。
    """
    paths = ["evidence.json"] if paths is None else paths
    if declared is None:
        declared = list(records) if records else ["demo-1"]
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "demo_op" / "test").mkdir(parents=True, exist_ok=True)
    (root / "evidence.json").parent.mkdir(parents=True, exist_ok=True)

    if records is None:
        records = {}
        for profile in declared:
            record: dict = {"level": evidence or "C2"}
            if record["level"] in ("C1", "C2"):
                record["basis"] = _C1_BASIS
            if record["level"] == "C2":
                record["report"] = "evidence.json"
            records[profile] = record
    c2_profiles = [
        p for p, r in records.items() if isinstance(r, dict)
        and r.get("level") == "C2"
    ]

    op: dict = {
        "name": "demo_op",
        "directory": "demo_op",
        "group": "core",
        "purpose": "demo",
        "key_result": "demo",
        "platforms": {
            "demo": {
                "profiles": declared,
                "evidence": [
                    dict(records[p], profile=p)
                    for p in declared if p in records
                ],
            },
        },
    }
    if evidence is not None:
        op["evidence"] = evidence
        if evidence == "C2":
            if basis is not None:
                op["evidence_basis"] = basis
            if paths is not None:
                op["evidence_paths"] = paths
    catalog = {
        "updated": updated,
        "platform_families": {
            "demo": {
                "display_name": "Demo 芯",
                "status": catalog_status,
                "current_profiles": sorted(
                    set(declared) | set(profiles or [])),
            },
        },
        "operators": [op],
    }
    (root / "ops" / "catalog.yaml").write_text(
        yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8"
    )
    if with_framework:
        (root / "ops" / "demo_op" / "test" / "framework_level.py").write_text(
            "def test_demo():\n    assert True\n", encoding="utf-8"
        )
    json_profile = c2_profiles[0] if c2_profiles else (
        declared[0] if declared else "demo-1")
    (root / "evidence.json").write_text(
        json.dumps({"profile": json_profile, "status": "PASS", "ok": True}),
        encoding="utf-8",
    )
    (root / "docs" / "operator-catalog.md").write_text(
        textwrap.dedent(
            f"""
            # catalog

            | Family key | 显示名 | 当前状态 |
            |---|---|---|
            | `demo` | Demo 芯 | {guide_status} |
            """
        ),
        encoding="utf-8",
    )
    if readme:
        (root / "ops").mkdir(exist_ok=True)
        status_cn = "部分交付" if catalog_status == "partial" else "规划中"
        (root / "ops" / "README.md").write_text(
            f"| Demo 芯（`demo`） | {status_cn} | 1 | demo-1 |\n",
            encoding="utf-8",
        )
    if report is not None:
        (root / "ops" / "demo_op" / "REPORT.md").write_text(
            report, encoding="utf-8"
        )
    if lock is not None:
        env_dir = root / "configs" / "env"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "demo.lock.yaml").write_text(
            yaml.safe_dump(lock, allow_unicode=True), encoding="utf-8"
        )
    for profile in sorted(set(declared) | set(profiles or [])):
        devices = root / "configs" / "devices"
        devices.mkdir(parents=True, exist_ok=True)
        (devices / f"{profile}.yaml").write_text(
            f"name: {profile}\nschema: 1\n", encoding="utf-8"
        )
    return root


def test_minimal_c2_repo_passes(tmp_path: Path):
    _make_repo(tmp_path)
    assert check(tmp_path) == []


def test_missing_evidence_field_is_rejected(tmp_path: Path):
    _make_repo(tmp_path, evidence=None)
    errors = check(tmp_path)
    assert any("evidence 必须是" in e for e in errors)


def test_c1_requires_framework_level_file(tmp_path: Path):
    _make_repo(tmp_path, evidence="C1", basis=None, paths=None,
               with_framework=False)
    errors = check(tmp_path)
    assert any("framework_level.py" in e for e in errors)


def test_c2_requires_basis_and_paths(tmp_path: Path):
    _make_repo(tmp_path, evidence="C2", basis=None, paths=[])
    errors = check(tmp_path)
    assert any("evidence_basis" in e for e in errors)
    assert any("evidence_paths" in e for e in errors)


def test_c2_rejects_missing_or_remote_paths(tmp_path: Path):
    _make_repo(tmp_path, evidence="C2",
               paths=["missing.json", "https://example.com/e.json"])
    errors = check(tmp_path)
    assert any("不存在: missing.json" in e for e in errors)
    assert any("仓库内路径" in e for e in errors)


def test_future_updated_date_is_rejected(tmp_path: Path):
    _make_repo(tmp_path, updated="2099-01-01")
    errors = check(tmp_path)
    assert any("updated 在未来" in e for e in errors)


def test_family_status_mismatch_is_rejected(tmp_path: Path):
    _make_repo(tmp_path, guide_status="planned", catalog_status="partial")
    errors = check(tmp_path)
    assert any("平台状态不一致 demo" in e for e in errors)


def test_readme_missing_family_row_is_rejected(tmp_path: Path):
    _make_repo(tmp_path, readme=False)
    errors = check(tmp_path)
    assert any("ops/README.md 缺失" in e for e in errors)


# ── 缺口 1：聚合报告的 FlagOS E2E 结果必须标 profile（#41 C 段）──

_E2E_ROW_PASS = (
    "# demo\n\n"
    "| 层级 | 入口 | 结果 |\n"
    "|---|---|---|\n"
    "| FlagOS E2E | `python3 script/e2e_flagos.py` | ☑ 4 层 mini-LLM PASS |\n"
)

_E2E_ROW_TAGGED = (
    "# demo\n\n"
    "| 层级 | 入口 | 结果 |\n"
    "|---|---|---|\n"
    "| FlagOS E2E | `python3 script/e2e_flagos.py --profile demo-1` | "
    "☑ 4 层 mini-LLM PASS（profile `demo-1`） |\n"
    "| FlagOS E2E | `script/e2e_flagos.py` | ✗ 缺证：栈未复测 |\n"
    "| FlagOS E2E | 叙述段落，未标结果 |\n"
)


def test_e2e_result_row_without_profile_is_rejected(tmp_path: Path):
    _make_repo(tmp_path, profiles=["demo-1"], report=_E2E_ROW_PASS)
    errors = check(tmp_path)
    assert any("FlagOS E2E 结果行未标明具体 profile" in e for e in errors)


def test_e2e_result_row_with_profile_passes(tmp_path: Path):
    _make_repo(tmp_path, profiles=["demo-1"], report=_E2E_ROW_TAGGED)
    assert check(tmp_path) == []


def _demo_lock(*, name="demo-1", scope=None, evidence=None) -> dict:
    lock: dict = {"name": name, "schema": 1, "collected_at": "2026-10-29"}
    if scope is not None:
        lock["scope"] = scope
    if evidence is not None:
        lock["evidence"] = evidence
    return lock


def test_lock_scope_and_evidence_crossref_passes(tmp_path: Path):
    _make_repo(
        tmp_path,
        records={"demo-1": {"level": "C2", "basis": _C1_BASIS,
                            "report": "ops/demo_op/REPORT.md"}},
        paths=["ops/demo_op/REPORT.md"],
        report="# demo\n\nFlagOS E2E 结果见 profile demo-1 复验。\n",
        lock=_demo_lock(
            scope=["ops/demo_op C2（真实 vLLM 双跑 2026-10-29）"],
            evidence=[{
                "path": "ops/demo_op/REPORT.md",
                "note": "聚合报告",
            }],
        ),
    )
    assert check(tmp_path) == []


def test_lock_scope_level_mismatch_with_catalog_is_rejected(tmp_path: Path):
    _make_repo(
        tmp_path,
        evidence="C1",
        basis=None,
        paths=None,
        report="# demo\n\nFlagOS E2E 见 profile demo-1。\n",
        lock=_demo_lock(
            scope=["ops/demo_op C2（真实 vLLM 双跑）"],
            evidence=[{
                "path": "ops/demo_op/REPORT.md",
                "note": "聚合报告",
            }],
        ),
    )
    errors = check(tmp_path)
    assert any("ops/demo_op C2 与 profile demo-1 记录 C1 不一致" in e
               for e in errors)


def test_lock_scope_claim_without_supporting_report_is_rejected(
        tmp_path: Path):
    _make_repo(
        tmp_path,
        lock=_demo_lock(
            scope=["ops/demo_op C2（真实 vLLM 双跑）"],
            evidence=[{"path": "docs/op-guide.md", "note": "通用文档"}],
        ),
    )
    (tmp_path / "docs" / "op-guide.md").write_text("# guide\n", encoding="utf-8")
    errors = check(tmp_path)
    assert any("没有对应报告支撑" in e for e in errors)


def test_lock_evidence_missing_report_is_rejected(tmp_path: Path):
    _make_repo(
        tmp_path,
        lock=_demo_lock(
            evidence=[{
                "path": "ops/demo_op/reports/gone.md",
                "note": "早已删除的报告",
            }],
        ),
    )
    errors = check(tmp_path)
    assert any("evidence 指向不存在的报告" in e for e in errors)


def test_lock_evidence_placeholder_entry_is_rejected(tmp_path: Path):
    _make_repo(tmp_path, lock=_demo_lock(evidence=[{"note": "占位证据"}]))
    errors = check(tmp_path)
    assert any("evidence 缺 path" in e for e in errors)


def test_lock_scope_unknown_operator_is_rejected(tmp_path: Path):
    _make_repo(tmp_path, lock=_demo_lock(scope=["ops/ghost_op backend"]))
    errors = check(tmp_path)
    assert any("catalog 没有登记的算子 ops/ghost_op" in e for e in errors)


# ── 预审新增 P2：锁证据 JSON 缺 profile/损坏不得无声跳过 ──

def test_lock_evidence_json_missing_profile_is_rejected(tmp_path: Path):
    """阴性（预审复现）：锁证据 JSON 缺 profile 单独报错（无 C2 scope 也报）。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports" / "noprof.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8")
    _make_repo(
        tmp_path,
        lock=_demo_lock(evidence=[{
            "path": "ops/demo_op/reports/noprof.json",
            "note": "缺 profile 的结构化证据",
        }]),
    )
    errors = check(tmp_path)
    assert any("缺少 profile 字段" in e and "不能作为支撑" in e
               for e in errors)


def test_lock_evidence_corrupt_json_is_rejected_and_not_supporting(
        tmp_path: Path):
    """阴性（预审复现）：损坏 JSON 单独报错，且不能成为 C2 支撑。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports" / "broken.json").write_text(
        "{ not valid json", encoding="utf-8")
    _make_repo(
        tmp_path,
        lock=_demo_lock(
            scope=["ops/demo_op C2（真实 vLLM 双跑）"],
            evidence=[{
                "path": "ops/demo_op/reports/broken.json",
                "note": "损坏报告",
            }],
        ),
    )
    errors = check(tmp_path)
    assert any("JSON 解析失败" in e and "不能作为支撑" in e for e in errors)
    assert any("没有对应报告支撑" in e for e in errors)


def test_lock_evidence_json_declaring_profile_passes(tmp_path: Path):
    """阳性：声明本锁 profile 的结构化证据正常通过（无 C2 scope）。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports" / "ok.json").write_text(
        json.dumps({"profile": "demo-1", "status": "PASS"}),
        encoding="utf-8")
    _make_repo(
        tmp_path,
        lock=_demo_lock(evidence=[{
            "path": "ops/demo_op/reports/ok.json",
            "note": "正常结构化报告",
        }]),
    )
    assert check(tmp_path) == []


# ── 审阅 #1 [P1]：证据等级必须绑定具体 profile（算子+profile 为最小单位）──

def test_per_profile_records_with_distinct_levels_pass(tmp_path: Path):
    """阳性：family 内两个 profile 各有独立记录（C1/C2 并存）且派生一致。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports"
     / "e2e_vllm_c2_demo-2.json").write_text(
        json.dumps({"profile": "demo-2", "status": "PASS"}),
        encoding="utf-8")
    _make_repo(
        tmp_path,
        evidence="C2",
        declared=["demo-1", "demo-2"],
        records={
            "demo-1": {"level": "C1", "basis": _C1_BASIS},
            "demo-2": {"level": "C2", "basis": "demo-2 引擎双跑",
                       "report": "ops/demo_op/reports/e2e_vllm_c2_demo-2.json"},
        },
        paths=["ops/demo_op/reports/e2e_vllm_c2_demo-2.json"],
        lock=_demo_lock(
            name="demo-2",
            scope=["ops/demo_op C2（demo-2 引擎双跑）"],
            evidence=[{
                "path": "ops/demo_op/reports/e2e_vllm_c2_demo-2.json",
                "note": "结构化报告",
            }],
        ),
    )
    assert check(tmp_path) == []


def test_missing_profile_record_is_rejected(tmp_path: Path):
    """阴性：已交付 profile 没有独立记录（共享/缺失都不行）。"""
    _make_repo(
        tmp_path,
        declared=["demo-1", "demo-2"],
        records={"demo-1": {"level": "C2", "basis": _C1_BASIS,
                            "report": "evidence.json"}},
    )
    errors = check(tmp_path)
    assert any("profile demo-2 缺独立证据记录" in e for e in errors)


def test_missing_family_evidence_records_is_rejected(tmp_path: Path):
    """阴性：family 完全没有逐 profile 证据记录（旧 schema）。"""
    _make_repo(tmp_path)
    catalog_path = tmp_path / "ops" / "catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text())
    catalog["operators"][0]["platforms"]["demo"].pop("evidence")
    catalog_path.write_text(
        yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")
    errors = check(tmp_path)
    assert any("缺逐 profile 证据记录" in e for e in errors)


def test_top_level_evidence_must_be_derived_from_records(tmp_path: Path):
    """阴性：顶层汇总 C2 但 profile 记录最高只有 C1（不能独立填写）。"""
    _make_repo(
        tmp_path,
        evidence="C2",
        records={"demo-1": {"level": "C1", "basis": _C1_BASIS}},
    )
    errors = check(tmp_path)
    assert any("与 profile 记录最高级 C1 不一致" in e for e in errors)


def test_nested_evidence_profiles_form_is_rejected(tmp_path: Path):
    """阴性：已停用的嵌套 `evidence.profiles` 形态必须报错。

    逐 profile 记录列表是唯一来源；嵌套形态会让多个环境共享未区分
    的等级（PR44 时代就是在这里出的洞），故直接拒绝而不是兼容读取。
    """
    _make_repo(tmp_path)
    catalog_path = tmp_path / "ops" / "catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text())
    catalog["operators"][0]["platforms"]["demo"]["evidence"] = {
        "profiles": {"demo-1": {"level": "C2", "basis": _C1_BASIS,
                                "report": "evidence.json"}},
    }
    catalog_path.write_text(
        yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")
    errors = check(tmp_path)
    assert any("嵌套形态已停用" in e for e in errors)


def test_c2_record_report_declaring_other_profile_is_rejected(
        tmp_path: Path):
    """阴性：C2 记录的报告 JSON 声明的是另一个环境的 profile。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports"
     / "e2e_flagos_p800-kunlunxin.json").write_text(
        json.dumps({"profile": "p800-kunlunxin", "ok": True}),
        encoding="utf-8")
    _make_repo(
        tmp_path,
        records={"demo-1": {
            "level": "C2", "basis": "声称 demo-2 栈双跑",
            "report": "ops/demo_op/reports/e2e_flagos_p800-kunlunxin.json",
        }},
    )
    errors = check(tmp_path)
    assert any(
        "报告声明 profile=p800-kunlunxin 与记录 profile=demo-1 不一致"
        in e for e in errors)


def test_c2_record_json_missing_status_is_rejected(tmp_path: Path):
    """阴性：结构化报告没有 status/ok 验证状态字段。"""
    (tmp_path / "report.json").write_text(
        json.dumps({"profile": "demo-1"}), encoding="utf-8")
    _make_repo(
        tmp_path,
        paths=["evidence.json", "report.json"],
        records={"demo-1": {"level": "C2", "basis": _C1_BASIS,
                            "report": "report.json"}},
    )
    errors = check(tmp_path)
    assert any("缺少 status/ok 验证状态字段" in e for e in errors)


# ── 审阅 #2 [P1]：锁的交叉校验必须绑定锁对应 profile ──

def test_lock_c2_scope_backed_by_same_profile_report_passes(tmp_path: Path):
    """阳性：锁 profile=demo-1 的 C2 声明由其 C2 记录 report（同源、PASS）支撑。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports"
     / "e2e_vllm_c2_demo-1.json").write_text(
        json.dumps({"profile": "demo-1", "status": "PASS",
                    "consumer_type": "vllm-enginecore"}),
        encoding="utf-8")
    _make_repo(
        tmp_path,
        records={"demo-1": {
            "level": "C2", "basis": "demo 引擎双跑 2026-10-29",
            "report": "ops/demo_op/reports/e2e_vllm_c2_demo-1.json",
        }},
        paths=["ops/demo_op/reports/e2e_vllm_c2_demo-1.json"],
        lock=_demo_lock(
            scope=["ops/demo_op C2（真实 vLLM EngineCore 消费）"],
            evidence=[{
                "path": "ops/demo_op/reports/e2e_vllm_c2_demo-1.json",
                "note": "C2 双跑",
            }],
        ),
    )
    assert check(tmp_path) == []


def test_lock_c2_scope_backed_by_fail_status_json_is_rejected(
        tmp_path: Path):
    """阴性（预审复现）：`status=FAIL` 的同 profile JSON 不能给 C2 背书。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports" / "bad.json").write_text(
        json.dumps({"profile": "demo-1", "status": "FAIL"}),
        encoding="utf-8")
    _make_repo(
        tmp_path,
        lock=_demo_lock(
            scope=["ops/demo_op C2（真实 vLLM 双跑）"],
            evidence=[{
                "path": "ops/demo_op/reports/bad.json",
                "note": "FAIL 报告",
            }],
        ),
    )
    errors = check(tmp_path)
    assert any("支撑报告不合格" in e and "状态非通过" in e for e in errors)


def test_lock_c2_scope_backed_by_statusless_json_is_rejected(
        tmp_path: Path):
    """阴性（预审复现）：缺 `status`/`ok` 的同 profile JSON 不能给 C2 背书。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports" / "bad.json").write_text(
        json.dumps({"profile": "demo-1"}), encoding="utf-8")
    _make_repo(
        tmp_path,
        lock=_demo_lock(
            scope=["ops/demo_op C2（真实 vLLM 双跑）"],
            evidence=[{
                "path": "ops/demo_op/reports/bad.json",
                "note": "无状态报告",
            }],
        ),
    )
    errors = check(tmp_path)
    assert any("支撑报告不合格" in e and "缺少 status/ok" in e
               for e in errors)


def test_lock_c2_scope_backed_by_unassociated_json_is_rejected(
        tmp_path: Path):
    """阴性：PASS 但与该 profile 的 C2 记录 report/consumer 无同源关联。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports" / "other.json").write_text(
        json.dumps({"profile": "demo-1", "status": "PASS"}),
        encoding="utf-8")
    _make_repo(
        tmp_path,
        lock=_demo_lock(
            scope=["ops/demo_op C2（真实 vLLM 双跑）"],
            evidence=[{
                "path": "ops/demo_op/reports/other.json",
                "note": "无关联报告",
            }],
        ),
    )
    errors = check(tmp_path)
    assert any("支撑报告不合格" in e and "无关联" in e for e in errors)


def test_lock_c2_scope_backed_by_other_profile_report_is_rejected(
        tmp_path: Path):
    """阴性（审阅复现）：PPU 锁的 C2 scope 用现存 P800 报告背书。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports"
     / "e2e_flagos_p800-kunlunxin.json").write_text(
        json.dumps({"profile": "p800-kunlunxin", "ok": True}),
        encoding="utf-8")
    _make_repo(
        tmp_path,
        lock=_demo_lock(
            scope=["ops/demo_op C2（真实 vLLM 双跑）"],
            evidence=[{
                "path": "ops/demo_op/reports/e2e_flagos_p800-kunlunxin.json",
                "note": "别环境的现存报告",
            }],
        ),
    )
    errors = check(tmp_path)
    assert any(
        "profile=p800-kunlunxin 与锁 profile=demo-1 不一致" in e
        for e in errors)
    assert any("没有对应报告支撑" in e for e in errors)


def test_lock_c1_claim_not_rejected_by_other_profiles_c2(tmp_path: Path):
    """阳性（审阅反例）：全局汇总 C2，但本锁 profile 记录 C1 → 不误杀。"""
    _make_repo(
        tmp_path,
        evidence="C2",
        declared=["demo-1", "demo-2"],
        records={
            "demo-1": {"level": "C1", "basis": _C1_BASIS},
            "demo-2": {"level": "C2", "basis": "demo-2 引擎双跑",
                       "report": "evidence.json"},
        },
        report="# demo\n\nFlagOS E2E 见 profile demo-1。\n",
        lock=_demo_lock(
            name="demo-1",
            scope=["ops/demo_op C1（mini consumer 基线/注入双跑）"],
            evidence=[{"path": "ops/demo_op/REPORT.md", "note": "报告"}],
        ),
    )
    assert check(tmp_path) == []


def test_lock_scope_level_mismatch_with_profile_record_is_rejected(
        tmp_path: Path):
    """阴性：scope C1 与本锁 profile 的 C2 记录不一致。"""
    _make_repo(
        tmp_path,
        report="# demo\n\nFlagOS E2E 见 profile demo-1。\n",
        lock=_demo_lock(
            scope=["ops/demo_op C1（自称只有 mini consumer）"],
            evidence=[{"path": "ops/demo_op/REPORT.md", "note": "报告"}],
        ),
    )
    errors = check(tmp_path)
    assert any("ops/demo_op C1 与 profile demo-1 记录 C2 不一致" in e
               for e in errors)


def test_lock_scope_claim_without_profile_record_is_rejected(
        tmp_path: Path):
    """阴性：catalog 在该锁 profile 上没有这个算子的记录。"""
    _make_repo(
        tmp_path,
        declared=["demo-1", "demo-2"],
        records={"demo-2": {"level": "C2", "basis": _C1_BASIS,
                            "report": "evidence.json"}},
        lock=_demo_lock(
            name="demo-1",
            scope=["ops/demo_op C2（无据声明）"],
            evidence=[{"path": "evidence.json", "note": "报告"}],
        ),
    )
    errors = check(tmp_path)
    assert any("没有 ops/demo_op 的证据记录" in e for e in errors)


def test_lock_with_level_claim_requires_known_device_profile(
        tmp_path: Path):
    """阴性：锁 name 不是 configs/devices 已知 profile，无法绑定记录。"""
    _make_repo(
        tmp_path,
        declared=[],
        lock=_demo_lock(
            name="demo-env",
            scope=["ops/demo_op C2（无 profile 可绑）"],
            evidence=[{"path": "evidence.json", "note": "报告"}],
        ),
    )
    errors = check(tmp_path)
    assert any("不是 configs/devices 已知 profile" in e for e in errors)


# ── 审阅 #3 [P2]：仓库内现存文件路径约束（统一解析函数）──

def test_absolute_path_is_rejected(tmp_path: Path):
    """阴性：绝对路径。"""
    _make_repo(
        tmp_path,
        lock=_demo_lock(evidence=[{"path": "/etc/passwd", "note": "外机文件"}]),
    )
    errors = check(tmp_path)
    assert any("不允许绝对路径" in e for e in errors)


def test_catalog_absolute_evidence_path_is_rejected(tmp_path: Path):
    """阴性：catalog evidence_paths 绝对路径。"""
    _make_repo(tmp_path, paths=["/tmp/outside.json"])
    errors = check(tmp_path)
    assert any("不允许绝对路径" in e for e in errors)


def test_path_escaping_repo_root_is_rejected(tmp_path: Path):
    """阴性：`ops/demo_op/../../../outside.md` 越界（外部文件存在）。"""
    outside = tmp_path.parent / f"{tmp_path.name}_outside.md"
    outside.write_text("# outside\n", encoding="utf-8")
    try:
        _make_repo(
            tmp_path,
            lock=_demo_lock(evidence=[{
                "path": f"ops/demo_op/../../../{outside.name}",
                "note": "越界文件",
            }]),
        )
        errors = check(tmp_path)
        assert any("越出仓库根目录" in e for e in errors)
    finally:
        outside.unlink(missing_ok=True)


def test_directory_as_evidence_path_is_rejected(tmp_path: Path):
    """阴性：目录不能当证据文件（catalog 侧）。"""
    _make_repo(tmp_path, paths=["ops"])
    errors = check(tmp_path)
    assert any("必须是文件，不能是目录" in e for e in errors)


def test_external_symlink_is_rejected(tmp_path: Path):
    """阴性：指向仓库外的 symlink。"""
    outside = tmp_path.parent / f"{tmp_path.name}_symlink_target.md"
    outside.write_text("# target\n", encoding="utf-8")
    try:
        _make_repo(tmp_path)
        link = tmp_path / "link.md"
        try:
            link.symlink_to(outside)
        except OSError:
            return  # 文件系统不支持 symlink 时跳过（CI Linux 支持）
        lock = _demo_lock(evidence=[
            {"path": "link.md", "note": "外部 symlink"}])
        env_dir = tmp_path / "configs" / "env"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "demo.lock.yaml").write_text(
            yaml.safe_dump(lock, allow_unicode=True), encoding="utf-8")
        errors = check(tmp_path)
        assert any("越出仓库根目录" in e for e in errors)
    finally:
        outside.unlink(missing_ok=True)


def test_valid_nested_evidence_path_passes(tmp_path: Path):
    """阳性：规范的仓库内相对路径文件。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports" / "ok.md").write_text(
        "# ok\n", encoding="utf-8")
    _make_repo(
        tmp_path,
        lock=_demo_lock(evidence=[
            {"path": "ops/demo_op/reports/ok.md", "note": "正常报告"}]),
    )
    assert check(tmp_path) == []


# ── 审阅 #4 [P2]：REPORT profile 必须完整 token 匹配 ──

def test_e2e_row_with_near_profile_name_is_rejected(tmp_path: Path):
    """阴性：`demo-10` 不能冒充 `demo-1`。"""
    report = (
        "# demo\n\n"
        "| FlagOS E2E | `script --profile demo-10` | ☑ PASS |\n"
    )
    _make_repo(tmp_path, report=report)
    errors = check(tmp_path)
    assert any("FlagOS E2E 结果行未标明具体 profile" in e for e in errors)


def test_e2e_row_with_invalid_wrapped_profile_is_rejected(tmp_path: Path):
    """阴性：`NOT-demo-1-invalid` 里的 demo-1 不算数。"""
    report = (
        "# demo\n\n"
        "| FlagOS E2E | `script NOT-demo-1-invalid` | ☑ PASS |\n"
    )
    _make_repo(tmp_path, report=report)
    errors = check(tmp_path)
    assert any("FlagOS E2E 结果行未标明具体 profile" in e for e in errors)


def test_e2e_success_row_with_empty_profile_set_is_rejected(
        tmp_path: Path):
    """阴性：已知 profile 集为空时，成功声明不能直接放过。"""
    _make_repo(
        tmp_path,
        declared=[],
        profiles=None,
        report=_E2E_ROW_PASS,
    )
    errors = check(tmp_path)
    assert any("已知 profile 集为空" in e for e in errors)


def test_e2e_row_with_full_token_profile_passes(tmp_path: Path):
    """阳性：完整 token（含文件名内嵌 profile）算命中。"""
    report = (
        "# demo\n\n"
        "| FlagOS E2E | `script --profile demo-1` | ☑ PASS "
        "| [reports/e2e_flagos_demo-1.json](reports/e2e_flagos_demo-1.json) |\n"
    )
    _make_repo(tmp_path, report=report)
    assert check(tmp_path) == []


def test_e2e_row_referenced_json_declaring_other_profile_is_rejected(
        tmp_path: Path):
    """阴性：成功行标 demo-1，所引结构化证据却声明 demo-2（范围不一致）。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports" / "e2e_other.json").write_text(
        json.dumps({"profile": "demo-2", "status": "PASS"}),
        encoding="utf-8")
    report = (
        "# demo\n\n"
        "| FlagOS E2E | `script --profile demo-1` | ☑ PASS "
        "| [reports/e2e_other.json](reports/e2e_other.json) |\n"
    )
    _make_repo(tmp_path, report=report)
    errors = check(tmp_path)
    assert any("成功行所引证据" in e and "范围不一致" in e for e in errors)


def test_e2e_row_referenced_json_matching_profile_passes(tmp_path: Path):
    """阳性：成功行标 demo-1，所引结构化证据声明 demo-1（范围一致）。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports" / "e2e_ok.json").write_text(
        json.dumps({"profile": "demo-1", "status": "PASS"}),
        encoding="utf-8")
    report = (
        "# demo\n\n"
        "| FlagOS E2E | `script --profile demo-1` | ☑ PASS "
        "| [reports/e2e_ok.json](reports/e2e_ok.json) |\n"
    )
    _make_repo(tmp_path, report=report)
    assert check(tmp_path) == []


# ── 审阅 #5 [P2]：C1 回填必须基于实际 consumer 证据 ──

def test_c0_contract_only_op_passes_without_consumer_evidence(
        tmp_path: Path):
    """阳性：仅契约/直调范围如实停在 C0，不冒充 C1。"""
    _make_repo(
        tmp_path,
        evidence="C0",
        basis=None,
        paths=None,
        with_framework=False,
        records={
            "demo-1": {"level": "C0",
                       "basis": "仅契约/直调范围，无 consumer 消费证据"},
        },
    )
    assert check(tmp_path) == []


def test_c1_record_without_basis_is_rejected(tmp_path: Path):
    """阴性：C1 记录没有 basis（文件存在不构成 consumer 证据）。"""
    _make_repo(tmp_path, records={"demo-1": {"level": "C1"}})
    errors = check(tmp_path)
    assert any("profile demo-1 C1 记录必须提供 basis" in e for e in errors)


def test_c2_record_without_report_is_rejected(tmp_path: Path):
    """阴性：C2 记录没有 report 路径。"""
    _make_repo(tmp_path, records={"demo-1": {"level": "C2",
                                             "basis": _C1_BASIS}})
    errors = check(tmp_path)
    assert any("profile demo-1 C2 记录必须提供 report" in e for e in errors)


def test_c1_top_level_with_only_c0_records_is_rejected(tmp_path: Path):
    """阴性：声称 C1 但所有 profile 记录都只有 C0（回填虚高）。"""
    _make_repo(
        tmp_path,
        evidence="C1",
        basis=None,
        paths=None,
        records={
            "demo-1": {"level": "C0", "basis": "契约/直调，无消费证据"},
        },
    )
    errors = check(tmp_path)
    assert any("evidence=C1 与 profile 记录最高级 C0 不一致" in e
               for e in errors)


# ── PR #46 后续整改：三轴语义（level / verdict / review）+ 派生更正 ──


def _write_report(root: Path, rel: str, payload) -> str:
    """落一份证据文件（str 原样写入，其他按 JSON），返回仓库相对路径。"""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return rel


def _write_redecide(
    root: Path,
    *,
    summary_rel: str,
    declared: str,
    now: str,
    sha: str | None = None,
    commit: str = "a" * 40,
    rel: str = "reports/acceptance_redecide.json",
) -> str:
    """落一份 acceptance_redecide.json（PR #44 口径），返回相对路径。"""
    if sha is None:
        sha = hashlib.sha256((root / summary_rel).read_bytes()).hexdigest()
    payload = {
        "kind": "c2-acceptance-redecide",
        "raw_material": {
            "summary_path": summary_rel,
            "summary_sha256": sha,
            "profile_field_original": declared,
        },
        "corrections": [{"field": "profile", "was": declared, "now": now}],
        "redecide": {"commit": commit},
    }
    return _write_report(root, rel, payload)


def _c2_repo(tmp_path: Path, rel: str, record_extra: dict | None = None,
             **kwargs) -> Path:
    """C2 记录 report 指向 rel 的最小仓库（paths 自动包含 rel）。"""
    record = {"level": "C2", "basis": _C1_BASIS, "report": rel}
    record.update(record_extra or {})
    return _make_repo(
        tmp_path,
        paths=["evidence.json", rel],
        records={"demo-1": record},
        **kwargs,
    )


def test_c2_record_json_status_fail_is_rejected(tmp_path: Path):
    """阴性：同 profile 但 `status=FAIL` 的报告不能支撑 C2 记录。"""
    rel = _write_report(tmp_path, "reports/fail.json",
                        {"profile": "demo-1", "status": "FAIL"})
    _c2_repo(tmp_path, rel)
    errors = check(tmp_path)
    assert any("报告验证状态非通过" in e for e in errors)


def test_c2_record_json_collect_only_is_rejected(tmp_path: Path):
    """阴性：仅采集完成（collect-only）没有实验结论，判定缺失。"""
    rel = _write_report(tmp_path, "reports/collect.json",
                        {"profile": "demo-1", "status": "collect-only"})
    _c2_repo(tmp_path, rel)
    errors = check(tmp_path)
    assert any("缺少 status/ok 验证状态字段" in e for e in errors)


def test_c2_record_json_without_profile_is_rejected(tmp_path: Path):
    """阴性：结构化报告缺 profile 字段（无法绑定环境）。"""
    rel = _write_report(tmp_path, "reports/noprof.json", {"status": "PASS"})
    _c2_repo(tmp_path, rel)
    errors = check(tmp_path)
    assert any("报告缺少 profile 字段" in e for e in errors)


def test_c2_record_json_corrupt_is_rejected(tmp_path: Path):
    """阴性：损坏的 JSON 报告单独报错，不静默放过。"""
    rel = _write_report(tmp_path, "reports/broken.json", "{ not valid json")
    _c2_repo(tmp_path, rel)
    errors = check(tmp_path)
    assert any("JSON 解析失败" in e for e in errors)


def test_c2_record_json_consumer_mismatch_is_rejected(tmp_path: Path):
    """阴性：记录声明的 consumer 与报告 consumer 不一致。"""
    rel = _write_report(
        tmp_path, "reports/cross_consumer.json",
        {"profile": "demo-1", "status": "PASS",
         "consumer_type": "other-engine"})
    _c2_repo(tmp_path, rel, {"consumer": "vllm-enginecore"})
    errors = check(tmp_path)
    assert any("与记录 consumer='vllm-enginecore' 不一致" in e
               for e in errors)


def test_record_review_pass_vs_report_pending_is_rejected(tmp_path: Path):
    """阴性：记录自称已审阅通过，报告 acceptance 仍是待审阅。"""
    rel = _write_report(
        tmp_path, "reports/pending.json",
        {"profile": "demo-1", "status": "PASS",
         "acceptance": {"status": "complete-pending-review"}})
    _c2_repo(tmp_path, rel, {"status": "pass"})
    errors = check(tmp_path)
    assert any("与报告审阅状态 pending 不一致" in e for e in errors)
    assert any("待审阅不能写成全量 PASS" in e for e in errors)


def test_record_review_pending_matches_report_pending_passes(
        tmp_path: Path):
    """阳性：complete-pending-review = verdict 通过 + 待审阅，两侧一致。"""
    rel = _write_report(
        tmp_path, "reports/pending.json",
        {"profile": "demo-1", "c2_status": "c2-hit",
         "acceptance": {"status": "complete-pending-review"}})
    _c2_repo(tmp_path, rel, {"status": "complete-pending-review"})
    assert check(tmp_path) == []


def test_record_unknown_review_word_is_rejected(tmp_path: Path):
    """阴性：记录 status 用未知词（审阅状态不许臆断）。"""
    rel = _write_report(tmp_path, "reports/ok.json",
                        {"profile": "demo-1", "status": "PASS"})
    _c2_repo(tmp_path, rel, {"status": "maybe"})
    errors = check(tmp_path)
    assert any("不是已知的审阅状态词" in e for e in errors)


def test_c2_record_redecide_corrects_profile_scope(tmp_path: Path):
    """阳性：报告保留旧 profile，核验通过的 redecide 采用更正范围。"""
    rel = _write_report(tmp_path, "ops/demo_op/reports/legacy.json",
                        {"profile": "legacy-demo", "status": "PASS"})
    redecide_rel = _write_redecide(
        tmp_path, summary_rel=rel, declared="legacy-demo", now="demo-1")
    _c2_repo(tmp_path, rel, {"redecide": redecide_rel})
    assert check(tmp_path) == []


def test_c2_record_redecide_bad_sha_is_rejected(tmp_path: Path):
    """阴性：redecide 记录的源 SHA256 与实际源文件不符。"""
    rel = _write_report(tmp_path, "ops/demo_op/reports/legacy.json",
                        {"profile": "legacy-demo", "status": "PASS"})
    redecide_rel = _write_redecide(
        tmp_path, summary_rel=rel, declared="legacy-demo", now="demo-1",
        sha="0" * 64)
    _c2_repo(tmp_path, rel, {"redecide": redecide_rel})
    errors = check(tmp_path)
    assert any("源文件 SHA256 不匹配" in e for e in errors)


def test_c2_record_redecide_missing_source_is_rejected(tmp_path: Path):
    """阴性：redecide 指向的源文件不存在（不能凭更正文件放行）。"""
    redecide_rel = _write_redecide(
        tmp_path, summary_rel="reports/gone.json",
        declared="legacy-demo", now="demo-1", sha="0" * 64)
    _c2_repo(tmp_path, "evidence.json", {"redecide": redecide_rel})
    errors = check(tmp_path)
    assert any("raw_material.summary_path 不存在: reports/gone.json" in e
               for e in errors)


def test_c2_record_redecide_bad_commit_is_rejected(tmp_path: Path):
    """阴性：redecide.commit 不是 40 位 commit。"""
    rel = _write_report(tmp_path, "ops/demo_op/reports/legacy.json",
                        {"profile": "legacy-demo", "status": "PASS"})
    redecide_rel = _write_redecide(
        tmp_path, summary_rel=rel, declared="legacy-demo", now="demo-1",
        commit="not-a-commit")
    _c2_repo(tmp_path, rel, {"redecide": redecide_rel})
    errors = check(tmp_path)
    assert any("redecide.commit 不是 40 位 commit" in e for e in errors)


def test_lock_evidence_json_backed_by_verified_redecide_passes(
        tmp_path: Path):
    """阳性：锁证据 JSON 声明旧 profile，但记录的 redecide 核验通过。"""
    rel = _write_report(tmp_path, "ops/demo_op/reports/legacy.json",
                        {"profile": "legacy-demo", "status": "PASS"})
    redecide_rel = _write_redecide(
        tmp_path, summary_rel=rel, declared="legacy-demo", now="demo-1")
    _c2_repo(
        tmp_path, rel, {"redecide": redecide_rel},
        lock=_demo_lock(
            scope=["ops/demo_op C2（更正后的范围）"],
            evidence=[{"path": rel, "note": "旧 profile 的结构化报告"}],
        ),
    )
    assert check(tmp_path) == []


def test_lock_evidence_json_redecide_bad_sha_is_rejected(tmp_path: Path):
    """阴性：锁证据侧的 redecide 源哈希不符，既报错也不能成为支撑。"""
    rel = _write_report(tmp_path, "ops/demo_op/reports/legacy.json",
                        {"profile": "legacy-demo", "status": "PASS"})
    redecide_rel = _write_redecide(
        tmp_path, summary_rel=rel, declared="legacy-demo", now="demo-1",
        sha="0" * 64)
    _c2_repo(
        tmp_path, rel, {"redecide": redecide_rel},
        lock=_demo_lock(
            scope=["ops/demo_op C2（坏哈希不能背书）"],
            evidence=[{"path": rel, "note": "源哈希不符"}],
        ),
    )
    errors = check(tmp_path)
    assert any("派生更正核验失败" in e and "SHA256 不匹配" in e
               for e in errors)
    assert any("没有对应报告支撑" in e for e in errors)


def test_lock_c2_scope_backed_by_collect_only_json_is_rejected(
        tmp_path: Path):
    """阴性：仅采集完成的 JSON 不能支撑锁的 C2 声明。"""
    rel = _write_report(tmp_path, "ops/demo_op/reports/collect.json",
                        {"profile": "demo-1", "status": "collect-only"})
    _make_repo(
        tmp_path,
        paths=["evidence.json", rel],
        lock=_demo_lock(
            scope=["ops/demo_op C2（只采了数据）"],
            evidence=[{"path": rel, "note": "collect-only 报告"}],
        ),
    )
    errors = check(tmp_path)
    assert any("支撑报告不合格" in e and "缺少 status/ok" in e
               for e in errors)


# ── PR #46 复审 P1：lock evidence[].path 入口结构化校验（与 catalog
#    record.report 同口径：可解析 / profile 一致含 redecide / verdict
#    通过 / consumer 一致；不通过即报「路径 + 原因」且不得成为支撑）──


def test_lock_evidence_fail_json_without_scope_is_rejected(
        tmp_path: Path):
    """阴性（帖称“空错误列表”的场景）：无 scope 时 FAIL JSON 也报错。"""
    rel = _write_report(tmp_path, "ops/demo_op/reports/fail.json",
                        {"profile": "demo-1", "status": "FAIL"})
    _make_repo(
        tmp_path,
        lock=_demo_lock(evidence=[{"path": rel, "note": "FAIL 报告"}]),
    )
    errors = check(tmp_path)
    assert any(f"evidence {rel}" in e and "状态非通过" in e
               and "不能作为支撑" in e for e in errors)


def test_lock_evidence_statusless_json_without_scope_is_rejected(
        tmp_path: Path):
    """阴性（帖称“空错误列表”的场景）：缺 status/ok 的 JSON 也报错。"""
    rel = _write_report(tmp_path, "ops/demo_op/reports/nostatus.json",
                        {"profile": "demo-1"})
    _make_repo(
        tmp_path,
        lock=_demo_lock(evidence=[{"path": rel, "note": "无判定报告"}]),
    )
    errors = check(tmp_path)
    assert any(f"evidence {rel}" in e and "缺少 status/ok" in e
               and "不能作为支撑" in e for e in errors)


def test_lock_evidence_collect_only_json_without_scope_is_rejected(
        tmp_path: Path):
    """阴性：仅采集完成（collect-only）没有实验结论，入口即拦。"""
    rel = _write_report(tmp_path, "ops/demo_op/reports/collect.json",
                        {"profile": "demo-1", "status": "collect-only"})
    _make_repo(
        tmp_path,
        lock=_demo_lock(evidence=[{"path": rel, "note": "只采了数据"}]),
    )
    errors = check(tmp_path)
    assert any(f"evidence {rel}" in e and "缺少 status/ok" in e
               and "不能作为支撑" in e for e in errors)


def test_lock_evidence_corrupt_json_without_scope_is_rejected(
        tmp_path: Path):
    """阴性（帖称“空错误列表”的场景）：损坏 JSON 入口单独报路径 + 原因。"""
    rel = _write_report(tmp_path, "ops/demo_op/reports/broken.json",
                        "{ not valid json")
    _make_repo(
        tmp_path,
        lock=_demo_lock(evidence=[{"path": rel, "note": "损坏报告"}]),
    )
    errors = check(tmp_path)
    assert any(f"evidence {rel}" in e and "JSON 解析失败" in e
               and "不能作为支撑" in e for e in errors)


def test_lock_evidence_consumer_mismatch_is_rejected(tmp_path: Path):
    """阴性：记录声明 consumer，报告 engine/consumer 不一致（入口即拦）。"""
    rel = _write_report(
        tmp_path, "ops/demo_op/reports/cross_consumer.json",
        {"profile": "demo-1", "status": "PASS", "engine": "other-engine"})
    _c2_repo(
        tmp_path, rel, {"consumer": "vllm-enginecore"},
        lock=_demo_lock(evidence=[{"path": rel, "note": "consumer 冲突"}]),
    )
    errors = check(tmp_path)
    assert any(
        f"evidence {rel}" in e
        and "报告 consumer='other-engine' 与记录 "
            "consumer='vllm-enginecore' 不一致" in e
        and "不能作为支撑" in e for e in errors)


def test_lock_evidence_json_with_matching_consumer_passes(tmp_path: Path):
    """阳性：合法报告（同 profile + PASS + consumer 与记录一致）通过。"""
    rel = _write_report(
        tmp_path, "ops/demo_op/reports/ok_consumer.json",
        {"profile": "demo-1", "status": "PASS",
         "consumer_type": "vllm-enginecore"})
    _c2_repo(
        tmp_path, rel, {"consumer": "vllm-enginecore"},
        lock=_demo_lock(evidence=[{"path": rel, "note": "合法报告"}]),
    )
    assert check(tmp_path) == []


def test_lock_c1_scope_backed_by_fail_json_is_rejected(tmp_path: Path):
    """阴性：FAIL 的同 profile JSON 不得成为 C1 声明的支撑。"""
    rel = _write_report(tmp_path, "ops/demo_op/reports/fail.json",
                        {"profile": "demo-1", "status": "FAIL"})
    _make_repo(
        tmp_path,
        evidence="C1",
        basis=None,
        paths=None,
        lock=_demo_lock(
            scope=["ops/demo_op C1（mini consumer 双跑）"],
            evidence=[{"path": rel, "note": "FAIL 报告"}],
        ),
    )
    errors = check(tmp_path)
    assert any(f"evidence {rel}" in e and "状态非通过" in e for e in errors)
    assert any("没有对应报告支撑" in e and "支撑报告不合格" in e
               for e in errors)


def test_lock_c1_scope_backed_by_statusless_json_is_rejected(
        tmp_path: Path):
    """阴性：缺 status/ok 的 JSON 不得成为 C1 声明的支撑。"""
    rel = _write_report(tmp_path, "ops/demo_op/reports/nostatus.json",
                        {"profile": "demo-1"})
    _make_repo(
        tmp_path,
        evidence="C1",
        basis=None,
        paths=None,
        lock=_demo_lock(
            scope=["ops/demo_op C1（mini consumer 双跑）"],
            evidence=[{"path": rel, "note": "无判定报告"}],
        ),
    )
    errors = check(tmp_path)
    assert any(f"evidence {rel}" in e and "缺少 status/ok" in e
               for e in errors)
    assert any("没有对应报告支撑" in e and "支撑报告不合格" in e
               for e in errors)


def test_lock_c1_scope_backed_by_md_without_profile_token_is_rejected(
        tmp_path: Path):
    """阴性：md 支撑按 scope 等级声明要求正文含本锁 profile 完整 token。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports" / "notes.md").write_text(
        "# notes\n\n只写环境描述，没有 profile 标记。\n", encoding="utf-8")
    _make_repo(
        tmp_path,
        evidence="C1",
        basis=None,
        paths=None,
        lock=_demo_lock(
            scope=["ops/demo_op C1（mini consumer 双跑）"],
            evidence=[{"path": "ops/demo_op/reports/notes.md",
                       "note": "无 profile token 的叙述"}],
        ),
    )
    errors = check(tmp_path)
    assert any("支撑材料不合格" in e
               and "正文未出现 profile demo-1 标记" in e for e in errors)
