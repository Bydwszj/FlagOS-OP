from __future__ import annotations

import textwrap
from pathlib import Path

import yaml

from scripts.evidence_check import check


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
) -> Path:
    paths = ["evidence.json"] if paths is None else paths
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "demo_op" / "test").mkdir(parents=True, exist_ok=True)
    (root / "evidence.json").parent.mkdir(parents=True, exist_ok=True)

    op: dict = {
        "name": "demo_op",
        "directory": "demo_op",
        "group": "core",
        "purpose": "demo",
        "key_result": "demo",
        "platforms": {"demo": {"profiles": ["demo-1"]}},
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
                "current_profiles": [],
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
    (root / "evidence.json").write_text("{}", encoding="utf-8")
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
    for profile in profiles or []:
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


# ── 缺口 2：环境锁 evidence/scope ↔ catalog evidence 交叉引用 ──

def _demo_lock(*, scope=None, evidence=None) -> dict:
    lock: dict = {"name": "demo-env", "schema": 1, "collected_at": "2026-10-29"}
    if scope is not None:
        lock["scope"] = scope
    if evidence is not None:
        lock["evidence"] = evidence
    return lock


def test_lock_scope_and_evidence_crossref_passes(tmp_path: Path):
    _make_repo(
        tmp_path,
        report="# demo\n",
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
        report="# demo\n",
        lock=_demo_lock(
            scope=["ops/demo_op C2（真实 vLLM 双跑）"],
            evidence=[{
                "path": "ops/demo_op/REPORT.md",
                "note": "聚合报告",
            }],
        ),
    )
    errors = check(tmp_path)
    assert any("ops/demo_op C2 与 catalog evidence=C1 不一致" in e
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
