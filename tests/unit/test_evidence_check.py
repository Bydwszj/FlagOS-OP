from __future__ import annotations

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
    family_evidence_level: str | None = None,
) -> Path:
    """搭一个最小可过的 demo 仓。

    - `declared`：`platforms.demo.profiles`（默认 `["demo-1"]`）；
      每个 declared profile 默认生成一条与顶层 `evidence` 同级的记录，
      C1/C2 记录自动带 `basis`，C2 记录 `report` 指向 `evidence.json`
      （内容声明该 profile 与 PASS 状态）；
    - `records`：整体替换 `evidence.profiles` 映射（不写报告文件，
      需要时测试自行落盘）；
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
                "evidence": {
                    "profiles": records,
                    **({"level": family_evidence_level}
                       if family_evidence_level is not None else {}),
                },
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


def test_missing_family_evidence_profiles_is_rejected(tmp_path: Path):
    """阴性：family 完全没有 evidence.profiles 记录（旧 schema）。"""
    _make_repo(tmp_path)
    catalog_path = tmp_path / "ops" / "catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text())
    catalog["operators"][0]["platforms"]["demo"].pop("evidence")
    catalog_path.write_text(
        yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")
    errors = check(tmp_path)
    assert any("缺 evidence.profiles" in e for e in errors)


def test_top_level_evidence_must_be_derived_from_records(tmp_path: Path):
    """阴性：顶层汇总 C2 但 profile 记录最高只有 C1（不能独立填写）。"""
    _make_repo(
        tmp_path,
        evidence="C2",
        records={"demo-1": {"level": "C1", "basis": _C1_BASIS}},
    )
    errors = check(tmp_path)
    assert any("与 profile 记录最高级 C1 不一致" in e for e in errors)


def test_family_level_field_must_be_derived_from_records(tmp_path: Path):
    """阴性：PR44 兼容的 family 汇总字段与记录不一致。"""
    _make_repo(
        tmp_path,
        records={"demo-1": {"level": "C1", "basis": _C1_BASIS}},
        family_evidence_level="C2",
    )
    # 顶层 evidence 仍为默认 C2 → 先触发派生不一致；再把顶层调成 C1
    catalog_path = tmp_path / "ops" / "catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text())
    catalog["operators"][0]["evidence"] = "C1"
    catalog_path.write_text(
        yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")
    errors = check(tmp_path)
    assert any("家族汇总必须由记录派生" in e for e in errors)


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
    """阳性：锁 profile=demo-1 的 C2 声明由声明 demo-1 的结构化报告支撑。"""
    (tmp_path / "ops" / "demo_op" / "reports").mkdir(
        parents=True, exist_ok=True)
    (tmp_path / "ops" / "demo_op" / "reports"
     / "e2e_vllm_c2_demo-1.json").write_text(
        json.dumps({"profile": "demo-1", "status": "PASS",
                    "consumer_type": "vllm-enginecore"}),
        encoding="utf-8")
    _make_repo(
        tmp_path,
        lock=_demo_lock(
            scope=["ops/demo_op C2（真实 vLLM EngineCore 消费）"],
            evidence=[{
                "path": "ops/demo_op/reports/e2e_vllm_c2_demo-1.json",
                "note": "C2 双跑",
            }],
        ),
    )
    assert check(tmp_path) == []


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
