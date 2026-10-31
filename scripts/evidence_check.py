#!/usr/bin/env python3
"""C1/C2 证据门禁（issue #41）。

检查六件事（任何一条失败即非零退出）：

1. `ops/catalog.yaml` 每个算子必须声明 `evidence: C0|C1|C2`；
   `C1`/`C2` 要求 `ops/<dir>/test/framework_level.py` 存在。
2. `C2` 必须给出 `evidence_basis`（一句范围诚实的说明）与
   `evidence_paths`（仓库内文件列表），且每个路径真实存在——
   证据声明必须锁到具体文件，防止只写结论不给产物。
3. 平台状态三方一致：`ops/catalog.yaml` 的 platform_families 状态
   ↔ `docs/operator-catalog.md` 状态表 ↔ 生成的 `ops/README.md`
   行（README 的两份生成口径由 `build_ops_index.py --check` 兜底，
   本脚本复核状态列）。
4. 日期字段（catalog `updated`、环境锁 `collected_at` /
   `hardware_collected_at`）必须是合法 `YYYY-MM-DD` 且不晚于今天。
5. 聚合报告 `ops/*/REPORT.md` 中宣称通过的 FlagOS E2E 结果行必须
   标明具体 profile/环境（`configs/devices` 已知 profile 之一）——
   P800 的结果不能自动代表 PPU。
6. 环境锁 `configs/env/*.lock.yaml`：`evidence` 条目必须给出仓库内
   且当前存在的报告路径（禁占位/外链）；`scope` 行里的
   `ops/<算子> C[012]` 声明必须与 catalog 的 `evidence` 等级一致，
   且该锁 `evidence` 中要有该算子的报告支撑。

用法::

    python3 scripts/evidence_check.py [--root PATH]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

import yaml

VALID_EVIDENCE = ("C0", "C1", "C2")
FAMILY_STATUS_CN = {"partial": "部分交付", "planned": "规划中"}
_GUIDE_TABLE_ROW = re.compile(
    r"^\|\s*`(?P<key>[a-z0-9_]+)`\s*\|\s*(?P<display>[^|]+?)\s*"
    r"\|\s*(?P<status>partial|planned)\b"
)
_E2E_MARK = re.compile(r"FlagOS\s+E2E|e2e_flagos", re.IGNORECASE)
_SUCCESS_MARKS = ("☑", "✅", "✔", "PASS", "通过", "全绿")
_FAILURE_MARKS = ("✗", "未通过", "缺证", "未跑", "失败")
_SCOPE_OP = re.compile(r"ops/(?P<op>[A-Za-z0-9_]+)")
_SCOPE_OP_LEVEL = re.compile(
    r"ops/(?P<op>[A-Za-z0-9_]+)\s+(?P<level>C[012])(?![A-Za-z0-9_])")
_EVIDENCE_OP_PATH = re.compile(r"^ops/(?P<op>[A-Za-z0-9_]+)/")


def _parse_date(value) -> _dt.date | None:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return _dt.date.fromisoformat(value)
    except ValueError:
        return None


def _load_locks(root: Path, errors: list[str]) -> list[tuple[Path, dict]]:
    """读入 `configs/env/*.lock.yaml`；解析失败记为错误并跳过该文件。"""
    locks: list[tuple[Path, dict]] = []
    env_dir = root / "configs" / "env"
    if not env_dir.is_dir():
        return locks
    for lock_path in sorted(env_dir.glob("*.lock.yaml")):
        try:
            lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            errors.append(f"{lock_path}: YAML 解析失败 {exc}")
            continue
        if not isinstance(lock, dict):
            errors.append(f"{lock_path}: 顶层不是映射")
            continue
        locks.append((lock_path, lock))
    return locks


def _check_dates(root: Path, catalog: dict,
                 locks: list[tuple[Path, dict]], errors: list[str]) -> None:
    today = _dt.date.today()
    updated = _parse_date(catalog.get("updated"))
    if updated is None:
        errors.append(f"ops/catalog.yaml: updated 非法: {catalog.get('updated')!r}")
    elif updated > today:
        errors.append(f"ops/catalog.yaml: updated 在未来: {updated}")

    for lock_path, lock in locks:
        for key in ("collected_at", "hardware_collected_at"):
            if key not in lock:
                continue
            day = _parse_date(lock.get(key))
            if day is None:
                errors.append(
                    f"{lock_path.name}: {key} 非法: {lock.get(key)!r}")
            elif day > today:
                errors.append(f"{lock_path.name}: {key} 在未来: {day}")


def _check_evidence(root: Path, catalog: dict, errors: list[str]) -> None:
    for op in catalog.get("operators", []):
        name = op.get("name", "?")
        level = op.get("evidence")
        if level not in VALID_EVIDENCE:
            errors.append(
                f"operator {name}: evidence 必须是 {VALID_EVIDENCE}，"
                f"当前 {level!r}")
            continue
        directory = op.get("directory", name)
        framework = root / "ops" / directory / "test" / "framework_level.py"
        if level in ("C1", "C2") and not framework.is_file():
            errors.append(
                f"operator {name}: evidence={level} 但缺少 "
                f"ops/{directory}/test/framework_level.py")
        if level == "C2":
            basis = op.get("evidence_basis")
            if not isinstance(basis, str) or not basis.strip():
                errors.append(f"operator {name}: C2 必须提供 evidence_basis")
            paths = op.get("evidence_paths")
            if not isinstance(paths, list) or not paths:
                errors.append(f"operator {name}: C2 必须提供 evidence_paths")
                continue
            for rel in paths:
                if not isinstance(rel, str) or not rel:
                    errors.append(f"operator {name}: evidence_paths 含非法项 {rel!r}")
                    continue
                if rel.startswith(("http://", "https://")):
                    errors.append(
                        f"operator {name}: evidence_paths 必须是仓库内路径，"
                        f"不允许外链 {rel}")
                    continue
                if not (root / rel).exists():
                    errors.append(
                        f"operator {name}: evidence_paths 不存在: {rel}")


def _guide_family_statuses(root: Path, errors: list[str]) -> dict[str, str]:
    guide = root / "docs" / "operator-catalog.md"
    found: dict[str, str] = {}
    if not guide.is_file():
        errors.append("docs/operator-catalog.md 缺失")
        return found
    in_table = False
    for line in guide.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Family key"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if set(line.replace("|", "").strip()) <= {"-", " "}:
                continue
            m = _GUIDE_TABLE_ROW.match(line)
            if m:
                found[m.group("key")] = m.group("status")
    if not found:
        errors.append("docs/operator-catalog.md: 未找到家族状态表")
    return found


def _check_family_consistency(root: Path, catalog: dict,
                              errors: list[str]) -> None:
    families = catalog.get("platform_families", {})
    guide = _guide_family_statuses(root, errors)
    for key, meta in families.items():
        want = meta.get("status")
        got = guide.get(key)
        if got is None:
            errors.append(f"docs/operator-catalog.md: 状态表缺少家族 {key}")
        elif got != want:
            errors.append(
                f"平台状态不一致 {key}: catalog={want} "
                f"operator-catalog.md={got}")
    for key in guide:
        if key not in families:
            errors.append(
                f"docs/operator-catalog.md: 状态表含 catalog 没有的家族 {key}")

    readme = root / "ops" / "README.md"
    if not readme.is_file():
        errors.append("ops/README.md 缺失（先运行 scripts/build_ops_index.py）")
        return
    text = readme.read_text(encoding="utf-8")
    for key, meta in families.items():
        status_cn = FAMILY_STATUS_CN.get(meta.get("status"))
        if status_cn is None:
            continue  # 非法状态由 build_ops_index 再报
        row = f"| {meta.get('display_name', key)}（`{key}`） | {status_cn} |"
        if row not in text:
            errors.append(f"ops/README.md 缺少状态行: {row}")


def _known_profiles(root: Path) -> set[str]:
    """已知 profile / 环境名。

    来源两处：`configs/devices/*.yaml`（`_template` 除外）与
    `configs/env/*.lock.yaml` 的 `name`——报告标 profile 时只能用
    这套机器可读的名字，写「P800」这类泛称不算数。
    """
    names: set[str] = set()
    devices = root / "configs" / "devices"
    if devices.is_dir():
        for path in devices.glob("*.yaml"):
            if not path.stem.startswith("_"):
                names.add(path.stem)
    env_dir = root / "configs" / "env"
    if env_dir.is_dir():
        for path in sorted(env_dir.glob("*.lock.yaml")):
            try:
                lock = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError:
                continue
            name = lock.get("name") if isinstance(lock, dict) else None
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _check_report_e2e_profiles(root: Path, errors: list[str]) -> None:
    """缺口 1：聚合报告里的 FlagOS E2E 结果必须标明具体 profile。

    只看 `ops/*/REPORT.md`（一页看全的聚合页）。同一行里既宣称通过
    （☑/PASS/通过…）又报 FlagOS E2E 结果的，必须出现已知 profile；
    纯叙述、纯数字、标 ✗ 的行不在此列。
    """
    ops_dir = root / "ops"
    if not ops_dir.is_dir():
        return
    profiles = _known_profiles(root)
    if not profiles:
        return
    for report in sorted(ops_dir.glob("*/REPORT.md")):
        rel = report.relative_to(root)
        text = report.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not _E2E_MARK.search(line):
                continue
            if not any(mark in line for mark in _SUCCESS_MARKS):
                continue
            if any(mark in line for mark in _FAILURE_MARKS):
                continue
            if any(profile in line for profile in profiles):
                continue
            errors.append(
                f"{rel}:{lineno}: FlagOS E2E 结果行未标明具体 profile/"
                f"环境（需含 configs/devices 已知 profile 之一，"
                f"如 {sorted(profiles)[0]}）")


def _check_lock_crossref(root: Path, catalog: dict,
                         locks: list[tuple[Path, dict]],
                         errors: list[str]) -> None:
    """缺口 2：环境锁 evidence/scope 与 catalog evidence 的交叉引用。

    - `evidence[].path` 必须是仓库内、当前存在的报告（禁外链/占位）；
      指向 `ops/<dir>/...` 时该算子必须在 catalog 登记；
    - `scope` 行里的 `ops/<dir>` 必须是 catalog 登记的算子；
    - `scope` 行声明 `ops/<dir> C[012]` 时，必须等于 catalog 的
      `evidence`，且本锁 `evidence` 里要有该算子的报告作为支撑。
    """
    ops_by_dir: dict[str, dict] = {}
    for op in catalog.get("operators", []):
        name = op.get("name", "?")
        ops_by_dir[op.get("directory", name)] = op

    for lock_path, lock in locks:
        rel = lock_path.relative_to(root)
        referenced: set[str] = set()

        entries = lock.get("evidence", [])
        if not isinstance(entries, list):
            errors.append(f"{rel}: evidence 必须是列表，当前 "
                          f"{type(entries).__name__}")
            entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                errors.append(f"{rel}: evidence 条目必须是映射 {entry!r}")
                continue
            path = entry.get("path")
            if not isinstance(path, str) or not path.strip():
                errors.append(f"{rel}: evidence 缺 path（占位证据）")
                continue
            if path.startswith(("http://", "https://")):
                errors.append(f"{rel}: evidence 必须是仓库内路径，"
                              f"不允许外链 {path}")
                continue
            if not (root / path).is_file():
                errors.append(f"{rel}: evidence 指向不存在的报告 {path}")
                continue
            m = _EVIDENCE_OP_PATH.match(path)
            if not m:
                continue
            op_name = m.group("op")
            if op_name in ops_by_dir:
                referenced.add(op_name)
            else:
                errors.append(
                    f"{rel}: evidence 引用 catalog 没有登记的算子 "
                    f"ops/{op_name}")

        scope = lock.get("scope", [])
        if not isinstance(scope, list):
            errors.append(f"{rel}: scope 必须是列表，当前 "
                          f"{type(scope).__name__}")
            scope = []
        for item in scope:
            if not isinstance(item, str):
                continue
            for m in _SCOPE_OP.finditer(item):
                op_name = m.group("op")
                if op_name not in ops_by_dir:
                    errors.append(
                        f"{rel}: scope 引用 catalog 没有登记的算子 "
                        f"ops/{op_name}")
            m = _SCOPE_OP_LEVEL.search(item)
            if m is None:
                continue
            op_name, claimed = m.group("op"), m.group("level")
            if op_name not in ops_by_dir:
                continue  # 未登记已由上面报过
            actual = ops_by_dir[op_name].get("evidence")
            if actual != claimed:
                errors.append(
                    f"{rel}: scope 声明 ops/{op_name} {claimed} 与 "
                    f"catalog evidence={actual} 不一致")
            if op_name not in referenced:
                errors.append(
                    f"{rel}: scope 的 ops/{op_name} {claimed} 声明在本锁 "
                    f"evidence 中没有对应报告支撑")


def check(root: Path) -> list[str]:
    errors: list[str] = []
    catalog_path = root / "ops" / "catalog.yaml"
    if not catalog_path.is_file():
        return [f"{catalog_path} 缺失"]
    try:
        catalog = yaml.safe_load(catalog_path.read_text())
    except yaml.YAMLError as exc:
        return [f"ops/catalog.yaml: YAML 解析失败 {exc}"]
    if not isinstance(catalog, dict):
        return ["ops/catalog.yaml: 顶层不是映射"]

    locks = _load_locks(root, errors)
    _check_dates(root, catalog, locks, errors)
    _check_evidence(root, catalog, errors)
    _check_family_consistency(root, catalog, errors)
    _check_report_e2e_profiles(root, errors)
    _check_lock_crossref(root, catalog, locks, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="仓库根目录（默认本脚本上级）",
    )
    args = ap.parse_args(argv)
    errors = check(args.root)
    if errors:
        print(f"evidence_check: {len(errors)} 个错误")
        for err in errors:
            print(f" - {err}")
        return 1
    print("evidence_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
