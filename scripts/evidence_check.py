#!/usr/bin/env python3
"""C1/C2 证据门禁（issue #41）。

检查六件事（任何一条失败即非零退出）：

1. `ops/catalog.yaml` 每个算子必须声明 `evidence: C0|C1|C2`，且该汇总
   必须等于各 `platforms.<family>.evidence.profiles` 记录的最高等级
   ——**算子 + profile 是证据记录的最小单位**，汇总只能由记录派生；
   `platforms.<family>.profiles` 里的每个 profile 都要有独立记录
   （单个 family 下多个 profile 不共享未区分的等级）；C1/C2 记录
   必须给 `basis`（consumer 证据范围）。`framework_level.py` 存在
   只是结构前提，不单独构成 C1（详见 docs/operator-catalog.md）。
2. `C2` 必须给出 `evidence_basis` 与 `evidence_paths`；所有证据路径
   经统一解析函数核验：禁外链、禁绝对路径、`resolve()` 后必须仍在
   仓库根内、必须是文件（目录、`..` 越界、仓库外 symlink 均拒绝）。
   C2 记录的 `report` 必须列入 `evidence_paths`；结构化 JSON 报告
   必须声明相同 `profile` 并给出通过状态（`status`/`ok`），不能拿
   别的环境的报告给本 profile 背书。
3. 平台状态三方一致：`ops/catalog.yaml` 的 platform_families 状态
   ↔ `docs/operator-catalog.md` 状态表 ↔ 生成的 `ops/README.md`
   行（README 的两份生成口径由 `build_ops_index.py --check` 兜底，
   本脚本复核状态列）。
4. 日期字段（catalog `updated`、环境锁 `collected_at` /
   `hardware_collected_at`）必须是合法 `YYYY-MM-DD` 且不晚于今天。
5. 聚合报告 `ops/*/REPORT.md` 中宣称通过的 FlagOS E2E 结果行必须
   出现已知 profile 的**完整 token**（边界匹配：`demo-10` /
   `NOT-demo-1-invalid` 都不算 `demo-1`）；已知 profile 集为空时，
   存在成功声明即报错——P800 的结果不能自动代表 PPU。行内引用的
   结构化 JSON 证据若声明 `profile`，必须与行内 profile 范围一致
   （声明了别的环境即报错；未声明 profile 的存量文件登记为已知
   缺口，不在本层臆断其范围）。
6. 环境锁 `configs/env/*.lock.yaml`：`evidence[].path` 用同一路径
   解析函数核验（禁占位/外链/越界）；锁 `name` 必须是
   `configs/devices` 已知 profile，且结构化证据必须可解析并声明
   本锁 `profile`（损坏/缺 profile 的 JSON 单独报错，不能静默跳过，
   也不能成为支撑）；`scope` 行的 `ops/<算子> C[012]` 必须等于
   **该锁 profile** 的记录等级（而不是 operator 全局汇总），C2
   声明的支撑报告必须**有效且通过**（JSON 可解析、声明本锁
   profile、status/ok 通过），并与该 profile 的 C2 记录
   `report`/`consumer` 同源关联。

用法::

    python3 scripts/evidence_check.py [--root PATH]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
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
_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_PATH_ERR = {
    "外链": "必须是仓库内路径，不允许外链 {rel}",
    "绝对路径": "必须是仓库内相对路径，不允许绝对路径 {rel}",
    "越界": "越出仓库根目录 {rel}",
    "目录": "必须是文件，不能是目录 {rel}",
    "不存在": "不存在: {rel}",
}


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


def _resolve_repo_file(
        root: Path, rel: str) -> tuple[Path | None, str | None, str | None]:
    """统一的仓库内证据路径解析。

    返回 ``(绝对路径, 规范化仓库相对路径, 错误类别)``。拒绝外链、
    绝对路径、逃出 repo root 的 `..` 越界与外部 symlink、目录；
    成功时给出规范化相对路径，供算子归属与跨引用基于同一字符串。
    """
    if not isinstance(rel, str) or not rel.strip():
        return None, None, "不存在"
    rel = rel.strip()
    if "://" in rel:
        return None, None, "外链"
    p = Path(rel)
    if p.is_absolute():
        return None, None, "绝对路径"
    try:
        resolved = (root / p).resolve()
    except OSError:
        return None, None, "越界"
    root_res = root.resolve()
    try:
        norm = resolved.relative_to(root_res).as_posix()
    except ValueError:
        return None, None, "越界"
    if resolved.is_dir():
        return None, None, "目录"
    if not resolved.is_file():
        return None, None, "不存在"
    return resolved, norm, None


def _path_error(kind: str, rel: str) -> str:
    return _PATH_ERR[kind].format(rel=rel)


def _token_in(line: str, token: str) -> bool:
    """完整 token 边界匹配。

    边界字符为 `[A-Za-z0-9-]`：`demo-10`、`NOT-demo-1-invalid`、
    `p800-kunlunxin-gems5` 里的 `p800-kunlunxin` 都不算命中；
    `_`/`.`/`/` 等分隔符后出现的完整 profile 名（如文件名
    `e2e_flagos_p800-kunlunxin.json` 内嵌的 profile）算命中。
    """
    return re.search(
        rf"(?<![A-Za-z0-9-]){re.escape(token)}(?![A-Za-z0-9-])",
        line) is not None


def _device_profiles(root: Path) -> set[str]:
    """`configs/devices/*.yaml` 的 profile 名（`_template` 除外）。"""
    names: set[str] = set()
    devices = root / "configs" / "devices"
    if devices.is_dir():
        for path in devices.glob("*.yaml"):
            if not path.stem.startswith("_"):
                names.add(path.stem)
    return names


def _known_profiles(root: Path) -> set[str]:
    """已知 profile / 环境名。

    来源两处：`configs/devices/*.yaml`（`_template` 除外）与
    `configs/env/*.lock.yaml` 的 `name`——报告标 profile 时只能用
    这套机器可读的名字，写「P800」这类泛称不算数。
    """
    names = _device_profiles(root)
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


def _profile_family_map(catalog: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for family, meta in catalog.get("platform_families", {}).items():
        if not isinstance(meta, dict):
            continue
        for profile in meta.get("current_profiles") or []:
            mapping[profile] = family
    return mapping


def _report_declares_profile(
        root: Path, resolved: Path, norm: str, profile: str,
        record: dict, where: str) -> list[str]:
    """结构化证据报告必须声明同一 profile/consumer/status。

    JSON：核对 `profile` 字段与通过状态（`status`/`ok`），记录里
    声明的 `consumer`/`status` 与报告一致；其他文本报告（md 等）：
    要求正文出现该 profile 的完整 token。避免要求程序理解任意
    自然语言，判定只走结构化字段或 token。
    """
    errors: list[str] = []
    if norm.endswith(".json"):
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            return [f"{where}: JSON 解析失败 {exc}"]
        if not isinstance(data, dict):
            return [f"{where}: JSON 顶层不是映射"]
        declared = data.get("profile")
        if not isinstance(declared, str) or not declared:
            errors.append(f"{where}: 报告缺少 profile 字段")
        elif declared != profile:
            errors.append(
                f"{where}: 报告声明 profile={declared} 与记录 "
                f"profile={profile} 不一致")
        status = data.get("status")
        if isinstance(status, str):
            if status.upper() not in ("PASS", "OK"):
                errors.append(f"{where}: 报告验证状态非通过: {status!r}")
        elif data.get("ok") is True:
            pass
        else:
            errors.append(f"{where}: 报告缺少 status/ok 验证状态字段")
        want_status = record.get("status")
        if isinstance(want_status, str) and isinstance(status, str):
            if want_status.upper() != status.upper():
                errors.append(
                    f"{where}: 记录 status={want_status!r} 与报告 "
                    f"status={status!r} 不一致")
        want_consumer = record.get("consumer")
        if isinstance(want_consumer, str) and want_consumer:
            declared_consumer = (
                data.get("consumer_type") or data.get("consumer")
                or data.get("engine"))
            if declared_consumer is not None and \
                    str(declared_consumer) != want_consumer:
                errors.append(
                    f"{where}: 报告 consumer={declared_consumer!r} 与记录 "
                    f"consumer={want_consumer!r} 不一致")
    else:
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return [f"{where}: 无法读取 {exc}"]
        if not _token_in(text, profile):
            errors.append(f"{where}: 报告正文未出现 profile {profile} 标记")
    return errors


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

        # ── 按 profile 的证据记录（最小单位：算子 + profile）──
        records: dict[str, dict] = {}
        platforms = op.get("platforms")
        if not isinstance(platforms, dict):
            errors.append(f"operator {name}: platforms 不是映射")
            platforms = {}
        for family, platform in platforms.items():
            if not isinstance(platform, dict):
                continue
            declared = platform.get("profiles") or []
            evidence = platform.get("evidence")
            if not isinstance(evidence, dict) or \
                    not isinstance(evidence.get("profiles"), dict):
                errors.append(
                    f"operator {name}/{family}: 缺 evidence.profiles "
                    f"按 profile 的证据记录（每个已交付 profile 必须"
                    f"有独立 level/basis）")
                continue
            fam_levels: dict[str, str] = {}
            for profile in declared:
                record = evidence["profiles"].get(profile)
                if not isinstance(record, dict):
                    errors.append(
                        f"operator {name}/{family}: profile {profile} "
                        f"缺独立证据记录")
                    continue
                rlevel = record.get("level")
                if rlevel not in VALID_EVIDENCE:
                    errors.append(
                        f"operator {name}: profile {profile} 记录 level "
                        f"必须是 {VALID_EVIDENCE}，当前 {rlevel!r}")
                    continue
                if rlevel in ("C1", "C2") and not (
                        isinstance(record.get("basis"), str)
                        and record["basis"].strip()):
                    errors.append(
                        f"operator {name}: profile {profile} {rlevel} 记录"
                        f"必须提供 basis（consumer 证据范围；仅存在 "
                        f"framework 文件不构成 C1）")
                if rlevel == "C2":
                    report = record.get("report")
                    if not isinstance(report, str) or not report.strip():
                        errors.append(
                            f"operator {name}: profile {profile} C2 记录"
                            f"必须提供 report")
                    else:
                        resolved, norm, kind = _resolve_repo_file(root, report)
                        if kind is not None:
                            errors.append(
                                f"operator {name}: profile {profile} C2 "
                                f"report {_path_error(kind, report)}")
                        else:
                            errors.extend(_report_declares_profile(
                                root, resolved, norm, profile, record,
                                where=f"operator {name}: profile {profile} "
                                      f"C2 report {norm}"))
                records[profile] = record
                fam_levels[profile] = rlevel
            family_level = evidence.get("level")
            if isinstance(family_level, str):
                # PR44 兼容的家族汇总字段：只能由 profile 记录派生。
                derived = max(fam_levels, key=lambda p: VALID_EVIDENCE.index(
                    fam_levels[p])) if fam_levels else None
                if derived is None or family_level != fam_levels[derived]:
                    errors.append(
                        f"operator {name}/{family}: evidence.level="
                        f"{family_level} 与 profile 记录最高级 "
                        f"{fam_levels.get(derived) if derived else '—'} "
                        f"不一致（家族汇总必须由记录派生）")

        if records:
            derived_top = max(
                records, key=lambda p: VALID_EVIDENCE.index(
                    records[p]["level"]))
            top = records[derived_top]["level"]
            if top != level:
                errors.append(
                    f"operator {name}: evidence={level} 与 profile 记录"
                    f"最高级 {top} 不一致（operator 顶层汇总只能由"
                    f"记录派生，不能独立填写）")
        else:
            errors.append(
                f"operator {name}: 没有任何 profile 证据记录"
                f"（platforms.<family>.evidence.profiles）")

        if level != "C2":
            continue

        # ── C2 顶层：basis + evidence_paths（统一路径解析）──
        basis = op.get("evidence_basis")
        if not isinstance(basis, str) or not basis.strip():
            errors.append(f"operator {name}: C2 必须提供 evidence_basis")
        paths = op.get("evidence_paths")
        if not isinstance(paths, list) or not paths:
            errors.append(f"operator {name}: C2 必须提供 evidence_paths")
            continue
        path_norms: set[str] = set()
        for rel in paths:
            if not isinstance(rel, str) or not rel:
                errors.append(f"operator {name}: evidence_paths 含非法项 {rel!r}")
                continue
            resolved, norm, kind = _resolve_repo_file(root, rel)
            if kind is not None:
                errors.append(
                    f"operator {name}: evidence_paths {_path_error(kind, rel)}")
                continue
            path_norms.add(norm)
        for profile, record in records.items():
            if record.get("level") == "C2" and \
                    isinstance(record.get("report"), str):
                _, norm, kind = _resolve_repo_file(root, record["report"])
                if kind is None and norm not in path_norms:
                    errors.append(
                        f"operator {name}: profile {profile} 的 C2 report "
                        f"{norm} 未列入 evidence_paths（报告路径必须与"
                        f"记录同源）")


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


def _line_json_refs(root: Path, line: str, base: Path) -> list[str]:
    """成功行通过 Markdown 链接引用的仓库内 `.json` 证据（规范相对路径）。

    只收仓库内、当前存在的 `.json` 文件；外链、绝对路径、越界与
    不存在的链接目标跳过（路径严格化由 catalog/锁侧统一解析负责，
    这里只取可核验的结构化证据做 profile 范围一致性检查）。
    """
    refs: list[str] = []
    root_res = root.resolve()
    for target in _MD_LINK.findall(line):
        if "://" in target or target.startswith(("#", "mailto:")):
            continue
        p = Path(target)
        if p.is_absolute():
            continue
        try:
            resolved = (base / p).resolve()
        except OSError:
            continue
        try:
            norm = resolved.relative_to(root_res).as_posix()
        except ValueError:
            continue
        if norm.endswith(".json") and resolved.is_file() and norm not in refs:
            refs.append(norm)
    return refs


def _check_report_e2e_profiles(root: Path, errors: list[str]) -> None:
    """缺口 1：聚合报告里的 FlagOS E2E 结果必须标明具体 profile。

    只看 `ops/*/REPORT.md`（一页看全的聚合页）。同一行里既宣称通过
    （☑/PASS/通过…）又报 FlagOS E2E 结果的，必须出现已知 profile 的
    **完整 token**（边界匹配，`demo-10` 不算 `demo-1`）；纯叙述、
    纯数字、标 ✗ 的行不在此列。已知 profile 集为空时，成功声明无法
    核验，直接报错而不是跳过。

    行内引用的结构化 JSON 证据还要与行内 profile 范围一致：JSON
    声明的 `profile` 必须以完整 token 出现在该行（成功行不能拿
    别的环境的报告给自己背书）；JSON 未声明 profile 时不在此层
    臆断范围，登记为已知缺口（见 docs/operator-catalog.md）。
    """
    ops_dir = root / "ops"
    if not ops_dir.is_dir():
        return
    profiles = _known_profiles(root)
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
            if not profiles:
                errors.append(
                    f"{rel}:{lineno}: 存在宣称通过的 FlagOS E2E 结果行，"
                    f"但已知 profile 集为空（configs/devices/*.yaml 与"
                    f"环境锁 name 均无），无法核验")
                continue
            line_profiles = [p for p in profiles if _token_in(line, p)]
            if not line_profiles:
                errors.append(
                    f"{rel}:{lineno}: FlagOS E2E 结果行未标明具体 profile/"
                    f"环境（需含 configs/devices 已知 profile 之一，"
                    f"如 {sorted(profiles)[0]}；子串近似名不算）")
                continue
            for norm in _line_json_refs(root, line, report.parent):
                resolved = root / norm
                try:
                    data = json.loads(resolved.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError, UnicodeDecodeError) \
                        as exc:
                    errors.append(
                        f"{rel}:{lineno}: 成功行所引证据 {norm} "
                        f"JSON 解析失败 {exc}")
                    continue
                if not isinstance(data, dict):
                    errors.append(
                        f"{rel}:{lineno}: 成功行所引证据 {norm} "
                        f"JSON 顶层不是映射")
                    continue
                declared = data.get("profile")
                if isinstance(declared, str) and declared and \
                        not _token_in(line, declared):
                    errors.append(
                        f"{rel}:{lineno}: 成功行所引证据 {norm} 声明 "
                        f"profile={declared} 与行内 profile 范围不一致"
                        f"（行内须标明该证据的 profile）")


def _lock_c2_support_ok(
        root: Path, norm: str, lock_name: str, record: dict,
        record_report_norm: str | None,
        path_norms: set[str]) -> tuple[bool, str]:
    """锁 C2 声明的支撑报告是否合格；不合格时给出原因。

    合格 = **有效且通过** + 与本 profile 的 C2 记录同源关联：

    - JSON：可解析、顶层是映射、声明本锁 `profile`、`status`/`ok`
      为通过态（与 catalog 侧报告通过状态检查同口径，挡
      `status=FAIL` / 缺 status 的假阳性背书）、记录声明
      `consumer` 时报告 consumer 不冲突；
    - 同源关联：路径等于该 profile C2 记录的 `report`、列入算子
      `evidence_paths`、或报告 consumer 与记录 `consumer` 一致
      （report/consumer 关联，任一即可）；
    - 非 JSON（md 等）：正文含本锁 profile 完整 token + 同源关联
      （通过状态无法结构化核验，只能由 catalog 侧记录报告校验兜底）。
    """
    resolved = root / norm
    associated = (
        (record_report_norm is not None and norm == record_report_norm)
        or norm in path_norms)
    want_consumer = record.get("consumer") if isinstance(record, dict) \
        else None
    if norm.endswith(".json"):
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            return False, f"JSON 解析失败 {exc}"
        if not isinstance(data, dict):
            return False, "JSON 顶层不是映射"
        declared = data.get("profile")
        if not isinstance(declared, str) or not declared:
            return False, "报告缺少 profile 字段"
        if declared != lock_name:
            return False, (
                f"报告声明 profile={declared} 与锁 profile={lock_name}"
                f" 不一致")
        status = data.get("status")
        if isinstance(status, str):
            if status.upper() not in ("PASS", "OK"):
                return False, f"报告验证状态非通过: {status!r}"
        elif data.get("ok") is True:
            pass
        else:
            return False, "报告缺少 status/ok 验证状态字段"
        got_consumer = (
            data.get("consumer_type") or data.get("consumer")
            or data.get("engine"))
        if isinstance(want_consumer, str) and want_consumer:
            if got_consumer is not None and \
                    str(got_consumer) != want_consumer:
                return False, (
                    f"报告 consumer={got_consumer!r} 与记录 "
                    f"consumer={want_consumer!r} 不一致")
            if str(got_consumer) == want_consumer:
                associated = True
        if not associated:
            return False, (
                f"与 profile {lock_name} 的 C2 记录 report/consumer "
                f"无关联")
        return True, ""
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return False, f"无法读取 {exc}"
    if not _token_in(text, lock_name):
        return False, f"正文未出现 profile {lock_name} 标记"
    if not associated:
        return False, (
            f"与 profile {lock_name} 的 C2 记录 report/consumer 无关联")
    return True, ""


def _check_lock_crossref(root: Path, catalog: dict,
                         locks: list[tuple[Path, dict]],
                         errors: list[str]) -> None:
    """缺口 2：环境锁 evidence/scope 与 catalog 按 profile 记录的交叉引用。

    - `evidence[].path` 必须经统一解析（仓库内、当前存在的文件，
      禁外链/占位/绝对路径/越界/目录）；结构化 JSON 证据必须可解析
      并声明与锁 `name` 一致的 `profile`（P800 报告不能给 PPU
      背书）——损坏、缺 profile、跨 profile 的 JSON **单独报错**，
      不再无声跳过，且不能成为任何支撑；
      指向 `ops/<dir>/...` 时按**规范化路径**做算子归属核对；
    - `scope` 行里的 `ops/<dir>` 必须是 catalog 登记的算子；
    - `scope` 行声明 `ops/<dir> C[012]` 时，锁 `name` 必须是
      `configs/devices` 已知 profile，且声明必须等于**该 profile**
      的记录等级（而非 operator 全局汇总），本锁 `evidence` 中要有
      该算子的合格报告支撑；C2 声明的支撑报告还必须**有效且通过**，
      并与该 profile 的 C2 记录 `report`/`consumer` 同源关联。
    """
    ops_by_dir: dict[str, dict] = {}
    for op in catalog.get("operators", []):
        name = op.get("name", "?")
        ops_by_dir[op.get("directory", name)] = op
    profile_family = _profile_family_map(catalog)
    device_profiles = _device_profiles(root)

    for lock_path, lock in locks:
        rel = lock_path.relative_to(root)
        lock_name = lock.get("name")
        referenced: set[str] = set()
        supporting: dict[str, list[str]] = {}

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
            if path.startswith(("http://", "https://")) or "://" in path:
                errors.append(f"{rel}: evidence 必须是仓库内路径，"
                              f"不允许外链 {path}")
                continue
            resolved, norm, kind = _resolve_repo_file(root, path)
            if kind is not None:
                if kind == "不存在":
                    errors.append(
                        f"{rel}: evidence 指向不存在的报告 {path}")
                else:
                    errors.append(
                        f"{rel}: evidence {_path_error(kind, path)}")
                continue
            support_ok = True
            if norm.endswith(".json"):
                # 结构化证据必须有效并声明本锁 profile：损坏/缺
                # profile/跨 profile 均单独报错，且不能成为支撑。
                try:
                    data = json.loads(
                        resolved.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError, UnicodeDecodeError) \
                        as exc:
                    errors.append(
                        f"{rel}: evidence {norm} JSON 解析失败 {exc}"
                        f"（损坏文件不能作为支撑）")
                    support_ok = False
                    data = None
                if data is not None and not isinstance(data, dict):
                    errors.append(
                        f"{rel}: evidence {norm} JSON 顶层不是映射"
                        f"（无效文件不能作为支撑）")
                    support_ok = False
                elif isinstance(data, dict):
                    declared = data.get("profile")
                    if not isinstance(declared, str) or not declared:
                        errors.append(
                            f"{rel}: evidence {norm} 缺少 profile 字段"
                            f"（结构化证据必须声明本锁 profile"
                            f"{' ' + lock_name if isinstance(lock_name, str) else ''}"
                            f"，无效文件不能作为支撑）")
                        support_ok = False
                    elif isinstance(lock_name, str) and \
                            declared != lock_name:
                        errors.append(
                            f"{rel}: evidence {norm} 声明 "
                            f"profile={declared} 与锁 profile="
                            f"{lock_name} 不一致（不能跨环境背书）")
                        support_ok = False
            m = _EVIDENCE_OP_PATH.match(norm)
            if not m:
                continue
            op_name = m.group("op")
            if op_name in ops_by_dir:
                referenced.add(op_name)
                if support_ok:
                    supporting.setdefault(op_name, []).append(norm)
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
            if not isinstance(lock_name, str) or \
                    lock_name not in device_profiles:
                errors.append(
                    f"{rel}: 锁 name={lock_name!r} 不是 configs/devices "
                    f"已知 profile，无法核对 ops/{op_name} 的 scope 等级"
                    f"声明（证据必须绑定具体 profile）")
                continue
            family = profile_family.get(lock_name)
            record = None
            if family is not None:
                platform = ops_by_dir[op_name].get("platforms")
                if isinstance(platform, dict) and \
                        isinstance(platform.get(family), dict):
                    evidence = platform[family].get("evidence")
                    if isinstance(evidence, dict) and \
                            isinstance(evidence.get("profiles"), dict):
                        record = evidence["profiles"].get(lock_name)
            if not isinstance(record, dict) or "level" not in record:
                errors.append(
                    f"{rel}: catalog 在 profile {lock_name} 上没有 "
                    f"ops/{op_name} 的证据记录，scope 声明无据")
                continue
            actual = record["level"]
            if actual != claimed:
                errors.append(
                    f"{rel}: scope 声明 ops/{op_name} {claimed} 与 "
                    f"profile {lock_name} 记录 {actual} 不一致"
                    f"（比较对象是本锁 profile 的记录，不是 operator "
                    f"全局汇总）")
            if op_name not in supporting:
                errors.append(
                    f"{rel}: scope 的 ops/{op_name} {claimed} 声明在本锁 "
                    f"evidence 中没有对应报告支撑")
            elif claimed == "C2":
                record_report_norm = None
                want_report = record.get("report")
                if isinstance(want_report, str) and want_report.strip():
                    _, record_report_norm, rkind = _resolve_repo_file(
                        root, want_report)
                    if rkind is not None:
                        record_report_norm = None
                path_norms: set[str] = set()
                op_paths = ops_by_dir[op_name].get("evidence_paths")
                if isinstance(op_paths, list):
                    for p in op_paths:
                        if isinstance(p, str) and p:
                            _, pnorm, pkind = _resolve_repo_file(root, p)
                            if pkind is None:
                                path_norms.add(pnorm)
                failures: list[str] = []
                qualified = False
                for norm in supporting[op_name]:
                    ok, why = _lock_c2_support_ok(
                        root, norm, lock_name, record,
                        record_report_norm, path_norms)
                    if ok:
                        qualified = True
                        break
                    failures.append(f"{norm}: {why}")
                if not qualified:
                    if failures:
                        errors.append(
                            f"{rel}: scope 的 ops/{op_name} C2 声明的"
                            f"支撑报告不合格（" + "；".join(failures) + "）")
                    else:
                        errors.append(
                            f"{rel}: scope 的 ops/{op_name} C2 声明在本锁 "
                            f"evidence 中没有对应报告支撑"
                            f"（需声明 profile {lock_name} 且与该 profile "
                            f"的 C2 记录 report/consumer 同源）")


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
