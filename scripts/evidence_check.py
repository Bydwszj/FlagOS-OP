#!/usr/bin/env python3
"""C1/C2 证据门禁（issue #41）。

逐 profile 记录的解析 / 路径归一化 / 通过判定统一走
``scripts/evidence_schema.py``（catalog、本脚本、目录生成共用，见该模块
docstring 的三轴语义：evidence level / 实验判定 verdict / 维护者审阅状态
review——``complete-pending-review`` 不是全量 PASS）。

检查六件事（任何一条失败即非零退出）：

1. `ops/catalog.yaml` 每个算子必须声明 `evidence: C0|C1|C2`，且该汇总
   必须等于各 `platforms.<family>.evidence[]` 逐 profile 记录的最高等级
   ——**算子 + profile 是证据记录的最小单位**，汇总只能由记录派生；
   `platforms.<family>.profiles` 里的每个 profile 都要有独立记录
   （单个 family 下多个 profile 不共享未区分的等级）；C1/C2 记录
   必须给 `basis`（consumer 证据范围）。`framework_level.py` 存在
   只是结构前提，不单独构成 C1（详见 docs/operator-catalog.md）。
2. `C2` 必须给出 `evidence_basis` 与 `evidence_paths`；所有证据路径
   经统一解析函数核验：禁外链、禁绝对路径、`resolve()` 后必须仍在
   仓库根内、必须是文件（目录、`..` 越界、仓库外 symlink 均拒绝）。
   C2 记录的 `report` 必须列入 `evidence_paths`；结构化 JSON 报告
   必须声明相同 `profile`（或经**核验过的派生更正**）并给出通过的
   实验判定，不能拿别的环境的报告给本 profile 背书；记录 `status`
   与报告审阅状态冲突时报错。
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
   `configs/devices` 已知 profile。每个结构化 JSON 证据在**入口**就
   走与 catalog `record.report` 同口径的校验：可解析、声明本锁
   `profile`（损坏/缺 profile/跨 profile 的 JSON 单独报「路径 +
   原因」，不能静默跳过，也不能成为支撑；唯一例外是记录声明且
   **核验通过的派生更正**——源 SHA256 / profile_field_original /
   corrections / commit 全对）、`status` 为 `PASS`/`OK` 或 `ok: true`
   （`FAIL`/缺判定/仅采集完成一律不成立）、记录声明 consumer/engine
   时与报告一致；md 等文本支撑在 `scope` **等级声明**核验时要求正文
   含本锁 profile 完整 token。锁 evidence 不必逐条指向该 profile 的
   `record.report`，但 C2 支撑必须同源；
   `scope` 行的 `ops/<算子> C[012]` 必须等于
   **该锁 profile** 的记录等级（而不是 operator 全局汇总），C2
   声明的支撑报告必须**有效且实验判定通过**（JSON 可解析、声明本锁
   profile、verdict=pass；`FAIL`/缺判定/仅采集完成一律不成立），
   并与该 profile 的 C2 记录
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

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evidence_schema import (  # noqa: E402  （脚本/包两种运行方式）
    C2_LEVELS,
    CONSUMER_LEVELS,
    REVIEW_PASS,
    REVIEW_PENDING,
    REVIEW_UNDECLARED,
    TOP_LEVELS,
    VERDICT_MISSING,
    VERDICT_PASS,
    consumer_problem,
    effective_profile,
    family_records,
    is_valid_level,
    load_json_object,
    load_redecide,
    operator_top_level,
    path_error,
    record_for,
    record_review_state,
    records_map,
    report_consumer,
    report_review_state,
    report_verdict,
    resolve_repo_file,
    review_word_known,
    top_level_from_records,
    verdict_problem,
)

VALID_EVIDENCE = TOP_LEVELS
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

# 统一路径解析（同一实现，勿在本文件另写一套）
_resolve_repo_file = resolve_repo_file
_path_error = path_error


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


def _report_declares_profile(
        root: Path, resolved: Path, norm: str, profile: str,
        record: dict, where: str) -> list[str]:
    """结构化证据报告必须落在本 profile 范围内，且实验判定通过。

    三轴分开（PR #46 意见 3）：

    - **范围**：JSON 用 `profile` 字段；不一致时若记录声明了
      `redecide`，走核验过的派生更正（源 SHA256 / 原始 profile /
      corrections / commit 全对才采用），否则报错。文本报告（md 等）
      要求正文出现该 profile 的完整 token。
    - **实验判定**：`verdict_problem()`（与环境锁侧同一函数）——
      `FAIL`、缺判定、仅采集完成一律不成立。
    - **审阅状态**：记录 `status` 与报告 acceptance 字段冲突时报错；
      `complete-pending-review` 不等于全量 PASS，但其中已通过的
      实验结果不丢（verdict 仍为通过）。
    consumer 只做一致性核对，不做自然语言匹配。
    """
    errors: list[str] = []
    if norm.endswith(".json"):
        data, err = load_json_object(resolved)
        if err is not None:
            return [f"{where}: {err}"]
        effective, err = effective_profile(
            root, resolved, norm, data, want_profile=profile,
            redecide_rel=record.get("redecide"))
        if err is not None:
            errors.append(f"{where}: {err}")
        problem = verdict_problem(data)
        if problem is not None:
            errors.append(f"{where}: {problem}")
        record_state = record_review_state(record)
        if not review_word_known(record):
            errors.append(
                f"{where}: 记录 status={record.get('status')!r} 不是已知的"
                f"审阅状态词（pass / complete-pending-review …）")
        report_state = report_review_state(data)
        if record_state != REVIEW_UNDECLARED and \
                report_state != REVIEW_UNDECLARED and \
                record_state != report_state:
            errors.append(
                f"{where}: 记录 status={record.get('status')!r}（审阅状态 "
                f"{record_state}）与报告审阅状态 {report_state} 不一致")
        if record_state == REVIEW_PASS and report_state == REVIEW_PENDING:
            errors.append(
                f"{where}: 记录自称已审阅通过，报告仍是 complete-pending-"
                f"review（待审阅不能写成全量 PASS）")
        want_consumer = record.get("consumer")
        problem = consumer_problem(data, want_consumer)
        if problem is not None:
            errors.append(f"{where}: {problem}")
        _ = effective  # 更正后的范围已用于一致性核验
    else:
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return [f"{where}: 无法读取 {exc}"]
        if not _token_in(text, profile):
            errors.append(f"{where}: 报告正文未出现 profile {profile} 标记")
        if not review_word_known(record):
            errors.append(
                f"{where}: 记录 status={record.get('status')!r} 不是已知的"
                f"审阅状态词（pass / complete-pending-review …）")
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
                errors.append(
                    f"operator {name}/{family}: platforms 条目不是映射")
                continue
            where = f"operator {name}/{family}"
            fam_records, fam_errors = family_records(platform, where=where)
            errors.extend(fam_errors)
            for record in fam_records:
                profile = record["profile"]
                rlevel = record["level"]
                if rlevel in CONSUMER_LEVELS and not (
                        isinstance(record.get("basis"), str)
                        and record["basis"].strip()):
                    errors.append(
                        f"operator {name}: profile {profile} {rlevel} 记录"
                        f"必须提供 basis（consumer 证据范围；仅存在 "
                        f"framework 文件不构成 C1）")
                if rlevel in C2_LEVELS:
                    report = record.get("report")
                    if not isinstance(report, str) or not report.strip():
                        errors.append(
                            f"operator {name}: profile {profile} {rlevel} "
                            f"记录必须提供 report")
                    else:
                        resolved, norm, kind = _resolve_repo_file(root, report)
                        if kind is not None:
                            errors.append(
                                f"operator {name}: profile {profile} "
                                f"{rlevel} report {_path_error(kind, report)}")
                        else:
                            errors.extend(_report_declares_profile(
                                root, resolved, norm, profile, record,
                                where=f"operator {name}: profile {profile} "
                                      f"{rlevel} report {norm}"))
                if isinstance(record.get("redecide"), str) and \
                        record["redecide"].strip():
                    # 派生更正必须自洽（源 SHA256 / 原始 profile /
                    # corrections / commit 全对），否则不能采用更正范围
                    _, rerr = load_redecide(
                        root, record["redecide"], expect_now=profile)
                    if rerr is not None:
                        errors.append(
                            f"operator {name}: profile {profile} 的 redecide "
                            f"{rerr}")
                records[profile] = record

        if records:
            top = operator_top_level(list(records.values()))
            if top != level:
                errors.append(
                    f"operator {name}: evidence={level} 与 profile 记录"
                    f"最高级 {top} 不一致（operator 顶层汇总只能由"
                    f"记录派生，不能独立填写）")
        else:
            errors.append(
                f"operator {name}: 没有任何 profile 证据记录"
                f"（platforms.<family>.evidence 逐 profile 记录列表）")

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
            if record.get("level") in C2_LEVELS and \
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


def _lock_json_evidence_problems(
        root: Path, resolved: Path, norm: str, lock_name, record,
) -> list[str]:
    """锁 `evidence[].path` 里结构化 JSON 的校验问题（空 = 合格）。

    与 catalog `record.report` **同口径**：判定函数全部来自
    `scripts/evidence_schema.py`，不另起一套——

    - 可解析：`load_json_object()`（损坏 / 顶层非映射单独报错）；
    - 范围：`effective_profile()` —— `profile` 与锁 `name` 一致，或
      记录声明且**核验通过**的派生更正（redecide）；
    - 实验判定：`verdict_problem()` —— `FAIL`、缺 `status`/`ok`、仅
      采集完成（`collect-only`）一律不成立；
    - consumer：记录声明 `consumer` 时，`consumer_problem()` 校验报告
      的 `consumer_type`/`consumer`/`engine` 一致。

    任一不合格都返回「路径 + 原因」级别的问题串，调用方据此报错并把该
    文件排除出支撑集合（不得成为任何 scope 等级声明的支撑）。
    """
    data, err = load_json_object(resolved)
    if err is not None:
        return [f"{err}（损坏文件不能作为支撑）"]
    problems: list[str] = []
    redecide_rel = record.get("redecide") if isinstance(record, dict) \
        else None
    declared = data.get("profile")
    if isinstance(declared, str) and declared and (
            not isinstance(lock_name, str) or declared == lock_name):
        pass
    elif redecide_rel:
        _, rerr = effective_profile(
            root, resolved, norm, data, want_profile=lock_name,
            redecide_rel=redecide_rel)
        if rerr is not None:
            problems.append(f"派生更正核验失败：{rerr}（不能跨环境背书）")
    elif not isinstance(declared, str) or not declared:
        problems.append(
            f"缺少 profile 字段（结构化证据必须声明本锁 profile"
            f"{' ' + lock_name if isinstance(lock_name, str) else ''}"
            f"，无效文件不能作为支撑）")
    else:
        problems.append(
            f"声明 profile={declared} 与锁 profile={lock_name} 不一致"
            f"（不能跨环境背书）")
    problem = verdict_problem(data)
    if problem is not None:
        problems.append(f"{problem}（不能作为支撑）")
    problem = consumer_problem(
        data, record.get("consumer") if isinstance(record, dict) else None)
    if problem is not None:
        problems.append(f"{problem}（不能作为支撑）")
    return problems


def _lock_c2_support_ok(
        root: Path, norm: str, lock_name: str, record: dict,
        record_report_norm: str | None,
        path_norms: set[str]) -> tuple[bool, str]:
    """锁 C2 声明的支撑报告是否合格；不合格时给出原因。

    合格 = **有效且通过** + 与本 profile 的 C2 记录同源关联：

    - 范围：JSON `profile` 字段与锁 `name` 一致；不一致时若记录
      声明 `redecide`，走核验过的派生更正（不一致且无/坏 redecide
      一律不合格，不能跨环境背书）；
    - 判定：`verdict_problem()` —— `FAIL`、缺判定、仅采集完成一律
      不成立（与 catalog 侧同一函数，挡 `status=FAIL` / 缺 status
      的假阳性背书）；记录声明 `consumer` 时报告 consumer 不冲突；
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
    redecide_rel = record.get("redecide") if isinstance(record, dict) \
        else None
    if norm.endswith(".json"):
        data, err = load_json_object(resolved)
        if err is not None:
            return False, err
        _, err = effective_profile(
            root, resolved, norm, data, want_profile=lock_name,
            redecide_rel=redecide_rel, want_label="锁")
        if err is not None:
            return False, err
        problem = verdict_problem(data)
        if problem is not None:
            return False, problem
        problem = consumer_problem(data, want_consumer)
        if problem is not None:
            return False, problem
        if isinstance(want_consumer, str) and want_consumer and \
                report_consumer(data) == want_consumer:
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
      禁外链/占位/绝对路径/越界/目录）；结构化 JSON 证据在**入口**就
      走与 catalog `record.report` 同口径的结构化校验
      （`_lock_json_evidence_problems`：可解析、`profile` 一致含
      redecide、`verdict_problem` 判定通过、记录声明 consumer/engine
      时一致）——损坏、缺 profile、跨 profile、`FAIL`、缺 `status`、
      `collect-only`、consumer 不一致的 JSON 一律**单独报「路径 +
      原因」**，不再无声跳过，且**不得成为任何 scope 等级声明的
      支撑**；指向 `ops/<dir>/...` 时按**规范化路径**做算子归属核对；
    - 非 JSON 支撑（md 等）：`scope` **等级声明**核验时要求正文含本锁
      `profile` 完整 token（通过状态无法结构化核验）。锁 evidence
      **不必**逐条指向该 profile 的 `record.report`；但 **C2 支撑必须
      同源**（记录 `report` / 列入 `evidence_paths` / 报告 consumer 与
      记录 consumer 一致，任一即可）；
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
    device_profiles = _device_profiles(root)

    for lock_path, lock in locks:
        rel = lock_path.relative_to(root)
        lock_name = lock.get("name")
        referenced: set[str] = set()
        supporting: dict[str, list[str]] = {}
        rejected: dict[str, list[str]] = {}

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
            problems: list[str] = []
            m = _EVIDENCE_OP_PATH.match(norm)
            op_name = m.group("op") if m else None
            if norm.endswith(".json"):
                # 结构化证据在入口就做与 catalog record.report 同口径的
                # 校验（可解析 / profile 一致含 redecide / 判定通过 /
                # consumer 一致）；任一不过即报「路径 + 原因」，且不得
                # 成为任何 scope 等级声明的支撑。
                record = record_for(catalog, op_name, lock_name) \
                    if op_name is not None and \
                    isinstance(lock_name, str) else None
                problems = _lock_json_evidence_problems(
                    root, resolved, norm, lock_name, record)
                if problems:
                    support_ok = False
                    for problem in problems:
                        errors.append(f"{rel}: evidence {norm} {problem}")
            if m is None:
                continue
            op_name = m.group("op")
            if op_name in ops_by_dir:
                referenced.add(op_name)
                if support_ok:
                    supporting.setdefault(op_name, []).append(norm)
                elif problems:
                    rejected.setdefault(op_name, []).append(
                        f"{norm}: {'；'.join(problems)}")
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
            record = record_for(catalog, op_name, lock_name)
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
                msg = (
                    f"{rel}: scope 的 ops/{op_name} {claimed} 声明在本锁 "
                    f"evidence 中没有对应报告支撑")
                reasons = rejected.get(op_name)
                if reasons:
                    # 入口校验已拦下这些文件（路径 + 原因），它们不得成为支撑
                    msg += f"（支撑报告不合格：{'；'.join(reasons)}）"
                elif claimed == "C2":
                    msg += (
                        f"（需声明 profile {lock_name} 且与该 profile "
                        f"的 C2 记录 report/consumer 同源）")
                errors.append(msg)
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
            else:
                # C0/C1 等级声明：结构化 JSON 已在入口做过同口径校验，
                # md 等文本支撑必须含本锁 profile 完整 token（正文范围
                # 核验；通过状态无法从文本结构化判定）。
                qualified = False
                md_failures: list[str] = []
                for norm in supporting[op_name]:
                    if norm.endswith(".json"):
                        qualified = True
                        break
                    try:
                        text = (root / norm).read_text(
                            encoding="utf-8", errors="replace")
                    except OSError as exc:
                        md_failures.append(f"{norm}: 无法读取 {exc}")
                        continue
                    if _token_in(text, lock_name):
                        qualified = True
                        break
                    md_failures.append(
                        f"{norm}: 正文未出现 profile {lock_name} 标记")
                if not qualified:
                    errors.append(
                        f"{rel}: scope 的 ops/{op_name} {claimed} 声明的"
                        f"支撑材料不合格（" + "；".join(md_failures) + "）")


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
