#!/usr/bin/env python3
"""Validate ops/catalog.yaml and generate ops/README.md.

The catalog is the scalable inventory source for a growing operator/platform
matrix. The generated README is checked in for direct GitHub browsing.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "ops" / "catalog.yaml"
OUTPUT_PATH = ROOT / "ops" / "README.md"
EXPECTED_SCHEMA = 1
VALID_PLATFORM_STATUSES = {"partial", "planned"}
VALID_GROUP_KINDS = {"core", "foundation"}


def load_catalog(path: Path = CATALOG_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def _require_mapping(value, name: str) -> dict:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return value


def validate_catalog(catalog: dict, root: Path = ROOT) -> None:
    """Validate schema, filesystem links, and incremental coverage semantics."""
    if catalog.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"ops/catalog.yaml schema must be {EXPECTED_SCHEMA}")

    families = _require_mapping(catalog.get("platform_families"), "platform_families")
    groups = _require_mapping(catalog.get("operator_groups"), "operator_groups")
    operators = catalog.get("operators")
    if not isinstance(operators, list) or not operators:
        raise ValueError("operators must be a non-empty list")

    expected_count = catalog.get("target_scope", {}).get("vendor_family_count")
    if expected_count != len(families):
        raise ValueError(
            "target_scope.vendor_family_count must equal "
            f"{len(families)}, got {expected_count!r}"
        )

    ranks: set[int] = set()
    for group, meta in groups.items():
        if meta.get("kind") not in VALID_GROUP_KINDS:
            raise ValueError(f"operator group {group} has unknown kind")
        rank = meta.get("rank")
        if not isinstance(rank, int):
            raise ValueError(f"operator group {group} lacks integer rank")
        if rank in ranks:
            raise ValueError("operator group ranks must be unique")
        ranks.add(rank)
        for field in ("title", "description"):
            if not isinstance(meta.get(field), str) or not meta[field]:
                raise ValueError(f"operator group {group} lacks {field}")

    family_counts: Counter[str] = Counter()
    for family, meta in families.items():
        if meta.get("status") not in VALID_PLATFORM_STATUSES:
            raise ValueError(
                f"unknown status for platform {family}: {meta.get('status')!r}"
            )
        if not isinstance(meta.get("display_name"), str) or not meta["display_name"]:
            raise ValueError(f"platform {family} lacks display_name")
        profiles = meta.get("current_profiles")
        if not isinstance(profiles, list):
            raise ValueError(f"platform {family} current_profiles must be a list")
        for profile in profiles:
            if not (root / "configs" / "devices" / f"{profile}.yaml").is_file():
                raise ValueError(
                    f"platform {family} profile has no configs/devices/{profile}.yaml"
                )
        if meta["status"] == "planned" and profiles:
            raise ValueError(f"planned platform {family} cannot declare profiles")

    names: set[str] = set()
    directories: set[str] = set()
    used_groups: set[str] = set()
    for op in operators:
        name = op.get("name")
        directory = op.get("directory")
        group = op.get("group")
        if not isinstance(name, str) or not name:
            raise ValueError("every operator needs a name")
        if name in names:
            raise ValueError(f"duplicate operator name: {name}")
        if not isinstance(directory, str) or not directory:
            raise ValueError(f"operator {name} lacks directory")
        if directory in directories:
            raise ValueError(f"duplicate operator directory: {directory}")
        if group not in groups:
            raise ValueError(f"operator {name} has unknown group {group!r}")
        for field in ("purpose", "key_result"):
            if not isinstance(op.get(field), str) or not op[field]:
                raise ValueError(f"operator {name} lacks {field}")
        if not (root / "ops" / directory).is_dir():
            raise ValueError(f"operator directory does not exist: {directory}")

        platforms = _require_mapping(op.get("platforms"), f"{name}.platforms")
        if groups[group]["kind"] == "foundation":
            routes = (
                [op["route"]]
                if op.get("route")
                else [platform.get("route") for platform in platforms.values()]
            )
            if not routes or any(not route for route in routes):
                raise ValueError(
                    f"foundation operator {name} lacks a top-level route "
                    "or a route for every platform"
                )
        for family, platform in platforms.items():
            if family not in families:
                raise ValueError(f"operator {name} has unknown family {family!r}")
            if families[family]["status"] == "planned":
                raise ValueError(
                    f"operator {name} declares {family}, but that family is planned"
                )
            profiles = platform.get("profiles")
            if not isinstance(profiles, list) or not profiles:
                raise ValueError(f"operator {name}/{family} lacks profiles")
            family_profiles = set(families[family]["current_profiles"])
            unknown_profiles = sorted(set(profiles) - family_profiles)
            if unknown_profiles:
                raise ValueError(
                    f"operator {name}/{family} has profiles outside its family: "
                    f"{unknown_profiles}"
                )
            family_counts[family] += 1
        if len(platforms) > 1:
            operator_dir = root / "ops" / directory
            if not (operator_dir / "PLATFORM.md").is_file():
                raise ValueError(f"multi-platform operator {name} lacks PLATFORM.md")
            if not (operator_dir / "MERGE.md").is_file():
                raise ValueError(f"multi-platform operator {name} lacks MERGE.md")
        names.add(name)
        directories.add(directory)
        used_groups.add(group)

    unused_groups = sorted(set(groups) - used_groups)
    if unused_groups:
        raise ValueError(f"operator groups have no operators: {unused_groups}")
    for family, meta in families.items():
        if meta["status"] == "partial" and family_counts[family] == 0:
            raise ValueError(f"partial platform {family} has no delivered operators")

    for contract_name, contract in catalog.get("registration_contracts", {}).items():
        for field in (
            "title", "scope", "canonical_key", "compatibility_key"
        ):
            if not isinstance(contract.get(field), str) or not contract[field]:
                raise ValueError(f"registration contract {contract_name} lacks {field}")
        if not isinstance(contract.get("registerable_count"), int):
            raise ValueError(
                f"registration contract {contract_name} lacks registerable_count"
            )

    actual_directories = {
        path.name for path in (root / "ops").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    missing = sorted(actual_directories - directories)
    if missing:
        raise ValueError(f"operator directories missing from catalog: {missing}")


def _operator_link(op: dict) -> str:
    return f"[{op['name']}]({op['directory']}/)"


def _profile_level(op: dict, family: str, profile: str) -> str | None:
    """算子在指定 family/profile 上的证据等级（无记录时 None）。"""
    platform = (op.get("platforms") or {}).get(family)
    if not isinstance(platform, dict):
        return None
    evidence = platform.get("evidence")
    if not isinstance(evidence, dict):
        return None
    profiles = evidence.get("profiles")
    if not isinstance(profiles, dict):
        return None
    record = profiles.get(profile)
    if isinstance(record, dict) and record.get("level") in ("C0", "C1", "C2"):
        return record["level"]
    return None


def _profile_labels(op: dict, family: str, profiles: list) -> list[str]:
    labels = []
    for profile in profiles:
        level = _profile_level(op, family, profile)
        labels.append(f"{profile} {level}" if level else str(profile))
    return labels


def _evidence_cell(op: dict) -> str:
    """证据列：顶层汇总 + 按 profile 的等级明细（按 profile 展示）。"""
    top = op.get("evidence") or "—"
    by_level: dict[str, list[str]] = {}
    ordered: list[tuple[str, str]] = []
    for family, platform in (op.get("platforms") or {}).items():
        if not isinstance(platform, dict):
            continue
        for profile in platform.get("profiles") or []:
            level = _profile_level(op, family, profile)
            if level is None:
                continue
            ordered.append((level, profile))
            by_level.setdefault(level, []).append(profile)
    if not ordered:
        return top
    if set(by_level) == {top}:
        return f"{top}（{' · '.join(p for _, p in ordered)}）"
    segs = [
        f"{level}（{' · '.join(by_level[level])}）"
        for level in sorted(
            by_level, key=lambda lv: -("C0", "C1", "C2").index(lv))
    ]
    return " / ".join(segs)


def _platform_summary(catalog: dict) -> list[str]:
    counts = Counter(
        family
        for op in catalog["operators"]
        for family in op["platforms"]
    )
    lines = [
        "| 目标平台 | 当前状态 | 已登记算子 | 当前 profile / 型号 |",
        "|---|---|---:|---|",
    ]
    for family, meta in catalog["platform_families"].items():
        profiles = [f"`{item}`" for item in meta.get("current_profiles", [])]
        variants = meta.get("known_variants", [])
        if variants:
            profiles.append("计划覆盖：" + " / ".join(variants))
        if not profiles:
            profiles = ["待首个交付算子确定"]
        status = "部分交付" if meta["status"] == "partial" else "规划中"
        lines.append(
            f"| {meta['display_name']}（`{family}`） | {status} | "
            f"{counts[family]} | {' · '.join(profiles)} |"
        )
    return lines


def _core_rows(catalog: dict) -> list[str]:
    families = catalog["platform_families"]
    rows = []
    for op in catalog["operators"]:
        if catalog["operator_groups"][op["group"]]["kind"] != "core":
            continue
        platforms = " · ".join(
            f"{families[family]['display_name']}"
            f"（{' / '.join(_profile_labels(op, family, platform['profiles']))}）"
            for family, platform in op["platforms"].items()
        )
        rows.append(
            f"| {_operator_link(op)} | {op['purpose']} | {platforms} | "
            f"{_evidence_cell(op)} | {op['key_result']} |"
        )
    return rows


def _foundation_rows(catalog: dict, group: str) -> list[str]:
    rows = []
    for op in catalog["operators"]:
        if op["group"] != group:
            continue
        routes = []
        for platform in op["platforms"].values():
            route = platform.get("route") or op.get("route")
            if route and route not in routes:
                routes.append(route)
        route = " / ".join(routes) or "—"
        rows.append(
            f"| {_operator_link(op)} | {op['purpose']} | {route} | "
            f"{_evidence_cell(op)} | {op['key_result']} |"
        )
    return rows


def render_index(catalog: dict) -> str:
    operators = catalog["operators"]
    groups = catalog["operator_groups"]
    core_count = sum(
        groups[op["group"]]["kind"] == "core" for op in operators
    )
    foundation_count = len(operators) - core_count
    lines: list[str] = [
        "# 已开发算子（ops/）",
        "",
        "<!--",
        "AUTO-GENERATED FROM ops/catalog.yaml",
        "Run: python3 scripts/build_ops_index.py",
        "Do not edit this file directly.",
        "-->",
        "",
        "这个目录存放按样板交付的算子。[ops/catalog.yaml](catalog.yaml) 是算子清单和平台",
        "覆盖的唯一事实来源；本页由脚本生成，避免手工维护长表格时漏改，也",
        "避免把局部平台结论误写成全局结论。",
        "维护方法和数据模型见[算子目录维护指南](../docs/operator-catalog.md)。",
        "",
        "## 新算子放进来",
        "",
        "```bash",
        "cp -r templates/operator ops/<你的算子名>/",
        "cd ops/<你的算子名>/",
        "grep -rl my_op . | xargs sed -i 's/my_op/<你的算子名>/g'",
        "```",
        "",
        "开发流程照[全链路指南](../docs/fullstack-guide.md)七步走。完成后：",
        "",
        "1. 运行 `python3 example.py <profile>`，确认三层测试全绿。",
        "2. 补齐 REPORT 和 development / test-report / accuracy / performance 分册。",
        "3. 在 `ops/catalog.yaml` 登记算子、用途、平台和证据。",
        "4. 运行 `python3 scripts/build_ops_index.py` 更新本页。",
        "5. 交付路径选择见[集成指南](../docs/integration.md)。",
        "",
        "## 平台覆盖口径",
        "",
        f"目标范围是 **{len(catalog['platform_families'])} 个厂商平台家族**。"
        f"当前登记 **{len(operators)} 个算子**：**{core_count} 个核心多平台算子**、"
        f"**{foundation_count} 个基础 ATen 算子**。",
        "这是一个增量交付子集，不是完整平台矩阵。`planned` 只表示目标范围，",
        "不表示支持完成；只有进入算子条目并带有测试与报告的平台才计为已交付。",
        "",
        *_platform_summary(catalog),
        "",
    ]

    ranked = sorted(groups.items(), key=lambda item: (item[1]["rank"], item[0]))
    core_groups = [(name, meta) for name, meta in ranked if meta["kind"] == "core"]
    foundation_groups = [
        (name, meta) for name, meta in ranked if meta["kind"] == "foundation"
    ]

    for name, meta in core_groups:
        lines.extend([
            f"## {meta['title']}",
            "",
            meta["description"].strip(),
            "",
            "| 算子 | 作用 | 平台 | 证据 | 关键结论 |",
            "|---|---|---|---|---|",
            *_core_rows(catalog),
            "",
        ])

    if foundation_groups:
        lines.extend([
            "## 基础 ATen 算子",
            "",
            "这些算子主要验证基础语义、执行链路和 A1 注册；应用消费者证据按报告标注为 mini consumer 或真实引擎。",
            "它们不一定以性能提升为首要目标，因此按用途分组，避免和核心计算",
            "算子混在同一张长表里。",
            "",
        ])
    for name, meta in foundation_groups:
        lines.extend([
            f"### {meta['title']}",
            "",
            meta["description"].strip(),
            "",
            "| 算子 | 作用 | 交付路线 | 证据 | 关键结论 |",
            "|---|---|---|---|---|",
            *_foundation_rows(catalog, name),
            "",
        ])

    lines.extend([
        "## 平台注册契约",
        "",
        "注册契约按平台族记录，不把某个平台的 dispatch key 推广成全局规则。",
        "",
    ])
    for contract in catalog.get("registration_contracts", {}).values():
        non_registerable = "、".join(
            f"`{name}`"
            for name in contract["non_registerable_operators"]
        )
        lines.extend([
            f"### {contract['title']}",
            "",
            f"适用范围：{contract['scope']}",
            "",
            "```python",
            "register_a1(",
            f"    dispatch_key=\"{contract['canonical_key']}\",",
            "    counter: dict | None = None,",
            "    platform=<explicit-platform>,",
            ")",
            "```",
            "",
            f"- Canonical key：`{contract['canonical_key']}`",
            f"- Compatibility key：`{contract['compatibility_key']}`",
            f"- 可注册算子：{contract['registerable_count']}",
            f"- 不可注册算子：{non_registerable}",
            *[f"- {note}" for note in contract.get("notes", [])],
            "",
        ])

    lines.extend([
        "## 维护规则",
        "",
        "- **目录名 = 算子名**，使用小写下划线。",
        "- **新增平台家族**先在 catalog 登记 display name、status 和 roadmap，",
        "  再随首个算子交付 profile；不要提前把规划平台写进 operator 条目。",
        "- **新增算子**同步更新 catalog、三层测试、golden、性能数据和报告。",
        "- **多平台算子**必须维护 `PLATFORM.md`；涉及平台合并流程时维护 `MERGE.md`。",
        "- **平台结论限定范围**：dtype、shape、dispatch key、运行时版本和",
        "  fallback 边界写在同一份证据链里。",
        "- **证据等级按 profile 记录**：`platforms.<family>.evidence.profiles`",
        "  是证据的最小单位（level/basis/report），顶层 `evidence` 只是这些",
        "  记录的派生汇总；单个 family 下多个 profile 不共享未区分的等级。",
        "- **包态导入**统一使用 `from ops.<operator> import register_a1`；脚本态",
        "  仅作为单算子目录内运行的兼容模式。",
        "- CI 的 `ops-packaging` 是 CPU 包入口扫描；`ops/test_packaging.py` 是",
        "  NPU 真机 E2E，两者口径不同。",
        "- 修改 catalog 后必须运行：",
        "",
        "  ```bash",
        "  python3 scripts/build_ops_index.py",
        "  ```",
        "",
        "- 合入前照[交付自查清单](../templates/operator/README.md)检查。",
        "",
        "<!-- Generated from ops/catalog.yaml; do not edit manually. -->",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail when ops/README.md is out of date")
    args = parser.parse_args()

    catalog = load_catalog()
    validate_catalog(catalog)
    rendered = render_index(catalog)
    if args.check:
        current = OUTPUT_PATH.read_text() if OUTPUT_PATH.exists() else ""
        if current != rendered:
            print("ops/README.md is out of date. Run scripts/build_ops_index.py.")
            return 1
        print("ops/README.md is up to date.")
        return 0

    OUTPUT_PATH.write_text(rendered)
    print(f"generated -> {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
