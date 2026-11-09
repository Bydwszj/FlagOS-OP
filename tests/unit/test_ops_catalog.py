from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import yaml

from scripts.build_ops_index import (
    OUTPUT_PATH,
    load_catalog,
    render_index,
    validate_catalog,
)


ROOT = Path(__file__).resolve().parents[2]


def test_catalog_models_six_incremental_platform_families():
    catalog = load_catalog()
    assert catalog["target_scope"]["vendor_family_count"] == 6
    assert set(catalog["platform_families"]) == {
        "ascend", "cambricon", "kunlunxin", "thead", "enflame", "biren"
    }

    planned = {
        name: meta
        for name, meta in catalog["platform_families"].items()
        if meta["status"] == "planned"
    }
    assert set(planned) == {"enflame", "biren"}
    assert all(not meta["current_profiles"] for meta in planned.values())

    thead = catalog["platform_families"]["thead"]
    assert thead["status"] == "partial"
    assert thead["current_profiles"] == ["thead-ppu", "thead-ppu-c2"]


def test_catalog_is_valid_and_covers_every_operator_directory():
    catalog = load_catalog()
    validate_catalog(catalog)

    actual = {
        path.name for path in (ROOT / "ops").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    cataloged = {op["directory"] for op in catalog["operators"]}
    assert cataloged == actual


def test_generated_ops_index_is_current():
    catalog = load_catalog()
    validate_catalog(catalog)
    assert OUTPUT_PATH.read_text() == render_index(catalog)


def test_generated_ops_index_shows_profile_scoped_evidence():
    """README 按 profile 展示证据等级，不显示无范围的全局 C2。"""
    catalog = load_catalog()
    text = render_index(catalog)
    # embedding：gems5/thead-ppu-c2 的 C2 与旧环境的 C1 分开标注
    assert "p800-kunlunxin-gems5 C2" in text
    assert "thead-ppu-c2 C2" in text
    assert "p800-kunlunxin C1" in text
    # 证据列给出按 profile 的明细，而不是单独一个无范围 C2
    assert "C2（ascend910-c2-vllm · p800-kunlunxin-gems5 · thead-ppu-c2）" in text
    # 按真实 consumer 证据回填后，empty/detach_/result_type 停在 C0
    assert "C0（cambricon · p800-kunlunxin）" in text


def test_foundation_route_may_come_from_every_platform_entry():
    catalog = deepcopy(load_catalog())
    foundation = next(
        op for op in catalog["operators"]
        if catalog["operator_groups"][op["group"]]["kind"] == "foundation"
    )
    original_route = foundation["route"]
    foundation.pop("route")
    for platform in foundation["platforms"].values():
        platform["route"] = original_route

    validate_catalog(catalog)
    row = next(
        row for row in render_index(catalog).splitlines()
        if f"({foundation['directory']}/)" in row
    )
    assert f"| {original_route} |" in row


def test_platform_claims_are_incremental_and_backed_by_profiles():
    catalog = load_catalog()
    operators_by_family = {
        family: [
            op for op in catalog["operators"] if family in op["platforms"]
        ]
        for family in catalog["platform_families"]
    }

    for family, meta in catalog["platform_families"].items():
        if meta["status"] == "planned":
            assert operators_by_family[family] == []
        else:
            assert operators_by_family[family]
            assert meta["current_profiles"]


def test_catalog_yaml_is_parseable_and_versioned():
    data = yaml.safe_load((ROOT / "ops/catalog.yaml").read_text())
    assert data["schema"] == 1
