#!/usr/bin/env python3
"""逐 profile 证据记录的公共解析 / 归一化 / 判定（issue #41，PR #46 整改 2/3）。

`ops/catalog.yaml`、环境锁交叉检查（`scripts/evidence_check.py`）、目录生成
（`scripts/build_ops_index.py`）共用本模块，避免三处各写一套「是否通过」。

三个正交维度（PR #46 复审意见 3）：

======================  =========================  =========================
维度                    来源                       含义
======================  =========================  =========================
evidence level          记录 ``level``             C0/C1/C2/C2-routing-boundary
实验判定 verdict        结构化报告 status/ok/...   pass / fail / missing
维护者审阅状态 review    记录 ``status``（或报告     pass / pending /
                        acceptance 字段）           undeclared
======================  =========================  =========================

``complete-pending-review`` 是 **pending**：证据已采集、实验判定可通过，
但**不等于全量 PASS**，也不抹掉其中已通过的实验结果。``FAIL``、缺判定、
仅采集完成（``collect-only``）一律不能支撑锁的 C2 声明。

结构来源唯一：``platforms.<family>.evidence`` 逐 profile 记录列表。
旧的单条 family 级 dict 兼容读取；无法唯一绑定 profile 时明确报错；
嵌套 ``evidence.profiles`` 已停用。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

# ── 证据等级（升序；rank 越大等级越高） ──
VALID_LEVELS: tuple[str, ...] = ("C0", "C1", "C2-routing-boundary", "C2")
LEVEL_RANK: dict[str, int] = {level: i for i, level in enumerate(VALID_LEVELS)}
# operator 顶层派生汇总只用这三个（routing-boundary 不冒充 C2）
TOP_LEVELS: tuple[str, ...] = ("C0", "C1", "C2")
C2_LEVELS: frozenset[str] = frozenset({"C2", "C2-routing-boundary"})
# 需要 consumer 证据范围（basis）的等级
CONSUMER_LEVELS: frozenset[str] = frozenset({"C1", "C2", "C2-routing-boundary"})

# ── 实验判定 ──
VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_MISSING = "missing"

# ── 维护者审阅状态 ──
REVIEW_PASS = "pass"
REVIEW_PENDING = "pending"
REVIEW_UNDECLARED = "undeclared"

_FAIL_WORDS: frozenset[str] = frozenset(
    {"fail", "failed", "failure", "error", "ng", "c2-miss", "miss"})
_PASS_WORDS: frozenset[str] = frozenset(
    {"pass", "passed", "ok", "success", "green", "c2-hit", "hit"})
# 实验判定通过、但维护者尚未审阅完（结果已拿到，审阅轴仍是 pending）
_PENDING_WORDS: frozenset[str] = frozenset({
    "complete-pending-review", "pending-review", "review-pending",
})
# 只是采集完成、还没有实验结论：**不能**算通过（审阅轴同样是 pending）
_NO_RESULT_WORDS: frozenset[str] = frozenset({
    "collect-only", "collection-complete", "collected", "collecting",
    "partial",
})
# 维护者审阅通过
_REVIEWED_WORDS: frozenset[str] = frozenset({
    "pass", "ok", "accepted", "approved", "reviewed", "review-complete",
    "complete", "completed", "full-pass",
})

# ── 统一的仓库内证据路径解析 ──
PATH_ERRORS: dict[str, str] = {
    "外链": "必须是仓库内路径，不允许外链 {rel}",
    "绝对路径": "必须是仓库内相对路径，不允许绝对路径 {rel}",
    "越界": "越出仓库根目录 {rel}",
    "目录": "必须是文件，不能是目录 {rel}",
    "不存在": "不存在: {rel}",
}

# ── 派生更正（#44 兼容） ──
REDECIDE_KIND = "c2-acceptance-redecide"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def level_rank(level) -> int | None:
    """证据等级的序位（未知/非法返回 None）。"""
    if isinstance(level, str):
        return LEVEL_RANK.get(level)
    return None


def is_valid_level(level) -> bool:
    return level_rank(level) is not None


def path_error(kind: str, rel: str) -> str:
    return PATH_ERRORS[kind].format(rel=rel)


def resolve_repo_file(
        root: Path, rel: str) -> tuple[Path | None, str | None, str | None]:
    """统一的仓库内证据路径解析。

    返回 ``(绝对路径, 规范化仓库相对路径, 错误类别)``。拒绝外链、绝对
    路径、逃出 repo root 的 ``..`` 越界与外部 symlink、目录；成功时给出
    规范化相对路径，供算子归属与跨引用基于同一字符串。
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


# ── 逐 profile 记录 ──

def family_records(platform: dict, *, where: str) -> tuple[list[dict], list[str]]:
    """``platforms.<family>`` 条目 → 逐 profile 记录列表 + 结构错误。

    - ``evidence`` 为 list：逐条记录；每条须带 ``profile``（该 family 只
      声明一个 profile 时可省略并绑定之）；
    - ``evidence`` 为 dict：旧的 family 级单条记录，兼容读取；未带
      ``profile`` 且该 family 声明多个 profile → 明确报错（无法唯一绑定
      环境，不能让多个环境共享未区分的等级）；
    - ``evidence`` 为含 ``profiles`` 键的 dict：已停用的嵌套形态 → 报错；
    - 每个已交付 profile 必须有独立记录；重复记录、非法 level 报错。
    """
    if not isinstance(platform, dict):
        return [], [f"{where}: platforms 条目不是映射"]
    declared = platform.get("profiles") or []
    if not isinstance(declared, list):
        declared = []
    evidence = platform.get("evidence")
    errors: list[str] = []

    if evidence is None:
        if declared:
            errors.append(
                f"{where}: 缺逐 profile 证据记录（evidence 必须是逐 profile "
                f"记录列表，每个已交付 profile 一条）")
        return [], errors
    if isinstance(evidence, dict) and "profiles" in evidence:
        errors.append(
            f"{where}: evidence.profiles 嵌套形态已停用，请改为逐 profile "
            f"记录列表（evidence: - profile: <profile> / level: ...）")
        return [], errors

    if isinstance(evidence, list):
        raw_entries = evidence
    elif isinstance(evidence, dict):
        raw_entries = [evidence]
    else:
        return [], [f"{where}.evidence 形态非法（应为逐 profile 记录列表或"
                    f"单条 dict），当前 {type(evidence).__name__}"]

    entries: list[tuple[str | None, dict]] = []
    for i, item in enumerate(raw_entries):
        if not isinstance(item, dict) or not item:
            errors.append(f"{where}.evidence[{i}] 不是有效映射 {item!r}")
            continue
        profile = item.get("profile")
        if profile is not None and (not isinstance(profile, str) or not profile):
            errors.append(f"{where}.evidence[{i}].profile 必须是非空字符串")
            continue
        entries.append((profile, dict(item)))

    records: list[dict] = []
    seen: set[str] = set()
    for profile, entry in entries:
        bound = profile
        if bound is None:
            if len(declared) == 1:
                bound = declared[0]
            else:
                errors.append(
                    f"{where}: evidence 记录缺 profile，且该条目声明 "
                    f"{len(declared)} 个 profile，无法唯一绑定环境（逐 "
                    f"profile 记录必须写 profile）")
                continue
        elif declared and bound not in declared:
            errors.append(
                f"{where}: 记录 profile={bound!r} 不在该条目 profiles "
                f"{declared} 中")
            continue
        if bound in seen:
            errors.append(f"{where}: 重复记录 profile={bound}")
            continue
        level = entry.get("level")
        if not is_valid_level(level):
            errors.append(
                f"{where}: profile {bound} 记录 level 必须是 "
                f"{list(VALID_LEVELS)}，当前 {level!r}")
            continue
        seen.add(bound)
        entry["profile"] = bound
        records.append(entry)

    for profile in declared:
        if profile not in seen:
            errors.append(
                f"{where}: profile {profile} 缺独立证据记录（每个已交付 "
                f"profile 必须有独立 level/basis，不共享未区分的等级）")
    return records, errors


def records_map(records: list[dict]) -> dict[str, dict]:
    return {r["profile"]: r for r in records if isinstance(r.get("profile"), str)}


def top_level_from_records(records: list[dict]) -> str | None:
    """记录派生的最高等级（operator 顶层汇总只能由记录派生）。"""
    best: str | None = None
    best_rank = -1
    for record in records:
        rank = level_rank(record.get("level"))
        if rank is not None and rank > best_rank:
            best, best_rank = record["level"], rank
    return best


def operator_top_level(records: list[dict]) -> str | None:
    """operator 顶层派生汇总（routing-boundary 归入 C2，不冒充独立等级）。"""
    top = top_level_from_records(records)
    return "C2" if top == "C2-routing-boundary" else top


def record_for(catalog: dict, op_dir: str, profile: str) -> dict | None:
    """按算子 directory + profile 取记录（环境锁交叉校验用）。"""
    for op in catalog.get("operators", []) or []:
        if op.get("directory", op.get("name")) != op_dir:
            continue
        for platform in (op.get("platforms") or {}).values():
            if not isinstance(platform, dict):
                continue
            records, _ = family_records(platform, where="catalog")
            found = records_map(records).get(profile)
            if found is not None:
                return found
    return None


# ── 结构化报告：实验判定 / 审阅状态 ──

def _word(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _classify(word: str | None) -> str | None:
    """状态词 → pass / fail / missing（None = 不认识该词）。"""
    if word is None:
        return None
    if word in _FAIL_WORDS:
        return VERDICT_FAIL
    if word in _PASS_WORDS:
        return VERDICT_PASS
    if word in _NO_RESULT_WORDS:
        return VERDICT_MISSING  # 仅采集完成不构成实验通过
    if word in _PENDING_WORDS:
        return VERDICT_PASS
    return None


def load_json_object(
        resolved: Path) -> tuple[dict | None, str | None]:
    """读取 JSON 顶层映射；失败返回 ``(None, 原因)``，不静默置 None。"""
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, f"JSON 解析失败 {exc}"
    if not isinstance(data, dict):
        return None, "JSON 顶层不是映射"
    return data, None


def _acceptance_block(data: dict) -> dict | None:
    for key in ("acceptance", "acceptance_decision", "decision", "redecide"):
        block = data.get(key)
        if isinstance(block, dict):
            return block
    return None


def report_verdict(data) -> str:
    """实验判定：该次实验通过 / 失败 / 缺判定。

    只认明确字段（不做自然语言匹配）：顶层 ``status``/``ok``、C2 harness
    的 ``c2_status``、``acceptance.*`` / ``acceptance_decision.*`` /
    ``redecide.*`` 的 ``acceptance_status``。``complete-pending-review``
    这类「已采集待审阅」词的**实验判定**为通过（其中的实验结果不丢），
    审阅状态由 :func:`report_review_state` 单独给出。
    """
    if not isinstance(data, dict):
        return VERDICT_MISSING
    word = _word(data.get("status"))
    if word is not None:
        verdict = _classify(word)
        if verdict is not None:
            return verdict
    if data.get("ok") is True:
        return VERDICT_PASS
    if data.get("ok") is False:
        return VERDICT_FAIL
    word = _word(data.get("c2_status"))
    if word is not None:
        verdict = _classify(word)
        if verdict is not None:
            return verdict
    block = _acceptance_block(data)
    if block is not None:
        for key in ("acceptance_status", "status", "verdict"):
            word = _word(block.get(key))
            if word is not None:
                verdict = _classify(word)
                if verdict is not None:
                    return verdict
    return VERDICT_MISSING


def report_consumer(data) -> str | None:
    """结构化报告声明的 consumer/engine（无则 None）。

    只取明确字段 ``consumer_type`` / ``consumer`` / ``engine``，不做自然
    语言匹配；catalog 记录与环境锁两侧共用，避免各写一套「consumer 是谁」。
    """
    if not isinstance(data, dict):
        return None
    for key in ("consumer_type", "consumer", "engine"):
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def consumer_problem(data, want_consumer) -> str | None:
    """记录声明 consumer 时，报告侧冲突给出原因；一致/未声明返回 None。

    与 :func:`verdict_problem` 一样是 catalog `record.report` 与锁
    `evidence[].path` 共用的判定口径：只做一致性核对，不猜语义。
    """
    if not (isinstance(want_consumer, str) and want_consumer.strip()):
        return None
    got = report_consumer(data)
    if got is not None and got != want_consumer:
        return (f"报告 consumer={got!r} 与记录 consumer="
                f"{want_consumer!r} 不一致")
    return None


def verdict_problem(data) -> str | None:
    """实验判定不合格时给出原因；合格返回 None（catalog 与锁共用）。"""
    verdict = report_verdict(data)
    if verdict == VERDICT_PASS:
        return None
    raw = None
    if isinstance(data, dict):
        for key in ("status", "c2_status"):
            if isinstance(data.get(key), str) and data[key].strip():
                raw = data[key]
                break
        if raw is None:
            block = _acceptance_block(data)
            if block is not None:
                for key in ("acceptance_status", "status", "verdict"):
                    value = block.get(key)
                    if isinstance(value, str) and value.strip():
                        raw = value
                        break
    if verdict == VERDICT_FAIL:
        return f"报告验证状态非通过: {raw!r}"
    return "报告缺少 status/ok 验证状态字段（结构化报告必须给出明确实验判定）"


def _review_from_word(word: str | None) -> str | None:
    if word is None:
        return None
    if word in _FAIL_WORDS:
        return REVIEW_PENDING  # 失败材料不构成「已审阅通过」
    if word in _PENDING_WORDS or word in _NO_RESULT_WORDS:
        return REVIEW_PENDING
    if word in _REVIEWED_WORDS:
        return REVIEW_PASS
    return None


def report_review_state(data) -> str:
    """报告侧给出的维护者审阅状态（无则 undeclared）。"""
    if not isinstance(data, dict):
        return REVIEW_UNDECLARED
    block = _acceptance_block(data)
    if block is not None:
        state = _review_from_word(_word(block.get("acceptance_status")))
        if state is None:
            state = _review_from_word(_word(block.get("status")))
        if state is not None:
            return state
    state = _review_from_word(_word(data.get("acceptance_status")))
    if state is not None:
        return state
    return REVIEW_UNDECLARED


def record_review_state(record) -> str:
    """记录 ``status``（维护者审阅状态的权威来源）。"""
    if isinstance(record, dict):
        state = _review_from_word(_word(record.get("status")))
        if state is not None:
            return state
        if _word(record.get("status")) is not None:
            return REVIEW_UNDECLARED
    return REVIEW_UNDECLARED


def review_word_known(record) -> bool:
    """记录 ``status`` 是否是已知词（未知词应报错而不是放过）。"""
    if not isinstance(record, dict) or _word(record.get("status")) is None:
        return True  # 未声明不算错
    return record_review_state(record) != REVIEW_UNDECLARED


# ── 派生更正（#44：原始 summary 保留旧 profile，redecide 记录更正） ──

def load_redecide(
        root: Path, redecide_rel: str, *, expect_now: str | None = None,
) -> tuple[dict | None, str | None]:
    """读取并核验派生更正文件。

    必须全部对上才采用更正后的范围：文件可解析、``kind``、源路径
    ``raw_material.summary_path`` 存在且是 JSON、源文件 **SHA256** 一致、
    ``profile_field_original`` 与源 profile 一致、``corrections`` 里有
    ``profile`` 的 was→now、``redecide.commit`` 是 40 位 commit。
    不覆写原始 JSON，也不因看到派生文件就跳过原始来源校验。
    """
    resolved, norm, kind = resolve_repo_file(root, redecide_rel)
    if kind is not None:
        return None, f"redecide {path_error(kind, redecide_rel)}"
    data, err = load_json_object(resolved)
    if err is not None:
        return None, f"{norm}: {err}"
    if data.get("kind") != REDECIDE_KIND:
        return None, f"{norm}: kind 必须是 {REDECIDE_KIND!r}，当前 " \
                     f"{data.get('kind')!r}"
    raw = data.get("raw_material")
    if not isinstance(raw, dict):
        return None, f"{norm}: 缺 raw_material（派生更正必须记录源文件）"
    source_rel = raw.get("summary_path")
    source, source_norm, source_kind = resolve_repo_file(
        root, source_rel if isinstance(source_rel, str) else "")
    if source_kind is not None:
        return None, f"{norm}: raw_material.summary_path {path_error(source_kind, source_rel)}"
    if source_norm != source_rel:
        return None, f"{norm}: raw_material.summary_path={source_rel!r} 与" \
                     f"规范化的源路径 {source_norm!r} 不一致"
    source_data, source_err = load_json_object(source)
    if source_err is not None:
        return None, f"{source_norm}: {source_err}"
    want_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    got_sha = raw.get("summary_sha256")
    if got_sha != want_sha:
        return None, (
            f"{norm}: 源文件 SHA256 不匹配（记录 {str(got_sha)[:16]}…，"
            f"实际 {want_sha[:16]}…）")
    original = raw.get("profile_field_original")
    declared = source_data.get("profile")
    if original != declared:
        return None, (
            f"{norm}: profile_field_original={original!r} 与源 "
            f"{source_norm} 的 profile={declared!r} 不一致")
    corrections = data.get("corrections")
    if not isinstance(corrections, list):
        return None, f"{norm}: 缺 corrections"
    now = None
    for item in corrections:
        if not isinstance(item, dict) or item.get("field") != "profile":
            continue
        if item.get("was") != original:
            return None, (
                f"{norm}: corrections.profile.was={item.get('was')!r} 与源 "
                f"profile={original!r} 不一致")
        now = item.get("now")
    if not isinstance(now, str) or not now:
        return None, f"{norm}: corrections 里没有 profile 更正"
    if expect_now is not None and now != expect_now:
        return None, (
            f"{norm}: 更正后的 profile={now!r} 与目标 {expect_now!r} 不一致")
    redecide_block = data.get("redecide")
    commit = redecide_block.get("commit") if isinstance(redecide_block, dict) \
        else data.get("commit")
    if not (isinstance(commit, str) and _COMMIT_RE.match(commit)):
        return None, f"{norm}: redecide.commit 不是 40 位 commit: {commit!r}"
    if raw.get("preserved_unmodified") is False:
        return None, f"{norm}: 源材料未保持原样（preserved_unmodified=false）"
    return data, None


def redecide_corrected_profile(data: dict) -> str | None:
    """从已核验的更正文件取更正后的 profile。"""
    for item in data.get("corrections") or []:
        if isinstance(item, dict) and item.get("field") == "profile":
            now = item.get("now")
            if isinstance(now, str) and now:
                return now
    return None


def effective_profile(
        root: Path, resolved: Path, norm: str, data: dict, *,
        want_profile: str, redecide_rel: str | None = None,
        want_label: str = "记录",
) -> tuple[str | None, str | None]:
    """结构化证据的**有效** profile（必要时经核验的派生更正）。

    返回 ``(有效 profile, 错误)``。直接声明一致 → 用之；不一致但记录声明了
    ``redecide`` → 核验更正（哈希/来源/commit 全对）后采用；否则报错。
    ``want_label`` 只是报错措辞（"记录" / "锁"）。
    """
    declared = data.get("profile")
    if isinstance(declared, str) and declared and declared == want_profile:
        return want_profile, None
    if isinstance(redecide_rel, str) and redecide_rel.strip():
        redecide, err = load_redecide(
            root, redecide_rel, expect_now=want_profile)
        if err is not None:
            return None, err
        return redecide_corrected_profile(redecide), None
    if not isinstance(declared, str) or not declared:
        return None, "报告缺少 profile 字段"
    return None, (f"报告声明 profile={declared} 与{want_label} "
                  f"profile={want_profile} 不一致")
