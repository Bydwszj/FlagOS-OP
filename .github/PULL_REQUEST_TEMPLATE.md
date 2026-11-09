<!--
PR 自查清单（2026-10-07 审计后新增——详见 templates/operator/README「交付自查清单」）
提交前逐项确认（新算子全项；平台扩展/修复按相关项）:
-->
## 自查

- [ ] 核心八件齐全: README / REPORT / `__init__.py`（包化入口） / reference / register / example / kernel/ / test/
- [ ] 三层测试绿: test/{kernel,op,framework}_level.py
- [ ] 性能对照: script/bench_perf.py（native 参照）+ reports/ 更新
- [ ] 黄金绿: goldendata/ + script/check_accuracy.py
- [ ] 包化回归绿: `python3 ops/test_packaging.py`（全算子导入扫描）
- [ ] 多平台变更时: PLATFORM.md / MERGE.md 已同步
- [ ] 证据等级同步: `ops/catalog.yaml` `evidence` 已回填（C1 需 `test/framework_level.py`；C2 需 `evidence_basis` + `evidence_paths` 且路径存在）；`python3 scripts/evidence_check.py` 通过
- [ ] 署名一致: `wt-YYYY-MM-DD-fix` + `# wt <邮箱>`，commit 作者相同

## 说明

<!-- 问题→思路→怎么用（或改动摘要）-->
