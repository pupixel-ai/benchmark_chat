#!/bin/bash
#
# Memory Engineering v2.0 - 目录结构重组脚本
# 2026-03-25 Claude 生成的历史参考稿；当前仓库不要直接照跑。
# 执行第 1-2 阶段：文档整理 + 脚本整理
#
# 使用方式：
#   bash restructure.sh
#
# ⚠️ 注意：这是 git 操作，执行前请确保工作区干净
#   git status 应该无 untracked 或 modified 文件
#

set -e  # 任何命令失败则停止

echo "========================================="
echo "Memory Engineering v2.0 - 重组脚本"
echo "========================================="
echo

# 检查 git 工作区状态
if ! git status --porcelain | grep -q .; then
    echo "✅ Git 工作区干净"
else
    echo "⚠️  Git 工作区有未提交的改动："
    git status --porcelain | head -10
    echo
    read -p "继续执行? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 1
    fi
fi

echo "========================================="
echo "第 1 阶段：文档整理"
echo "========================================="
echo

# 创建 docs 子目录
mkdir -p docs/analysis docs/guides
echo "✅ 创建 docs/ 子目录"

# 移动文档
git mv AGENTS.md docs/agents.md 2>/dev/null && echo "✅ 移动 AGENTS.md → docs/agents.md" || echo "⚠️  AGENTS.md 不存在或已移动"
git mv HANDOVER.md docs/handover.md 2>/dev/null && echo "✅ 移动 HANDOVER.md → docs/handover.md" || echo "⚠️  HANDOVER.md 不存在或已移动"
git mv AUDIT_REPORT.md docs/analysis/audit_report.md 2>/dev/null && echo "✅ 移动 AUDIT_REPORT.md → docs/analysis/" || echo "⚠️  AUDIT_REPORT.md 不存在或已移动"
git mv CRITICAL_ISSUES.md docs/analysis/critical_issues.md 2>/dev/null && echo "✅ 移动 CRITICAL_ISSUES.md → docs/analysis/" || echo "⚠️  CRITICAL_ISSUES.md 不存在或已移动"
git mv CLEANUP_GUIDE.md docs/guides/cleanup_guide.md 2>/dev/null && echo "✅ 移动 CLEANUP_GUIDE.md → docs/guides/" || echo "⚠️  CLEANUP_GUIDE.md 不存在或已移动"
git mv QUICK_CLEANUP.md docs/guides/quick_cleanup.md 2>/dev/null && echo "✅ 移动 QUICK_CLEANUP.md → docs/guides/" || echo "⚠️  QUICK_CLEANUP.md 不存在或已移动"

echo

# 创建 docs/README.md
echo "创建 docs/README.md"
cat > docs/README.md << 'EOF'
# Memory Engineering v2.0 文档导航

## 核心文档
- [Agent 行为规范](agents.md) - Codex Agent 工作指南（必读）
- [交接文档](handover.md) - v2.0 最后更新状态、冻结决策、暂停项目

## 问题分析报告（2026-03-25 生成）
- [完整审计报告](analysis/audit_report.md) - 链路、数据流、文件问题的深度分析（P0-P3）
- [P0 级问题摘要](analysis/critical_issues.md) - 3 个链路断裂问题、3 个设计决策问题

## 清理指南
- [完整清理指南](guides/cleanup_guide.md) - 详细的清理方案（按优先级 P0-P3）
- [快速清理参考](guides/quick_cleanup.md) - 一行命令版、速查表

## 策略文档
- [主链路策略](../docs/主链路策略/README.md) - 整个记忆工程的核心策略、LLM 角色体系

---

## 文档用途指南

**新开发者必读顺序**:
1. agents.md - 了解 Agent 规范
2. handover.md - 了解当前状态和冻结决策
3. 主链路策略/README.md - 了解核心架构

**问题排查**:
- 链路有问题？→ analysis/critical_issues.md
- 想深入理解？→ analysis/audit_report.md

**清理和维护**:
- 想清理项目？→ guides/cleanup_guide.md
- 想快速查命令？→ guides/quick_cleanup.md

EOF
echo "✅ 创建 docs/README.md"

echo

echo "========================================="
echo "第 2 阶段：脚本整理"
echo "========================================="
echo

# 创建 scripts 子目录
mkdir -p scripts/eval scripts/debug
echo "✅ 创建 scripts/ 子目录"

# 移动脚本
git mv test_chen_meiyi_tool_strategy.py scripts/eval/ 2>/dev/null && echo "✅ 移动 test_chen_meiyi_tool_strategy.py → scripts/eval/" || echo "⚠️  test_chen_meiyi_tool_strategy.py 不存在或已移动"
git mv debug_llm_response.py scripts/debug/ 2>/dev/null && echo "✅ 移动 debug_llm_response.py → scripts/debug/" || echo "⚠️  debug_llm_response.py 不存在或已移动"

echo

# 创建 scripts/README.md
echo "创建 scripts/README.md"
cat > scripts/README.md << 'EOF'
# 脚本库

## 生产脚本

### run_lp3_fresh.py
Official Gemini API 从头运行 LP3 profile generation

```bash
python3 run_lp3_fresh.py --photos /path/to/photos --user-name username --reset-cache
```

### run_lp3_bundle_openrouter.py
使用 OpenRouter 的 bundled profile generation

```bash
OPENROUTER_API_KEY=xxx python3 run_lp3_bundle_openrouter.py --photos /path --user-name xxx
```

### upgrade_vlm_cache_format.py
升级旧版 VLM 缓存格式

```bash
python3 upgrade_vlm_cache_format.py --cache-dir cache/default/
```

---

## 调试脚本

### debug/debug_llm_response.py
测试 LLM 对单个字段的响应，用于 LP3 调试

```bash
python3 debug/debug_llm_response.py
```

---

## 评测脚本

### eval/test_chen_meiyi_tool_strategy.py
验证新 Tool 策略（location_stats, brand_stats, career_stats）是否改善陈美伊的字段质量

```bash
python3 eval/test_chen_meiyi_tool_strategy.py
```

EOF
echo "✅ 创建 scripts/README.md"

echo

echo "========================================="
echo "提交变更"
echo "========================================="
echo

git add .
git commit -m "chore: reorganize docs and scripts into subdirectories

Phase 1: Consolidate documentation
- Create docs/ directory structure with subdirectories for analysis and guides
- Move AGENTS.md → docs/agents.md
- Move HANDOVER.md → docs/handover.md
- Move AUDIT_REPORT.md → docs/analysis/audit_report.md
- Move CRITICAL_ISSUES.md → docs/analysis/critical_issues.md
- Move CLEANUP_GUIDE.md → docs/guides/cleanup_guide.md
- Move QUICK_CLEANUP.md → docs/guides/quick_cleanup.md
- Create docs/README.md with navigation guide

Phase 2: Organize utility scripts
- Create scripts/eval and scripts/debug subdirectories
- Move test_chen_meiyi_tool_strategy.py → scripts/eval/
- Move debug_llm_response.py → scripts/debug/
- Create scripts/README.md with usage documentation

Result:
- Root directory cleaned up (from 10+ docs to 3 core files)
- Documentation management centralized in docs/
- Utility scripts organized by purpose in scripts/
- Zero code changes or import modifications

References:
- docs/README.md - Navigation guide for all documentation
- docs/analysis/critical_issues.md - 3 P0 issues requiring attention
- docs/guides/cleanup_guide.md - Detailed cleanup strategy"

echo
echo "✅ 提交完成"

echo
echo "========================================="
echo "重组结果"
echo "========================================="
echo

echo "✅ 第 1 阶段（文档整理）完成"
echo "   ├── docs/agents.md"
echo "   ├── docs/handover.md"
echo "   ├── docs/analysis/audit_report.md"
echo "   ├── docs/analysis/critical_issues.md"
echo "   ├── docs/guides/cleanup_guide.md"
echo "   ├── docs/guides/quick_cleanup.md"
echo "   └── docs/README.md (导航)"

echo
echo "✅ 第 2 阶段（脚本整理）完成"
echo "   ├── scripts/eval/test_chen_meiyi_tool_strategy.py"
echo "   ├── scripts/debug/debug_llm_response.py"
echo "   └── scripts/README.md (说明)"

echo
echo "📊 根目录现在包含："
ls -1 | grep -E "\.py$|\.md$" || echo "   (仅 config.py, main.py 等核心文件)"

echo
echo "✅ 重组完成！"
echo
echo "下一步建议："
echo "  1. 运行清理命令：rm -rf cache/ output/ __pycache__/"
echo "  2. 验证项目大小：du -sh ."
echo "  3. 查看文档：cat docs/README.md"
