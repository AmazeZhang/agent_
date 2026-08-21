# OpenSearch-VL Agentic RL 学习与复现项目

本目录用于审计和缩小复现 OpenSearch-VL 的多模态搜索 Agent 训练链路，重点包括 Agentic SFT、多工具在线 rollout、Fatal-aware GRPO、故障 token masking 和 one-sided advantage clamp。

## 当前状态

- 已完成官方仓库固定版本的源码审计。
- 已建立 24 节系统学习路线，当前完成第 1 节。
- 尚未完成本地 SFT、RL 或论文结果复现；文档中的论文指标均明确标为上游结果。
- 后续目标是先跑通官方 8B checkpoint 推理，再做 100～500 条 SFT cold start 和短程 fatal-aware GRPO 消融。

## 目录

- `docs/SOURCE_AUDIT_2026-08-17.md`：代码、数据、训练配置、开放边界与复现风险审计。
- `docs/LEARNING_ROADMAP.md`：系统学习路线和进度台账。
- `docs/01-项目背景与研究动机.md`：已完成的第一节复习材料。
- `vendor/OpenSearch-VL/`：官方仓库的 Git submodule，固定到审计 commit `c5c02a49780e26ae9cb6f1fb56731d1e594d59f0`。

## 获取源码

```bash
git submodule update --init project4-opensearch-vl-rl/vendor/OpenSearch-VL
```

## 归属

OpenSearch-VL 官方源码及许可证归原作者所有，本项目不复制或冒充其代码；本仓库新增内容是学习文档、源码审计与后续实验记录。
