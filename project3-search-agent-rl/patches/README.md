# Upstream patches

对`vendor/verl-agent`的算法修改应保留为可审查Patch或独立Commit，并记录：

- 上游submodule commit；
- 修改的Reward、Advantage或Loss公式；
- 对应配置与随机种子；
- 单元测试和Smoke结果；
- 与未修改上游基线的差异。

不要直接提交模型、数据、索引、Checkpoint或实验缓存。
