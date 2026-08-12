# 项目三资源计划

开发阶段、日期、模块接口、验收门槛和服务器SOP以
[`DEVELOPMENT_SPEC_2026-08-11.md`](DEVELOPMENT_SPEC_2026-08-11.md)为准；本文件集中记录
硬件与扩容口径。各Profile的机器可读初值见`../configs/experiment_profiles.yaml`。

## 最小可运行定义

这里将“可运行”分为三层，避免把能推理误认为能训练：

| 定义 | 验收条件 | 最小显存 |
|---|---|---:|
| Inference Smoke | 完成一次搜索工具调用并得到规则Reward | 12GB |
| Training Smoke | 至少完成一次非零梯度更新、保存Checkpoint并可恢复 | 24GB |
| Useful Run | Reward有组内方差，完成20–100 Step并得到可解释曲线 | 48GB总显存 |

单张24GB的Training Smoke不是原论文复现。必须使用Qwen2.5-1.5B-Instruct、LoRA、
Gradient Checkpointing、Reference/Optimizer Offload、短上下文、小Batch和Group 2；Retriever
使用CPU BM25、CPU ANN或小型Corpus。

## 推荐租机规格

### 最省钱验证

```text
1×RTX 4090 24GB
16+ CPU cores
64GB RAM
150GB+ NVMe free
```

目标仅为环境、Rollout、Reward、Advantage和Backward闭环，不追求最终指标。

### 稳定开发

```text
2×RTX 4090 24GB，或1×A6000/L40S 48GB
24+ CPU cores
128GB RAM
250GB+ NVMe free
```

适合1.5B GRPO/GiGPO和小规模Reward、KL、Entropy消融。单张48GB部署简单，但两张4090
通常吞吐更高；实际选择取决于租价和PCIe拓扑。

### 简历级主实验

```text
4×RTX 4090 24GB起
32+ CPU cores
128–256GB RAM
300GB+ NVMe free
```

适合Qwen2.5-3B、100–300个更新Step以及GRPO/GiGPO主对比。若服务器可用物理GPU 1–7，
使用7张卡可以运行完整Dense Retriever并提高训练余量。

## Dense Retriever额外成本

- `wiki-18-e5-index`：约64.6GB；
- `wiki-18-corpus`压缩文件：约5.12GB，解压后更大；
- FAISS GPU模式会以FP16将索引分片到所有可见GPU；
- 上游文档按多卡环境估算每张GPU约额外占用6GB；
- 下载缓存、拼接文件、解压Corpus和最终文件可能短期同时存在，磁盘不能只按成品大小准备。

因此完整Dense Retriever不属于单卡Smoke。租4卡以下机器时，优先使用CPU BM25/ANN或
Corpus子集，把算法链路与检索系统容量解耦。

## 当前8卡服务器GPU约束

- 物理GPU 0承载图形界面和远程会话，训练与Retriever进程禁止使用；
- 物理GPU 5有多次掉卡记录，默认不加入`CUDA_VISIBLE_DEVICES`；
- 默认稳定卡集合为`1,2,3,4,6,7`，对应`server_6x24` Profile；
- GPU 5只允许在资源确实不足、能够接受任务失败且有人值守时，通过
  `ALLOW_UNSTABLE_GPU5=1`显式解锁；
- 完整Dense Retriever仍需先做20 Step容量与稳定性测试，不能因为总显存充足就直接长跑。

## 扩容判据

完成20个Training Step后记录：

- 峰值显存及保留余量；
- Rollout、Retriever、Log-prob和Backward各阶段耗时；
- Reward均值、标准差、全对/全错Group比例；
- Episode/Step Advantage均值、标准差和极值；
- KL、Entropy、Clip Fraction与Gradient Norm；
- 有效轨迹率、搜索次数和最终答案准确率。

只有无OOM、无NaN、存在有效Reward方差且Checkpoint可恢复，才增加模型、Group、上下文或
训练Step。预计总时间使用实测外推：

```text
总墙钟时间 ≈ 前20 Step平均时间 × 计划Step数 × 1.2
GPU Hours = 总墙钟时间 × GPU数量
```
