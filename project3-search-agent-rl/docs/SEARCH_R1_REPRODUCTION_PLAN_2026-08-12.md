# Search-R1复现优先实施计划

- 制定日期：2026-08-12
- 当前状态：P0/P1/P2通过；下一步执行P2.5真实Corpus Retriever门禁
- 核心框架：`langfengQ/verl-agent`内置的veRL训练栈
- 核心任务：Search-R1多轮搜索环境
- 原则：先复现、再诊断、后改进；不把上游能力或论文数字写成个人结果

## 1. 项目定位

项目三不是简单运行一个开源Shell脚本，而是在veRL框架下完成Search-R1的分层复现：

```text
Search-R1数据与Retriever
        ↓
多轮Search/Answer环境与规则Reward
        ↓
veRL Rollout、GRPO Advantage、PPO-style Loss
        ↓
真实参数更新、Checkpoint与冻结评测
        ↓
问题诊断、GiGPO对照与个人改进
```

上游Search-R1、verl-agent和veRL提供复现基础；本项目的个人工作包括环境适配、可复现配置、
Reward到梯度的仪表化、失败诊断、安全实验系统、严格对照以及后续算法改进。

## 2. 三层复现口径

### R1：功能复现

- 一条Search-R1样本可被加载；
- 模型至少执行一次`search`并收到Retriever结果；
- 模型提交最终答案；
- Exact Match规则Reward与人工计算一致；
- 环境错误与模型错误被分开记录。

达到R1只能声称“Search-R1推理链路跑通”，不能声称完成RL复现。

### R2：训练闭环复现

- 使用veRL完成至少一个非零GRPO参数更新；
- 能从Reward追踪到Return、Group Advantage、Token Mask、Policy Loss和Gradient Norm；
- Retrieved/Observation Token不参与Policy Loss；
- Checkpoint可保存、加载并继续训练；
- 1、5、20 Step逐级通过，无OOM、NaN、Inf和进程残留。

达到R2可以声称“在veRL下复现Search-R1训练闭环”。

### R3：结果复现

- Base/Instruct与GRPO使用同一模型、数据切分、Retriever和评测预算；
- 完成足以观察趋势的训练曲线；
- 报告与公开结果的配置差异，不把缩小版结果称为论文数字复现；
- 至少两个Seed后才形成稳定性结论；
- 原始日志能够自动生成表格和图。

只有配置、模型、数据和评测口径均与原工作对齐时，才使用“论文结果复现”；否则使用
“缩小版复现”或“方法复现”。

## 3. 阶段计划

### P0：机器、存储与复现边界

任务：

1. 使用已挂载在`/media/imc/data`的`/dev/nvme0n1`，只创建项目专用目录，不格式化；
2. 将模型、数据、索引、缓存、日志和Checkpoint统一放到`PROJECT3_DATA_ROOT`；
3. 记录根仓库Commit、verl-agent submodule Commit和原始Search-R1参考版本；
4. 固定Python、CUDA、PyTorch、vLLM、FlashAttention、Ray、veRL依赖；
5. 为数据集、Corpus和模型建立来源、版本、哈希与许可Manifest；
6. 所有GPU命令统一经过`preflight.sh`和`run_managed.sh`。

退出门槛：

- 数据盘已挂载且可写，剩余容量满足Profile；
- GPU0被硬隔离、GPU5默认禁用；
- 环境可重建，依赖版本不再漂移；
- 尚未启动训练。

产物：环境Lock、数据/模型Manifest、机器预检记录。

状态（2026-08-12）：环境Lock、data盘目录、GPU门禁、受管进程组、tmux包装及GPU1基础
CUDA/FlashAttention验证已完成；该阶段结束时未下载模型、数据或索引，未启动训练。数据
Manifest已在P1建立；模型Manifest将在P2首次下载前建立。

### P1：CPU数据、Retriever与Reward闭环

任务：

1. 审计上游Search-R1数据预处理脚本与字段；
2. 生成8条训练样本和16条验证样本的固定Smoke切分；
3. 先使用CPU BM25或裁剪Corpus Retriever，不加载完整GPU E5 Index；
4. 独立测试`search(query, topk)`协议、超时和错误分类；
5. 为答案抽取、标准化和Exact Match Reward添加单元测试；
6. 保存一条不依赖模型的确定性样例Trace。

退出门槛：

- 数据可重复生成且无训练/验证泄漏；
- Retriever返回结构稳定；
- Reward测试与人工结果一致；
- Retriever失败不会被计为模型回答错误。

产物：Smoke数据、Manifest、Reward测试、Retriever测试和样例Trace。

状态（2026-08-12）：已完成。真实上游问答Parquet已下载至data盘；8/16固定Smoke、来源与
哈希Manifest、模型无关确定性Trace、Reward/动作/HTTP Retriever/SearchEnv共12项CPU测试均
通过。已增加纯观测补丁区分Retriever错误与模型错误。完整Wikipedia Corpus/E5 Index未下载，
未进行模型推理或训练。

### P2：Base模型Search-R1推理复现

任务：

1. 在物理GPU1上加载`Qwen2.5-1.5B-Instruct`；
2. 依次运行1条、4条、16条冻结样本；
3. 记录Prompt、Reasoning、Query、Retrieved Documents、Answer和Reward；
4. 统计格式错误、无效动作、重复Query、搜索次数、延迟和峰值显存；
5. 验证进程退出后GPU1显存恢复且无Ray/vLLM残留。

退出门槛：达到R1功能复现，并形成Base/Instruct原始能力报告。

产物：Inference配置、原始Trace、S0基线指标和显存/延迟记录。

状态（2026-08-12）：已在物理GPU1完成1/4/16逐级冻结推理，并通过独立强制搜索诊断和
自然16条Run验证模型、动作投影、SearchEnv、Retriever、历史回填与Reward闭环。正式16条
Run为9个投影search、8次实际成功检索、2条Fixture Reward=1；所有Run退出后无GPU进程残留。
由于Fixture由Ground Truth衍生，该结果只满足R1功能复现，不构成S0质量基线。

### P2.5：真实Corpus Retriever门禁

任务：

1. 固定上游Wikipedia Corpus、E5 Retriever、索引revision、许可证和文件哈希；
2. 使用与答案无关的真实文档完成8/16条检索协议和延迟测试；
3. 检查Query、Top-k文档、Retriever失败与答案之间不存在Fixture式泄漏；
4. 冻结P3要使用的Retriever服务启动和停止方式。

退出门槛：真实Retriever可重复启动，8/16请求无基础设施错误，训练不引用Fixture Corpus。

产物：Corpus/Index Manifest、真实检索Trace、延迟与错误报告。

### P3：veRL GRPO单步训练闭环

任务：

1. 将规划Profile翻译成可直接传给Hydra的版本化配置；
2. 使用1.5B、LoRA、Group 2、短上下文完成一次真实Backward；
3. 记录Reward、Return、Group标准差、Advantage、Ratio、KL、Entropy、Clip Fraction和Gradient Norm；
4. 审计Action Token与Observation Token的Loss Mask；
5. 保存Checkpoint，重新启动进程并继续一个Step；
6. 保存失败Run，不覆盖、不删除。

退出门槛：达到R2训练闭环复现；Checkpoint恢复前不进入20 Step。

产物：S1-1step配置、Reward→Loss审计Trace、Checkpoint恢复证据。

### P4：Search-R1 GRPO缩小版基线

任务：

1. 严格按1 Step、5 Step、20 Step晋级；
2. 20 Step后报告吞吐、峰值显存、Retriever P50/P95和预计GPU Hours；
3. 检查组内Reward方差和有效Group比例；
4. 若信号有效，再扩展到50–100 Step；
5. 比较Base与GRPO，固定任务切分和评测预算；
6. 冻结配置后补Seed 1，关键结果再决定是否补Seed 2。

暂停条件：Reward长期全零/全一、Mask错误、NaN/Inf、Checkpoint不可恢复、Retriever错误率过高、
进程或显存残留。

退出门槛：达到R3中的“缩小版结果复现”，得到可信的S0 vs S1报告。

产物：训练曲线、评测表、吞吐与成本报告、失败分类。

### P5：GiGPO复现与问题诊断

前置条件：P4的GRPO基线验收通过。

任务：

1. 在相同模型、切分和预算下运行Exact-state GiGPO；
2. 再运行上游Similarity-based GiGPO；
3. 记录Episode Advantage、Step Advantage、Group大小、单元素组和错误合并；
4. 解释GRPO和GiGPO在稀疏奖励下的差异；
5. 不以单次Seed或不同预算比较方法优劣。

退出门槛：GiGPO可重跑，Step Advantage不退化，S1/S2/S3比较公平。

产物：算法复现报告、状态分组审计、GRPO/GiGPO对照。

### P6：个人改进与消融

主方案为Structured Anchor State；仅当重复状态不足且有失败证据时，才切换Dynamic Sampling。

任务：

1. 将Query、文档ID、Evidence Signature、Turn和Action Type组成结构化状态；
2. 保留关闭改进的严格对照；
3. 优先消融State On/Off、Similarity Threshold和Step Advantage Weight；
4. 至少两个Seed；
5. 同时报告准确率、样本效率、搜索成本和稳定性，不只挑最好指标。

退出门槛：个人代码边界清晰、有单元测试、有公平对照；无正结果时如实报告失败原因。

### P7：秋招交付

- 中文技术报告：问题、原理、复现、失败、改进和结论；
- 英文README：安装、最小Demo、配置和结果来源；
- Reward→Advantage→Loss流程图；
- Base/GRPO/GiGPO/个人改进结果与消融图；
- 一条可演示Search轨迹；
- 简历Bullet中的每个数字绑定原始Run和自动生成报告；
- 明确标注Search-R1、verl-agent、veRL上游贡献与个人贡献。

## 4. GPU与破坏性操作门禁

- P0/P1默认不使用GPU；
- P2/P3首轮只使用物理GPU1；
- 未通过20 Step前不扩到多卡；
- 超过20 Step、启用完整E5 GPU Index或使用4张以上GPU前需要用户确认；
- 不自动挂载、格式化、清理磁盘或删除失败实验；
- 不使用GPU0，不全局结束Python/Ray进程，不结束身份无法确认的显卡进程；
- 所有Run目录只追加新实验，不覆盖已有结果。

## 5. 当前建议口径

1. 首个目标是1.5B + CPU/裁剪Retriever的R2闭环，不追求论文分数；
2. 第二目标是1.5B GRPO的20–100 Step缩小版R3结果；
3. 之后升级3B并完成GRPO/GiGPO公平对照；
4. Structured Anchor State最后进入，不能替代Search-R1基线复现；
5. 完整E5 Index是结果对齐阶段的资源选项，不是功能复现前置条件。

## 6. 需要共同确认的决策

- 是否接受“1.5B功能/训练复现 → 3B结果复现”的模型升级顺序；
- 是否接受“CPU/裁剪Retriever → 完整E5 Retriever”的检索升级顺序；
- 秋招主叙事是否确定为“veRL下Search-R1复现、链路解释、问题诊断与信用分配改进”；
- 数据根目录固定为`/media/imc/data`，盘上既有数据一律视为必须保留。
