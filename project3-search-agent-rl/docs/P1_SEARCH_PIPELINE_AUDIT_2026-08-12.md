# P1 Search数据、Retriever与Reward审计

## 结论

P1的模型无关CPU闭环已跑通：真实上游Parquet可转换为veRL所需字段，固定Smoke切分可重建，
测试Retriever遵循上游HTTP协议，SearchEnv能够收到Observation并在最终答案上计算严格EM。
这不是模型推理或训练结果，也不是Wikipedia Retriever质量结果。

## 上游链路

```text
PeterJinGo/nq_hotpotqa_train train/test.parquet
  -> preprocess_search_r1_dataset.py
  -> prompt + reward_model + extra_info + env_kwargs
  -> SearchMultiProcessEnv.reset(env_kwargs)
  -> SearchEnv.step(<search>query</search>)
  -> POST /retrieve {query, topk, return_scores}
  -> {result: [[{document, score}, ...]]}
  -> <information>...</information>
  -> SearchEnv.step(<answer>answer</answer>)
  -> compute_score: last complete answer + normalized strict EM
```

上游处理后训练集169,615条、测试集51,713条。训练来源为NQ和HotpotQA；测试来源为NQ、
HotpotQA、PopQA、2WikiMultihopQA、TriviaQA、MuSiQue和Bamboogle。

## Smoke切分

- 训练8条：NQ 4、HotpotQA 4；
- 验证16条：NQ 2、HotpotQA 2、PopQA 3、2WikiMultihopQA 3、TriviaQA 2、
  MuSiQue 2、Bamboogle 2；
- 选择方法：对`source + normalized_question`加固定域前缀后做SHA256，按哈希升序取每源配额；
- 同一切分问题去重，Smoke训练与验证的规范化问题交集为0；
- 原始行索引、问题、答案、输入/输出哈希全部写入`manifest.json`。

Smoke协议语料由所选Ground Truth生成，只允许测试格式、超时、Reward和环境接口。它会让正确
文档可被确定性检索到，因此严禁用于准确率、Retriever Recall或模型能力声明。

## Reward语义

1. 只识别完整的`<answer>...</answer>`；
2. 拼接全部Chat History后取最后一个完整Answer；
3. 小写、移除英文冠词、ASCII标点和多余空白；
4. 与`ground_truth["target"]`任一答案完全一致时得1，否则得0；
5. 中间Search步骤Reward恒为0；达到`max_turns`时当步直接终止，Search不会再执行。

## 已证实的上游风险

1. 数据处理脚本对每个Split捕获宽泛`Exception`但不重新抛出，两个Split都失败时仍可正常退出；
   自动化必须额外检查文件存在、行数和哈希，不能只看退出码。
2. 原始数据存在9条训练集内部重复、1,219条测试集内部重复，以及10个规范化问题跨
   train/test重叠。正式评测前需定义去重策略，不能默认无泄漏。
3. Retriever对连接、超时和5xx最多重试10次，线性退避累计45秒，另加每次请求超时；故障时
   单轮可能长时间阻塞。训练前应将重试策略版本化并测P95/P99。
4. 原始`SearchToolGroup`生成`api_error/no_results/processing_error`等状态后会丢弃。P1增加了纯观测
   补丁`patches/0001-search-retrieval-status-observability.patch`，在不改变Observation和Reward的前提下，
   透传精简状态到`SearchEnv.metadata.retrieval`及`retrieval_failed`；已覆盖成功和模拟超时测试。
5. 动作Projection是大小写不敏感的，SearchEnv自身解析是大小写敏感的；当前Rollout依赖先做
   Projection来规避不一致。直接调用环境时不能假定大小写兼容。
6. SearchEnv中的Action后处理被注释，非法的Search+Answer组合仍可能进入History并影响最后答案
   抽取；训练依赖Projection valid mask和invalid-action penalty处理这一层。
7. 上游基于Gym 0.26，导入时会发出未维护警告。当前固定NumPy 1.26避免NumPy 2兼容问题；
   首轮复现不擅自迁移Gymnasium，以免改变环境语义。

## P1验证

`tests/test_search_p1.py`共12项：

- Manifest行数、切分隔离和Fixture标识；
- Answer归一化、缺失Tag、错误答案、最后Answer语义；
- Search/Answer动作投影及非法动作判定；
- Retriever排序和上游嵌套响应结构；
- 全部24条Fixture问题的Top-1文档ID与对应样本一致；
- 真实localhost HTTP下的SearchEnv Search→Observation→Answer→Reward闭环；
- 最大轮次终止且不执行最后一次Search。
- 模拟Retriever超时时标记`api_error/retrieval_failed`，不静默归因给模型。

2026-08-12结果：`12 passed in 2.54s`，`CUDA_VISIBLE_DEVICES=''`；Smoke重新构建前后5个
核心文件SHA256完全一致。

## 下一步门禁

Retriever错误状态透传门禁已满足。完整E5索引和Wikipedia Corpus仍未下载；P2单样本模型功能验证
可以继续使用明确标注的Fixture/裁剪Retriever，
但任何模型质量评测必须切换到真实Corpus Retriever。
