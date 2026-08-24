# 官方 Search-VL SFT 数据状态（2026-08-24）

## 结论

用于工程复现的首个官方数据门禁已通过。当前不是自建/合成 SFT，而是从官方
`OpenSearch-VL/Search-VL-SFT-36K` 的固定 revision 中取得 `wiki_en` 子集，保持原始 system、tools、
conversation、observation 和答案不变，只做确定性抽样与一个不完整样本的排除。

这一步足以进入 Base 模型的 1→5→20→50 optimizer-step LoRA SFT 验证，但不能声称已经复现完整
SFT-36K 混合训练。其余语言与任务目录仍未下载。

## 固定来源与完整性

- 数据集 revision：`2c1c460af4fa15bd63210cbf426a96664b959944`
- JSON：`wiki_en/wiki_en_llama_factory_filtered.json`
  - size：`131910169`
  - SHA256：`a22a44c6a04d79d6dfd0064c89d8a792045278eed70a8e27c14b7c5e2f4850e3`
- 图片：`wiki_en/images.zip`
  - size：`104683505`
  - SHA256：`3576d4349aa8cca66f246b7dcdcf658ad8274567fc8d9eec16697ce34d264d0b`
- ZIP 审计：4,084 files、2 directories、105,121,529 uncompressed bytes、最大压缩比约
  `10.286`；CRC、路径、重复成员、符号链接、加密与膨胀限制均通过。

两项资产均由强制忽略环境代理的 Range 下载器取得，没有使用 Clash 7890/7891。只有 size 与 SHA256
同时匹配后才原子发布。原始数据和中断/并发事故证据均保留在项目四数据盘，不进入 Git。

## 全量结构审计

官方 `wiki_en` 有 3,503 条、4,084 个唯一图片引用。所有行使用同一份工具声明：

```text
crop, image_search, layout_parsing, perspective_correct,
sharpen, super_resolution, text_search, web_search
```

实际工具调用总数：

```text
crop=422, image_search=3424, layout_parsing=411, sharpen=25,
super_resolution=133, text_search=4603, web_search=5
```

发现官方索引 `1900` 以 `observation` 结束，没有最终 `gpt` 消息。源 JSON 的官方 SHA256 正确，因此这
是源数据内容问题，不是下载损坏。构建器不伪造答案、不截断修补，明确记录并从训练候选池排除该行；
剩余 3,502 条可训练。

## 固定 1,000 条工程子集

输出：

```text
/media/imc/data/yzy/agent/project4-opensearch-vl-rl/datasets/processed/
  search-vl-sft-wiki-en-official-1000-r2c1c460/
```

- selection seed：`opensearch-vl-official-sft-v1`
- 选择方法：对 seed、源索引和规范化完整行内容做 SHA256 排序，取前 1,000 条；不是易受 JSON 顺序
  之外随机状态影响的运行时 shuffle。
- 精确索引列表及 SHA256 写入 `manifest.json`；索引 1900 不在其中。
- 样本：1,000；复制图片：1,155；目录约 67 MiB。
- `wiki_en_official_1000.json` SHA256：
  `af5eb4adc2a9e4fcc0529ed2c6cfc523fca3753740aec1178c57acd54b4a3dd7`
- `dataset_info.json` SHA256：
  `6b065a7c32ddc1ac5ac79c4575fe84cc71322fd73f53739ac62d9443d0b3641f`
- `selected_indices` SHA256：
  `3195eafee69202c74cfb382cb7572fc198a471a357ec6af625f58d653d072018`

子集调用覆盖：`image_search=982`、`text_search=1318`、`crop=115`、`layout_parsing=97`、
`super_resolution=33`、`sharpen=7`。官方全量仅有 5 次的 `web_search` 没有被确定性子集抽中；
`perspective_correct` 在官方 `wiki_en` 全量中也没有实际调用。工具声明保持官方原样，没有给模型暴露
本地 provider、SQLite、entity ID 或离线语料 revision。

## 进入 SFT 前的剩余门禁

1. 用固定 LLaMA Factory 环境加载 `dataset_info.json`，完成少量 CPU/preprocess 检查；
2. 冻结 SFT launcher、模型、数据、seed、LoRA、cutoff 和输出目录约束；
3. 先在受管 tmux 中运行 1 step，再依次 5、20；每个 Run 使用新 ID，只用空闲物理 GPU1，GPU0/GPU5
   均不参与；
4. 20-step 证据通过后，按用户已经批准的目标执行 50 optimizer steps，并保留 checkpoint、loss、GPU
   与 cleanup 证据；出现 OOM、NaN/Inf、Xid、图片/工具 schema 错误立即停止，不自动扩卡。

50-step 只证明官方数据上的训练链路和 checkpoint 正常，不等价于官方完整 SFT 收敛或论文指标复现。

## 首次 1-step 发现的 cutoff 门禁

首次 Run `official-sft-wiki-en-1step-20260824` 使用 2,048 cutoff。LLaMA Factory 成功读取和转换全部
1,000 条，并加载模型，但首个 forward 在 optimizer update 前报：

```text
ValueError: Image features and image tokens do not match, tokens: 0, features: 77
```

原因不是图片或工具 schema：官方 system prompt 加完整 8 工具 JSON 声明超过 4K token。CPU 同样本审计
显示 2,048/3,072/4,096 均在图像占位符之前截断，到 5,120 才保留图像 token。对全 1,000 条以 5,120
重新预处理后：

- zero image-token rows：0；
- zero supervised-label rows：0；
- 每行 image token 数：64～245；
- 997/1,000 行仍在 5,120 截断，因此这是单卡闭环配置，不是完整轨迹 SFT。

失败 Run `exit_code=1`，GPU1 前后均 18 MiB、无 compute process，46°C；GPU0/GPU5 未参与。失败 Run、
traceback、配置和 cleanup 证据全部保留。launcher cutoff 已提升到 5,120；再次 GPU 尝试仍只允许新的
1-step Run，若 OOM 则停止并重新评估单卡内存方案，不自动降回会破坏图像占位符的 cutoff。

第二次 Run `official-sft-wiki-en-1step-v2-20260824` 保留了 image token，成功进入模型 forward，并在
全词表 logits/cross-entropy 处 OOM：进程已用约 23.21 GiB，额外申请 2.90 GiB。Run `exit_code=1`，
GPU1 cleanup 后回到 18 MiB、无 compute process；GPU0/GPU5 未参与。

Liger/fused CE 当前不是可用的小改动：环境未安装，且固定 LLaMA Factory wrapper 没有 `qwen3_vl`
Liger 路由。因此下一版 profile 改为标准 NF4 QLoRA：保持同一数据、template、5,120 cutoff 和 LoRA
target，只把冻结基座改为 4-bit NF4 double quantization。固定安装 `bitsandbytes==0.48.1`；Linux x86-64
wheel 官方 PyPI SHA256 为 `3e72cf07ba6d2169e69a61282a6f072fc675efee86049e56a33de099a0363ef2`。
下载显式清空所有代理并使用华为云直连镜像，未经过 Clash 7890/7891。

独立受管 Run `bnb-nf4-smoke-20260824` 已在物理 GPU1 通过 4096×4096 NF4 Linear 前向/反向；
loss `0.32608724`，峰值分配 117,891,072 B，`exit_code=0`，cleanup 后 GPU1 为 18 MiB 且无 compute
process。GPU0/GPU5 未参与。该结果只证明 NF4 kernel 可用，下一步仍从新的 1-step QLoRA Run 开始。

首次 QLoRA Run `official-sft-wiki-en-qlora-1step-20260824` 没有真正量化：launcher 写入了
`quantization_method: bitsandbytes`，但固定 LLaMA Factory revision 的唯一合法枚举是 `bnb`。其 parser
保留未知字符串且未报错，量化分支因此被静默跳过；日志没有 `Quantizing model ...` 标志，参数统计仍为
8,788,947,184，显存与 BF16 Run 相同，并在同一 logits/loss 位置 OOM。该 Run 未执行 optimizer update、
`exit_code=1`；GPU1 cleanup 后 18 MiB、无 compute process，GPU0/GPU5 未参与，失败证据完整保留。

launcher 已改为 `quantization_method: bnb`，profile 升级为 `official-wiki-en-qlora-v3`，并新增 fail-closed
配置门禁，禁止旧字符串或非 NF4/double-quant 配置继续启动。CPU 离线 loader 诊断必须看到
`BitsAndBytesConfig(load_in_4bit=True)` 后，才允许新的 1-step Run；v2 失败 Run 不可作为 resume 来源。
该诊断现已通过：`load_in_4bit=True`、`quant_type=nf4`、`double_quant=True`，且没有加载模型权重。

新的受管 Run `official-sft-wiki-en-qlora-v3-1step-20260824` 在物理 GPU1 完成了 1 次 optimizer
update：训练 loss `1.1252400875`、grad norm `1.016`、runtime `5.5213s`、`global_step=1`、
`exit_code=0`。日志明确出现 `Quantizing model to 4 bit with bitsandbytes`；加载后显存约 16.6 GiB，
没有复现 BF16 的 logits OOM。`checkpoint-1` 同时包含 adapter、optimizer、scheduler、trainer state 和
RNG state；adapter SHA256 为 `7e57a0834d2819655b75c40ae194a96a67624e42535f0bb9230eb3c5fe92c4fb`。
GPU1 before/after 均 18 MiB，cleanup 为 `compute_processes=none`；GPU0/GPU5 未参与。精确 tmux 仅在确认
pane dead、status 0 后关闭。该 checkpoint 现可作为同 profile 的 5-step resume 来源。

受管 Run `official-sft-wiki-en-qlora-v3-step5-20260824` 已从上述 `checkpoint-1` 正确恢复；Trainer 日志
明确记录 `Continuing training from global step 1`，最终 `global_step=5`、`exit_code=0`。step 1～5 loss
依次为 `1.12524, 1.35522, 1.16823, 1.31633, 1.08457`，grad norm 均有限。`checkpoint-5` adapter
SHA256 为 `600b9910c0651e05b6f85d5f0ec0e2c54d2f0b2310cba0e8a17c4aba7acba2c8`；optimizer 与
trainer state 均存在。GPU1 cleanup 后 18 MiB、无 compute process，GPU0/GPU5 未参与；已退出的精确
tmux status 0 后关闭。下一门为从 checkpoint-5 resume 到 20 steps。
