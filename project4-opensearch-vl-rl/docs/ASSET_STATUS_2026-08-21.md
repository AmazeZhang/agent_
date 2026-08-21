# OpenSearch-VL 资产状态

> 更新日期：2026-08-21

## 8B 基座：已完成并校验

本地路径：

```text
/media/imc/data/yzy/agent/project4-opensearch-vl-rl/models/Qwen3-VL-8B-Instruct
```

权威标识：

```text
repo: Qwen/Qwen3-VL-8B-Instruct
HF revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
manifest bytes: 17,545,915,883
```

下载没有使用 Clash 7890/7891。hf-mirror 的 Xet 路径返回 401，普通 LFS 链路吞吐过低；
最终从 ModelScope 的 Qwen 同名快照分段直连，并用固定 HF revision 清单验收内容：

- 4 个 safetensors shard 全部通过 HF LFS SHA256；
- 其余 11 个功能文件通过 size + Git blob SHA1；
- 实际校验 15 个文件、17,545,914,364 字节；
- ModelScope 改写了 `.gitattributes`（2,449 B，而 HF 为 1,519 B），该非运行元数据被显式忽略，
  没有被描述为一致；
- ModelScope 增加 `configuration.json`。早期失败下载留下 `.incomplete` 和 `.cache/` 证据，
  未经删除授权没有清理；模型加载器不会把它们当作 shard。

固定清单位于 `manifests/opensearch-vl-assets.json`，复核命令使用
`scripts/verify_asset_manifest.py`。

## Search-VL-SFT-36K：清单完成，文件未完成

权威标识：

```text
repo: OpenSearch-VL/Search-VL-SFT-36K
HF revision: 2c1c460af4fa15bd63210cbf426a96664b959944
published bytes: 13,073,649,606
```

结果边界：

- 固定 revision、24 个文件的大小、LFS SHA256/Git blob SHA1 已写入同一 manifest；
- ModelScope 官方目录和关键词搜索均未找到该数据集；
- hf-mirror 的 LFS 下载重定向到外部对象存储后，8 路和单路 Range 均长时间为 0 字节；
- 卡住的测试仅终止精确独立进程组，未使用全局停止，零字节 part 目录作为失败证据保留；
- 未使用 Clash，因此当前不能声称 SFT-36K 已下载或可用于训练。

发布目录名与上游 `SFT/data/dataset_info.json` 也不完全一致，例如 `fvqa`/`new_fvqa`、
`livevqa`/`new_livevqa`、`webqa`/`WebQA`、`wiki_art`/`wikiart`。完整下载后必须通过显式的
非覆盖 staging/relocation 步骤解决，不能直接把发布根目录当作可训练目录。

## 校验工具

`scripts/download_hf_asset.py` 提供：

- 强制直连（`requests.Session.trust_env = False`）；
- HTTPS 和公网地址检查；
- 1～16 路有界 Range；
- part 断点续传；
- 目标文件拒绝覆盖；
- size + SHA256 校验后才原子发布。

工具使用华为云 PyPI 上 10,286,522 B 的 ruff wheel 做了真实四路测试，官方 SHA256
`f2d812e482f5a7e02eee26cd73d2a37ebbdf47d795ea63ba1b89110ae93e9fb3` 校验通过；再次执行只报告
`already verified`，没有覆盖文件。CPU 单测覆盖 Range 完整性、SHA256、Git blob SHA1 和输出目录逃逸拒绝。

## 对后续 SFT 的影响

模型加载和离线推理不受阻。为继续工程链路，可以先使用明确标记的合成 agentic smoke 数据执行
1-step/恢复测试；这只能证明训练管线，不可替代公开 SFT-36K，也不可形成效果结论。真实小样本
SFT 仍以获取并校验官方数据为进入条件。
