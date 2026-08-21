# OpenSearch-VL SFT 环境基线

> 冻结日期：2026-08-21
> 环境位置：`/media/imc/data/yzy/agent/project4-opensearch-vl-rl/envs/sft-py311`
> 完整版本：`environments/sft-py311.freeze.txt`

## 结论

项目使用数据盘上的独立 Python 3.11 虚拟环境，不复用系统 Python、项目二环境或项目三环境。
核心栈已经完成 CPU 导入和受管物理 GPU1 正反向 smoke：

- Python 3.11.15；
- PyTorch 2.6.0+cu124，TorchVision/TorchAudio 0.21.0/2.6.0；
- Transformers 5.2.0、Datasets 4.0.0、Accelerate 1.11.0；
- PEFT 0.18.1、TRL 0.24.0、TorchData 0.11.0；
- DeepSpeed 0.18.4、Ray 2.34.0；
- FlashAttention 2.7.4.post1；
- 上游 SFT 包以 editable 方式固定到 submodule commit
  `c5c02a49780e26ae9cb6f1fb56731d1e594d59f0`。

## 下载与构建来源

- 所有安装命令均显式清空大小写代理变量，没有使用 Clash 7890/7891。
- PyTorch 官方 CUDA wheel 源和 PyPI 直连可建立连接，但大文件吞吐过低。
- 最终 PyPI 包从华为云 PyPI 镜像下载；PyTorch 2.6 的 Linux wheel 对应 CUDA 12.4 依赖。
- FlashAttention 官方 GitHub release 元数据可达，但 release 资产 CDN 直连超时。最终从 PyPI
  镜像取得 `2.7.4.post1` 源码，使用本机 CUDA 12.4 强制构建。
- FlashAttention 设置 `FLASH_ATTN_CUDA_ARCHS=80`。该版本官方构建脚本不提供 sm89 选项，
  官方 torch2.6 wheel 同样依赖 sm80 cubin 覆盖 RTX 4090 的 Ada 8.x 兼容路径；真实 sm89
  正反向 smoke 已通过。

关键构建参数：

```text
MAX_JOBS=16
NVCC_THREADS=2
FLASH_ATTENTION_FORCE_BUILD=TRUE
FLASH_ATTN_CUDA_ARCHS=80
```

## 上游安装说明偏差

上游 README 建议执行：

```text
pip install -e ".[torch,metrics,deepspeed,ray]"
```

但当前 `SFT/pyproject.toml` 没有声明任何 optional dependency extras。实际安装因此使用：

1. 先固定 PyTorch/TorchVision/TorchAudio；
2. editable 安装 `SFT/` 的声明依赖；
3. 显式安装 DeepSpeed、Ray、qwen-vl-utils、decord；
4. 单独构建 FlashAttention。

## 验证证据

CPU 导入验证覆盖 torch、torchvision、transformers、datasets、accelerate、PEFT、TRL、
DeepSpeed、Ray、qwen-vl-utils、decord、LLaMA Factory 和 FlashAttention。

真实 GPU 验证通过项目四受管启动器执行：

```text
Run ID: p1-gpu-stack-smoke-20260821
物理 GPU: 1
进程内可见设备: 1 张逻辑 cuda:0
设备: NVIDIA GeForce RTX 4090 D, compute capability 8.9
测试: BF16 FlashAttention forward + backward
结果: output/gradient 均 finite，exit_code=0
```

Run 证据保存在数据盘 `runs/p1-gpu-stack-smoke-20260821/`，未提交大日志到 Git。

## 尚未覆盖

- 此步骤只证明软件栈和单卡 CUDA 扩展可用，不证明模型可加载或训练可收敛。
- 尚未做多卡 NCCL、模型加载、数据预处理、SFT 1-step 或 resume。
- GPU5 尚未执行专项健康门禁，本步骤没有使用 GPU5。
