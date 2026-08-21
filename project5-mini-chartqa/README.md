# MiniChartQA 视觉工具 RL 实验原型

本项目是基于 VTool-R1 / veRL / EasyR1 训练栈裁剪和修改的 ChartQA 工具调用强化学习原型，不是对底层框架的个人原创声明。第三方归属见 [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md)。

当前仅完成本地代码、配置和数据预处理入口整理，尚未形成可公开验证的训练结果。提交中不包含模型、checkpoint、运行输出、API 密钥或本地训练数据压缩包。

## 环境准备

建议先在独立环境中核对 CUDA、PyTorch、vLLM 和 FlashAttention 的版本兼容性：

```bash
pip install -r requirements.txt
```

## 数据准备

将合法获取的 ChartQA/VTool-R1 数据放入 `data/`，再运行：

```bash
cd data
bash preprocess_data.sh
```

## 运行

默认脚本使用 4 张 GPU，并遵守当前服务器约束，只选择物理 GPU `1,2,3,4`：

```bash
bash train.sh
```

## 状态边界

- 训练入口和配置尚需在目标服务器执行 preflight/smoke 后才能称为可运行。
- `judge/judge_info.json` 只保留占位符，真实密钥必须通过本地安全配置提供。
- 任何后续效果都应使用 Base/GRPO、同协议评测和可复验日志报告，不能把 VTool-R1 上游结果写成本项目结果。
