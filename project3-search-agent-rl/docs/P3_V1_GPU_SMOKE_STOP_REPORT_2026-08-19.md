# P3 Search-aware GRPO v1 阶段一启动停止报告（2026-08-19）

状态：**硬停止**（fail-closed 生效，未进入训练）。本报告记录首次真实配置加载发现的
patch 0008 struct 写入缺陷、完整诊断、待决策修复选项与资源状态。

## 1. 背景与已完成部分

1. **Wrapper 补丁门禁修复**（commit `d5b3a86`，已推送，仅含
   `scripts/run_p3_grpo_search_aware_v1.sh`）：
   - 原 0001–0007 逐补丁 `git apply --reverse --check` 对链式补丁不可判定（0007/0008
     与 0001/0002/0004 改同一批文件、hunk 漂移，reverse-check 误报"未应用"），
     导致 Phase 4B 以来所有 GPU 启动被错误拦停；
   - 新门禁：确认 vendor HEAD == 20bd331b → `git archive HEAD` 到 scratch
     （`/tmp/p3patch.XXXXXX`）→ 0001→0008 严格顺序 `git apply --check` + `git apply`
     → 重建树与 vendor 工作树全树 diff（仅排除 `.git/__pycache__/*.pyc/
     .pytest_cache/*.egg-info`）→ 任意失败 exit 15；EXIT trap + 范围校验清理；
   - 提交前 CPU 门禁测试 5 项全过（正确状态 PASS / 改 1 字节 exit 15 并恢复 /
     0008 未应用 exit 15 / 0008 缺失 exit 15 / 缓存存在仍 PASS / 每次 PASS 与 FAIL
     后无 `/tmp/p3patch.*` 残留），另做真 wrapper 端到端测试（补丁门禁通过后按设计
     在 retriever 门禁失败）。

2. **阶段一启动**（六卡 1 步工程 smoke）：
   - run ID `p3-grpo-v1-eng-smoke-fsdp6-b66-n5-s0-20260819a`
   - GPUs 1,2,3,4,6,7（GPU0/GPU5 禁），tmux + `run_managed.sh`
   - 全部 wrapper 门禁通过（upstream pin、CPU 内存、GPU 集合、**新补丁门禁**、
     retriever health 21,015,324 向量 / limit 64）
   - **未进入训练**：`make_envs` 处配置加载失败，run exit 1

## 2. 缺陷（首次真实配置加载暴露）

### 2.1 错误

```
omegaconf.errors.ConfigAttributeError: Key 'search_aware_step_reward' is not in struct
    full_key: env.search.search_aware_step_reward
    object_type=dict
```

位置：`SearchMultiProcessEnv.__init__` ← `build_search_envs` ← `make_envs`（Ray
TaskRunner，pid 4105633）。完整 traceback 见
`runs/p3-grpo-v1-eng-smoke-fsdp6-b66-n5-s0-20260819a/stderr.log`。

### 2.2 根因

patch 0008 传播代码 `agent_system/environments/env_package/search/envs.py` L80-88：

```python
search_cfg  = env_config.search              # config.env.search —— hydra struct 节点
...
for idx in range(self.batch_size):
    cfg_i = deepcopy(search_cfg)             # struct 的 deepcopy 仍带 struct 标志
    cfg_i.search_url = search_urls[idx % n_clients]   # 已存在键，OK
    cfg_i.search_aware_step_reward = bool(getattr(env_config, "search_aware_step_reward", False))
    self.envs.append(SearchEnv(cfg_i))
```

`env.search` 节点 schema（`verl/trainer/config/ppo_trainer.yaml`）只有
`log_requests / search_url / topk / timeout`，且为 struct 模式；向不存在的键
`search_aware_step_reward` 写入即抛 ConfigAttributeError。顶层开关由 override
`+env.search_aware_step_reward=true` 提供（`+` 允许向 struct 加键，加载期不报错），
读取侧正常；**失败仅在向嵌套 `env.search` 节点写入传播值时**。

### 2.3 旁证（已核对）

- 全部其他 v1 flag 访问均只读且安全：
  - `env.py:32` `bool(getattr(env_config, "search_aware_step_reward", False))`
  - `main_ppo.py:169` `config.reward_model.get("search_aware_step_reward", False)`
  - `main_ppo.py:182/186` 构造器 kwargs；`episode.py:42` 构造器参数
  - **`envs.py:87` 是唯一一处 struct 写入**
- CPU 测试缺口：`tests/test_v1_env_question_passthrough.py` 用手工
  `DictConfig({...})` 构造 env_config，非 struct 模式，写入不报错；真实 hydra
  struct 路径从未被执行（门禁缺陷拦停了所有 GPU 启动）。

## 3. 修复选项（待决策，均超出"门禁修复"已批范围）

| 选项 | 内容 | 变更面 | 配置指纹 |
|---|---|---|---|
| A（推荐） | 代码侧：`with open_dict(cfg_i):` 包裹 L87 写入（import 加 `open_dict`，envs.py 已 import omegaconf） | 新 patch 0009 | 不变 |
| B | schema 侧：`ppo_trainer.yaml` `env.search` 加默认键 `search_aware_step_reward: false` | 新 patch 0009 | 不变 |
| C | wrapper overrides 加 `+env.search.search_aware_step_reward=false` 预声明键 | 仅 wrapper | 变（与"配置不变"冲突） |

无论选哪种，建议补 CPU 测试：用真实 `ppo_trainer.yaml` + wrapper overrides 经
hydra compose 后构造 `SearchMultiProcessEnv`，验证 struct 路径写入不报错，堵住
同类测试缺口。

## 4. 资源状态（清理完整）

- run `exit_code=1`（metadata.env），`finished_at` 已记录；
- 无残留进程（verl / ray / vllm / main_ppo 均无），无 `/tmp/p3r.*`；
- 六卡回 18MiB 基线，GPU0 387MiB 桌面不受影响；
- run 目录完整保留（stdout.log / stderr.log / metadata.env / hydra / ray 归档）
  作为证据；tmux 会话 `p3-p3-grpo-v1-eng-smoke-fsdp6-b66-n5-s0-20260819a`
  已结束（remain-on-exit 保留输出）。

## 5. 后续路径

1. 用户选定修复选项（A/B/C）→ 按选项实现（A/B 走 patch 0009 流程：补丁、
   README、CPU 测试、门禁测试、提交推送）；
2. 重新启动阶段一（全新 run ID，eng-smoke，1 步，配置与批准一致）；
3. 阶段一 PASS 后按既有授权自动接阶段二（10 步行为 smoke，从 Base 新启动）。
