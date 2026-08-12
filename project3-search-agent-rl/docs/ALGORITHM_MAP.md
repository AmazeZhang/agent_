# Reward到Loss代码地图

## 主链路

```text
SearchEnv._get_reward
  → rollout中的step/episode rewards
  → ray_trainer.compute_advantage
  → GRPO或GiGPO advantage tensor
  → core_algos.compute_policy_loss
  → entropy bonus / reference KL penalty
  → loss.backward
```

## GRPO

同一个问题采样一组轨迹，按组内Reward计算：

```text
A_i = (R_i - group_mean) / (group_std + epsilon)
```

如果一组轨迹全对或全错，Advantage接近零，无法提供有效相对学习信号。这是Dynamic
Sampling和课程学习的主要动机之一。

## GiGPO

```text
A(i,t) = A_episode(i) + step_advantage_w × A_step(i,t)
```

Episode Advantage衡量完整搜索轨迹；Step Advantage比较相同或相似Anchor State下不同
动作的后续Return。Search文本很少完全一致，因此Similarity Threshold与State表示是项目
三最重要的算法切入点。

## Policy Loss、KL与Entropy

```text
ratio = exp(current_log_prob - old_log_prob)
policy_loss = -min(ratio × A, clip(ratio) × A)
total_loss = policy_loss + kl_coef × reference_kl - entropy_coef × entropy
```

- PPO近似KL：Current Policy与Old Policy的变化，用于监控和Clip；
- Reference KL：Current Policy与Reference Model的距离，用于防止策略漂移；
- Entropy：策略分布的不确定性，奖励Entropy可鼓励探索，但过大导致随机搜索；
- Retrieved Token必须从Policy Loss中Mask，只更新模型自己生成的Reasoning、Query和Answer。

## DPO边界

DPO输入离线`chosen/rejected`偏好对，直接优化相对对数概率，不显式计算环境Reward和
Advantage。因此DPO可以作为搜索格式或优质轨迹Warm Start，但不代替本项目的在线
GRPO/GiGPO主实验。
