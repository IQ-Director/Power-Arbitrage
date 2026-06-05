> **赛事链接**：[第四届世界科学智能大赛：电力市场交易赛道：储能电站收益优化](https://competition.ai4s.com.cn/p/9?channelCode=dw)

# v4 — 监督模仿学习（Imitation Learning）

## 概述

在 v3 全部功能基础上，新增**监督模仿学习**管线：跳过电价预测，直接输出充放电决策。模型学习买入/卖出概率分布，用 Policy Gradient 最大化 Oracle 期望收益。两条管线可独立运行。

## 相比 v3 的新增内容

### 模仿学习模型

```
特征 (B, 96, 107) → ConvBiLSTM backbone (243K, 缩减版)
  → charge_head:   Linear(2H→1) on ts 0..80 → u[tc] → softmax(-u) → P_charge (81维买入概率)
  → discharge_head: Linear(2H→1) on ts 8..89 → v[td] → softmax(v)  → P_discharge (81维卖出概率)
  → P_joint = P_charge^T × P_discharge → mask(td≥tc+8) → argmax → (tc, td) → power schedule
```

**为什么不是真正的 RL**：训练时 Oracle 已知——真实电价 + 枚举即得最优 (tc, td)。没有未知状态、没有探索必要。这是 Behavioral Cloning + Policy Gradient，不是 RL。真正的 RL 需要从环境反馈中学习，而我们直接有正确答案。

**架构要点**：
- 两个独立 Linear 头（514 参数），不直接输出 3321 维——利润天然分解为"放电收益 − 充电成本"
- 独立 softmax 优于 joint softmax：边际约束防止概率全压在单一组合上
- 小模型(243K)优于大模型(824K)：PG 对参数敏感，少参数→少过拟合→策略更稳

### 损失函数：Policy Gradient + 熵正则

```python
P_joint = P_charge^T × P_discharge            # 联合概率
E_profit = Σ P_joint × Oracle_profit          # 期望收益
H = -Σ P_charge log P_charge - Σ P_discharge log P_discharge  # 边际熵

Loss = -E_profit - 0.5 × H    # 最大化收益 + 防坍缩
```

梯度天然为优势函数：高于期望的动作概率上升，低于期望的下降。所有 3321 个动作的收益均已知——Oracle 用真实价格预计算。

**为什么 PG 而不是 SoftCE**：SoftCE 模仿 Oracle 的完整 Q 分布（52% profit ratio），PG 只关心"比平均好"（63%）。SoftCE 浪费容量学习坏动作间的差异。

### 推理与策略

```
价格管线: 特征 → 96个电价 → enumerate 3321对 → argmax → power → output.csv
模仿管线: 特征 → P_charge + P_discharge → 外积 → argmax → power → output.csv
                                 ↑ 决策在模型内部一次完成
```

### 实验迭代

| 版本 | 配置 | Train Profit比 |
|------|------|---------------|
| 大模型(824K) + 浅头 + 独立softmax | PG | 60.0% |
| 大模型 + 深头(MLP) + gap_bias + joint | PG | 59.2% |
| SoftCE 监督模仿 (τ=3.0) | KL | 52.0% |
| **小模型(243K) + 浅头 + 独立softmax + PG** | **PG** | **63.0%** |

关键发现：小模型更好、独立softmax > joint、PG > SoftCE、对比预训练在 PG 中反而有害。

## 运行

```bash
# 价格预测管线（v3 主力方案）
python main.py --mode final --model bilstm --contrastive

# 模仿学习管线（v4 新增）
python main.py --mode final --model rl

# 其他模型
python main.py --mode train --model transformer --contrastive
```

## 特征（107 维，同 v3）

移除全部价格特征，仅依赖气象与基本面。PDP 拐点分箱 + 排列重要性迭代校准。

## 模型（价格预测）

ConvBiLSTM 单头（~823K）+ 可选 FeatureAttention/TemporalAttention + 对比预训练。支持 MoE/WTA/Router/HardRoute/Transformer 变体。

## 文件结构（v4 新增/修改标记 ★）

```
├── main.py ★                   # + --model rl 分支
├── src/
│   ├── config.py ★             # + RL_HIDDEN_DIM/RL_ENTROPY_WEIGHT/RL_MODEL_PATH
│   ├── models/
│   │   ├── model_rl.py ★       # ConvBiLSTM_RL（模仿学习模型）
│   │   ├── model.py
│   │   ├── model_moe.py
│   │   ├── model_hardroute.py
│   │   ├── model_transformer.py
│   │   ├── train.py ★          # + compute_q_labels/pg_loss/validate_rl/_train_rl_impl
│   │   ├── predict.py ★        # + predict_rl
│   │   └── ...
│   └── strategy/
│       └── optimizer.py ★      # + generate_strategy_from_decisions
└── CHANGELOG.md ★              # 2026-05-27 完整实验记录
```
