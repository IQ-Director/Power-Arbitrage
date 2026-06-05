# Changelog

## 2026-05-27

### 监督模仿学习（Imitation Learning）：跳过电价预测，直接输出充放电决策

#### 动机：为什么不做真正的 RL

讨论后明确了三个事实：

1. **这不是 RL，是模仿学习（Behavioral Cloning + Policy Gradient）**。真正 RL 需要从环境反馈中探索未知策略，而我们的场景中 Oracle（最优动作）极易获取——训练时有真实电价，枚举 3321 个 (tc,td) 组合即得每个动作的精确收益。没有未知状态、没有分布外泛化、没有探索的必要。

2. **逐步 RL（每 15 分钟一个决策）不可行**。一天 96 步中真正自由的决策点只有两个——什么时候充电（tc）和什么时候放电（td）。其余 94 步要么在强制充电（不可中断）、要么在等待、要么已完成交易。逐步决策不增加自由度，反而引入延迟奖励（44 步后才知盈亏）和状态机约束的复杂性。

3. **日级决策是这个问题的最优粒度**。以"天"为决策单元，模型输入 `(96, F)` 特征，输出充放电时间点的概率分布，一次性完成决策。

#### 架构设计：两个独立输出头 + 因子化概率

利润天然可分解：`Q(tc, td) = 放电收益[td] - 充电成本[tc]`。不直接预测 3321 维联合概率（参数爆炸且丢失时间步精细信息），而是用两个独立的小头分别预测买入和卖出概率：

```
ConvBiLSTM backbone (243K) → (B, 96, 2H)

  ├─ charge_head:   Linear(2H→1) on timesteps 0..80  → u[tc] → softmax(-u) → P_charge[tc]  (买入概率, 81维)
  └─ discharge_head: Linear(2H→1) on timesteps 8..89 → v[td] → softmax(v) → P_discharge[td] (卖出概率, 81维)

P_joint[tc, td] = P_charge[tc] × P_discharge[td_idx]   (81×81 外积)
mask: 上三角 (td ≥ tc + 8)，屏蔽非法组合
argmax → (tc, td) → power schedule → output.csv
```

**为什么用两个独立头而不是一个联合头**：

- 参数效率：514（2×Linear(256→1)）vs ~850K（Linear(256→3321)）
- 归纳偏置：利润天然是加法——不是"(10,50)这个组合值钱"，而是"10点充电便宜 + 50点放电贵"加在一起值钱。因式分解把 3321 个组合压缩成 162 个基础量
- BiLSTM 特征对齐：双向 LSTM 在 tc 位置的隐状态已包含整天上下文，charge_head 直接取 `h[tc]` 就能学到"在这个时间点充电的总成本"

**为什么独立 softmax 优于 joint softmax**：

独立 softmax（P_charge 和 P_discharge 分别归一化）让两个边际分布各自为 1——模型必须明确选择"哪段时间整体适合充电、哪段时间整体适合放电"。Joint softmax（整个 81×81 矩阵一起归一化）更容易坍缩——模型可以直接把概率全压在某个单一 (tc,td) 对上。实验也证实了这一点。

#### 损失函数：Policy Gradient + 熵正则

```python
P_charge = softmax(-u)          # 买入概率
P_discharge = softmax(v)        # 卖出概率
P_joint = P_charge^T × P_discharge   # 联合概率

E_profit = Σ P_joint[tc,td] × Oracle_profit[tc,td]   # 期望收益
H = -(P_charge log P_charge) - (P_discharge log P_discharge)  # 熵正则

Loss = -E_profit - 0.5 × H     # 最大化收益 + 防止坍缩
```

梯度天然是优势函数形式：`∂L/∂logit_i = P_i × (E[R] - R_i)`。收益高于当前策略期望的动作概率上升，低于期望的下降。所有 3321 个动作的 Oracle 收益均已知，无需采样。

**为什么 PG 而不是 SoftCE**：SoftCE（KL 散度模仿 Oracle Q 分布）在实验中被 PG 完胜（52% vs 63%）。SoftCE 强迫模型匹配 Oracle 的完整分布形状，但 Oracle 分布中 80% 的动作都是"坏的"——模型浪费大量容量学习坏动作之间的差异。PG 只关心"哪些动作比平均好"——信号更聚焦。

#### 实验迭代

| 版本 | 配置 | Train Profit比 | 问题 |
|------|------|---------------|------|
| v1 | 大模型(824K) + 浅头 + 独立softmax | 60.0% | TC仅5种值，TD有11种 |
| v2 | 大模型 + 深头(MLP) + gap_bias + joint softmax + 对比预训练 | 59.2% | TC坍缩(std=0) |
| v3 | SoftCE 监督模仿 (τ=3.0) + 同上 | 52.0% | TC+TD全部坍缩 |
| **v4** | **小模型(243K) + 浅头 + 独立softmax + PG** | **63.0%** | TD std=12.4 ✓ |

**关键发现**：

- 小模型(243K)反而优于大模型(824K)——PG 对参数敏感，少参数 → 少过拟合 → 策略更稳
- 独立 softmax > joint softmax：因子化边际分布比联合矩阵归一化更抗坍缩
- PG (63%) > SoftCE (52%)：直接优化收益优于模仿 Oracle 分布形状
- 对比预训练在 PG 中反而有害（v2 的 TC 坍缩比 v1 更严重）——自监督表征学习学到的特征分布与 PG 的优化目标可能不兼容
- 深头(MLP) + gap_bias 也没有帮助——额外参数增加了优化难度但没带来新的信息
- 全量训练监控 Profit 比 59%，TC std≈0（跨季节坍缩），但线上测试可能优于本地——模型直接优化决策，不依赖价格预测精度
- TC 头始终趋向集中（最优充电时间受负荷和光伏主导，日间变化确实较小），TD 头有更好的多样性（放电时机更依赖当日天气和价格条件）

#### 与价格预测管线对比

```
价格预测: 特征 → 96个电价 → 枚举3321对 → 算收益 → argmax → power
模仿学习: 特征 → 81个买入概率 + 81个卖出概率 → 外积 → argmax → power
                                 ↑ 跳过电价，直接学到"哪个时段值钱"
```

核心区别不是输出维度（96 vs 162），而是**优化目标**。价格预测的梯度 80% 花在"让 96 个点都准"——但 80% 的时间点的价格误差对策略收益没影响。模仿学习的梯度 100% 花在"让最优动作排第一"。


#### 决策策略修改

模仿学习模型不产价格预测，推理和策略生成流程与价格预测管线完全不同：

**推理阶段（predict_rl）**：

```
特征(96,F) → backbone → u[tc](81维) + v[td](81维)
  → P_charge = softmax(-u)   # 买入概率分布
  → P_discharge = softmax(v)  # 卖出概率分布
  → P_joint = P_charge^T × P_discharge   # 81×81 联合概率矩阵
  → mask（td < tc+8 置零）
  → argmax(P_joint) → (tc, td)
  → power[tc:tc+8] = -1000, power[td:td+8] = +1000
```

不是"预测价格 → 枚举 → 算收益 → argmax"，而是"特征 → 概率 → argmax"。决策在模型内部一次完成，无需外部枚举 3321 对。

**策略生成（generate_strategy_from_decisions）**：

```
价格管线: df_price(96维价格) → optimize_day(枚举3321对 × 8步窗口求和) → power schedule → output.csv
模仿学习: power schedule(96维功率) → 直接格式化 → output.csv
                                 ↑ 跳过整个optimize_day
```

`generate_strategy_from_decisions` 直接把 power schedule 写入提交 CSV，不需要：
- 算 3321 个组合的收益（每对要加 8 步价格）
- tc/td 合法性检查（模型输出已保证 td ≥ tc+8）
- profit > 0 判断（模型选择最优组合，利润可能为负值则输出为0）

**main.py 分支差异**：

```python
# 价格预测管线（Step 4 + Step 5）
predictions = model(x_test)                          # → (B, 96) 电价
df_price = predictions_to_dataframe(predictions)     # → 写入 price_predictions.csv
generate_full_strategy(df_price)                     # → 枚举 → output.csv

# 模仿学习管线（Step 4 + Step 5）
power, tc_arr, td_arr = predict_rl(x_test)           # → (B, 96) 功率 + 决策统计
generate_strategy_from_decisions(power)              # → 直接写入 output.csv
```

模仿学习管线中 `predictions` 为 `None`，Step 4 不生成 `price_predictions.csv`，Step 5 直接使用 power schedule。


**新增文件**：`src/models/model_rl.py`
**修改文件**：`src/models/train.py`（+6 函数）、`src/models/predict.py`（+predict_rl）、`src/strategy/optimizer.py`（+generate_strategy_from_decisions）、`src/config.py`（+5 配置）、`main.py`（+`--model rl`）

---

## 2026-05-26

### 注意力机制实验

**实验设置**：ConvBiLSTM 单头 + Trend-Aware + 时间点级对比预训练 + 107 维特征，`--mode train`（前 29 天验证）

**方案1：SENet 特征注意力**
- 在 `input_norm` 之后插入 `FeatureAttention`：GlobalAvgPool→FC(64→16)→ReLU→FC(16→64)→Sigmoid 逐特征加权
- 参数增量：~2K
- 结果：Val RMSE 1.1714（基线 1.1547），+1.4%
- **结论**：轻微有害。当前 107 维特征已经过多轮排列重要性筛选，学习特征权重只是增加噪声。

**方案2：时序自注意力**
- 在 Conv 和 BiLSTM 之间插入 `MultiheadAttention(embed_dim=128, heads=8)` + 残差 + LayerNorm
- 参数增量：~65K
- 结果：Val RMSE 1.2012（同时开启三个方案时），+4.0%
- 单独对比（方案1+3组合实验）：整体 RMSE 从 1.15 升至 1.20，方案2 是主要退化来源
- **结论**：明确有害。334 天数据不足支撑自注意力学习跨时间步依赖，严重过拟合。

**方案4：对比预训练类型④（价格峰谷正样本对）**
- 同一天内，价格 Top-25% 时间点两两互为正样本（峰值），Bottom-25% 同理（谷值）
- 权重 0.25，作为类型①②③的补充
- 结果：Val RMSE 1.1543 vs 基线 1.1547，差异仅 -0.03%
- **结论**：中性。峰谷时间点之间表征已经足够相似（类型③ 相邻步已覆盖），额外加正样本对不提供新信息。
- 配置开关 `CONTRASTIVE_USE_TYPE4 = True` 保留了该功能，全量训练监控 RMSE 0.7534

**整体结论**：三个方案在当前 334 天数据量下均未超越基线。特征工程（排列重要性筛选 + PDP 分箱）和对比预训练已经逼近数据量能支撑的性能上限，进一步的架构改进需要更多数据。

### 对比预训练类型④ 全量训练

- 配置：`USE_FEATURE_ATTENTION=False`、`USE_TEMPORAL_ATTENTION=False`、`CONTRASTIVE_USE_TYPE4=True`
- 全量训练监控 RMSE：**0.7534**（epoch 9）
- 类型④ 峰谷正样本对保留在对比预训练管线中（虽单次实验为中性，但不增加推理成本）
- 新增配置：`USE_FEATURE_ATTENTION`、`FEATURE_ATTN_REDUCTION`、`USE_TEMPORAL_ATTENTION`、`TEMPORAL_ATTN_HEADS`、`TEMPORAL_ATTN_DROPOUT`、`CONTRASTIVE_USE_TYPE4`
- 模型更新：`ConvBiLSTM` 和 `ConvBiLSTM_Contrastive` 均支持可选注意力模块（通过 config 开关控制）
- 文件：`src/models/model.py`（新增 `FeatureAttention` 类）、`src/models/train.py`、`src/config.py`、`main.py`、`src/models/predict.py`、`analyze_features.py`、`src/models/pdp_analysis.py`、`src/models/shap_torch.py`

### Transformer 替换 BiLSTM

新建 `ConvTransformer` 模型：Conv1d 前端 + PositionalEncoding + TransformerEncoder 后端，替换 BiLSTM。配置开关 `USE_TRANSFORMER`，命令行 `--model transformer`。

**架构**：`input → input_proj(64) → Conv1d×2 → conv_proj(128) → tf_proj(d_model) → PosEnc → TransformerEncoder(N, d_model, heads) → +residual → output`

**大版本**（N=4, d_model=256, heads=8, ff=512, dropout=0.2）：
- 参数量：2,306K（BiLSTM 的 2.8 倍）
- Train 模式：Val RMSE **1.0934**（BiLSTM 1.1547，+5.3%）
- Final 模式：监控 RMSE **0.812**（BiLSTM 0.753，-7.8%）
- 训练极不稳定，RMSE 在 1.09–1.35 间剧烈震荡
- 评估：Train 模式表现最好但 Final 模式最差 → 典型的大模型小数据跨季节过拟合

**小版本**（N=2, d_model=128, heads=4, ff=256, dropout=0.3）：
- 参数量：437K（BiLSTM 的 53%）
- Train 模式：Val RMSE **1.1392**（+1.3% vs BiLSTM）
- Final 模式：监控 RMSE **0.798**（BiLSTM 0.753，-6.0%）
- 训练更稳定，但全量训练仍不如 BiLSTM

**对比预训练**：两种 Transformer 的对比预训练 Loss 均优于 BiLSTM（5.48 vs 6.40 起始，5.48 vs 5.48 终值），说明 Transformer 在无监督表征学习上收敛更快。

**结论**：
- Transformer 在训练分布内（1月验证集）有优势，但在跨季节泛化（12月监控集）上不如 BiLSTM
- BiLSTM 的循环结构对季节偏移更稳健——LSTM 的隐状态能更好地处理分布变化
- 334 天数据不足以训练稳定的大 Transformer；即使缩小到 437K（小于 BiLSTM），泛化仍不如
- **BiLSTM 仍是当前最佳选择**

**新增文件**：`src/models/model_transformer.py`（`ConvTransformer`、`ConvTransformer_Contrastive`、`PositionalEncoding`）
**修改文件**：`src/models/train.py`（新增 `train_model_transformer`、`train_final_transformer`、`contrastive_pretrain_transformer`）、`src/models/predict.py`（新增 `predict_transformer`）、`src/config.py`（新增 6 个 Transformer 配置）、`main.py`（新增 `--model transformer`）
**新增配置**：`USE_TRANSFORMER`、`TRANSFORMER_NUM_LAYERS`、`TRANSFORMER_HEADS`、`TRANSFORMER_DIM`、`TRANSFORMER_FF_DIM`、`TRANSFORMER_DROPOUT`

---

## 2026-05-17

### I. 移除价格特征

- 价格相关特征在测试时全部填 0，训练/测试存在分布偏移
- 移除 10 个价格特征：price_lag_d1/d7、price_roll_mean/std_4/96、price_diff_1、price_lag24_diff
- 移除 4 个 SHAP 价格分箱：is_vs_yesterday_up/big_up、is_current_rising、is_near_zero_price
- 移除 `_add_zero_placeholder` 函数（不再需要测试集填 0）
- 清理 assemble_dataset.py 中所有价格相关的 import 和调用
- 特征维度：118 → 107
- 影响：本地监控 RMSE 从 0.76 升至 0.99（训练失去价格信号），但训练/测试分布一致，泛化预期更好
- 文件：`src/features/assemble_dataset.py`

### II. 分箱阈值第三次校准（107 维无价格特征）

- 重新运行排列重要性（ConvBiLSTM 单头 + 对比预训练权重，106 维）
- 排列重要性 Top-6：v100_std(25.0%)、v100_p90(14.6%)、v100_max(11.7%)、v100_p10(8.5%)、v100_min(8.5%)、v100_mean(6.5%)
- wind_speed 衍生（max/std/mean）首次进入 Top-10，无价格特征后 NWP 风特征完全主导
- PDP 拐点更新：
  - `is_high_pressure`: 1.89 → **2.05**（sp_mean 拐点上移）
  - `is_high_solar`: 1.56 → **3.05**（ghi_p90 拐点大幅右移）
  - `is_high_eff_ghi`: **移除**（Δ=0.009，无拐点）
  - `is_high_hdd`: **新增**（HDD > 25.80，Δ=0.095）
  - `is_high_net_load`: **新增**（net_load > median，Δ=0.017，单调无拐点但持续有效）
  - `is_low_pressure`、`is_extreme_pressure`：确认不变
- 分箱特征 5→6，总特征 106→107
- 文件：`src/features/feature_engineering.py`、`analyze_features.py`

### III. SimCLR 对比预训练（天级）

- 在预测训练前增加无标签对比预训练阶段
- 模型：`ConvBiLSTM_Contrastive` = ConvBiLSTM 骨干 + 投影头 MLP(256→128→128)
- 正样本对：同一天 + 两种高斯噪声（σ=0.15），两次前向得到两个视图
- 负样本对：batch 内其他所有天（等权推开）
- InfoNCE 温度 τ=0.7，预训练 30 轮，LR=1e-3
- 微调：加载预训练骨干权重 → Trend-Aware full train
- 监控 RMSE：0.81 → **0.76**（+6%）
- 新配置：`CONTRASTIVE_TEMPERATURE`、`CONTRASTIVE_AUG_STD`、`CONTRASTIVE_EPOCHS`、`CONTRASTIVE_EMBED_DIM`
- 用法：`python main.py --mode final --model bilstm --contrastive`
- 文件：`src/models/train.py`（ConvBiLSTM_Contrastive、info_nce_loss、contrastive_pretrain）、`main.py`

### IV. Regime-Weighted 对比预训练

- 问题：SimCLR 平等推开所有负样本——1 月高压天和 12 月高压天被当负样本推开，但 regime 相似的天表征不应太远
- 方案：用 regime 特征（sp_mean/HDD 等 7 维日均值）计算天级余弦相似度
- 负样本权重 = 0.3 + 0.7×(1-regime_sim)：regime 相似的天降低推开力度
- 预训练 Loss 3.40→**2.20**（负样本约束更合理）
- 监控 RMSE：0.76（与纯 SimCLR 持平）
- 文件：`src/models/train.py`（regime_weighted_nt_xent、_compute_regime_similarity）

### V. 时间点级对比预训练

- 核心洞察：预测任务是 96 个时间点，但对比预训练把整天压成 1 个向量——两个任务粒度不匹配
- 改为逐时间点表征：每个 15 分钟步一个 128 维向量，LSTM 输出不做 pool
- 样本量：363 天 → 34,848 个时间点；每 batch 负样本 ~30 → ~3070（+100×）
- 正样本对三种（时间点级）：
  - ① **同时刻+双视图**（w=1.0）：同一步的两个噪声版本——主力
  - ② **同时刻+相邻天**（w=0.5）：D1 08:00 ↔ D2 08:00——日间周期
  - ③ **同天+相邻步**（w=0.2）：08:00 ↔ 08:15——短期平滑
- 每个时间点 ~4 个正样本 vs ~3070 个负样本，正负比 1:750
- 损失使用 (2N, 2N) 掩码矩阵一次性计算所有正负对关系
- 预训练 Loss 从 2.20→5.53（负样本暴增，任务变难），监控 RMSE **0.749**（新最佳）
- 文件：`src/models/train.py`（ConvBiLSTM_Contrastive.forward 改为输出 (B,96,D)、timepoint_nt_xent）

## 2026-05-16

### 线上反馈：Trend-Aware > Huber

- 本地回测 Huber 161k > Trend 131k，但线上提交后 Trend 明显更好
- 说明 Trend-Aware 的方向一致性 + 窗口重叠约束在测试集分布上泛化更强
- 本地验证集（1月）与测试集（12月）的季节性差异导致 Huber 过拟合本地模式
- **当前线上最佳**：单头 ConvBiLSTM + Trend-Aware + 新分箱阈值

### 损失函数系统对比（单头 + 新分箱）

| 损失 | 本地RMSE | 本地收益 | 线上 |
|------|----------|----------|------|
| Huber | 1.22 | 161k | — |
| TopK-Weighted | 1.24 | 140k | — |
| Trend-Aware | 1.23 | 131k | **最佳** |

- Huber 本地最优但线上不如 Trend——本地过拟合，Trend 的方向约束限制了这种过拟合
- TopK 关注极端价格点，但极端点的模式在验证/测试集间不一致
- Trend 的方向+窗口约束是一种更通用的归纳偏置，跨季节迁移更好

### 架构简化：移除 Gate 回到单头

- 放弃 MoE/WTA/Router 所有多专家方案，回到纯 ConvBiLSTM 单头
- 原因：多专家的增益（~14%）来自隐式集成降低方差，不是真正的条件路由
- 当单头+更好特征就能超过 MoE 时，多专家的额外复杂度不值得
- 单头 824K 参数 vs MoE 924K，训练更稳定，无 Gate 坍塌风险

### 分箱阈值更新（第二次 PDP 分析）

- 重新运行排列重要性 + PDP 拐点分析（MoE 模型，118 维）
- 排列重要性 Top-5 全为 NWP 风速/温度：t2m_min(15.1%)、v100_max(14.6%)、v100_mean(14.2%)、v100_min(11.6%)、v100_std(11.3%)
- PDP 拐点新发现：ghi_p90@1.56(Δ=0.083)、eff_ghi_mean@1.57(Δ=0.026)、sp_min@-5.81(Δ=0.029)
- **关键洞察**：风速决定重要性，气压/辐照决定方向——排列重要性说 v100 最重要，PDP 说 ghi/sp 边际效应最大，两者互补
- 更新：
  - `is_high_pressure`: 1.24 → **1.89**（拐点上移 52%）
  - 新增 `is_low_pressure`: sp_min < **-5.81**
  - 新增 `is_high_solar`: ghi_p90 > **1.56**
  - 新增 `is_high_eff_ghi`: eff_ghi_mean > **1.57**
  - 移除 `is_high_pressure_max`（sp_max 无明确拐点）
  - 保留 `is_extreme_pressure`（sp_p90 无拐点但 Δ=0.080）
- NWP 分箱特征 3→5，总特征 116→118
- 文件：`src/features/feature_engineering.py:add_nwp_pdp_bins()`

### Router 分流模块实验 ⭐

- 按用户要求设计：分箱特征聚类打标 → Router 分类器 → 硬分配到对应头
- 预训练 Router 准确率仅 49%（regime 特征无法可靠预测分箱聚类标签）
- 改用 Gumbel-Softmax straight-through 让预测损失回传 → 仍坍缩
- 回测收益 151k，与 HardRoute 持平，未超过 Soft MoE(157k)
- **失败根因**：分箱特征和 regime 特征之间存在本质信息差异——分箱特征包含价格自身信息（is_vs_yesterday_up 等），而 Router 训练和推理时看不到这些。聚类标签中的信息 Router 无法学到。

### 聚类伪标签实验

- KMeans 给训练集天打 cluster 标签 → Gate 加交叉熵损失
- CE 权重 0.3：收益 115k（CE 干扰 Huber）
- CE 权重 0.05：RMSE 1.13，但 E1 仍死亡（0.02）
- **失败根因**：CE 告诉 Gate "这是什么 cluster"，但没说"不同 cluster 应该用不同专家"。KMeans 聚出的 cluster 与最优预测策略之间没有必然联系。

## 2026-05-16

### MoE-WTA（赢者通吃）实验

- 尝试方案 1（TODO 推荐首选）：每个样本只更新 Gate 选中的专家，其余冻结
- 共 4 轮迭代均未能稳定 Gate：
  - v1：基础 WTA (tau=0.5) → E0 死亡，E2 垄断 28/29 天
  - v2：同权初始化 + tau=2.0 → 失去探索，E0 垄断全部 29 天
  - v3：软启动预热 8 轮（软 MoE → WTA）→ Gate 在 E0/E1/E2 间振荡后崩溃
  - v4：epsilon 平滑 (eps=0.1) + tau=1.0 + 负载均衡 0.5 → 同 v3
- 最佳 epoch：RMSE 1.12（模型最佳），回测收益 139k
- **失败根因**：鸡生蛋蛋生鸡——专家需要差异化才能让 Gate 学到有意义路由，但差异化需要 Gate 先把不同样本分给不同专家。334 天的数据不足以打破这个循环。每个专家只能从 ~111 天学预测，样本量不足导致专家间方差大、训练不稳定。
- 新增配置：`WTA_TEMPERATURE`、`WTA_BALANCE_WEIGHT`、`WTA_WARMUP_EPOCHS`、`WTA_EPS`
- 文件：`src/models/model_moe.py`、`src/models/train.py`、`src/models/predict.py`、`src/config.py`、`main.py`

### MoE 专家退化问题的本质（总结）

经过 WTA、伪标签、Router 三种方案的尝试，核心问题已明确：
1. **Huber 不奖励分化**：所有梯度指向"预测更准"，专家可以学成一样，Gate 无信号区分
2. **小数据乘数效应**：334 天/3 专家 = 每个 ~111 天，不足以学到稳定的特定 regime 模式
3. **缺少分类标签**：没有标准的分类标签来分别训练不同的专家，只能依赖Gate进行分类，而Gate无法有效学习分类标准
3. **RNN 内化时序**：BiLSTM 已经把跨天时序模式内化了，专家头之间没有真正的分工空间
4. **MoE 的 14% 增益**：来自隐式集成（多个头投票降低方差），不是条件路由。
5. **分箱特征才是真正的杠杆**：新阈值让单头收益从 138k→161k(+17%)，超过了 MoE 的 157k。特征工程的收益比架构复杂度更可靠

## 2026-05-15

### NWP 时区回退测试

- 尝试将 NWP 当作北京时间（去除 period+32 对齐），验证"时区修正是否真的必要"
- 回测收益骤降至 72k（对齐版本 150k），证实时区修正不可或缺
- **结论**：恢复 UTC→北京时间对齐，确认 `_merge_nwp()` 中 period+32 修正有效
- 文件：`src/features/assemble_dataset.py:_merge_nwp()`

## 2026-05-14

### HardRoute 全量训练与提交

- 使用 sp_mean 中位数硬分流，2 专家，全量训练后提交
- 验证集表现：RMSE 1.13，收益 151k
- 最终提交收益接近 0——测试集 regime 分布与训练集不匹配（Dec 18 天中 2/16 拆分 vs 训练集 50/50）
- 确认 HardRoute 在分布偏移下失效
- 文件：`src/models/model_hardroute.py`

### MoE（熵正则）提交

- 3 专家 + 熵正则 + 负载均衡，本地回测收益 157k，RMSE 1.19
- Gate 做隐式集成（非条件路由），本地收益最高
- 效果：确实避免了专家垄断和专家死亡，但是依然不能推动专家分化，本质只是扩大了模型参数
- **线上测试：不如单头**——MoE 隐式集成降低方差的效果仅在训练分布内有效，跨季节泛化反而更差
- 文件：`src/models/model_moe.py`

## 2026-05-13

### HardRoute（规则硬分流）

- 不学 Gate，用 sp_mean 中位数直接切割训练集
- 共享 ConvBiLSTM 主干 + 两个独立输出头，根据 sp_mean 日均值 > 中位数路由
- 训练集天然 50/50（167/167 天），验证集 20/9 天
- 无梯度冲突，无坍塌风险
- 效果：不好，人为设定的硬规则不能有效划分数据
- 文件：`src/models/model_hardroute.py`、`src/models/train.py`

### MoE Gate 退化问题分析

- 纯 Huber 下 Gate 退化为单专家垄断（E0=0.96，E1=0.04），3 专家时 E2 始终死亡（0.01）
- 根因：
  - Huber 无动力让专家分化，只能保证最终预测结果的损失降低，但没有有效标签监督不同专家之间的分化训练
  - 低权重专家收不到梯度 → 死亡循环；
  - 数据量小（334 天）
- TODO 方案：WTA / 对比学习辅助 / 输出多样性+硬门控
- 文件：`plans/todos/TODO-MoE专家退化.md`

### MoE 架构

- 共享 ConvBiLSTM 主干 + Gate MLP(7→32→N) + N 个专家头 + 共享头（80/20 融合）
- Regime 特征：sp_mean、sp_max、sp_p90、HDD、t2m_max、renew_ratio、net_load
- Gate 模式：soft / noisy_soft / gumbel_top1 / gumbel_top2
- 理想效果：各个专家分别学习不同类型的电价变化规律（如高压天、低压天）
- 实际效果：无法有效学习
- 参数量：924K（3 专家）
- 文件：`src/models/model_moe.py`、`src/models/train.py`

### 回测方法论修正

- **问题**：曾用预测价格评估策略收益，自我验证循环
- **修正**：策略在真实电价上计算收益（预测驱动决策，真实价格算钱）
- 移除 optimizer.py 中基于预测价格的"总预期收益"打印
- 文件：`src/strategy/optimizer.py`

## 2026-05-12

### SHAP 分箱与 PDP 拐点分析

- 训练 LightGBM → TreeSHAP 计算特征重要性 → 检测 PDP 拐点
- PDP 拐点处设置分箱特征：较大地放大预测差异
- SHAP 分箱：is_vs_yesterday_up/big_up、is_current_rising、is_near_zero_price
- NWP PDP 分箱：is_high_pressure@1.24、is_extreme_pressure、is_high_pressure_max
- ConvBiLSTM 排列重要性：NWP 占 69%，价格仅 7%（与 LightGBM 结果截然相反）
- 文件：`src/models/shap_calibrate.py`、`src/models/shap_torch.py`、`src/models/pdp_analysis.py`、`src/features/feature_engineering.py`

### 特征精简（排列重要性）

- ConvBiLSTM 排列重要性驱动，199 维精简至 116 维
- 去除低效特征：boundary_ratios、boundary_diff、forecast_error、market_features、price_jump 等
- 保留边界条件关键滚动统计（6 列，排列重要性 Top-30）
- 文件：`src/features/feature_engineering.py`、`src/features/assemble_dataset.py`

## 2026-05-11

### 特征扩展

- 参考微信文章[《特征工程指南》](https://mp.weixin.qq.com/s/qlYapVIxAvwajJ5hvXRF1g)和[《SHAP 案例分析》](https://mp.weixin.qq.com/s/1EE95MAJ4NxcKzzz-3xqUg)大幅扩充特征
- 边界条件：滞后（lag 1/2/7 天）、滚动统计（窗口 4/12/24/96）、差分、比率
- 新能源：renew_ratio、net_load 及衍生特征
- NWP：体感温度（CDD/HDD）、变化率、交叉特征（temp×wind、pressure×wind 等）
- 价格：滞后（1/2/3/7/14 天）、滚动、差分、昨日对比、跳变检测
- 时间：cyclical hour/day/month、peak_hour、month_boundary、节假日（2024 年）
- 特征维度：85 → 199
- 文件：`src/features/feature_engineering.py`、`src/config.py`

### 损失函数实验

- TopK-Weighted Huber：每日最高/最低 k% 价格点加权 ×3——波动大，效果不稳定
- Trend-Aware：0.3×方向 + 0.5×窗口 + 0.2×Huber——效果不如纯 Huber
- 均恢复为 Huber
- 文件：`src/models/train.py`、`src/config.py`

## 2026-05-10

### v1 基线恢复

- 从 backup/v1 恢复 ConvBiLSTM 架构：Conv1d(64→128→256) + 2×BiLSTM + residual + LayerNorm
- 824K 参数，Huber Loss
- 保留两项修复：
  1. NWP 时区修正（UTC→北京时间，period+32）
  2. 验证集时序划分（前 29 天 = 验证集，不可随机打乱）
- 回测收益：112k，RMSE 1.59
- 文件：`src/models/model.py`、`src/models/train.py`、`src/features/assemble_dataset.py`

### NWP 时区修正

- NWP 数据 UTC 时间，边界条件/电价为北京时间，存在 8 小时偏差
- `_merge_nwp()` 中 period +32（UTC 00:00 → 北京 08:00），跨天时 date+1
- 文件：`src/features/assemble_dataset.py:_merge_nwp()`

### 验证集切分策略

- 随机切分泄漏未来信息（时序数据）
- 前 29 天（1月3-31日）固定为验证集，剩余 334 天为训练集
- 提交前全量训练：最后 1/20 作为监控
- 文件：`src/models/train.py:temporal_split()`

### 学习率调整

- 初始 LR 较高导致最佳 epoch 过早出现（epoch 1-3）
- 逐步降低至 5e-4，配合 CosineAnnealingLR + 早停 patience=20
- 文件：`src/config.py`、`src/models/train.py`

## v1 

- 来源：`backup/v1/`，项目最早的完整实现
- 模型：ConvBiLSTM（参数量 824K）
  - 输入投影 Linear(85→64) → LayerNorm
  - Conv1d(64→128→256, kernel=5) → 2×BiLSTM(128, bidir=256, 2层)
  - Residual(LayerNorm) → Linear(256→1) → (B,96,1)
- 损失函数：Huber Loss
- 学习率：1e-3，CosineAnnealingLR (T_max=100)，EarlyStopping patience=20
- 特征工程（85 维）：
  - 边界条件预测值 7 列（系统负荷/风光总加/联络线/风电/光伏/水电/非市场化机组）
  - 边界条件实际值 7 列（同上）
  - 边界条件滞后 d1 7 列（前一天同一时刻）
  - NWP 原始特征 ~49 列（7 通道 × 6 统计量 + 衍生）
  - 时间特征 12 列（hour/min/dayofweek/month/dayofyear 的 sin/cos + is_weekend + is_daylight）
  - 价格滞后 4 列（d1/d2/d3/d7）
  - 价格滚动 8 列（窗口 4/8/24/96 的 mean/std）
- NWP 对齐：按 `cumcount()` 分配 period（**无时区修正，存在 UTC/北京 8 小时偏差**）
- 验证集切分：**随机切分**（时间序列，存在未来信息泄漏）
- 策略：枚举 (tc, td)，BLOCK_LEN=8，CHARGE=-1000，DISCHARGE=+1000
- 已知问题：
  - NWP 时区未对齐（UTC vs 北京时间）
  - 随机验证切分泄漏未来信息
  - 学习率偏高导致最佳 epoch 过早
  - 特征维度较少，缺少天气体感、新能源比率、节假日等关键信息
- 文件：`backup/v1/` 下所有文件

---

## v0（2026-05 之前）

### 特征工程探索

**NWP 气象数据处理**：

- 方案一（推荐）：对 104×225 网格计算全局统计量（mean/std/min/max/p10/p90），7 通道 × 6 统计量 = 42 维。额外衍生风速 √(u²+v²)、有效辐照度 ghi×(1−tcc) 等变量各自的统计量，共 154 维。"该处理验证有效，十分推荐"
- 方案二（不推荐）：4×4=16 分区均匀切分，每个分区计算 6 统计量。"效果不显著且维度暴涨，不建议"
- 方案三（酌情参考）：残差 CNN 将 NWP 网格提取为 128 维特征。单独使用效果不好，叠加全局统计量后依然不好。推测"边界条件本身是气象局使用专业模型基于气象数据得到的，普通模型重复该过程可能引入噪声"

**边界条件组合特征**：

- 净负荷 = 系统负荷 − 风光总加。新能源渗透率 = (风电+光伏+水电) / 系统负荷。"有效果，建议使用"
- 尝试用小模型根据边界条件预测值 + NWP 预测实际值，再用预测的实际值做电价预测——"效果不好，不建议"。如果小模型能准确预测实际值且实际值有效，主模型也应能直接从预测值学到映射，且气象局提供的预测值理应更准确，"小模型反而引入噪声"

**StandardScaler 标准化**：

- 所有特征做 z-score 标准化后，价差消失、收益暴跌。"十分不建议"进行减小价差的标准化操作

**时间编码**：

- 用 sin/cos 处理 0→24 小时周期性 + 布尔特征（is_weekend, is_daylight），有显著改进。整数编码无法表达周期性，sin/cos 是更好的选择。"建议使用"

**历史电价滞后**：

- 增加 D-1/D-2/D-3/D-7 滞后 + 4 窗口滚动统计（mean/std），提供价格自回归信息。"不是最关键的维度（删除后收益基本持平）"，大模型帮助有限，特征偏少时有帮助。"酌情考虑"

### 评价指标探索

核心发现：**RMSE 和收益不相关。** 多次实验一致表明 RMSE 最低的模型收益往往不是最高。模型最需要知道的是"最便宜的 2 小时在哪、最贵的 2 小时在哪"，不关心 96 个点的平均精度。预测值全部接近均值（保守预测）时 RMSE 好但价差信号消失，策略无法盈利。

- 训练时将验证集真实收益作为核心指标，RMSE 仅供参考
- "禁止用预测价格自评收益"（本项目的回测方法论修正即基于此经验）
- 建议在训练集中分出一个月的本地测试集，用真实电价计算收益作为评价

### 模型选型探索

**LightGBM**：稳定、下限高，容易达到 4500 分。但"突破 5000 分较为困难"——树模型无法捕获时序依赖、对特征不敏感，很难从特征处理上取得有效突破。"只建议新手入门"

**模型演进路线**：LightGBM → TCN → ConvBiLSTM

**TCN**：RMSE 优于 BiLSTM，但"收益反而更差"。TCN 的局部感受野不如 BiLSTM 的双向全局上下文适合捕捉全天电价分布（本项目的 ConvBiLSTM 最终选择即基于此经验）。

**Transformer**：365 天样本量偏少，容易过拟合（本项目的 Transformer 实验已确认此结论——训练分布内优于 BiLSTM，跨季节泛化更差）。

**Squeeze-and-Excitation**（通道注意力）：本地测试效果更好，但线上测试集得分反而更低（本项目 SENet 实验已确认——+1.4% RMSE，无增益）。

在当前数据集规模下，"不建议继续增加模型参数与复杂度"。

### 损失函数探索

强烈建议引入**排序相关的损失**。表现最优的是：

> 方向一致性 + Top-K + Huber 混合

- Huber：锚定基线
- 方向一致性：惩罚真实涨跌方向错误，而非将所有非最优解平等对待
- Top-K：学习最优解以外时间点的相对优劣

"尤其不建议只使用 MSE"——这正好违背了"RMSE 和收益不相关"的结论。本项目的 Trend-Aware 损失（dir=0.3, win=0.5, huber=0.2）验证了此观点——虽本地 RMSE 略高但泛化更好。

---

## 当前最佳配置

| 配置项 | 值 |
|--------|-----|
| NWP 时区 | UTC → 北京时间（period+32） |
| 特征维度 | 107（无价格特征，统一训练/测试分布） |
| 模型 | ConvBiLSTM 单头（826K） + 时间点级对比预训练 + 类型④峰谷样本对 |
| 注意力模块 | SENet=False, TemporalSelfAttn=False（实验确认无增益） |
| 损失函数 | Trend-Aware (dir=0.3, win=0.5, huber=0.2) |
| 学习率 | 5e-4 |
| 调度器 | CosineAnnealingLR (T_max=100) |
| 早停 | patience=20 |
| 全量训练监控RMSE | **0.753**（本次），历史最佳 0.749 |
| 分箱特征 | is_high_pressure(>2.05)、is_extreme_pressure、is_low_pressure(<-5.81)、is_high_solar(>3.05)、is_high_hdd(>25.80)、is_high_net_load(>median) |
| 用法 | `python main.py --mode final --model bilstm --contrastive` |
| 对比预训练 | 类型①②③ + 类型④（峰谷价格正样本对，配置开关 `CONTRASTIVE_USE_TYPE4`） |

## 关键经验

1. **训练/测试特征必须统一**：价格特征在训练时可用但测试时填 0，等效于特征空间不一致。移除后虽本地 RMSE 上升，但泛化更好。
2. **对比预训练有效降低 RMSE**：无预训练 0.81 → 天级 0.76 → 时间点级 **0.749**。预训练粒度必须与预测任务对齐（96 点输出 → 96 点对比）。
3. **线上 > 本地**：本地验证集（1月）与测试集（12月）季节分布不同，Trend-Aware 的方向约束限制了本地过拟合，跨季节泛化更好。
4. **特征 > 架构**：分箱阈值更新 + 对比预训练的增益远大于 MoE/注意力等架构复杂度，且特征工程不引入训练不稳定性。
5. **MoE 不适合小数据时序预测**：334 天无法支撑多专家学习稳定路由，Gate 坍缩不可避。
6. **PDP 拐点必须随特征变更重校准**：移除价格特征后 NWP 重要性分布大幅变化，阈值需同步更新。
7. **简洁 > 复杂**：单头 826K + 好特征 + 对比预训练 > 多头 924K + 坏路由。
8. **注意力机制需要更多数据**：SENet 和时序自注意力在 334 天数据上均未带来增益，注意力不是万能药。
9. **峰谷正样本对为中性**：类型④ 对 RMSE 几乎无影响，但不增加推理成本，已保留在管线中。
10. **Transformer 不适合当前数据量**：ConvTransformer 在训练分布内优于 BiLSTM（+5.3%），但跨季节泛化不如（Final 模式 -7.8%）。BiLSTM 的循环结构对时间序列分布偏移更稳健。
11. **模仿学习可直接优化决策**：跳过电价预测，两个独立头输出买入/卖出概率，PG 直接最大化期望收益。小模型(243K) + 独立 softmax + PG 达到 63% Oracle 收益比。相比预测价格的间接路径，梯度 100% 聚焦在"让最优动作排第一"。但策略容易坍缩到少数动作，跨季节泛化仍不足。
