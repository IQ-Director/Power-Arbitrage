# v3 — 对比预训练 + 特征成熟期

## 概述

回归 ConvBiLSTM 单头，精简特征至 107 维（移除价格特征，统一训练/测试分布），加入时间点级对比预训练、PDP 拐点分箱、Trend-Aware 损失。验证了 MoE、注意力、Transformer 等架构方向，均放弃——BiLSTM 仍是当前最佳。

## 特征（107 维，无价格特征）

价格特征在测试时全部填 0，训练/测试存在分布偏移——全部移除。模型只依赖气象与基本面信息。

| 类别 | 维度 | 内容 |
|------|------|------|
| 边界条件预测值 | 7 | 系统负荷、风光总加、联络线、风电、光伏、水电、非市场化机组 |
| 边界条件 D-1 滞后 | 7 | 各预测值 shift(96) |
| 边界条件滚动统计 | 6 | 风光总加 12/96、风电 12/24、非市场化机组 96、系统负荷 4 |
| NWP 原始统计 | ~42 | 7 通道(u100/v100/t2m/tp/tcc/sp/ghi) × 6 统计量(mean/std/min/max/p10/p90) |
| 风速衍生 | ~14 | wind_speed(√(u²+v²)) × 6 统计量 + eff_ghi(ghi×(1-tcc)) 等 |
| NWP 体感温度 | 5 | CDD(>26°C)/HDD(<18°C)/is_hot(>32°C)/is_cold(<0°C)/t2m_change |
| NWP 交叉特征 | 4 | temp_wind(风寒)/temp_over_wind/pressure_wind/temp_pressure |
| 时间特征 | 20 | hour/min/dayofweek/month/dayofyear 的 sin/cos + is_weekend/is_daylight/is_peak_hour + month_boundary + holidays(2024) + shift_workday |
| 供需基本面 | 5 | renew_ratio(新能源渗透率)/net_load/net_load_diff/net_load_roll_4/96 |
| PDP 分箱 | 6 | is_high_pressure(>2.05)/is_extreme_pressure/is_low_pressure(<-5.81)/is_high_solar(>3.05)/is_high_hdd(>25.80)/is_high_net_load |

## 模型（~823K 参数）

```
Input (B, 96, 107) → Linear(107→64) → LayerNorm
  → [FeatureAttention: 可选, 默认关闭]
  → Conv1d(64→128, k5) → ReLU → Dropout(0.2)
  → Conv1d(128→128, k5) → ReLU → Dropout(0.2)
  → Linear(128→128)
  → [MultiheadAttention(128, 8头): 可选, 默认关闭]
  → 2×BiLSTM(128, bidir) → Dropout(0.2)
  → Residual(64→256) → LayerNorm → Linear(256→1) → (B, 96)
```

### 可选模型

| 命令 | 模型 | 参数量 |
|------|------|--------|
| `--model bilstm` | ConvBiLSTM（默认） | 823K |
| `--model transformer` | ConvTransformer（Conv1d + PositionalEncoding + TransformerEncoder） | 437K~2.3M |
| `--model moe` | ConvBiLSTM_MoE（3 专家门控） | 924K |
| `--model hardroute` | 规则硬分流（sp_mean 中位数） | ~823K×2 |
| `--model wta` | MoE-WTA（赢者通吃） | 924K |
| `--model router` | Router 分流模块 | 924K |

## 对比预训练

30 轮无监督 SimCLR 预训练，时间点级 InfoNCE：

```
特征 + 高斯噪声(σ=0.15) → ConvBiLSTM_Contrastive → projection head → (B,96,128)
四种正样本对:
  ① 同时刻+双噪声视图 (w=1.0)    主力信号
  ② 同时刻+相邻天     (w=0.5)    日间连续性
  ③ 同天+相邻步       (w=0.2)    时间平滑性
  ④ 同天+同价格档     (w=0.25)   峰谷结构
Regime 加权负样本: 相似天气日之间降低推开力度
```

| 预训练 | 监控 RMSE |
|--------|----------|
| 无 | 0.81 |
| 天级 SimCLR | 0.76 |
| **时间点级 + Regime 加权** | **0.749** |

用法：`python main.py --mode final --model bilstm --contrastive`

## Loss: Trend-Aware

```
Loss = 0.3 × 方向一致性（tanh 软化涨跌方向）
     + 0.5 × Top-K窗口重叠（softmax 匹配 8段最优窗口）
     + 0.2 × Huber（绝对值锚定）
```

## 训练

```
数据划分: 前 29 天验证 / 剩余训练（时序切分，不可随机打乱）
提交前: 全量训练，最后 1/20 监控
Optimizer: AdamW(lr=5e-4, wd=1e-4)
Scheduler: CosineAnnealingLR(T_max=100)
Early Stop: patience=20
噪声增强: Gaussian(σ=0.01) 仅训练集
```

## 策略

枚举 tc∈[0,80], td∈[tc+8,88]，8 步窗口，max(放电窗口和 − 充电窗口和) × 1000。

## 已测试并放弃的方向

| 方向 | 结果 | 原因 |
|------|------|------|
| MoE (3 专家) | Gate 坍缩 | 334 天无法支撑多专家路由；WTA/pseudo-label/Router 均失败 |
| SENet 特征注意力 | RMSE +1.4% | 特征已通过排列重要性精选，额外权重=噪声 |
| 时序自注意力 | RMSE +4.0% | 65K 参数在 334 天上严重过拟合 |
| Transformer 大 (N=4,d=256) | Train +5.3%, Final -7.8% | 跨季节泛化差，2.3M 参数过拟合 |
| Transformer 小 (N=2,d=128) | Final -6.0% | 437K 仍不如 BiLSTM |
| 类型④峰谷正样本对 | 中性（RMSE 差 0.03%） | 已保留在管线中 |

## 文件结构

```
├── main.py                   # 流水线入口 (full/cache/train/predict/final)
├── analyze_features.py       # 排列重要性 + PDP 分析
├── src/
│   ├── config.py             # 集中配置
│   ├── features/
│   │   ├── assemble_dataset.py    # 特征拼接
│   │   ├── build_nwp_cache.py     # NWP .nc → parquet
│   │   ├── feature_engineering.py # 特征管线 (13 类函数)
│   │   └── nwp_processor.py       # NWP 统计量提取
│   ├── models/
│   │   ├── model.py              # ConvBiLSTM + FeatureAttention
│   │   ├── model_moe.py          # MoE + WTA + Router
│   │   ├── model_hardroute.py    # 规则硬分流
│   │   ├── model_transformer.py  # ConvTransformer
│   │   ├── train.py              # 训练 + 对比预训练 + 6种 Loss
│   │   ├── predict.py            # 推理 (支持 6 种模型)
│   │   ├── dataset.py            # PyTorch Dataset
│   │   ├── shap_torch.py         # SHAP 特征重要性
│   │   ├── shap_calibrate.py     # SHAP 校准
│   │   └── pdp_analysis.py       # PDP 拐点检测
│   └── strategy/
│       └── optimizer.py          # 枚举优化 + 回测
├── example/                  # LightGBM/sklearn baseline
└── CHANGELOG.md              # 完整迭代记录
```
