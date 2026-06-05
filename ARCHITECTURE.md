# v2 架构说明

## 概述

CNN+BiLSTM 端到端版本：保留手工 NWP 作为锚定，增加 CNN 从原始气象网格提取空间特征，Trend-Aware Loss 直接优化趋势。

## 模型输入特征（239 维表格 + 128 维 CNN 嵌入 = 367 维）

### 特征总览

```
239 维 = 7边界 + 7边界滞后 + 159NWP(全局+分区+衍生)
       + 42修正预报 + 12时间 + 4价格滞后 + 8价格滚动

+ 128 维 CNN 嵌入（从原始 (7,104,225) 网格提取）

→ 拼接后输入 BiLSTM: (B, 96, 367)
```

### 1. 边界条件预测值（7 维）

| 列名 | 含义 |
|------|------|
| 系统负荷预测值 | 系统总负荷日前预测 |
| 风光总加预测值 | 风电+光伏总出力预测 |
| 联络线预测值 | 跨区联络线功率预测 |
| 风电预测值 | 风电出力预测 |
| 光伏预测值 | 光伏出力预测 |
| 水电预测值 | 水电出力预测 |
| 非市场化机组预测值 | 非市场化机组出力预测 |

### 2. 边界条件滞后（7 维）

```
对每个边界预测值取 D-1 同时间步滞后: {col}_lag_d1
```

即 7 个边界列各自的 `_lag_d1`，共 7 维。

### 3. NWP 气象统计特征（159 维）

#### 3a. 全局空间统计（7 通道 × 6 统计量 = 42 维）

对每个通道在 104×225 网格上计算（过滤 1%/99% 异常值后）：

| 通道 | 含义 | 统计量 |
|------|------|--------|
| u100 | 100m 纬向风 | mean, std, min, max, p10, p90 |
| v100 | 100m 经向风 | mean, std, min, max, p10, p90 |
| t2m | 2m 温度 | mean, std, min, max, p10, p90 |
| tp | 总降水量 | mean, std, min, max, p10, p90 |
| tcc | 总云量 | mean, std, min, max, p10, p90 |
| sp | 地表气压 | mean, std, min, max, p10, p90 |
| ghi | 地表水平辐照度 | mean, std, min, max, p10, p90 |

#### 3b. 分区均值（7 通道 × 16 子区域 = 112 维）

将 104×225 网格切分为 4×4=16 个子区域，每通道每子区域计算均值：

```
{通道}_q0{0-3}_mean, ..., {通道}_q3{0-3}_mean
```

> v2 相比 v1 新增：捕获 NWP 的空间分布信息

#### 3c. 衍生特征（5 维）

| 特征 | 公式 | 含义 |
|------|------|------|
| wind_speed_mean | mean(√(u100²+v100²)) | 合成风速均值 |
| wind_speed_std | std(√(u100²+v100²)) | 合成风速标准差 |
| wind_speed_max | max(√(u100²+v100²)) | 合成风速最大值 |
| eff_ghi_mean | mean(ghi × (1-tcc)) | 有效辐照度均值（扣除云量遮挡） |
| eff_ghi_max | max(ghi × (1-tcc)) | 有效辐照度最大值 |

**NWP 总计: 42 + 112 + 5 = 159 维**

> 每小时的 NWP 网格不同，因此逐时间点的 159 维统计量也不同。24 小时 → 三次样条插值 → 96 步（15分钟粒度）。

### 4. 修正预报特征（7 变量 × 6 = 42 维）

用历史预报误差修正当前预报值。对每个边界变量，计算 3 种偏差估计：

| 偏差类型 | 计算方式 |
|----------|----------|
| bias_3d | (D-1 + D-2 + D-3 误差) / 3 |
| bias_7d | (D-1 + ... + D-7 误差) / 7 |
| bias_ema | 0.5×D-1 + 0.25×D-2 + 0.125×D-3 + 0.125×D-4 |

每种偏差生成 2 类特征：

| 特征类型 | 公式 | 含义 |
|----------|------|------|
| {col}_corr_{3d,7d,ema} | 今日预测值 + bias | 偏差修正后的预测值 |
| {col}_corr_{3d,7d,ema}_mag | \|bias\| | 预报不确定性（修正幅度） |

**共 7 变量 × (3 修正值 + 3 幅度) = 42 维**

> v2 相比 v1 新增：让模型学习预报的系统性偏差

### 5. 时间特征（12 维）

| 特征 | 编码方式 |
|------|----------|
| hour_sin, hour_cos | 小时 (0-23) → sin/cos |
| minute_sin, minute_cos | 分钟 (0-59) → sin/cos |
| dayofweek_sin, dayofweek_cos | 周几 (0-6) → sin/cos |
| month_sin, month_cos | 月份 (1-12) → sin/cos |
| dayofyear_sin, dayofyear_cos | 年中第几天 (1-366) → sin/cos |
| is_weekend | 布尔: 周六/周日=1 |
| is_daylight | 布尔: 6:00-18:59=1 |

### 6. 价格滞后特征（4 维）

```
price_lag_d1, price_lag_d2, price_lag_d3, price_lag_d7
```

前 1/2/3/7 天同一时间步的电价。

### 7. 价格滚动统计（8 维）

当日窗口内的统计量：

| 窗口 | 特征 |
|------|------|
| 4 步 (1h) | price_roll_mean_4, price_roll_std_4 |
| 8 步 (2h) | price_roll_mean_8, price_roll_std_8 |
| 24 步 (6h) | price_roll_mean_24, price_roll_std_24 |
| 96 步 (全天) | price_roll_mean_96, price_roll_std_96 |

### 8. CNN 嵌入（128 维）

从原始 NWP `.nc` 文件加载 `(7, 104, 225)` 网格，逐小时送入 CNN 编码器：

```
输入 (B, 7, 104, 225)
  → AvgPool(2) → (B, 7, 52, 113)
  → Conv(7→32, k5) → BN → ReLU → (B, 32, 52, 113)
  → Stage1: 2×ResBlock(32) → Down(32→32, s2) → (B, 32, 26, 57)
  → Stage2: 2×ResBlock(32) → Down(32→32, s2) → (B, 32, 13, 29)
  → Stage3: 2×ResBlock(32) → Down(32→32, s2) → (B, 32, 7, 15)
  → Head: Flatten → FC(3136→128)
  → 输出 (B, 128)
```

24 小时 → 24 个 128 维向量 → 线性插值到 96 步 → `(B, 96, 128)`，与表格特征拼接。

> 手工 NWP 统计作为锚定保留在表格特征中，CNN 嵌入提供补充的空间纹理信息。

---

## 模型：ConvBiLSTMCNN (~3.8M 参数)

### CNN 编码器 (~575K)

同上 §8，`src/models/nwp_cnn.py`。

### BiLSTM 主干 (~3.3M)

```
表格特征 (B, 96, 239) + CNN 嵌入 (B, 96, 128) → (B, 96, 367)

  → Linear(367→128) → LayerNorm
  → Conv1d(128→256, k5) → Conv1d(256→256, k5) → ReLU → Dropout
  → 2×BiLSTM(256, bidir) → Residual(128→512)
  → LayerNorm → Linear(512→1)
  → Output (B, 96)
```

## Loss: Trend-Aware（默认）

```
Loss = 0.3 × 方向一致性    (相邻时段涨跌方向, tanh 软化)
     + 0.5 × Top-K窗口     (最贵/最便宜 8 段重叠度, softmax 匹配)
     + 0.2 × Huber         (绝对值锚定, 防止预测缩零)
```

BLOCK=8 与策略充放电窗口对齐。

## 训练

```
数据划分: 前11月训练(332天) / 12月验证(31天)
Optimizer: AdamW(lr=1e-3, wd=1e-4)
Scheduler: CosineAnnealing(T_max=100)
Batch: 8 (CNN 模式显存较大)
Early Stop: patience=20
噪声增强: Gaussian(σ=0.01) 仅训练集
```

## 其他支持的 Loss（配置项 LOSS_TYPE）

| LOSS_TYPE | 组成 |
|-----------|------|
| huber | 纯 Huber |
| weighted_huber | Huber × 偏离均值权重 |
| ranking | 成对排序损失 |
| decision_aware | 软策略收益优化 |
| trend_aware | 方向+TopK+Huber (默认) |
| hybrid | 0.6×ranking + 0.4×huber |

## 其他支持的模型（配置项 MODEL_TYPE）

| MODEL_TYPE | 说明 |
|------------|------|
| bilstm | ConvBiLSTM (USE_CNN_NWP=False 时使用，~830K) |
| tcn | TCNPricePredictor |

## 策略（同 v1）

枚举 tc∈[0,80], td∈[tc+8,88]，每段 8 步，选最大收益，每天至多 1 次充放电。

## 相比 v1 的主要变化

| 项目 | v1 | v2 |
|------|----|----|
| 特征维度 | 85 | 239 |
| NWP 空间粒度 | 仅全局统计 | 全局 + 16 分区 |
| 修正预报 | 无 | 3 种偏差修正 (42维) |
| CNN 编码器 | 无 | ResBlock CNN (128维嵌入) |
| 模型参数 | ~830K | ~3.8M |
| 默认 Loss | Huber | Trend-Aware |
| Batch Size | 16 | 8 |

## 文件结构

```
v2/
├── main.py              # 流水线入口
└── src/
    ├── config.py         # 路径、常量、超参数、开关
    ├── features/
    │   ├── assemble_dataset.py   # 特征拼接 → (N,96,F) 数组
    │   ├── feature_engineering.py # 时间编码、滞后、修正预报
    │   ├── nwp_processor.py      # .nc → 空间统计 + 分区
    │   ├── build_nwp_cache.py    # 批量处理 424 .nc → parquet
    │   └── nwp_raw_loader.py     # 加载原始 NWP 到内存 (CNN 用)
    ├── models/
    │   ├── model.py      # ConvBiLSTM + ConvBiLSTMCNN
    │   ├── model_tcn.py  # TCN 备选
    │   ├── nwp_cnn.py    # NWPCNNEncoder (残差)
    │   ├── dataset.py    # PriceDataset + NWPPriceDataset
    │   ├── train.py      # 训练循环 + 6种 Loss
    │   └── predict.py    # 推理 (支持 CNN/TCN)
    └── strategy/
        └── optimizer.py  # 策略枚举 + 回测函数
```
