# v1 架构说明

## 概述

基础版本：ConvBiLSTM + 手工 NWP 特征 + Huber Loss。

## 特征工程

```
85 维 = 7边界 + 7边界滞后 + 47NWP全局统计 + 12时间 + 4价格滞后 + 8价格滚动
```

- **NWP 处理**: 对 7 通道 × 24 小时的 104×225 网格取全局统计 (mean/std/min/max/p10/p90) + 衍生特征 (风速、有效辐照度)，三次样条插值到 15 分钟。每个时间点（15分钟粒度）的 NWP 特征**不同**，24h→96 步由三次样条平滑过渡。

### 47 维 NWP 特征明细

**基础统计 (7 通道 × 6 统计量 = 42 维)**

| 通道 | 含义 | 统计量 |
|------|------|--------|
| u100 | 100m 纬向风 | mean, std, min, max, p10, p90 |
| v100 | 100m 经向风 | mean, std, min, max, p10, p90 |
| t2m | 2m 温度 | mean, std, min, max, p10, p90 |
| tp | 总降水量 | mean, std, min, max, p10, p90 |
| tcc | 总云量 | mean, std, min, max, p10, p90 |
| sp | 地表气压 | mean, std, min, max, p10, p90 |
| ghi | 地表水平辐照度 | mean, std, min, max, p10, p90 |

**衍生特征 (5 维)**

| 特征 | 公式 | 含义 |
|------|------|------|
| wind_speed_mean | mean(√(u100²+v100²)) | 合成风速均值 |
| wind_speed_std | std(√(u100²+v100²)) | 合成风速标准差 |
| wind_speed_max | max(√(u100²+v100²)) | 合成风速最大值 |
| eff_ghi_mean | mean(ghi × (1-tcc)) | 有效辐照度均值（扣除云量遮挡） |
| eff_ghi_max | max(ghi × (1-tcc)) | 有效辐照度最大值 |

**总计: 42 + 5 = 47 维**

> 注意：这 47 维是对整个 104×225 网格取**全局**空间统计，未区分子区域。
- **边界条件**: 仅用 7 个预测值列，D-1 滞后
- **预测误差**: 无
- **修正预报**: 无
- **价格特征**: D-1/D-2/D-3/D-7 滞后 + 4 窗口滚动统计

## 模型：ConvBiLSTM (~830K 参数)

```
Input (B, 96, 85)
  → Linear(85→64) → LayerNorm
  → Conv1d(64→128, k5) → ReLU → Dropout(0.2)
  → Conv1d(128→128, k5) → ReLU → Dropout(0.2)
  → Linear(128→128)
  → 2×BiLSTM(128, bidir) → Dropout(0.2)
  → Residual(64→256)
  → LayerNorm → Linear(256→1)
  → Output (B, 96)
```

## 训练

```
数据划分: 前11月训练(332天) / 12月验证(31天)
Loss: Huber
Optimizer: AdamW(lr=1e-3, wd=1e-4)
Scheduler: CosineAnnealing(T_max=100)
Batch: 16
Early Stop: patience=20
```

## 策略

枚举 tc∈[0,80], td∈[tc+8,88]，每段 8 步，选最大收益，每天至多 1 次充放电。

## 文件结构

```
v1_backup/
├── main.py              # 流水线入口
└── src/
    ├── config.py         # 路径、常量、超参数
    ├── features/
    │   ├── assemble_dataset.py   # 特征拼接 → (N,96,F) 数组
    │   ├── feature_engineering.py # 时间编码、滞后、滚动
    │   ├── nwp_processor.py      # .nc → 空间统计 → 15min插值
    │   └── build_nwp_cache.py    # 批量处理 424 .nc → parquet
    ├── models/
    │   ├── model.py      # ConvBiLSTM
    │   ├── dataset.py    # PyTorch Dataset
    │   ├── train.py      # 训练循环
    │   └── predict.py    # 推理
    └── strategy/
        └── optimizer.py  # 策略枚举
```
