# v1 — ConvBiLSTM 基线

## 概述

项目第一个完整版本。Conv1d + BiLSTM 架构，85 维手工特征，Huber Loss。回测收益约 112k，RMSE 1.59。

## 特征（85 维）

- 边界条件预测值 7 维：系统负荷、风光总加、联络线、风电、光伏、水电、非市场化机组
- 边界条件滞后 D-1 7 维
- NWP 全局统计 47 维：7 通道 × 6 统计量 + 5 衍生（风速、有效辐照度），三次样条 24h→96 步
- 时间特征 12 维：hour/min/dayofweek/month/dayofyear 的 sin/cos + is_weekend + is_daylight
- 价格滞后 4 维：D-1/D-2/D-3/D-7
- 价格滚动 8 维：4/8/24/96 窗口的 mean/std

## 模型（~830K 参数）

```
Input (B, 96, 85) → Linear(85→64) → LayerNorm
  → Conv1d(64→128, k5) → ReLU → Dropout
  → Conv1d(128→128, k5) → ReLU → Dropout
  → Linear(128→128)
  → 2×BiLSTM(128, bidir) → Dropout
  → Residual(64→256) → LayerNorm → Linear(256→1)
  → Output (B, 96)
```

## 训练

- 数据划分：前 11 月训练 / 12 月验证
- Loss: Huber
- Optimizer: AdamW(lr=1e-3, wd=1e-4)
- Scheduler: CosineAnnealingLR(T_max=100)
- Early Stop: patience=20

## 已知问题

- NWP 时区未修正（UTC vs 北京 8 小时偏差）
- 验证集随机切分（存在未来信息泄漏）
- 特征维度较少，缺少体感温度、新能源比率、节假日等

## 用法

```bash
python main.py --mode full
```
