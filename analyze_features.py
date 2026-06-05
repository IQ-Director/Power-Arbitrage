"""综合分析：排列重要性 + PDP 拐点"""
import numpy as np
import torch
import pandas as pd
from src.models.model import ConvBiLSTM
from src.features.assemble_dataset import build_train_dataset
from src.models.train import temporal_split
from src.features.feature_engineering import (
    get_time_feature_cols, get_nwp_pdp_bin_cols,
)
from src.config import (
    HIDDEN_DIM, NUM_LSTM_LAYERS, LSTM_DROPOUT, CONV_KERNEL_SIZE, EMBED_DIM,
    MODEL_PATH, OUTPUT_DIR,
    USE_FEATURE_ATTENTION, FEATURE_ATTN_REDUCTION,
    USE_TEMPORAL_ATTENTION, TEMPORAL_ATTN_HEADS,
)

np.set_printoptions(precision=3, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# ==================== 1. 数据 ====================
X, y, feature_cols = build_train_dataset()
X_train, X_val, y_train, y_val = temporal_split(X, y)

# ==================== 2. 加载模型 ====================
model = ConvBiLSTM(
    input_dim=len(feature_cols), hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LSTM_LAYERS, dropout=0.0,
    conv_kernel=CONV_KERNEL_SIZE, embed_dim=EMBED_DIM,
    use_feature_attention=USE_FEATURE_ATTENTION,
    feat_attn_reduction=FEATURE_ATTN_REDUCTION,
    use_temporal_attention=USE_TEMPORAL_ATTENTION,
    temp_attn_heads=TEMPORAL_ATTN_HEADS,
    temp_attn_dropout=0.0,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
is_moe = False
print("加载 ConvBiLSTM 模型")
model.eval()

def model_predict(X_batch):
    x_t = torch.FloatTensor(X_batch).to(device)
    pred = model(x_t)
    return pred.detach().cpu().numpy()

# ==================== 3. 排列重要性（快速版：Top-50 特征） ====================
print("\n" + "=" * 60)
print("排列重要性分析 (快速版, Top-50)")
print("=" * 60)

baseline_pred = model_predict(X_val)
baseline_rmse = np.sqrt(np.mean((y_val.ravel() - baseline_pred.ravel()) ** 2))
print(f"基线 RMSE: {baseline_rmse:.4f}")

# 分析 Top-60 特征
n_features = min(60, len(feature_cols))
imp_results = {}
rng = np.random.RandomState(42)

for f_idx in range(n_features):
    f_name = feature_cols[f_idx]
    X_perm = X_val.copy()
    flat = X_perm[:, :, f_idx].ravel()
    rng.shuffle(flat)
    X_perm[:, :, f_idx] = flat.reshape(X_val.shape[0], X_val.shape[1])
    pred = model_predict(X_perm)
    rmse = np.sqrt(np.mean((y_val.ravel() - pred.ravel()) ** 2))
    imp_results[f_name] = rmse - baseline_rmse
    if (f_idx + 1) % 10 == 0:
        print(f"  进度: {f_idx + 1}/{n_features}")

items = sorted(imp_results.items(), key=lambda x: x[1], reverse=True)
print(f"\n{'排名':<5} {'特征':<35} {'重要性':>10} {'占比':>8}")
print("-" * 60)
pos_sum = sum(v for _, v in items if v > 0)
for i, (name, val) in enumerate(items[:30]):
    pct = (val / pos_sum * 100) if pos_sum > 0 and val > 0 else 0
    print(f"{i+1:<5} {name:<35} {val:>10.5f} {pct:>7.1f}%")

# ==================== 4. PDP 拐点分析 ====================
print("\n" + "=" * 60)
print("PDP 拐点分析")
print("=" * 60)

key_features = [
    'v100_max', 'HDD', 't2m_max', 'sp_max', 'v100_std',
    't2m_p10', 'sp_p90', 'wind_speed_mean', 'v100_p90', 'sp_mean',
    'net_load', 'renew_ratio', 'sp_min', 'tp_max',
    'tcc_mean', 'eff_ghi_mean', 'ghi_p90', 't2m_min',
    'u100_max', 'u100_p90', 'v100_mean', 'v100_min',
    'tp_p90', 'CDD', 'ghi_mean',
]
key_features = [f for f in key_features if f in feature_cols]
print(f"分析 {len(key_features)} 个关键特征\n")

pdp_results = []
for f_name in key_features:
    f_idx = feature_cols.index(f_name)
    f_vals = X_val[:, :, f_idx].ravel()
    grid = np.percentile(f_vals, np.linspace(5, 95, 30))
    preds = []
    with torch.no_grad():
        for g in grid:
            X_mod = X_val.copy()
            X_mod[:, :, f_idx] = g
            pred = model_predict(X_mod)
            preds.append(pred.mean())
    pdp = np.array(preds)
    pdp_range = pdp.max() - pdp.min()
    d1 = np.gradient(pdp, grid)
    d2 = np.gradient(d1, grid)
    inflections = []
    for i in range(1, len(d2) - 1):
        if d2[i] * d2[i + 1] < 0:
            inflections.append(i)
    for i in range(1, len(d1) - 1):
        if abs(d1[i] - d1[i - 1]) > np.std(d1) * 3:
            if i not in inflections:
                inflections.append(i)
    if pdp_range > 0.003:
        inf_vals = sorted(set([grid[i] for i in inflections[:4]]))
        pdp_results.append({
            "feature": f_name, "delta": pdp_range,
            "inflections": inf_vals,
            "min_val": grid[0], "max_val": grid[-1],
        })

for r in sorted(pdp_results, key=lambda x: x["delta"], reverse=True):
    inf_str = f"拐点≈ {[f'{v:.3f}' for v in r['inflections']]}" if r['inflections'] else "无明显拐点"
    print(f"  [{r['feature']:20s}] Δ={r['delta']:.4f}  "
          f"范围[{r['min_val']:.3f}, {r['max_val']:.3f}]  {inf_str}")

# ==================== 5. 汇总 ====================
print("\n" + "=" * 60)
print("汇总")
print("=" * 60)

print("\n【NWP 核心特征】（排列重要性 Top-10 中占比 >90%）")
nwp_top = [
    "v100_max", "HDD", "t2m_max", "sp_max", "v100_std",
    "t2m_p10", "sp_p90", "wind_speed_mean", "v100_p90", "sp_mean",
]
for f in nwp_top:
    if f in feature_cols:
        rank = next((i + 1 for i, (n, v) in enumerate(items) if n == f and v > 0), "-")
        pdp = next((r for r in pdp_results if r["feature"] == f), None)
        pdp_info = f"Δ={pdp['delta']:.4f}" if pdp else ""
        print(f"  rank={rank} {f}  {pdp_info}")

print("\n【需要分箱的特征】（PDP 有明确拐点）")
binned = [r for r in pdp_results if r["inflections"]]
if binned:
    for r in binned:
        for thr in r["inflections"]:
            print(f"  {r['feature']}: is_{r['feature']}_gt_{thr:.2f}  (阈值={thr:.3f})")
else:
    print("  未检测到显著拐点")

print(f"\n总特征数: {len(feature_cols)}")
print(f"分析脚本: analyze_features.py")
