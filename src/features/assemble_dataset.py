"""
拼接边界条件 + NWP 缓存 + 价格标签 → 时序特征数组 (N_days, 96, F)
"""
import os
import pandas as pd
import numpy as np
from src.config import (
    TRAIN_FEATURE_PATH, TRAIN_LABEL_PATH, TEST_FEATURE_PATH,
    NWP_CACHE_PATH, BOUNDARY_FORECAST_COLS, BOUNDARY_ACTUAL_COLS,
    TARGET_COL, STEPS_PER_DAY, SCALER_PATH,
)
from src.features.feature_engineering import (
    add_cyclical_time_features, add_boundary_lags,
    add_price_lags, add_price_rolling, add_corrected_forecast_features,
    get_time_feature_cols, get_price_lag_cols, get_price_rolling_cols,
    get_forecast_error_cols,
)


def _load_boundary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["times"] = pd.to_datetime(df["times"])
    df = df.sort_values("times").reset_index(drop=True)
    return df


def _load_labels() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_LABEL_PATH)
    df["times"] = pd.to_datetime(df["times"])
    df = df.sort_values("times").reset_index(drop=True)
    return df


def _load_nwp_cache() -> pd.DataFrame:
    df = pd.read_parquet(NWP_CACHE_PATH)
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


def _merge_nwp(df: pd.DataFrame, nwp: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["times"].dt.strftime("%Y-%m-%d")
    df["period"] = df.groupby("date").cumcount()

    nwp = nwp.copy()
    nwp["date"] = nwp["target_date"].dt.strftime("%Y-%m-%d")

    nwp_feat = nwp.drop(columns=["target_date"], errors="ignore")
    df = df.merge(nwp_feat, on=["date", "period"], how="left")
    df = df.drop(columns=["date", "period"])
    return df


def _to_sequence_array(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    n_total = len(df)
    n_days = n_total // STEPS_PER_DAY
    if n_days == 0:
        raise ValueError(f"数据不足 96 行，总行数: {n_total}")
    trimmed = n_days * STEPS_PER_DAY
    data = df[feature_cols].iloc[:trimmed].values.astype(np.float32)
    return data.reshape(n_days, STEPS_PER_DAY, len(feature_cols))


def _compute_scaler_params(X: np.ndarray) -> tuple:
    """
    对 3D 数组 (N, 96, F) 的每个特征维度 (F) 计算 mean 和 std
    在 N 和 96 两个维度上展开后计算
    """
    N, T, F = X.shape
    flat = X.reshape(-1, F)  # (N*96, F)
    mean = np.mean(flat, axis=0, keepdims=False)  # (F,)
    std = np.std(flat, axis=0, keepdims=False)     # (F,)
    std = np.where(std < 1e-8, 1.0, std)            # 防止除零
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """对 3D 数组 (N, 96, F) 做 (x - mean) / std"""
    return (X - mean) / std


def _save_scaler(mean: np.ndarray, std: np.ndarray):
    """保存 scaler 参数"""
    np.savez(SCALER_PATH, mean=mean, std=std)
    print(f"  Scaler 参数已保存: {SCALER_PATH}")


def _load_scaler() -> tuple:
    """加载 scaler 参数"""
    data = np.load(SCALER_PATH)
    return data["mean"], data["std"]


def _get_feature_cols(df: pd.DataFrame, extra_test_cols: list = None) -> list:
    """提取特征列名，保持与 build_train_dataset 一致的顺序"""
    boundary_lag_cols = [f"{c}_lag_d1" for c in BOUNDARY_FORECAST_COLS]
    time_cols = get_time_feature_cols()
    price_lag_cols = get_price_lag_cols([1, 2, 3, 7])
    price_rolling_cols = get_price_rolling_cols([4, 8, 24, 96])
    forecast_error_cols = get_forecast_error_cols()

    known_cols = set(
        BOUNDARY_FORECAST_COLS + boundary_lag_cols + time_cols +
        price_lag_cols + price_rolling_cols + forecast_error_cols +
        BOUNDARY_ACTUAL_COLS + [TARGET_COL, "times"]
    )
    if extra_test_cols:
        known_cols.update(extra_test_cols)

    nwp_feature_cols = [c for c in df.columns if c not in known_cols]

    feature_cols = (
        BOUNDARY_FORECAST_COLS + boundary_lag_cols + nwp_feature_cols +
        time_cols + forecast_error_cols + price_rolling_cols + price_lag_cols
    )
    feature_cols = [c for c in feature_cols if c in df.columns]
    return feature_cols


def _get_tab_feature_cols(df: pd.DataFrame) -> list:
    """CNN 模式下同时保留 NWP 手工统计（锚定）+ 表格特征"""
    return _get_feature_cols(df)


def build_train_dataset() -> tuple:
    """
    构建训练数据集
    返回: X_train (N_days, 96, F), y_train (N_days, 96), feature_cols list
    """
    print("构建训练数据集...")

    df_feat = _load_boundary(TRAIN_FEATURE_PATH)
    df_label = _load_labels()
    nwp = _load_nwp_cache()

    df = df_feat.merge(df_label, on="times", how="inner")
    print(f"  边界条件: {df_feat.shape}, 标签: {df_label.shape}, 合并后: {df.shape}")

    df = _merge_nwp(df, nwp)
    nwp_cols = [c for c in df.columns if c not in ["times", TARGET_COL] + BOUNDARY_FORECAST_COLS + BOUNDARY_ACTUAL_COLS]
    df[nwp_cols] = df[nwp_cols].fillna(0)

    df = add_cyclical_time_features(df)
    df = add_boundary_lags(df, BOUNDARY_FORECAST_COLS, STEPS_PER_DAY)
    df = add_corrected_forecast_features(df, BOUNDARY_ACTUAL_COLS, BOUNDARY_FORECAST_COLS)
    df = add_price_lags(df, [1, 2, 3, 7])
    df = add_price_rolling(df, [4, 8, 24, 96])

    # LightGBM 预测误差修正（Phase 1）—— 效果不佳，暂时禁用
    # from src.features.error_predictor import predict_errors
    # df = predict_errors(df)

    df = df.fillna(0)

    from src.config import USE_CNN_NWP
    if USE_CNN_NWP:
        feature_cols = _get_tab_feature_cols(df)
    else:
        feature_cols = _get_feature_cols(df)
    print(f"  总特征维度: {len(feature_cols)}")

    X = _to_sequence_array(df, feature_cols)
    y = _to_sequence_array(df, [TARGET_COL]).squeeze(-1)

    # 提取每天的目标日期（用于索引 NWP）
    n_days = len(X)
    dates = df["times"].dt.strftime("%Y-%m-%d").values[::STEPS_PER_DAY][:n_days]

    # 计算交易标签：每天最优收益是否 > 0
    from src.strategy.optimizer import compute_trade_labels
    trade_labels = compute_trade_labels(y)
    print(f"  交易天数: {trade_labels.sum()}/{len(trade_labels)}")

    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_cols, trade_labels, dates


def build_test_dataset() -> tuple:
    """
    构建测试数据集（无价格标签）
    返回: X_test (N_days, 96, F), times_list, feature_cols list
    """
    print("构建测试数据集...")

    df = _load_boundary(TEST_FEATURE_PATH)
    nwp = _load_nwp_cache()

    df = _merge_nwp(df, nwp)
    nwp_cols = [c for c in df.columns if c not in ["times"] + BOUNDARY_FORECAST_COLS]
    df[nwp_cols] = df[nwp_cols].fillna(0)

    df = add_cyclical_time_features(df)
    df = add_boundary_lags(df, BOUNDARY_FORECAST_COLS, STEPS_PER_DAY)

    # LightGBM 预测误差修正 —— 效果不佳，暂时禁用
    # from src.features.error_predictor import predict_errors
    # df = predict_errors(df)

    # 测试集没有实际值 → 偏差填0 → 修正值=原始预测值, 修正幅度=0
    for c in BOUNDARY_FORECAST_COLS:
        fc_val = df[c]
        for suffix in ["corr_3d", "corr_7d", "corr_ema"]:
            df[f"{c}_{suffix}"] = fc_val  # 无偏差修正 = 原始预测
        for suffix in ["corr_3d_mag", "corr_7d_mag", "corr_ema_mag"]:
            df[f"{c}_{suffix}"] = 0.0      # 不确定性=0
    for d in [1, 2, 3, 7]:
        df[f"price_lag_d{d}"] = 0.0
    for w in [4, 8, 24, 96]:
        df[f"price_roll_mean_{w}"] = 0.0
        df[f"price_roll_std_{w}"] = 0.0

    df = df.fillna(0)

    from src.config import USE_CNN_NWP
    if USE_CNN_NWP:
        feature_cols = _get_tab_feature_cols(df)
    else:
        feature_cols = _get_feature_cols(df)
    print(f"  总特征维度: {len(feature_cols)}")

    times_list = df["times"].values
    X = _to_sequence_array(df, feature_cols)

    # 加载训练集的 scaler 参数并应用 — 已禁用
    # if os.path.exists(SCALER_PATH):
    #     mean, std = _load_scaler()
    #     F = min(len(mean), X.shape[-1])
    #     X[:, :, :F] = _apply_scaler(X[:, :, :F], mean[:F], std[:F])

    n_days = len(X)
    dates = df["times"].dt.strftime("%Y-%m-%d").values[::STEPS_PER_DAY][:n_days]

    print(f"  X shape: {X.shape}")
    return X, times_list, feature_cols, dates
