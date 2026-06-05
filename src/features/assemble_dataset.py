"""
拼接边界条件 + NWP 缓存 + 价格标签 → 时序特征数组 (N_days, 96, F)
"""
import pandas as pd
import numpy as np
from src.config import (
    TRAIN_FEATURE_PATH, TRAIN_LABEL_PATH, TEST_FEATURE_PATH,
    NWP_CACHE_PATH, BOUNDARY_FORECAST_COLS, BOUNDARY_ACTUAL_COLS,
    TARGET_COL, STEPS_PER_DAY,
)
from src.features.feature_engineering import (
    add_cyclical_time_features, add_boundary_lags,
    add_price_lags, add_price_rolling,
    get_time_feature_cols, get_price_lag_cols, get_price_rolling_cols,
)


def _load_boundary(path: str) -> pd.DataFrame:
    """加载边界条件数据，解析时间列"""
    df = pd.read_csv(path)
    df["times"] = pd.to_datetime(df["times"])
    df = df.sort_values("times").reset_index(drop=True)
    return df


def _load_labels() -> pd.DataFrame:
    """加载价格标签"""
    df = pd.read_csv(TRAIN_LABEL_PATH)
    df["times"] = pd.to_datetime(df["times"])
    df = df.sort_values("times").reset_index(drop=True)
    return df


def _load_nwp_cache() -> pd.DataFrame:
    """加载 NWP 缓存"""
    df = pd.read_parquet(NWP_CACHE_PATH)
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


def _merge_nwp(df: pd.DataFrame, nwp: pd.DataFrame) -> pd.DataFrame:
    """
    将 NWP 特征按 (日期, 时段) 合并到主表
    df 需要有 'date' 和 'period' 列
    """
    df = df.copy()
    df["date"] = df["times"].dt.strftime("%Y-%m-%d")
    df["period"] = df.groupby("date").cumcount()

    nwp = nwp.copy()
    nwp["date"] = nwp["target_date"].dt.strftime("%Y-%m-%d")

    # 去掉 NWP 中已有的 date/period 列，避免重复
    nwp_feat = nwp.drop(columns=["target_date"], errors="ignore")

    df = df.merge(nwp_feat, on=["date", "period"], how="left")
    df = df.drop(columns=["date", "period"])
    return df


def _to_sequence_array(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    将 flat DataFrame (N_total, F) 转换为 3D 数组 (N_days, 96, F)
    自动处理不足 96 步的最后一天
    """
    n_total = len(df)
    n_days = n_total // STEPS_PER_DAY

    if n_days == 0:
        raise ValueError(f"数据不足 96 行，总行数: {n_total}")

    trimmed = n_days * STEPS_PER_DAY
    data = df[feature_cols].iloc[:trimmed].values.astype(np.float32)
    return data.reshape(n_days, STEPS_PER_DAY, len(feature_cols))


def build_train_dataset() -> tuple:
    """
    构建训练数据集
    返回: X_train (N_days, 96, F), y_train (N_days, 96), feature_cols list
    """
    print("构建训练数据集...")

    # 加载数据
    df_feat = _load_boundary(TRAIN_FEATURE_PATH)
    df_label = _load_labels()
    nwp = _load_nwp_cache()

    # 合并标签
    df = df_feat.merge(df_label, on="times", how="inner")
    print(f"  边界条件: {df_feat.shape}, 标签: {df_label.shape}, 合并后: {df.shape}")

    # 合并 NWP
    df = _merge_nwp(df, nwp)
    # 补全缺失的 NWP 列（文件缺失时用 0 填充）
    nwp_cols = [c for c in df.columns if c not in ["times", TARGET_COL] + BOUNDARY_FORECAST_COLS]
    df[nwp_cols] = df[nwp_cols].fillna(0)

    # 时间特征
    df = add_cyclical_time_features(df)

    # 滞后特征（按时间顺序）
    df = add_boundary_lags(df, BOUNDARY_FORECAST_COLS, STEPS_PER_DAY)
    df = add_price_lags(df, [1, 2, 3, 7])
    df = add_price_rolling(df, [4, 8, 24, 96])

    # 填充因滞后产生的 NaN（前几天的数据无历史）
    df = df.fillna(0)

    # 确定特征列顺序
    boundary_lag_cols = [f"{c}_lag_d1" for c in BOUNDARY_FORECAST_COLS]
    time_cols = get_time_feature_cols()
    price_lag_cols = get_price_lag_cols([1, 2, 3, 7])
    price_rolling_cols = get_price_rolling_cols([4, 8, 24, 96])

    # 所有非 NWP 特征
    base_cols = BOUNDARY_FORECAST_COLS + boundary_lag_cols + time_cols + price_lag_cols + price_rolling_cols
    # NWP 特征（df 中的列减去已知的非 NWP 列）
    known_cols = set(base_cols + BOUNDARY_ACTUAL_COLS + [TARGET_COL, "times"])
    nwp_feature_cols = [c for c in df.columns if c not in known_cols]

    feature_cols = BOUNDARY_FORECAST_COLS + boundary_lag_cols + nwp_feature_cols + time_cols + price_rolling_cols + price_lag_cols

    # 只保留存在的列
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"  总特征维度: {len(feature_cols)}")

    # 转为 3D 数组
    X = _to_sequence_array(df, feature_cols)
    y = _to_sequence_array(df, [TARGET_COL]).squeeze(-1)  # (N_days, 96)

    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_cols


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

    # 测试集没有价格，填充占位列
    for d in [1, 2, 3, 7]:
        df[f"price_lag_d{d}"] = 0.0
    for w in [4, 8, 24, 96]:
        df[f"price_roll_mean_{w}"] = 0.0
        df[f"price_roll_std_{w}"] = 0.0

    df = df.fillna(0)

    boundary_lag_cols = [f"{c}_lag_d1" for c in BOUNDARY_FORECAST_COLS]
    time_cols = get_time_feature_cols()
    price_lag_cols = get_price_lag_cols([1, 2, 3, 7])
    price_rolling_cols = get_price_rolling_cols([4, 8, 24, 96])

    base_cols = BOUNDARY_FORECAST_COLS + boundary_lag_cols + time_cols + price_lag_cols + price_rolling_cols
    known_cols = set(base_cols + ["times"])
    nwp_feature_cols = [c for c in df.columns if c not in known_cols]

    feature_cols = BOUNDARY_FORECAST_COLS + boundary_lag_cols + nwp_feature_cols + time_cols + price_rolling_cols + price_lag_cols
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"  总特征维度: {len(feature_cols)}")

    times_list = df["times"].values
    X = _to_sequence_array(df, feature_cols)
    print(f"  X shape: {X.shape}")

    return X, times_list, feature_cols
