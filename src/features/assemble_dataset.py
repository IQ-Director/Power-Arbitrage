"""
拼接边界条件 + NWP + 价格 → 时序特征数组
基于 ConvBiLSTM 排列重要性精简：NWP 核心 + 供需基本面 + SHAP 分箱
"""
import pandas as pd
import numpy as np
from src.config import (
    TRAIN_FEATURE_PATH, TRAIN_LABEL_PATH, TEST_FEATURE_PATH,
    NWP_CACHE_PATH, BOUNDARY_FORECAST_COLS, BOUNDARY_ACTUAL_COLS,
    TARGET_COL, STEPS_PER_DAY,
)
from src.features.feature_engineering import (
    add_cyclical_time_features, add_peak_hour, add_month_boundary,
    add_holiday_features, get_time_feature_cols,
    add_boundary_lags, add_key_boundary_rolling,
    get_boundary_lag_cols, get_key_boundary_rolling_cols,
    add_renewable_ratio, add_net_load_features,
    get_renewable_cols, get_net_load_cols,
    add_nwp_thermal, add_nwp_change, add_nwp_cross_features,
    get_nwp_thermal_cols, get_nwp_cross_cols,
    add_nwp_pdp_bins, get_nwp_pdp_bin_cols,
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
    """
    将 NWP 从 UTC 转为北京时间后，按 (date, period) 与边界数据对齐。
    NWP period 0 = 00:00 UTC = 08:00 北京 → period 32
    NWP period 64 = 16:00 UTC = 00:00 北京（次日）→ period 0, date+1
    """
    df = df.copy()
    df["date"] = df["times"].dt.strftime("%Y-%m-%d")
    df["period"] = df["times"].dt.hour * 4 + df["times"].dt.minute // 15

    nwp = nwp.copy()
    nwp["period"] = nwp["period"] + 32
    nwp["date_shift"] = nwp["period"] // 96
    nwp["period"] = nwp["period"] % 96
    nwp["date"] = (pd.to_datetime(nwp["target_date"]) + pd.to_timedelta(nwp["date_shift"], unit="D")).dt.strftime("%Y-%m-%d")
    nwp_feat = nwp.drop(columns=["target_date", "date_shift"], errors="ignore")

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


def _get_nwp_feature_cols(df: pd.DataFrame, known_cols: set) -> list:
    return [c for c in df.columns if c not in known_cols]


def _build_feature_cols(df: pd.DataFrame) -> list:
    boundary_lag_cols = get_boundary_lag_cols()
    boundary_rolling_cols = get_key_boundary_rolling_cols()
    renewable_cols = get_renewable_cols()
    net_load_cols = get_net_load_cols()
    time_cols = get_time_feature_cols()
    nwp_thermal_cols = get_nwp_thermal_cols()
    nwp_cross_cols = get_nwp_cross_cols()
    nwp_bin_cols = get_nwp_pdp_bin_cols()

    base_cols = (
        BOUNDARY_FORECAST_COLS + boundary_lag_cols +
        boundary_rolling_cols +
        renewable_cols + net_load_cols +
        time_cols + nwp_thermal_cols + nwp_cross_cols +
        nwp_bin_cols
    )
    known_set = set(base_cols + BOUNDARY_ACTUAL_COLS + [TARGET_COL, "times"])
    nwp_feature_cols = _get_nwp_feature_cols(df, known_set)

    feature_cols = (
        BOUNDARY_FORECAST_COLS +
        boundary_lag_cols +
        boundary_rolling_cols +
        renewable_cols +
        net_load_cols +
        nwp_feature_cols +
        time_cols +
        nwp_thermal_cols +
        nwp_cross_cols +
        nwp_bin_cols
    )
    feature_cols = [c for c in feature_cols if c in df.columns]
    return feature_cols


def _apply_features(df: pd.DataFrame) -> pd.DataFrame:
    """统一特征管线"""
    df = add_boundary_lags(df, BOUNDARY_FORECAST_COLS, STEPS_PER_DAY)
    df = add_key_boundary_rolling(df)
    df = add_renewable_ratio(df)
    df = add_net_load_features(df)

    df = add_cyclical_time_features(df)
    df = add_peak_hour(df)
    df = add_month_boundary(df)
    df = add_holiday_features(df)

    df = add_nwp_thermal(df)
    df = add_nwp_change(df)
    df = add_nwp_cross_features(df)
    df = add_nwp_pdp_bins(df)

    df = df.fillna(0)
    return df


def build_train_dataset() -> tuple:
    print("构建训练数据集...")

    df_feat = _load_boundary(TRAIN_FEATURE_PATH)
    df_label = _load_labels()
    nwp = _load_nwp_cache()

    df = df_feat.merge(df_label, on="times", how="inner")
    print(f"  边界条件: {df_feat.shape}, 标签: {df_label.shape}, 合并后: {df.shape}")

    df = _merge_nwp(df, nwp)
    df = _apply_features(df)

    feature_cols = _build_feature_cols(df)
    print(f"  总特征维度: {len(feature_cols)}")

    X = _to_sequence_array(df, feature_cols)
    y = _to_sequence_array(df, [TARGET_COL]).squeeze(-1)

    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_cols


def build_test_dataset() -> tuple:
    print("构建测试数据集...")

    df = _load_boundary(TEST_FEATURE_PATH)
    nwp = _load_nwp_cache()

    df = _merge_nwp(df, nwp)
    df = _apply_features(df)
    df = df.fillna(0)

    feature_cols = _build_feature_cols(df)
    print(f"  总特征维度: {len(feature_cols)}")

    times_list = df["times"].values
    X = _to_sequence_array(df, feature_cols)
    print(f"  X shape: {X.shape}")

    return X, times_list, feature_cols
