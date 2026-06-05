"""
特征工程：基于 ConvBiLSTM 排列重要性 + 微信文章(3)(5) 体系
核心发现：NWP 气象特征贡献 69%，价格自回归靠 LSTM 内部捕获
"""
import numpy as np
import pandas as pd
from src.config import STEPS_PER_DAY


# ==================== 时间特征 ====================

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["times"]
    result = {}
    hour_frac = dt.dt.hour + dt.dt.minute / 60.0
    result["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)
    result["minute_sin"] = np.sin(2 * np.pi * dt.dt.minute / 60)
    result["minute_cos"] = np.cos(2 * np.pi * dt.dt.minute / 60)
    result["dayofweek_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    result["dayofweek_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    result["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    result["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    result["dayofyear_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 366)
    result["dayofyear_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 366)
    result_df = pd.DataFrame(result, index=df.index)
    result_df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(float)
    result_df["is_daylight"] = ((dt.dt.hour >= 6) & (dt.dt.hour < 19)).astype(float)
    return pd.concat([df, result_df], axis=1)


def add_peak_hour(df: pd.DataFrame) -> pd.DataFrame:
    hour = df["times"].dt.hour
    result = pd.DataFrame({
        "is_peak_hour": hour.isin([8, 9, 10, 11, 12, 17, 18, 19, 20, 21]).astype(float),
    }, index=df.index)
    return pd.concat([df, result], axis=1)


def add_month_boundary(df: pd.DataFrame) -> pd.DataFrame:
    day = df["times"].dt.day
    result = pd.DataFrame({
        "is_month_start": (day <= 3).astype(float),
        "is_month_end": (day >= 27).astype(float),
    }, index=df.index)
    return pd.concat([df, result], axis=1)


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.config import HOLIDAY_DATES_2024
    dates = df["times"].dt.date
    holiday_set = set(pd.to_datetime(HOLIDAY_DATES_2024).date)
    shift_workdays_2024 = {
        pd.Timestamp("2024-02-04").date(), pd.Timestamp("2024-02-18").date(),
        pd.Timestamp("2024-04-07").date(), pd.Timestamp("2024-04-28").date(),
        pd.Timestamp("2024-05-11").date(), pd.Timestamp("2024-09-14").date(),
        pd.Timestamp("2024-09-29").date(), pd.Timestamp("2024-10-12").date(),
    }
    dow = df["times"].dt.dayofweek
    is_holiday = np.array([d in holiday_set for d in dates], dtype=float)
    is_shift = np.array([d in shift_workdays_2024 for d in dates], dtype=float)
    is_workday = ((dow < 5) & (~np.array([d in holiday_set for d in dates]))) | is_shift.astype(bool)
    is_workday = is_workday.astype(float)
    holiday_series = pd.Series(is_holiday, index=df.index)
    day_before = ((holiday_series == 0) & (holiday_series.shift(-1) == 1)).astype(float)
    day_after = ((holiday_series == 0) & (holiday_series.shift(1) == 1)).astype(float)
    result = pd.DataFrame({
        "is_holiday": is_holiday, "is_workday": is_workday,
        "is_shift_workday": is_shift,
        "day_before_holiday": day_before, "day_after_holiday": day_after,
    }, index=df.index)
    return pd.concat([df, result], axis=1)


# ==================== 边界条件特征（精简——仅保留有排列重要性的） ====================

def add_boundary_lags(df: pd.DataFrame, columns: list,
                      lag_periods: int = STEPS_PER_DAY) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_lag_d1"] = df[col].shift(lag_periods)
    return df


def add_key_boundary_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """仅保留排列重要性 Top-30 中的边界滚动特征"""
    df = df.copy()
    # 来自排列重要性 rank 14/23/24/25/28/29
    key_features = [
        ("风光总加预测值", [12, 96]),
        ("风电预测值", [12, 24]),
        ("非市场化机组预测值", [96]),
        ("系统负荷预测值", [4]),
    ]
    for col, windows in key_features:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
    return df


# ==================== 供需基本面特征 ====================

def add_renewable_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "风光总加预测值" in df.columns and "系统负荷预测值" in df.columns:
        load = df["系统负荷预测值"].replace(0, np.nan)
        df["renew_ratio"] = df["风光总加预测值"] / load
    else:
        df["renew_ratio"] = 0.0
    return df


def add_net_load_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    need = ["系统负荷预测值", "风光总加预测值", "水电预测值"]
    if not all(c in df.columns for c in need):
        return df
    net_load = df[need[0]] - df[need[1]] - df[need[2]]
    df["net_load"] = net_load
    df["net_load_diff"] = net_load.diff()
    df["net_load_roll_4"] = net_load.rolling(4).mean()
    df["net_load_roll_96"] = net_load.rolling(96).mean()
    return df


# ==================== NWP 气象衍生特征 ====================

def add_nwp_thermal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t2m_col = None
    for c in df.columns:
        if "t2m" in c and "mean" in c:
            t2m_col = c
            break
    if t2m_col is None:
        return df
    t2m = df[t2m_col]
    df["CDD"] = (t2m - 299.15).clip(lower=0)    # >26°C
    df["HDD"] = (291.15 - t2m).clip(lower=0)    # <18°C
    df["is_hot"] = (t2m > 305.15).astype(float)  # >32°C
    df["is_cold"] = (t2m < 273.15).astype(float)  # <0°C
    return df


def add_nwp_change(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if "t2m" in c and "mean" in c:
            df["t2m_change"] = df[c].diff()
            return df
    df["t2m_change"] = 0.0
    return df


def add_nwp_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """NWP 交叉特征：温度×风、气压×风 —— 排列重要性 Top 全是 NWP，深挖交互"""
    df = df.copy()
    t2m_mean = next((c for c in df.columns if "t2m" in c and "mean" in c), None)
    sp_mean = next((c for c in df.columns if c == "sp_mean"), None)
    ws_mean = next((c for c in df.columns if c == "wind_speed_mean"), None)
    v100_max = next((c for c in df.columns if c == "v100_max"), None)

    if t2m_mean and ws_mean:
        df["temp_wind"] = df[t2m_mean] * df[ws_mean]          # 风寒效应
        df["temp_over_wind"] = df[t2m_mean] / (df[ws_mean] + 0.1)
    if sp_mean and v100_max:
        df["pressure_wind"] = df[sp_mean] * df[v100_max]       # 气压梯度→风
    if t2m_mean and sp_mean:
        df["temp_pressure"] = df[t2m_mean] * df[sp_mean]

    return df


# ==================== 价格特征（精简——仅保留排列重要性正贡献的） ====================

def add_price_lags(df: pd.DataFrame) -> pd.DataFrame:
    """仅保留有信号的滞后：d7（rank 30），d1 也为基本结构保留"""
    df = df.copy()
    if "A" not in df.columns:
        return df
    for d in [1, 7]:
        df[f"price_lag_d{d}"] = df["A"].shift(d * STEPS_PER_DAY)
    return df


def add_price_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """精简版滚动统计"""
    df = df.copy()
    if "A" not in df.columns:
        return df
    for w in [4, 96]:
        roll = df["A"].rolling(w, center=False)
        df[f"price_roll_mean_{w}"] = roll.mean()
        df[f"price_roll_std_{w}"] = roll.std()
    return df


def add_price_diff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "A" not in df.columns:
        return df
    df["price_diff_1"] = df["A"].diff()
    return df


def add_price_yesterday_compare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "A" not in df.columns:
        return df
    lag24 = df["A"].shift(96)
    df["price_lag24_diff"] = df["A"] - lag24  # rank 17
    return df


# ==================== SHAP 拐点校准分箱 ====================

def add_shap_price_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "A" not in df.columns:
        return df
    lag24d = df["A"] - df["A"].shift(96)
    df["is_vs_yesterday_up"] = (lag24d > 0).astype(float)
    df["is_vs_yesterday_big_up"] = (lag24d > 1.0).astype(float)
    df["is_current_rising"] = (df["A"].diff() > 0).astype(float)
    df["is_near_zero_price"] = (df["A"] < 0.05).astype(float)  # PDP 拐点 @0
    return df


def add_nwp_pdp_bins(df: pd.DataFrame) -> pd.DataFrame:
    """PDP 拐点分箱（2026-05-17 校准）"""
    df = df.copy()
    if "sp_mean" in df.columns:
        df["is_high_pressure"] = (df["sp_mean"] > 2.05).astype(float)
    else:
        df["is_high_pressure"] = 0.0
    if "sp_p90" in df.columns:
        median = df["sp_p90"].median()
        df["is_extreme_pressure"] = (df["sp_p90"] > median).astype(float)
    else:
        df["is_extreme_pressure"] = 0.0
    if "sp_min" in df.columns:
        df["is_low_pressure"] = (df["sp_min"] < -5.81).astype(float)
    else:
        df["is_low_pressure"] = 0.0
    if "ghi_p90" in df.columns:
        df["is_high_solar"] = (df["ghi_p90"] > 3.05).astype(float)
    else:
        df["is_high_solar"] = 0.0
    if "HDD" in df.columns:
        df["is_high_hdd"] = (df["HDD"] > 25.80).astype(float)
    else:
        df["is_high_hdd"] = 0.0
    # net_load: PDP Δ=0.017，单调递增无明确拐点，用中位数捕捉高负荷场景
    if "net_load" in df.columns:
        median = df["net_load"].median()
        df["is_high_net_load"] = (df["net_load"] > median).astype(float)
    else:
        df["is_high_net_load"] = 0.0
    return df


# ==================== 列名获取 ====================

def get_time_feature_cols() -> list:
    return [
        "hour_sin", "hour_cos", "minute_sin", "minute_cos",
        "dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos",
        "dayofyear_sin", "dayofyear_cos",
        "is_weekend", "is_daylight", "is_peak_hour",
        "is_month_start", "is_month_end",
        "is_holiday", "is_workday", "is_shift_workday",
        "day_before_holiday", "day_after_holiday",
    ]


def get_boundary_lag_cols() -> list:
    from src.config import BOUNDARY_FORECAST_COLS
    return [f"{c}_lag_d1" for c in BOUNDARY_FORECAST_COLS]


def get_key_boundary_rolling_cols() -> list:
    return [
        "风光总加预测值_roll_mean_12", "风光总加预测值_roll_mean_96",
        "风电预测值_roll_mean_12", "风电预测值_roll_mean_24",
        "非市场化机组预测值_roll_mean_96",
        "系统负荷预测值_roll_mean_4",
    ]


def get_renewable_cols() -> list:
    return ["renew_ratio"]


def get_net_load_cols() -> list:
    return ["net_load", "net_load_diff", "net_load_roll_4", "net_load_roll_96"]


def get_nwp_thermal_cols() -> list:
    return ["CDD", "HDD", "is_hot", "is_cold", "t2m_change"]


def get_nwp_cross_cols() -> list:
    return ["temp_wind", "temp_over_wind", "pressure_wind", "temp_pressure"]


def get_price_lag_cols() -> list:
    return ["price_lag_d1", "price_lag_d7"]


def get_price_rolling_cols() -> list:
    return ["price_roll_mean_4", "price_roll_std_4",
            "price_roll_mean_96", "price_roll_std_96"]


def get_price_derived_cols() -> list:
    return ["price_diff_1", "price_lag24_diff"]


def get_shap_price_bin_cols() -> list:
    return ["is_vs_yesterday_up", "is_vs_yesterday_big_up",
            "is_current_rising", "is_near_zero_price"]


def get_nwp_pdp_bin_cols() -> list:
    return ["is_high_pressure", "is_extreme_pressure",
            "is_low_pressure", "is_high_solar",
            "is_high_hdd", "is_high_net_load"]
