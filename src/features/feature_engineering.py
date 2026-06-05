"""
特征工程：时间编码、滞后特征、滚动统计
"""
import numpy as np
import pandas as pd
from src.config import STEPS_PER_DAY


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    用 sin/cos 对编码周期性时间特征，替代原始整数
    """
    dt = df["times"]
    result = {}

    # 小时: 0-23, 包含分钟的小数部分
    hour_frac = dt.dt.hour + dt.dt.minute / 60.0
    result["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)

    # 分钟: 0-59
    result["minute_sin"] = np.sin(2 * np.pi * dt.dt.minute / 60)
    result["minute_cos"] = np.cos(2 * np.pi * dt.dt.minute / 60)

    # 周几: 0=Mon, 6=Sun
    result["dayofweek_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    result["dayofweek_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # 月: 1-12
    result["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    result["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)

    # 年中第几天: 1-366
    result["dayofyear_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 366)
    result["dayofyear_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 366)

    result_df = pd.DataFrame(result, index=df.index)

    # 布尔特征
    result_df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(float)
    result_df["is_daylight"] = ((dt.dt.hour >= 6) & (dt.dt.hour < 19)).astype(float)

    return pd.concat([df, result_df], axis=1)


def add_boundary_lags(df: pd.DataFrame, columns: list, lag_periods: int = STEPS_PER_DAY) -> pd.DataFrame:
    """
    添加前一天同一时刻的边界条件预测值作为滞后特征
    按天分组，每个 lag_periods 步（96）为一天
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_lag_d1"] = df[col].shift(lag_periods)
    return df


def add_forecast_error_lags(df: pd.DataFrame, actual_cols: list, forecast_cols: list,
                             lag_days: list = [1, 2, 3]) -> pd.DataFrame:
    """
    已弃用 — 保留兼容。改由 add_corrected_forecast_features 替代。
    """
    return df


def add_corrected_forecast_features(df: pd.DataFrame, actual_cols: list,
                                     forecast_cols: list) -> pd.DataFrame:
    """
    用历史预报误差修正当前预报值，生成有物理含义的特征。

    对每个变量计算三种偏差估计：
      bias_3d  = (d1+d2+d3误差) / 3
      bias_7d  = (d1+...+d7误差) / 7
      bias_ema = 0.5*d1 + 0.25*d2 + 0.125*d3 + 0.125*d4

    特征：
      修正预测值_3d/7d/ema  = 今天预测值 + bias
      修正幅度_3d/7d/ema     = |bias|  (预报不确定性)
    """
    df = df.copy()
    for fc_col in forecast_cols:
        act_col = fc_col.replace("预测值", "实际值")
        if act_col not in df.columns or fc_col not in df.columns:
            continue

        error = df[act_col] - df[fc_col]  # 当前时刻的预报误差
        spd = STEPS_PER_DAY

        # 三种偏差估计
        e_d1 = error.shift(spd)
        e_d2 = error.shift(2 * spd)
        e_d3 = error.shift(3 * spd)
        e_d4 = error.shift(4 * spd)
        e_d5 = error.shift(5 * spd)
        e_d6 = error.shift(6 * spd)
        e_d7 = error.shift(7 * spd)

        bias_3d = (e_d1 + e_d2 + e_d3) / 3.0
        bias_7d = (e_d1 + e_d2 + e_d3 + e_d4 + e_d5 + e_d6 + e_d7) / 7.0
        bias_ema = 0.5 * e_d1 + 0.25 * e_d2 + 0.125 * e_d3 + 0.125 * e_d4

        # 修正预测值
        fc_today = df[fc_col]
        df[f"{fc_col}_corr_3d"] = fc_today + bias_3d
        df[f"{fc_col}_corr_7d"] = fc_today + bias_7d
        df[f"{fc_col}_corr_ema"] = fc_today + bias_ema

        # 修正幅度（不确定性）
        df[f"{fc_col}_corr_3d_mag"] = bias_3d.abs()
        df[f"{fc_col}_corr_7d_mag"] = bias_7d.abs()
        df[f"{fc_col}_corr_ema_mag"] = bias_ema.abs()

    return df


def add_price_lags(df: pd.DataFrame, lag_days: list = [1, 2, 3, 7]) -> pd.DataFrame:
    """
    添加滞后电价特征：前 N 天同一时间步的价格
    """
    df = df.copy()
    target_col = "A"
    if target_col not in df.columns:
        return df
    for d in lag_days:
        df[f"price_lag_d{d}"] = df[target_col].shift(d * STEPS_PER_DAY)
    return df


def add_price_rolling(df: pd.DataFrame, windows: list = [4, 8, 24, 96]) -> pd.DataFrame:
    """
    添加价格滚动统计（当日窗口）
    """
    df = df.copy()
    target_col = "A"
    if target_col not in df.columns:
        return df
    for w in windows:
        df[f"price_roll_mean_{w}"] = df[target_col].rolling(w, center=False).mean()
        df[f"price_roll_std_{w}"] = df[target_col].rolling(w, center=False).std()
    return df


def get_forecast_error_cols() -> list:
    """返回修正预报特征列名 (corr_3d/7d/ema + magnitude × 7变量)"""
    from src.config import BOUNDARY_FORECAST_COLS
    cols = []
    for c in BOUNDARY_FORECAST_COLS:
        for suffix in ["corr_3d", "corr_7d", "corr_ema",
                       "corr_3d_mag", "corr_7d_mag", "corr_ema_mag"]:
            cols.append(f"{c}_{suffix}")
    return cols


def get_time_feature_cols() -> list:
    """返回所有时间特征列名"""
    return [
        "hour_sin", "hour_cos", "minute_sin", "minute_cos",
        "dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos",
        "dayofyear_sin", "dayofyear_cos",
        "is_weekend", "is_daylight",
    ]


def get_price_lag_cols(lag_days: list = [1, 2, 3, 7]) -> list:
    """返回所有价格滞后列名"""
    return [f"price_lag_d{d}" for d in lag_days]


def get_price_rolling_cols(windows: list = [4, 8, 24, 96]) -> list:
    """返回所有价格滚动统计列名"""
    cols = []
    for w in windows:
        cols.append(f"price_roll_mean_{w}")
        cols.append(f"price_roll_std_{w}")
    return cols
