"""
储能充放电策略优化：枚举所有合法的 (tc, td) 组合，求最大收益
"""
import numpy as np
import pandas as pd
from src.config import (
    CHARGE_POWER, DISCHARGE_POWER, BLOCK_LEN, STEPS_PER_DAY,
    TC_MIN, TC_MAX, TD_MIN_OFFSET, TD_MAX,
    OUTPUT_POWER_PATH,
)


def optimize_day(prices: np.ndarray) -> tuple:
    """
    对单日 96 个电价寻找最优充放电策略
    返回: (tc, td, profit, power_array)
    若所有策略收益 ≤ 0 则 tc=td=-1, profit=0, power_array 全零
    """
    best_profit = 0.0
    best_tc = -1
    best_td = -1

    for tc in range(TC_MIN, TC_MAX + 1):
        charge_sum = np.sum(prices[tc:tc + BLOCK_LEN])
        for td in range(tc + TD_MIN_OFFSET, TD_MAX + 1):
            discharge_sum = np.sum(prices[td:td + BLOCK_LEN])
            profit = (discharge_sum - charge_sum) * DISCHARGE_POWER
            if profit > best_profit:
                best_profit = profit
                best_tc = tc
                best_td = td

    power = np.zeros(STEPS_PER_DAY)
    if best_tc >= 0 and best_td >= 0:
        power[best_tc:best_tc + BLOCK_LEN] = CHARGE_POWER
        power[best_td:best_td + BLOCK_LEN] = DISCHARGE_POWER

    return best_tc, best_td, best_profit, power


def generate_full_strategy(price_df: pd.DataFrame, save_path: str = OUTPUT_POWER_PATH) -> pd.DataFrame:
    """
    处理所有天，生成最终提交文件
    price_df: 包含 'times' 和 'A'（预测电价）列
    """
    price_df = price_df.copy()
    price_df["times"] = pd.to_datetime(price_df["times"])
    price_df["date"] = price_df["times"].dt.date

    results = []
    total_profit = 0
    trade_days = 0

    for date, group in price_df.groupby("date"):
        prices = group["A"].values
        times = group["times"].values

        if len(prices) != STEPS_PER_DAY:
            print(f"警告: {date} 数据点 {len(prices)}, 期望 {STEPS_PER_DAY}")

        tc, td, profit, power = optimize_day(prices[:STEPS_PER_DAY])

        if tc >= 0:
            total_profit += profit
            trade_days += 1

        for i in range(len(prices)):
            results.append({
                "times": times[i] if i < len(times) else None,
                "实时价格": prices[i] if i < len(prices) else 0.0,
                "power": power[i] if i < STEPS_PER_DAY else 0,
            })

    df_out = pd.DataFrame(results)
    df_out.to_csv(save_path, index=False)

    n_days = len(price_df["date"].unique())

    print(f"策略已保存: {save_path}")
    print(f"总天数: {n_days}, 交易天数: {trade_days}")

    return df_out


def backtest_profit(predictions: np.ndarray, actual_prices: np.ndarray) -> dict:
    """
    回测：用预测价格生成的策略在实际价格上的收益
    predictions: (N_days, 96) 预测电价
    actual_prices: (N_days, 96) 实际电价
    """
    total_profit = 0.0
    trade_days = 0

    for day in range(len(predictions)):
        pred = predictions[day]
        actual = actual_prices[day]

        # 基于预测价格找最优策略
        _, _, _, power = optimize_day(pred)

        # 用实际价格计算收益
        profit = np.sum(actual * power)
        total_profit += profit
        if np.any(power != 0):
            trade_days += 1

    n_days = len(predictions)
    return {
        "total_profit": total_profit,
        "avg_profit": total_profit / n_days if n_days > 0 else 0,
        "trade_days": trade_days,
    }
