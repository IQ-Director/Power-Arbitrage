"""
批量处理所有 NWP .nc 文件 → 单个 parquet 缓存
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import glob
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
from src.features.nwp_processor import NWPProcessor, interpolate_to_15min
from src.config import NWP_DIR, NWP_CACHE_PATH, STEPS_PER_DAY


def parse_target_date(filename: str) -> datetime:
    """
    文件名 YYYYMMDD.nc → 目标日期 YYYY-MM-(DD+1)
    例如: 20250101.nc → 2025-01-02
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    issue_date = datetime.strptime(basename, "%Y%m%d")
    target_date = issue_date + timedelta(days=1)
    return target_date


def build_cache():
    """
    遍历所有 .nc 文件，处理每个文件，输出单个 parquet
    """
    nc_files = sorted(glob.glob(os.path.join(NWP_DIR, "*.nc")))
    if not nc_files:
        print(f"未找到 .nc 文件: {NWP_DIR}")
        return

    print(f"找到 {len(nc_files)} 个 .nc 文件")
    processor = NWPProcessor()

    all_dfs = []
    skipped = 0

    for nc_file in tqdm(nc_files, desc="处理 NWP 文件"):
        target_date = parse_target_date(nc_file)
        try:
            hourly_df = processor.process_file(nc_file)  # (24, F)
            df_15min = interpolate_to_15min(hourly_df)    # (96, F)
            df_15min["target_date"] = target_date.strftime("%Y-%m-%d")
            df_15min["period"] = range(STEPS_PER_DAY)
            all_dfs.append(df_15min)
        except Exception as e:
            print(f"处理失败 {nc_file}: {e}")
            skipped += 1

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_parquet(NWP_CACHE_PATH, index=False)
        print(f"缓存已保存: {NWP_CACHE_PATH}")
        print(f"形状: {combined.shape}, 目标日期范围: {combined['target_date'].min()} ~ {combined['target_date'].max()}")
        print(f"成功: {len(nc_files) - skipped}, 失败: {skipped}")
    else:
        print("没有成功处理的文件")


if __name__ == "__main__":
    build_cache()
