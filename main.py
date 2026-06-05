"""
端到端流水线：NWP 缓存 → 特征构建 → 模型训练 → 预测 → 策略生成
用法:
  python main.py --mode full       # 全流程
  python main.py --mode cache      # 仅重建 NWP 缓存
  python main.py --mode train      # 训练 + 预测 + 策略
  python main.py --mode predict    # 仅推理（使用已有模型）
"""
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="电价预测与储能优化流水线")
    parser.add_argument("--mode", choices=["full", "cache", "train", "predict"],
                        default="full", help="运行模式")
    args = parser.parse_args()

    # 确保 src 在 path 中
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.config import NWP_CACHE_PATH, MODEL_PATH, OUTPUT_PRICE_PATH, OUTPUT_POWER_PATH

    # ========== Step 1: NWP 缓存 ==========
    if args.mode in ("full", "cache"):
        if os.path.exists(NWP_CACHE_PATH) and args.mode != "cache":
            print(f"[cache] NWP 缓存已存在: {NWP_CACHE_PATH}，跳过")
        else:
            print("=" * 60)
            print("[1/5] 构建 NWP 缓存...")
            print("=" * 60)
            from src.features.build_nwp_cache import build_cache
            build_cache()

    if args.mode == "cache":
        print("缓存构建完成")
        return

    # ========== Step 2: 特征工程 ==========
    print("\n" + "=" * 60)
    print("[2/5] 构建特征数据集...")
    print("=" * 60)
    from src.features.assemble_dataset import build_train_dataset, build_test_dataset

    X_train, y_train, feature_cols = build_train_dataset()
    X_test, test_times, _ = build_test_dataset()

    feature_dim = X_train.shape[-1]
    print(f"特征维度: {feature_dim}")

    # ========== Step 3: 模型训练 ==========
    if args.mode in ("full", "train"):
        print("\n" + "=" * 60)
        print("[3/5] 训练 ConvBiLSTM 模型...")
        print("=" * 60)
        from src.models.train import train_model

        model, val_metrics = train_model(X_train, y_train, feature_dim)

    # ========== Step 4: 预测 ==========
    print("\n" + "=" * 60)
    print("[4/5] 测试集推理...")
    print("=" * 60)
    from src.models.predict import predict, predictions_to_dataframe

    predictions = predict(X_test, feature_dim, MODEL_PATH)
    df_price = predictions_to_dataframe(predictions, test_times)
    df_price.to_csv(OUTPUT_PRICE_PATH, index=False)
    print(f"电价预测已保存: {OUTPUT_PRICE_PATH}")
    print(f"预测形状: {predictions.shape}, 均值: {predictions.mean():.4f}")

    # ========== Step 5: 策略优化 ==========
    print("\n" + "=" * 60)
    print("[5/5] 生成充放电策略...")
    print("=" * 60)
    from src.strategy.optimizer import generate_full_strategy

    generate_full_strategy(df_price, OUTPUT_POWER_PATH)

    print("\n" + "=" * 60)
    print("流水线完成！")
    print(f"提交文件: {OUTPUT_POWER_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
