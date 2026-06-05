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
    parser.add_argument("--mode", choices=["full", "cache", "train", "predict", "final"],
                        default="full", help="运行模式")
    parser.add_argument("--model", choices=["bilstm", "moe", "hardroute", "wta", "router", "transformer"], default="bilstm",
                        help="模型架构: bilstm | moe | hardroute | wta | router | transformer")
    parser.add_argument("--contrastive", action="store_true",
                        help="使用对比预训练（bilstm / transformer）")
    args = parser.parse_args()

    # 确保 src 在 path 中
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.config import NWP_CACHE_PATH, MODEL_PATH, OUTPUT_PRICE_PATH, OUTPUT_POWER_PATH, OUTPUT_DIR

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
    if args.mode in ("full", "train", "final"):
        print("\n" + "=" * 60)
        if args.model == "moe":
            label = "MoE (多专家门控)"
            print(f"[3/5] {'全量' if args.mode == 'final' else ''}训练 {label}...")
            print("=" * 60)
            from src.models.train import train_final_moe, train_model_moe
            if args.mode == "final":
                model, val_metrics = train_final_moe(X_train, y_train, feature_cols)
            else:
                model, val_metrics = train_model_moe(X_train, y_train, feature_cols)
        elif args.model == "hardroute":
            label = "HardRoute (阈值硬分流)"
            print(f"[3/5] {'全量' if args.mode == 'final' else ''}训练 {label}...")
            print("=" * 60)
            from src.models.train import train_final_hardroute, train_model_hardroute
            if args.mode == "final":
                model, val_metrics = train_final_hardroute(X_train, y_train, feature_cols)
            else:
                model, val_metrics = train_model_hardroute(X_train, y_train, feature_cols)
        elif args.model == "wta":
            label = "MoE-WTA (赢者通吃)"
            print(f"[3/5] {'全量' if args.mode == 'final' else ''}训练 {label}...")
            print("=" * 60)
            from src.models.train import train_final_moe_wta, train_model_moe_wta
            if args.mode == "final":
                model, val_metrics = train_final_moe_wta(X_train, y_train, feature_cols)
            else:
                model, val_metrics = train_model_moe_wta(X_train, y_train, feature_cols)
        elif args.model == "router":
            label = "Router (分流模块)"
            print(f"[3/5] {'全量' if args.mode == 'final' else ''}训练 {label}...")
            print("=" * 60)
            from src.models.train import train_final_router, train_model_router
            if args.mode == "final":
                model, val_metrics = train_final_router(X_train, y_train, feature_cols)
            else:
                model, val_metrics = train_model_router(X_train, y_train, feature_cols)
        elif args.model == "transformer":
            label = "ConvTransformer"
            pretrained_path = None
            if args.contrastive:
                label += " + 对比预训练"
                pretrained_path = os.path.join(OUTPUT_DIR, "models", "contrastive_pretrained.pt")
                print(f"[3/5] {'全量' if args.mode == 'final' else ''}对比预训练 (Transformer)...")
                print("=" * 60)
                from src.models.train import contrastive_pretrain_transformer
                contrastive_pretrain_transformer(X_train, feature_dim, pretrained_path,
                                                  feature_cols=feature_cols,
                                                  y_train=y_train)
                print("对比预训练完成，开始微调...")

            print(f"[3/5] {'全量' if args.mode == 'final' else ''}训练 {label}...")
            print("=" * 60)
            from src.models.train import train_final_transformer, train_model_transformer
            if args.mode == "final":
                model, val_metrics = train_final_transformer(X_train, y_train, feature_dim,
                                                              pretrained_path=pretrained_path)
            else:
                model, val_metrics = train_model_transformer(X_train, y_train, feature_dim,
                                                              pretrained_path=pretrained_path)
        else:
            label = "ConvBiLSTM"
            pretrained_path = None
            if args.contrastive:
                label += " + 对比预训练"
                pretrained_path = os.path.join(OUTPUT_DIR, "models", "contrastive_pretrained.pt")
                print(f"[3/5] {'全量' if args.mode == 'final' else ''}对比预训练...")
                print("=" * 60)
                from src.models.train import contrastive_pretrain
                contrastive_pretrain(X_train, feature_dim, pretrained_path,
                                      feature_cols=feature_cols,
                                      y_train=y_train)
                print("对比预训练完成，开始微调...")

            print(f"[3/5] {'全量' if args.mode == 'final' else ''}训练 {label}...")
            print("=" * 60)
            from src.models.train import train_final, train_model
            if args.mode == "final":
                model, val_metrics = train_final(X_train, y_train, feature_dim,
                                                  pretrained_path=pretrained_path)
            else:
                model, val_metrics = train_model(X_train, y_train, feature_dim,
                                                  pretrained_path=pretrained_path)

    # ========== Step 4: 预测 ==========
    print("\n" + "=" * 60)
    print("[4/5] 测试集推理...")
    print("=" * 60)
    from src.models.predict import predict, predict_transformer, predict_moe, predict_moe_wta, predict_hardroute, predict_router, predictions_to_dataframe

    elif args.model == "moe":
        predictions = predict_moe(X_test, feature_cols, MODEL_PATH)
    elif args.model == "hardroute":
        predictions = predict_hardroute(X_test, feature_cols, MODEL_PATH)
    elif args.model == "wta":
        predictions = predict_moe_wta(X_test, feature_cols, MODEL_PATH)
    elif args.model == "router":
        predictions = predict_router(X_test, feature_cols, MODEL_PATH)
    elif args.model == "transformer":
        predictions = predict_transformer(X_test, feature_dim, MODEL_PATH)
    else:
        predictions = predict(X_test, feature_dim, MODEL_PATH)

    df_price = predictions_to_dataframe(predictions, test_times)
    df_price.to_csv(OUTPUT_PRICE_PATH, index=False)
    print(f"电价预测已保存: {OUTPUT_PRICE_PATH}")
    print(f"预测形状: {predictions.shape}, 均值: {predictions.mean():.4f}")

    # ========== Step 5: 策略优化 ==========
    print("\n" + "=" * 60)
    print("[5/5] 生成充放电策略...")
    print("=" * 60)

    else:
        from src.strategy.optimizer import generate_full_strategy
        generate_full_strategy(df_price, OUTPUT_POWER_PATH)

    print("\n" + "=" * 60)
    print("流水线完成！")
    print(f"提交文件: {OUTPUT_POWER_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
