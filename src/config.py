"""
所有路径、常量、特征列名、模型超参数
"""
import os

# ==================== 路径配置 ====================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_ROOT = ROOT_DIR  # power/
DATA_DIR = os.path.join(REAL_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
NWP_DIR = os.path.join(DATA_DIR, "all_nc")
NWP_CACHE_PATH = os.path.join(DATA_DIR, "nwp_cache.parquet")
OUTPUT_DIR = os.path.join(REAL_ROOT, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

TRAIN_FEATURE_PATH = os.path.join(TRAIN_DIR, "mengxi_boundary_anon_filtered.csv")
TRAIN_LABEL_PATH = os.path.join(TRAIN_DIR, "mengxi_node_price_selected.csv")
TEST_FEATURE_PATH = os.path.join(TEST_DIR, "test_in_feature_ori.csv")
OUTPUT_PRICE_PATH = os.path.join(OUTPUT_DIR, "price_predictions.csv")
OUTPUT_POWER_PATH = os.path.join(OUTPUT_DIR, "output.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== 特征列名 ====================
BOUNDARY_FORECAST_COLS = [
    "系统负荷预测值", "风光总加预测值", "联络线预测值",
    "风电预测值", "光伏预测值", "水电预测值", "非市场化机组预测值",
]

BOUNDARY_ACTUAL_COLS = [
    "系统负荷实际值", "风光总加实际值", "联络线实际值",
    "风电实际值", "光伏实际值", "水电实际值", "非市场化机组实际值",
]

TARGET_COL = "A"

# ==================== NWP 配置 ====================
NWP_CHANNELS = ["u100", "v100", "t2m", "tp", "tcc", "sp", "ghi"]
NWP_STATS = ["mean", "std", "min", "max", "p10", "p90"]
NWP_LEAD_TIMES = 24  # 逐小时预报，共24步

# ==================== 储能参数 ====================
BATTERY_CAPACITY = 8000
CHARGE_POWER = -1000
DISCHARGE_POWER = 1000
BLOCK_LEN = 8  # 充放电持续时间（15分钟步数）
STEPS_PER_DAY = 96
TC_MIN, TC_MAX = 0, 80
TD_MIN_OFFSET = 8
TD_MAX = 88

# ==================== 模型超参数 ====================
HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
CONV_KERNEL_SIZE = 5
EMBED_DIM = 64

# ==================== 训练超参数 ====================
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
T_MAX_SCHEDULER = 100
NOISE_STD = 0.01

# ==================== 特征工程常量 ====================
HOLIDAY_DATES_2024 = [
    "2024-01-01",  # 元旦
    "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13",  # 春节
    "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17",
    "2024-04-04", "2024-04-05", "2024-04-06",  # 清明节
    "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",  # 劳动节
    "2024-06-08", "2024-06-09", "2024-06-10",  # 端午节
    "2024-09-15", "2024-09-16", "2024-09-17",  # 中秋节
    "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05",  # 国庆节
    "2024-10-06", "2024-10-07",
]

BOUNDARY_ROLLING_WINDOWS = [4, 12, 24, 96]
PRICE_WINDOWS = [4, 8, 24, 96]
PRICE_LAG_DAYS = [1, 2, 3, 7, 14]

# ==================== MoE 配置 ====================
MOE_NUM_EXPERTS = 3
MOE_REGIME_FEATURES = ["sp_mean", "sp_max", "sp_p90", "HDD", "t2m_max", "renew_ratio", "net_load"]
GATE_MODE = "noisy_soft"    # "soft" | "noisy_soft" | "gumbel_top1" | "gumbel_top2"
GATE_NOISE_STD = 1.0        # noisy_soft 模式：训练时 Gate logits 加高斯噪声
GATE_BALANCE_WEIGHT = 0.1   # 负载均衡权重
WTA_BALANCE_WEIGHT = 0.5   # WTA 模式：更强的负载均衡权重
WTA_TEMPERATURE = 1.0       # WTA 模式：Gumbel-Softmax 温度（越低越硬）
WTA_WARMUP_EPOCHS = 8       # WTA 模式：前 N 轮用软 MoE 让专家分化
WTA_EPS = 0.1               # WTA 模式：soft gate 最小权重（防梯度消失）

# ==================== 注意力机制 ====================
USE_FEATURE_ATTENTION = False
FEATURE_ATTN_REDUCTION = 4

USE_TEMPORAL_ATTENTION = False
TEMPORAL_ATTN_HEADS = 8
TEMPORAL_ATTN_DROPOUT = 0.2

# ==================== Transformer 替换 BiLSTM ====================
USE_TRANSFORMER = True
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_HEADS = 8
TRANSFORMER_DIM = 256
TRANSFORMER_FF_DIM = 512
TRANSFORMER_DROPOUT = 0.2

# ==================== 对比预训练 ====================
CONTRASTIVE_TEMPERATURE = 0.7   # InfoNCE 温度
CONTRASTIVE_EMBED_DIM = 128     # 对比投影维度
CONTRASTIVE_EPOCHS = 30         # 预训练轮数
CONTRASTIVE_AUG_STD = 0.15      # 输入噪声标准差（构造正样本对）
CONTRASTIVE_USE_TYPE4 = False   # 类型④ 价格峰谷正样本对（需真实 y_train）
PSEUDO_LABEL_WEIGHT = 0.05  # 交叉熵损失权重（控制 Gate 对 cluster 标签的拟合强度）
PSEUDO_LABEL_K = 3          # KMeans 聚类数（= 专家数）
RANDOM_SEED = 42

# ==================== Router（分流模块） ====================
ROUTER_K = 3                # 分组数
ROUTER_FREEZE = False       # 预测训练时是否冻结 Router
ROUTER_CE_WEIGHT = 0.3      # 联合训练时 Router CE 损失权重

# ==================== Hard Route（软 MoE 替代方案） ====================
HARD_ROUTE_FEATURE = "sp_mean"     # 路由特征（日均值）— PDP 拐点 @1.24，Δ=0.17
HARD_ROUTE_THRESHOLD = "median"     # "median"=中位数, 或具体数值

# ==================== 损失函数 ====================
LOSS_TYPE = "trend_aware"   # "huber" | "topk_weighted" | "trend_aware"
LOSS_K_RATIO = 0.15          # topk_weighted 模式：每日 top/bottom k% 极端值加权
LOSS_EXTREME_WEIGHT = 3.0    # topk_weighted 模式：极端值权重倍数
TA_DIR_WEIGHT = 0.3          # trend_aware: 方向一致性权重
TA_WINDOW_WEIGHT = 0.5       # trend_aware: Top-K窗口重叠权重
TA_HUBER_WEIGHT = 0.2        # trend_aware: Huber 锚定权重（稳定性）

