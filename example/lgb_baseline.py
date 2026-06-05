"""
LightGBM 基线：根据边界条件预测节点电价 A
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== 路径配置 ====================
# /xxx为存储数据的根目录
train_feature_path = '/xxx/train/mengxi_boundary_anon_filtered.csv'
train_label_path = '/xxx/train/mengxi_node_price_selected.csv'
test_feature_path = '/xxx/test/test_in_feature_ori.csv'
output_price_path = '/xxx/output_price/lgb_baseline_output.csv'
output_power_path = '/xxx/output_power/lgb_baseline_output.csv'

# 边界条件特征列（与测试集对齐，仅使用预测值列）
feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
target_col = 'A'

# ==================== 1. 数据准备 ====================
df_feat = pd.read_csv(train_feature_path)
df_label = pd.read_csv(train_label_path)

# 按 times 内连接对齐
df_train = pd.merge(df_feat, df_label, on='times', how='inner')
df_train['times'] = pd.to_datetime(df_train['times'])


# 添加时间特征
def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df


df_train = add_time_features(df_train)
all_features = feature_cols + ['hour', 'minute', 'dayofweek', 'month']

X = df_train[all_features].values
y = df_train[target_col].values

# 按时间顺序划分，最后20%做验证
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# ==================== 2. 模型训练 ====================
train_set = lgb.Dataset(X_train, label=y_train, feature_name=all_features)
val_set = lgb.Dataset(X_val, label=y_val, feature_name=all_features, reference=train_set)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'verbose': -1,
}

model = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
    valid_sets=[val_set],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

# 验证集评估
y_val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)
print(f'\n验证集 RMSE: {rmse:.6f}, MAE: {mae:.6f}')

# ==================== 3. 测试集推理 ====================
df_test = pd.read_csv(test_feature_path)
df_test['times'] = pd.to_datetime(df_test['times'])
df_test = add_time_features(df_test)

X_test = df_test[all_features].values
y_test_pred = model.predict(X_test)

df_out = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
df_out.to_csv(output_price_path, index=False)
print(f'推理结果已保存: {output_price_path}, shape={df_out.shape}')


# ==================== 4. 充放电策略生成 ====================
def generate_strategy(price_csv, save_path="output_profit_15min.csv"):
    """
    根据预测的实时价格确定充放电策略，此处略
    """
    pass


generate_strategy(output_price_path, output_power_path)
