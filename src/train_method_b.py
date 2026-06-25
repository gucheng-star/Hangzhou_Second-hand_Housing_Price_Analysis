# -*- coding: utf-8 -*-
"""
方法 B：随机森林 (300棵树, Bagging)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', '预处理_杭州二手房信息汇总.xlsx')
IMG_PATH = os.path.join(PROJECT_DIR, 'images', 'method_b_result.png')

feature_cols = ['建筑面积', '室数', '厅数', '厨数', '卫数',
                '楼层类别', '总楼层数', '建筑类型_编码', '房屋朝向_编码',
                '建筑结构_编码', '装修情况_编码']

# 加载
start_time = time.time()
data = pd.read_excel(DATA_PATH)
data = data.dropna(subset=feature_cols + ['总价']).reset_index(drop=True)
print(f"[方法B] 数据: {len(data)} 条, 耗时: {time.time()-start_time:.2f}s")

X = data[feature_cols]
Y = data['总价']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=85)

# 训练
start_time = time.time()
model = RandomForestRegressor(
    n_estimators=300, max_depth=15, min_samples_split=10,
    random_state=85, n_jobs=-1
)
model.fit(Xtrain, Ytrain)
print(f"[方法B] 训练耗时: {time.time()-start_time:.2f}s")

# 评估
yhat = model.predict(Xtest)
r2 = model.score(Xtest, Ytest)
rmse = np.sqrt(mean_squared_error(Ytest, yhat))
mae = mean_absolute_error(Ytest, yhat)

print(f"\n{'='*50}")
print(f"  方法B — 随机森林")
print(f"{'='*50}")
print(f"  R^2 = {r2:.4f}")
print(f"  RMSE = {rmse:.2f} 万元")
print(f"  MAE  = {mae:.2f} 万元")

imp_df = pd.DataFrame({'特征': X.columns, '重要性': model.feature_importances_})
imp_df = imp_df.sort_values('重要性', ascending=False)
for _, row in imp_df.iterrows():
    print(f"    {row['特征']:12s}: {row['重要性']:.4f}")

# 出图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
n = 100
axes[0].plot(range(n), Ytest.values[:n], 'b-', label='Actual', linewidth=1.5)
axes[0].plot(range(n), yhat[:n], 'g--', label='Predicted', linewidth=1.5)
axes[0].set_title(f'Method B: Predicted vs Actual (first {n})')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].scatter(Ytest, yhat, alpha=0.3, s=5, color='green')
lims = [min(Ytest.min(), yhat.min()), max(Ytest.max(), yhat.max())]
axes[1].plot(lims, lims, 'r--', linewidth=1)
axes[1].set_title(f'Method B: Scatter (R^2={r2:.4f})')
axes[1].grid(True, alpha=0.3)

axes[2].barh(imp_df['特征'][::-1], imp_df['重要性'][::-1], color='steelblue')
axes[2].set_title('Method B: Feature Importance')

plt.tight_layout()
plt.savefig(IMG_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[方法B] 图表已保存至: {IMG_PATH}")
