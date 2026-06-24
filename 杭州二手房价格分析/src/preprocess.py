# -*- coding: utf-8 -*-
"""
数据预处理脚本
对房屋户型、所在楼层、建筑类型、朝向、结构、装修进行分类编码。
户型拆分为室数/厅数/厨数/卫数，其他特征标签编码。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import re
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
INPUT_FILE = os.path.join(PROJECT_DIR, 'data', 'raw', '提取后_杭州二手房汇总.xlsx')
OUTPUT_FILE = os.path.join(PROJECT_DIR, 'data', 'processed', '预处理_杭州二手房信息汇总.xlsx')
OUTPUT_CSV = os.path.join(PROJECT_DIR, 'data', 'processed', '预处理_杭州二手房信息汇总.csv')

# ---------------------- 1. 读取原始数据 ----------------------
data = pd.read_excel(INPUT_FILE)
print(f"读取数据: {data.shape[0]} 行, {data.shape[1]} 列")
print(f"列名: {list(data.columns)}")

# ---------------------- 1.5 建筑面积 → 数值 ----------------------
if '建筑面积' in data.columns:
    # 清洗 "124.71㎡" → 124.71
    data['建筑面积'] = data['建筑面积'].astype(str).str.replace('㎡', '', regex=False)
    data['建筑面积'] = pd.to_numeric(data['建筑面积'], errors='coerce')
    print(f'建筑面积 → float (缺失: {data["建筑面积"].isna().sum()} 条)')

# ---------------------- 2. 房屋户型 → 拆分为 室数、厅数、厨数、卫数 ----------------------
if '房屋户型' in data.columns:
    pattern = r'(\d+)室(\d+)厅(\d+)厨(\d+)卫'
    extracted = data['房屋户型'].str.extract(pattern).astype(float)
    extracted.columns = ['室数', '厅数', '厨数', '卫数']
    data = pd.concat([data, extracted], axis=1)
    data = data.drop('房屋户型', axis=1)
    print(f'房屋户型 → 室数, 厅数, 厨数, 卫数')

# ---------------------- 3. 所在楼层 → 楼层类别(标签编码) + 总楼层数 ----------------------
if '所在楼层' in data.columns:
    data['楼层类别'] = data['所在楼层'].str.extract(r'(低楼层|中楼层|高楼层)').fillna('未知楼层')
    data['总楼层数'] = data['所在楼层'].str.extract(r'共(\d+)层').astype(float)
    floor_mapping = {'低楼层': 0, '中楼层': 1, '高楼层': 2, '未知楼层': 3}
    data['楼层类别'] = data['楼层类别'].map(floor_mapping)
    data = data.drop('所在楼层', axis=1)
    print('所在楼层 → 楼层类别(0-3), 总楼层数')

# ---------------------- 4. 建筑类型 → 标签编码 ----------------------
if '建筑类型' in data.columns:
    labels, uniques = pd.factorize(data['建筑类型'])
    mapping = {uniques[i]: i for i in range(len(uniques))}
    data['建筑类型_编码'] = labels
    data = data.drop('建筑类型', axis=1)
    print(f'建筑类型 → 编码 (类别数={len(uniques)}): {mapping}')

# ---------------------- 5. 房屋朝向 → 标签编码 ----------------------
if '房屋朝向' in data.columns:
    labels, uniques = pd.factorize(data['房屋朝向'])
    data['房屋朝向_编码'] = labels
    data = data.drop('房屋朝向', axis=1)
    print(f'房屋朝向 → 编码 (类别数={len(uniques)})')

# ---------------------- 6. 建筑结构 → 标签编码 ----------------------
if '建筑结构' in data.columns:
    labels, uniques = pd.factorize(data['建筑结构'])
    mapping = {uniques[i]: i for i in range(len(uniques))}
    data['建筑结构_编码'] = labels
    data = data.drop('建筑结构', axis=1)
    print(f'建筑结构 → 编码 (类别数={len(uniques)}): {mapping}')

# ---------------------- 7. 装修情况 → 有序标签编码 ----------------------
if '装修情况' in data.columns:
    decoration_order = {'毛坯': 0, '简装': 1, '精装': 2}
    data['装修情况_编码'] = data['装修情况'].map(decoration_order).fillna(3).astype(int)
    data = data.drop('装修情况', axis=1)
    print('装修情况 → 序编码: 毛坯=0, 简装=1, 精装=2')

# ---------------------- 8. 保存 ----------------------
print(f'\n处理后: {data.shape[0]} 行, {data.shape[1]} 列')
print(f'列名: {list(data.columns)}')
data.to_excel(OUTPUT_FILE, index=False)
data.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f'已保存: {OUTPUT_FILE}')
print(f'已保存: {OUTPUT_CSV}')
