# -*- coding: utf-8 -*-
"""
杭州二手房数据合并脚本
遍历 data/raw/ 目录下各区 Excel 文件，提取指定列并合并为单一文件。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
INPUT_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
OUTPUT_FILE = os.path.join(PROJECT_DIR, 'data', 'raw', '提取后_杭州二手房汇总.xlsx')

target_columns = [
    "核心卖点", "小区名称", "区域位置", "总价", "单价",
    "关注度", "房屋户型", "所在楼层", "建筑面积",
    "建筑类型", "房屋朝向", "建筑结构", "装修情况"
]

all_extracted_data = []
excel_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".xlsx") and not f.startswith("提取后")]

if not excel_files:
    print(f"警告：在 {INPUT_DIR} 目录下未找到各区的 Excel 文件！")
else:
    for file in excel_files:
        file_path = os.path.join(INPUT_DIR, file)
        try:
            excel = pd.ExcelFile(file_path)
            for sheet_name in excel.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                existing_cols = [col for col in target_columns if col in df.columns]
                if existing_cols:
                    df_filtered = df[existing_cols].copy()
                    df_filtered["来源文件"] = file
                    df_filtered["来源工作表"] = sheet_name
                    all_extracted_data.append(df_filtered)
                    print(f"提取 {file} - {sheet_name}: {len(existing_cols)} 列")
        except Exception as e:
            print(f"处理 {file} 时出错：{e}，已跳过")

if all_extracted_data:
    combined_df = pd.concat(all_extracted_data, ignore_index=True)
    print(f"合并完成: {len(combined_df)} 条记录 → {OUTPUT_FILE}")
    combined_df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
else:
    print("未提取到有效数据")
