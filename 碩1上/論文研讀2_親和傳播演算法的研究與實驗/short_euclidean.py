import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.metrics import pairwise_distances

# 创建一个示例数据集
data = pd.read_excel("ecoli.xlsx")
X = data.iloc[:, 1:8]

# 計算兩兩數據點之間的距離（使用歐幾里得距離）
distances = pairwise_distances(X, metric='euclidean')

# 將距離矩陣轉換成 pandas DataFrame
df = pd.DataFrame(distances)

# 創建一個新的 Excel 工作簿
workbook = Workbook()

# 選擇工作簿的活動工作表
worksheet = workbook.active

# 使用 openpyxl 的功能將 DataFrame 寫入工作表
for row in dataframe_to_rows(df, index=False, header=False):
    worksheet.append(row)

# 儲存 Excel 文件
workbook.save("歐式距離計算2.xlsx")
