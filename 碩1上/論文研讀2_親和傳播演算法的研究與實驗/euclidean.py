import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# 创建一个示例数据集
data = pd.read_excel("ecoli.xlsx")
X = data.iloc[:, 1:8]

# 計算兩兩數據點之間的距離（使用歐幾里得距離）
distances = pdist(X, 'euclidean')

# 將一維距離數組轉換為二維距離矩陣
distance_matrix = squareform(distances)

#計算出矩陣的中值
median_distance = np.median(distance_matrix)
print(median_distance)


# 將距離矩陣轉換成 pandas DataFrame
df = pd.DataFrame(distance_matrix)

# 創建一個新的 Excel 工作簿
workbook = Workbook()

# 選擇工作簿的活動工作表
worksheet = workbook.active

# 使用 openpyxl 的功能將 DataFrame 寫入工作表
for row in dataframe_to_rows(df, index=False, header=False):
    worksheet.append(row)

# 儲存 Excel 文件
workbook.save("歐式距離計算.xlsx")
