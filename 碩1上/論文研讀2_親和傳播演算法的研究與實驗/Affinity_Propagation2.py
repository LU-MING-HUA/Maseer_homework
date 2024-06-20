import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# 步骤 1: 准备相似度矩阵S
# 从 "ecoli.xlsx" 文件中读取数据
data = pd.read_excel("ecoli.xlsx")

# 提取数据集的特征，这里假设数据集的第1列至第7列包含了数据
X = data.iloc[:, 1:8].to_numpy()  # 转换为 NumPy 数组

# 计算两两数据点之间的距离（使用欧氏距离）
S = pairwise_distances(X, metric='euclidean')

# 步骤 2: 初始化Responsibility矩阵R和Availability矩阵A
n_samples = 272
R = np.zeros((n_samples, n_samples))  # 初始化Responsibility矩阵
A = np.zeros((n_samples, n_samples))  # 初始化Availability矩阵

max_iters = 100  # 指定最大迭代次數
tolerance = 1e-5  # 指定停止條件，當 R 和 A 矩陣變化小於 tolerance 時停止迭代

damping = 0.8
convergence_iter=15

# 步骤 3: 迭代更新R和A矩阵
R_previous = np.zeros((n_samples, n_samples))  # 初值設定為全零
A_previous = np.zeros((n_samples, n_samples))  # 初值設定為全零

for iteration in range(max_iters):
        # Update responsibilities
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    max_val = max(A[i, k]+S[i, k] for k in range(n_samples) if k != j)
                    R[i, j] = S[i, j] - max_val
                else:
                    R[i, j] = 0

        # Update availabilities
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    total_positive_R = sum(max(0, R[k, j]) for k in range(n_samples) if k != i)
                    A[i, j] = min(0, R[j, j] + total_positive_R)

        # Damping to prevent numerical oscillations
        R = damping * R + (1 - damping) * np.copy(R)
        A = damping * A + (1 - damping) * np.copy(A)

        # Check for convergence
        if iteration % convergence_iter == 0:
            labels = np.argmax(R + A, axis=1)
            if np.all(labels == labels[0]):
                break

exemplars = [np.where(labels == i)[0][0] if i in labels else -1 for i in range(n_samples)]
print(exemplars)