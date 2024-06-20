import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# 從 "ecoli.xlsx" 資料中讀取數值
data = pd.read_excel("ecoli.xlsx")

# 提取數據集的特徵
X = data.iloc[:, 1:8].to_numpy()  # 轉換成 NumPy 數組型態

# 計算兩數據點之間的距離（使用歐基里德距離公式）並得出一個矩陣
distances = pairwise_distances(X, metric='euclidean')
S = -np.square(distances)
prefer = np.mean(S)
# 使用相似度传播算法
af = AffinityPropagation(preference=prefer, verbose=True, damping=0.55 ,max_iter=200, convergence_iter=1)
af.fit(X)

labels = af.fit_predict(X)
cluster_center = af.cluster_centers_

# 获取聚类标签
# cluster_centers_indices = af.cluster_centers_indices_
# n_clusters = len(cluster_centers_indices)
# labels = af.labels_

# 计算 Silhouette 指标
silhouette_avg = metrics.silhouette_score(X, labels, metric='euclidean')
print("Silhouette Score:", silhouette_avg)

# # 绘制数据点
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
# # 绘制聚类中心
# cluster_centers = X[cluster_centers_indices]
# plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, marker='X', c='red', label='center')
# plt.legend()
# plt.show()
