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

# 使用相似度傳播算法
af = AffinityPropagation(affinity="precomputed", preference=0.499, verbose=True, damping=0.5,max_iter=200, convergence_iter=1)
af.fit(distances)

# 聚類標籤
cluster_centers_indices = af.cluster_centers_indices_
n_clusters = len(cluster_centers_indices)
labels = af.labels_

# 計算 Silhouette 指標
silhouette_avg = metrics.silhouette_score(distances, labels, metric='precomputed')
print("Silhouette Score:", silhouette_avg)

# 繪製數據點
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
# 繪製聚類中心
cluster_centers = X[cluster_centers_indices]
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, marker='X', c='red', label='center')
plt.legend()
plt.show()
