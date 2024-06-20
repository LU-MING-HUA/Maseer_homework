import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances
import pandas as pd


data = pd.read_excel("歐式距離計算_對角線為中值.xlsx") #準備資料集
# X = data.iloc[:, 1:8].to_numpy()  # 轉換為 NumPy 數組型態
# S = -np.square(pairwise_distances(X)) #相似度矩陣
# S = pairwise_distances(X, metric='euclidean') #計算歐基里德相似度矩陣
# prefer = np.mean(S) #找出S矩陣的中值
model = AffinityPropagation(affinity="precomputed", preference = 0.499)
model.fit(data)
labels = model.fit_predict(X)
print(labels)
# cluster_center = model.cluster_centers_
# print(cluster_center)

# from sklearn.metrics import silhouette_score
# silhouette_avg = silhouette_score(X, labels, metric='precomputed')
# print("Silhouette Score:", silhouette_avg)

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# class_cp = X[:143, :]
# class_im = X[144:219, :]
# class_pp = X[220:272, :]
# ax.scatter(class_cp[:, 0], class_cp[:, 1], class_cp[:, 2], label = "cp")
# ax.scatter(class_im[:, 0], class_im[:, 1], class_im[:, 2], label = "im")
# ax.scatter(class_pp[:, 0], class_pp[:, 1], class_pp[:, 2], label = "pp")
# ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# ax.set_yticks([1, 0.8, 0.6, 0.4])
# ax.set_zticks([1, 0.8, 0.6, 0.4, 0.2, 0])
# ax.legend()
# plt.show()


