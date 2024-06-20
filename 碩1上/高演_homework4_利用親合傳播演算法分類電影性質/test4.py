import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

#設定50個樣本點每個樣本點有2個特徵(2個向量)並大致分為3群
X,y = datasets.make_blobs(n_samples=30, centers=3, n_features=2, random_state= 20, cluster_std = 1.5)


S = -np.square(pairwise_distances(X)) #相似度矩陣S
prefer = np.mean(S) #設定p偏向參數

#Affinity Propagation 模型
model = AffinityPropagation(preference = prefer)
model.fit(X)
labels = model.fit_predict(X)
cluster_center = model.cluster_centers_

silhouette_avg = silhouette_score(S, labels)
print("Silhouette分數為：",silhouette_avg)

#劃出中心與樣本點
plt.figure()
plt.scatter(X[:,0], X[:,1], c = labels)
plt.scatter(cluster_center[:,0], cluster_center[:,1], c = 'r')
plt.axis('equal')
plt.title('Prediction')
plt.show()


