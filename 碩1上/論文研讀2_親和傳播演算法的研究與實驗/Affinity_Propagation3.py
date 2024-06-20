import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from openpyxl import Workbook
from sklearn import datasets
from openpyxl.utils.dataframe import dataframe_to_rows

X,y = datasets.make_blobs(n_samples=10, centers=2, n_features=2, random_state= 20, cluster_std = 1.5)

# 計算數據點之間的距離(歐式距離)
S = pairwise_distances(X, metric='euclidean')

# 步骤 2: 初始化Responsibility矩陣R和Availability矩陣A
n_samples = 10
R = np.zeros((n_samples, n_samples))  
A = np.zeros((n_samples, n_samples))  

max_iters = 100  # 指定最大迭代次數
tolerance = 1e-5  # 指定停止條件，當 R 和 A 矩陣變化小於 tolerance 時停止迭代

damping = 0.6
convergence_iter=15


for iteration in range(max_iters):
        # Update responsibilities
        for i in range(n_samples):
            for k in range(n_samples):
                if i != k:
                    max_val = -np.inf
                    for j in range(n_samples):
                        if j != i:
                            max_val = max(max_val, A[i, j] + S[i, j])
                    R[i, k] = S[i, k] - max_val

        # Update availabilities
        for i in range(n_samples):
            for k in range(n_samples):
                if i != k:
                    total_positive_R = sum(max(0, R[j, k]) for j in range(n_samples) if j != i)
                    A[i, k] = min(0, R[k, k] + total_positive_R)

        # Damping to prevent numerical oscillations
        R = damping * R + (1 - damping) * np.copy(R)
        A = damping * A + (1 - damping) * np.copy(A)
        print(A)
        

        # Check for convergence
        if iteration % convergence_iter == 0:
            labels = np.argmax(R + A, axis=1)
            if np.all(labels == labels[0]):
                break

exemplars = [np.where(labels == i)[0][0] if i in labels else -1 for i in range(n_samples)]
print(exemplars)