#-----4_6------
import numpy as np
import pandas as pd

data = pd.read_csv("diabetes.csv")
X = data.iloc[:,0:8]
Y = data.iloc[:,8]

# print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape)
print(Y_train.shape)

#-----4_7------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
params = knn.get_params()
print("\n列出knn所有設定參數")
# print(params.items()) #列印參數用
for param, value in params.items(): #列印參數用
    print(f"{param}: {value}")
print()
print("訓練分數：",knn.score(X_train, Y_train))
print("測試分數：",knn.score(X_test, Y_test))

#-----4_8------
data_predict = pd.read_csv("diabetes_test.csv")
# X_pred = data_predict.iloc[:,0:7]
# Y_pred = data_predict.iloc[:,8]
p = data_predict.iloc[:, 0:8]
# print(p)
print(knn.predict(p))
