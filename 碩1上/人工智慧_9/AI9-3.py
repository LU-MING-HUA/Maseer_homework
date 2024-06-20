import numpy as np
import pandas as pd

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
# print(X_train.shape)
# print(Y_train.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
# print(model.score(X_train, Y_train))
# print(model.score(X_test, Y_test))

# print(X_test[0])
p_data = np.array([[6.96215, 0, 18.1, 0, 0.7, 5.713, 97, 1.9265, 24, 666, 20.2, 394.43, 17.11]])
result = model.predict(p_data)
print(result)
# print(Y_test)