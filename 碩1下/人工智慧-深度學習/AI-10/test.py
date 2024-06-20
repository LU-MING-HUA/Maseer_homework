import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, LSTM

# 讀取CSV文件並刪除任何包含NaN的行
stockdf = pd.read_csv("dataset.csv", index_col=0)
stockdf.dropna(how='any', inplace=True)

# 初始化MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()

# 建立新的DataFrame來存儲縮放後的數據
newdf = stockdf.copy()
newdf['open'] = min_max_scaler.fit_transform(stockdf.open.values.reshape(-1, 1))
newdf['low'] = min_max_scaler.fit_transform(stockdf.low.values.reshape(-1, 1))
newdf['high'] = min_max_scaler.fit_transform(stockdf.high.values.reshape(-1, 1))
newdf['close'] = min_max_scaler.fit_transform(stockdf.close.values.reshape(-1, 1))
newdf['volume'] = min_max_scaler.fit_transform(stockdf.volume.values.reshape(-1, 1))

# 構建時間序列數據
datavalue = newdf.values
result = []
time_frame = 10
for index in range(len(datavalue) - (time_frame + 1)):
    result.append(datavalue[index: index + (time_frame + 1)])

result = np.array(result)

# 切分訓練集和測試集
number_train = round(0.9 * result.shape[0])
X_train = result[:int(number_train), : -1, :]
Y_train = result[:int(number_train), -1, -1]
Y_train_onehot = to_categorical(Y_train)

X_test = result[int(number_train):, : -1, :]
Y_test = result[int(number_train):, -1, -1]
Y_test_onehot = to_categorical(Y_test)

# 構建LSTM模型
model = Sequential()
model.add(LSTM(256, input_shape=(10, 5), return_sequences = True, activation="tanh"))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=False, activation="tanh"))
model.add(Dropout(0.5))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(units=3, activation="softmax"))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(X_train, Y_train_onehot, batch_size=8, epochs=10, validation_split=0.2, verbose=1)
score = model.evaluate(X_test, Y_test_onehot)
# model.save('lstm.h5')
print(score[1])

# 編譯模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, Y_train_onehot, batch_size=8, epochs=10, validation_split=0.2, verbose=1)

# 保存模型
model.save('lstm.h5')

# 評估模型
score = model.evaluate(X_test, Y_test_onehot)
print("Test accuracy:", score[1])

# 加載模型
loaded_model = load_model('lstm.h5')

# 使用 loaded_model 來獲取預測並使用 np.argmax 來確定類別
predictions = loaded_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)
print(Y_test)
