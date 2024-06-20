import pandas as pd

stockdf = pd.read_csv("dataset.csv", index_col=0)
stockdf.dropna(how='any', inplace=True)

# print(stockdf)

from sklearn import preprocessing
from keras.utils import to_categorical
min_max_scaler = preprocessing.MinMaxScaler()

newdf = stockdf.copy()
flagdf = stockdf.copy()

newdf['open'] = min_max_scaler.fit_transform(stockdf.open.values.reshape(-1,1))
newdf['low'] = min_max_scaler.fit_transform(stockdf.low.values.reshape(-1,1))
newdf['high'] = min_max_scaler.fit_transform(stockdf.high.values.reshape(-1,1))
newdf['close'] = min_max_scaler.fit_transform(stockdf.close.values.reshape(-1,1))
newdf['volume'] = min_max_scaler.fit_transform(stockdf.volume.values.reshape(-1,1))

# print(stockdf)
# print(newdf)

import numpy as np
datavalue = newdf.values
result = []

# print(datavalue)

time_frame = 10
for index in range(len(datavalue) - (time_frame+1)):
    result.append(datavalue[index: index+(time_frame+1)])

# print(result)
result = np.array(result)
# print(result.shape[0])
# print(result[0])
# print(result[1])

number_train = round(0.9 * result.shape[0])
X_train = result[:int(number_train), : -1, 0:5]
# print(X_train)
Y_train = result[:int(number_train),-1][:,-1]
# print(Y_train)
Y_train_onehot = to_categorical(Y_train)
# print(Y_train_onehot)

X_test = result[int(number_train):, : -1, 0:5]
Y_test = result[int(number_train):,-1][:,-1]
Y_test_onehot = to_categorical(Y_test)
# print(X_test)
# print(Y_test)
# print(Y_test_onehot)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM


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
# print(score[1])

import matplotlib.pyplot as plt
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.plot("Train History")
plt.ylabel("train")
plt.xlabel("epoch")
plt.legend(['train','validation'], loc = 'upper right')
plt.show()

plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.plot("loss History")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(['train','validation'], loc = 'upper right')
plt.show()