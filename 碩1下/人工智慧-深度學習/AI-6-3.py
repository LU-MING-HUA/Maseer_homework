import np_utils
import numpy as np
from keras.datasets import mnist
from keras.src.utils.np_utils import to_categorical

(x_train_image, y_train_label),(x_test_image, y_test_label) = mnist.load_data()

x_Train = x_train_image.reshape(60000, 784).astype("float32")
x_Train_normalize = x_Train/255
print(x_train_image[0])
print(x_Train[0])

y_Train_OneHot = to_categorical(y_train_label)
print(y_train_label[0])
print(y_Train_OneHot[0])

#分隔~~~~~~~~~~~~
x_Test = x_test_image.reshape(10000, 784).astype("float32")
x_Test_normalize = x_Test/255

y_Test_OneHot = to_categorical(y_test_label)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) #784輸入層 256隱藏層
model.add(Dense(units=10,kernel_initializer='normal', activation='softmax'))
print(model.summary())