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
print(x_test_image[0])
print(x_Test[0])

y_Test_OneHot = to_categorical(y_test_label)
print(y_test_label[0])
print(y_Test_OneHot[0])


#同5-8