import np_utils
import numpy as np
from keras.datasets import mnist

(x_train_image, y_train_label),(x_test_image, y_test_label) = mnist.load_data()

print(y_train_label[100])