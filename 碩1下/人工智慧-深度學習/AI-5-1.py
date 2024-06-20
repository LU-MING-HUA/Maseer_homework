import numpy as numpy
import pandas as pd
import np_utils
from keras.datasets import mnist
(x_train_image, y_train_label),(x_test_image, y_test_label) = mnist.load_data()
print("train data = ", len(x_train_image))
print("test data = ", len(x_test_image))
