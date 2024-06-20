import np_utils
import numpy as np
import pandas as pd
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train_image, y_train_label),(x_test_image, y_test_label) = mnist.load_data()
print(y_train_label[100])
print(y_train_label[20])

fig = plt.gcf()
fig.set_size_inches(2,2)
plt.imshow(x_train_image[100], cmap="binary")
plt.show()
plt.imshow(x_train_image[20], cmap="binary")
plt.show()