import numpy as numpy
import pandas as pd
import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train_image, y_train_label),(x_test_image, y_test_label) = mnist.load_data()
print("train data = ", len(x_train_image))
print("test data = ", len(x_test_image))

fig = plt.gcf()
fig.set_size_inches(2,2)
plt.imshow(x_test_image[61], cmap="binary")
plt.show()

print(x_train_image.shape)
print(x_train_image[61])
print(x_train_image[61][5][9])

x = [
    [0,0,0,0,0,0,0,0],
    [0,0,0,255,255,0,0,0],
    [0,0,0,255,255,0,0,0],
    [0,0,0,255,255,0,0,0],
    [0,0,0,255,255,0,0,0],
    [0,0,0,255,255,0,0,0],
    [0,0,0,255,255,0,0,0],
    [0,0,0,0,0,0,0,0]
]

fig = plt.gcf()
fig.set_size_inches(2,2)
plt.imshow(x, cmap="binary")
plt.show()