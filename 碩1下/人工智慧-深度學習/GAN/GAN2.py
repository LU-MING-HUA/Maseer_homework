from keras.datasets import mnist
from keras.layers import Dense,Dropout,Input
from keras.models import Model,Sequential
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    (x_train, y_train),(_,_)=mnist.load_data()
    x_train=(x_train.astype(np.float32)-127.5)/127.5
    x_train=x_train.reshape(60000, -1)
    return (x_train, y_train)

x_train, y_train = load_data()
print(x_train.shape, y_train.shape)